use std::mem::MaybeUninit;
use std::ptr::NonNull;
use std::sync::OnceLock;
use std::time::Instant;

use flate2::{Decompress, FlushDecompress, Status};

use super::metadata::VariantRecord;
use super::profile::{ThreadLocalProfileSnapshot, elapsed_nanoseconds};
use super::sample_selection::SampleSelection;
use super::simd;
use super::trusted;
use super::{BgenError, CompressionType};
use crate::buffer::raw_pointer::OutputBufferAddress;
use crate::preprocess;

mod config;

pub(super) use config::decode_tile_variant_count;
use config::row_major_direct_write_enabled;
pub use config::{set_decode_tile_variant_count, set_row_major_direct_write_enabled};

#[cfg(test)]
mod tests;

const MISSING_SAMPLE_FLAG_MASK: u8 = 0x80;
const PLOIDY_MASK: u8 = 0x3F;

pub(super) struct VariantMajorOutputMatrix<Value> {
    pointer: NonNull<Value>,
    row_value_count: usize,
    row_context: &'static str,
}

impl<Value> VariantMajorOutputMatrix<Value> {
    /// Builds a typed view over a caller-owned variant-major output matrix.
    ///
    /// # Safety
    ///
    /// `output_pointer_address` must point to writable memory with enough initialized
    /// storage for every row requested through this helper. Concurrent workers must
    /// request disjoint variant rows for the same allocation.
    pub(super) unsafe fn from_pointer_address(
        output_pointer_address: OutputBufferAddress,
        row_value_count: usize,
        row_context: &'static str,
    ) -> Result<Self, BgenError> {
        if row_value_count == 0 {
            return Err(BgenError::Range(format!("{row_context} output row length must be positive.")));
        }
        let value_alignment = std::mem::align_of::<Value>();
        let output_pointer_address = output_pointer_address.get();
        if !output_pointer_address.is_multiple_of(value_alignment) {
            return Err(BgenError::Range(format!(
                "{row_context} output pointer is not aligned to {value_alignment} bytes.",
            )));
        }
        let pointer = NonNull::new(output_pointer_address as *mut Value)
            .ok_or_else(|| BgenError::Range(format!("{row_context} output pointer is null.")))?;
        Ok(Self { pointer, row_value_count, row_context })
    }

    pub(super) fn row_mut(&mut self, variant_index: usize) -> Result<&mut [Value], BgenError> {
        let row_offset = variant_index.checked_mul(self.row_value_count).ok_or_else(|| {
            BgenError::Range(format!("Integer overflow while locating {} output row.", self.row_context))
        })?;
        let row_pointer = unsafe {
            // Constructor callers guarantee that the backing allocation spans the requested rows.
            self.pointer.as_ptr().add(row_offset)
        };
        Ok(unsafe { std::slice::from_raw_parts_mut(row_pointer, self.row_value_count) })
    }
}

struct RowMajorOutputMatrix<Value> {
    pointer: NonNull<Value>,
    row_value_count: usize,
    row_context: &'static str,
}

impl<Value> RowMajorOutputMatrix<Value> {
    /// Builds a typed view over a caller-owned row-major output matrix.
    ///
    /// # Safety
    ///
    /// `output_pointer_address` must point to writable memory with enough initialized
    /// storage for every row requested through this helper. Concurrent workers must
    /// request disjoint variant columns or row ranges for the same allocation.
    unsafe fn from_pointer_address(
        output_pointer_address: OutputBufferAddress,
        row_value_count: usize,
        row_context: &'static str,
    ) -> Result<Self, BgenError> {
        if row_value_count == 0 {
            return Err(BgenError::Range(format!("{row_context} output row length must be positive.")));
        }
        let value_alignment = std::mem::align_of::<Value>();
        let output_pointer_address = output_pointer_address.get();
        if !output_pointer_address.is_multiple_of(value_alignment) {
            return Err(BgenError::Range(format!(
                "{row_context} output pointer is not aligned to {value_alignment} bytes.",
            )));
        }
        let pointer = NonNull::new(output_pointer_address as *mut Value)
            .ok_or_else(|| BgenError::Range(format!("{row_context} output pointer is null.")))?;
        Ok(Self { pointer, row_value_count, row_context })
    }

    fn row_mut(&mut self, row_index: usize) -> Result<&mut [Value], BgenError> {
        let row_offset = row_index.checked_mul(self.row_value_count).ok_or_else(|| {
            BgenError::Range(format!("Integer overflow while locating {} output row.", self.row_context))
        })?;
        let row_pointer = unsafe {
            // Constructor callers guarantee that the backing allocation spans the requested rows.
            self.pointer.as_ptr().add(row_offset)
        };
        Ok(unsafe { std::slice::from_raw_parts_mut(row_pointer, self.row_value_count) })
    }

    fn row_range_mut(
        &mut self,
        row_index: usize,
        column_start: usize,
        value_count: usize,
    ) -> Result<&mut [Value], BgenError> {
        let row_context = self.row_context;
        let column_stop = column_start.checked_add(value_count).ok_or_else(|| {
            BgenError::Range(format!("Integer overflow while locating {row_context} output row range."))
        })?;
        let row_values = self.row_mut(row_index)?;
        row_values
            .get_mut(column_start..column_stop)
            .ok_or_else(|| BgenError::Range(format!("{row_context} output row range exceeds the row length.")))
    }

    fn column_mut(&mut self, column_index: usize) -> Result<RowMajorOutputColumnMut<'_, Value>, BgenError> {
        if column_index >= self.row_value_count {
            return Err(BgenError::Range(format!(
                "{} output column {column_index} exceeds the row length {}.",
                self.row_context, self.row_value_count,
            )));
        }
        Ok(RowMajorOutputColumnMut { matrix: self, column_index })
    }
}

struct RowMajorOutputColumnMut<'a, Value> {
    matrix: &'a mut RowMajorOutputMatrix<Value>,
    column_index: usize,
}

impl<Value> RowMajorOutputColumnMut<'_, Value> {
    /// Writes one value in the validated column.
    ///
    /// # Safety
    ///
    /// `row_index` must be within the caller-owned matrix row count covered by the
    /// constructor safety contract. Parallel callers must own disjoint columns or
    /// row spans.
    unsafe fn write_unchecked(&mut self, row_index: usize, value: Value) {
        let value_offset = (row_index * self.matrix.row_value_count) + self.column_index;
        let value_pointer = unsafe {
            // The matrix safety contract covers all rows written through this column view.
            self.matrix.pointer.as_ptr().add(value_offset)
        };
        unsafe {
            value_pointer.write(value);
        }
    }
}

#[derive(Debug)]
pub(super) struct VariantDecodeResult {
    pub(super) profile_snapshot: ThreadLocalProfileSnapshot,
    pub(super) selected_dosage_total: f32,
    pub(super) selected_dosage_square_total: f32,
    pub(super) selected_observation_count: i32,
    pub(super) has_missing_values: bool,
    pub(super) zero_count: i32,
    pub(super) nonzero_count: i32,
    pub(super) homozygous_reference_count: i32,
    pub(super) heterozygous_count: i32,
    pub(super) homozygous_alternate_count: i32,
}

#[derive(Debug)]
pub(super) struct DosageTileDecodeResult {
    pub(super) profile_snapshot: ThreadLocalProfileSnapshot,
    pub(super) selected_dosage_totals: Vec<f32>,
}

pub(super) struct VariantMajorTileStatsMut<'a> {
    pub(super) dosage_sum: &'a mut [f32],
    pub(super) dosage_square_sum: &'a mut [f32],
    pub(super) observation_count: &'a mut [i32],
    pub(super) zero_count: &'a mut [i32],
    pub(super) nonzero_count: &'a mut [i32],
    pub(super) homozygous_reference_count: &'a mut [i32],
    pub(super) heterozygous_count: &'a mut [i32],
    pub(super) homozygous_alternate_count: &'a mut [i32],
}

#[derive(Debug)]
pub(super) struct VariantMajorTileDecodeResult {
    pub(super) profile_snapshot: ThreadLocalProfileSnapshot,
    pub(super) has_missing_values: bool,
}

fn build_variant_decode_result(
    profile_snapshot: ThreadLocalProfileSnapshot,
    selected_dosage_total: f32,
) -> VariantDecodeResult {
    VariantDecodeResult {
        profile_snapshot,
        selected_dosage_total,
        selected_dosage_square_total: 0.0,
        selected_observation_count: 0,
        has_missing_values: false,
        zero_count: 0,
        nonzero_count: 0,
        homozygous_reference_count: 0,
        heterozygous_count: 0,
        homozygous_alternate_count: 0,
    }
}

fn selected_sample_count_to_i32(selected_sample_count: usize) -> Result<i32, BgenError> {
    i32::try_from(selected_sample_count).map_err(|_| {
        BgenError::Range(format!(
            "Selected sample count {selected_sample_count} exceeds the supported i32 statistics range.",
        ))
    })
}

pub(super) fn unphased_eight_bit_dosage_lookup() -> &'static [f32] {
    static UNPHASED_EIGHT_BIT_DOSAGE_LOOKUP: OnceLock<Vec<f32>> = OnceLock::new();
    UNPHASED_EIGHT_BIT_DOSAGE_LOOKUP.get_or_init(|| {
        let reciprocal_scale = 1.0_f32 / 255.0_f32;
        let mut dosage_lookup = Vec::with_capacity(usize::from(u16::MAX) + 1);
        for packed_probability_index in 0..=u16::MAX {
            let homozygous_reference_probability = f32::from(
                u8::try_from(packed_probability_index & 0x00FF).expect("low packed probability byte should fit u8"),
            ) * reciprocal_scale;
            let heterozygous_probability = f32::from(
                u8::try_from((packed_probability_index & 0xFF00) >> 8)
                    .expect("high packed probability byte should fit u8"),
            ) * reciprocal_scale;
            dosage_lookup.push(2.0_f32 - ((2.0_f32 * homozygous_reference_probability) + heterozygous_probability));
        }
        dosage_lookup
    })
}

fn exact_eight_bit_probability_pairs(packed_probability_bytes: &[u8]) -> &[[u8; 2]] {
    let (probability_pairs, []) = packed_probability_bytes.as_chunks::<2>() else {
        unreachable!("8-bit BGEN probability byte slices are built from two bytes per sample");
    };
    probability_pairs
}

pub(super) fn packed_eight_bit_probability_index(
    [homozygous_reference_probability_byte, heterozygous_probability_byte]: [u8; 2],
) -> usize {
    usize::from(homozygous_reference_probability_byte) | (usize::from(heterozygous_probability_byte) << 8)
}

pub(super) fn read_eight_bit_probability_pair(buffer: &[u8], offset: usize) -> Result<[u8; 2], BgenError> {
    let probability_bytes = read_exact_bytes(buffer, offset, 2)?;
    let ([probability_pair], []) = probability_bytes.as_chunks::<2>() else {
        unreachable!("selected 8-bit BGEN probability reads request exactly two bytes");
    };
    Ok(*probability_pair)
}

pub(super) struct ThreadScratch {
    zlib_decompressor: Decompress,
    decompressed_probability_block: Vec<u8>,
    dosage_tile: Vec<f32>,
}

impl Default for ThreadScratch {
    fn default() -> Self {
        Self {
            zlib_decompressor: Decompress::new(true),
            decompressed_probability_block: Vec::new(),
            dosage_tile: Vec::new(),
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn decode_variant_dosage_tile_into_row_major_matrix(
    mmap: &[u8],
    compression_type: CompressionType,
    sample_count: usize,
    sample_selection: &SampleSelection,
    variant_record_chunk: &[VariantRecord],
    output_pointer_address: OutputBufferAddress,
    selected_variant_count: usize,
    tile_variant_start_index: usize,
    profiling_enabled: bool,
    trusted_no_missing_diploid: bool,
    collect_dosage_totals: bool,
    thread_scratch: &mut ThreadScratch,
) -> Result<DosageTileDecodeResult, BgenError> {
    if row_major_direct_write_enabled(profiling_enabled, sample_selection) {
        return decode_variant_dosage_tile_direct_into_row_major_matrix(
            mmap,
            compression_type,
            sample_count,
            sample_selection,
            variant_record_chunk,
            output_pointer_address,
            selected_variant_count,
            tile_variant_start_index,
            profiling_enabled,
            trusted_no_missing_diploid,
            collect_dosage_totals,
            thread_scratch,
        );
    }

    let tile_variant_count = variant_record_chunk.len();
    let tile_value_count = sample_selection
        .selected_sample_count
        .checked_mul(tile_variant_count)
        .ok_or_else(|| BgenError::Range("Integer overflow while allocating a BGEN dosage decode tile.".to_string()))?;
    if thread_scratch.dosage_tile.capacity() < tile_value_count {
        thread_scratch.dosage_tile.reserve(tile_value_count - thread_scratch.dosage_tile.capacity());
    }
    unsafe {
        // Every tile element is overwritten during decode before any reads occur.
        thread_scratch.dosage_tile.set_len(tile_value_count);
    }

    let tile_pointer_address = OutputBufferAddress::from_mut_ptr(thread_scratch.dosage_tile.as_mut_ptr());
    let mut thread_local_profile_snapshot = ThreadLocalProfileSnapshot::default();
    let mut selected_dosage_totals = if collect_dosage_totals { vec![0.0_f32; tile_variant_count] } else { Vec::new() };
    for (tile_variant_index, variant_record) in variant_record_chunk.iter().enumerate() {
        let variant_decode_result = decode_variant_dosages_into_row_major_matrix(
            mmap,
            compression_type,
            sample_count,
            sample_selection,
            variant_record,
            tile_pointer_address,
            tile_variant_index,
            tile_variant_count,
            profiling_enabled,
            trusted_no_missing_diploid,
            collect_dosage_totals,
            thread_scratch,
        )?;
        if collect_dosage_totals {
            selected_dosage_totals[tile_variant_index] = variant_decode_result.selected_dosage_total;
        }
        if profiling_enabled {
            thread_local_profile_snapshot.merge_from(&variant_decode_result.profile_snapshot);
        }
    }
    if profiling_enabled {
        thread_local_profile_snapshot.decode_tile_count += 1;
    }

    let copy_tile_start_time = profiling_enabled.then(Instant::now);
    let mut output_matrix = unsafe {
        RowMajorOutputMatrix::<f32>::from_pointer_address(
            output_pointer_address,
            selected_variant_count,
            "row-major BGEN dosage",
        )?
    };
    for selected_sample_index in 0..sample_selection.selected_sample_count {
        let tile_row_start = selected_sample_index * tile_variant_count;
        let tile_row_stop = tile_row_start + tile_variant_count;
        let output_row_range =
            output_matrix.row_range_mut(selected_sample_index, tile_variant_start_index, tile_variant_count)?;
        output_row_range.copy_from_slice(&thread_scratch.dosage_tile[tile_row_start..tile_row_stop]);
    }
    if let Some(copy_tile_start_time) = copy_tile_start_time {
        thread_local_profile_snapshot.output_write_ns += elapsed_nanoseconds(copy_tile_start_time);
        thread_local_profile_snapshot.output_write_count += 1;
        thread_local_profile_snapshot.output_byte_count +=
            u64::try_from(tile_value_count.checked_mul(std::mem::size_of::<f32>()).ok_or_else(|| {
                BgenError::Range("Integer overflow while profiling BGEN tile copy bytes.".to_string())
            })?)
            .unwrap_or(u64::MAX);
    }

    Ok(DosageTileDecodeResult { profile_snapshot: thread_local_profile_snapshot, selected_dosage_totals })
}

#[allow(clippy::too_many_arguments)]
fn decode_variant_dosage_tile_direct_into_row_major_matrix(
    mmap: &[u8],
    compression_type: CompressionType,
    sample_count: usize,
    sample_selection: &SampleSelection,
    variant_record_chunk: &[VariantRecord],
    output_pointer_address: OutputBufferAddress,
    selected_variant_count: usize,
    tile_variant_start_index: usize,
    profiling_enabled: bool,
    trusted_no_missing_diploid: bool,
    collect_dosage_totals: bool,
    thread_scratch: &mut ThreadScratch,
) -> Result<DosageTileDecodeResult, BgenError> {
    let tile_variant_count = variant_record_chunk.len();
    let mut thread_local_profile_snapshot = ThreadLocalProfileSnapshot::default();
    let mut selected_dosage_totals = if collect_dosage_totals { vec![0.0_f32; tile_variant_count] } else { Vec::new() };
    for (tile_variant_index, variant_record) in variant_record_chunk.iter().enumerate() {
        let variant_decode_result = decode_variant_dosages_into_row_major_matrix(
            mmap,
            compression_type,
            sample_count,
            sample_selection,
            variant_record,
            output_pointer_address,
            tile_variant_start_index + tile_variant_index,
            selected_variant_count,
            profiling_enabled,
            trusted_no_missing_diploid,
            collect_dosage_totals,
            thread_scratch,
        )?;
        if collect_dosage_totals {
            selected_dosage_totals[tile_variant_index] = variant_decode_result.selected_dosage_total;
        }
        if profiling_enabled {
            thread_local_profile_snapshot.merge_from(&variant_decode_result.profile_snapshot);
        }
    }
    if profiling_enabled {
        thread_local_profile_snapshot.decode_tile_count += 1;
    }

    Ok(DosageTileDecodeResult { profile_snapshot: thread_local_profile_snapshot, selected_dosage_totals })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn decode_variant_major_dosage_tile(
    mmap: &[u8],
    compression_type: CompressionType,
    sample_count: usize,
    sample_selection: &SampleSelection,
    variant_record_chunk: &[VariantRecord],
    output_pointer_address: OutputBufferAddress,
    selected_sample_count: usize,
    tile_variant_start_index: usize,
    profiling_enabled: bool,
    trusted_no_missing_diploid: bool,
    tile_stats: &mut VariantMajorTileStatsMut<'_>,
    thread_scratch: &mut ThreadScratch,
) -> Result<VariantMajorTileDecodeResult, BgenError> {
    validate_variant_major_tile_stats_lengths(tile_stats, variant_record_chunk.len())?;
    let mut thread_local_profile_snapshot = ThreadLocalProfileSnapshot::default();
    let mut has_missing_values = false;
    for (tile_variant_index, variant_record) in variant_record_chunk.iter().enumerate() {
        let variant_decode_result = decode_variant_dosages_into_variant_major_matrix(
            mmap,
            compression_type,
            sample_count,
            sample_selection,
            variant_record,
            output_pointer_address,
            tile_variant_start_index + tile_variant_index,
            selected_sample_count,
            profiling_enabled,
            trusted_no_missing_diploid,
            thread_scratch,
        )?;
        tile_stats.dosage_sum[tile_variant_index] = variant_decode_result.selected_dosage_total;
        tile_stats.dosage_square_sum[tile_variant_index] = variant_decode_result.selected_dosage_square_total;
        tile_stats.observation_count[tile_variant_index] = variant_decode_result.selected_observation_count;
        has_missing_values |= variant_decode_result.has_missing_values;
        tile_stats.zero_count[tile_variant_index] = variant_decode_result.zero_count;
        tile_stats.nonzero_count[tile_variant_index] = variant_decode_result.nonzero_count;
        tile_stats.homozygous_reference_count[tile_variant_index] = variant_decode_result.homozygous_reference_count;
        tile_stats.heterozygous_count[tile_variant_index] = variant_decode_result.heterozygous_count;
        tile_stats.homozygous_alternate_count[tile_variant_index] = variant_decode_result.homozygous_alternate_count;
        if profiling_enabled {
            thread_local_profile_snapshot.merge_from(&variant_decode_result.profile_snapshot);
        }
    }
    if profiling_enabled {
        thread_local_profile_snapshot.decode_tile_count += 1;
    }
    Ok(VariantMajorTileDecodeResult { profile_snapshot: thread_local_profile_snapshot, has_missing_values })
}

pub(super) fn validate_variant_major_tile_stats_lengths(
    tile_stats: &VariantMajorTileStatsMut<'_>,
    variant_count: usize,
) -> Result<(), BgenError> {
    if tile_stats.dosage_sum.len() == variant_count
        && tile_stats.dosage_square_sum.len() == variant_count
        && tile_stats.observation_count.len() == variant_count
        && tile_stats.zero_count.len() == variant_count
        && tile_stats.nonzero_count.len() == variant_count
        && tile_stats.homozygous_reference_count.len() == variant_count
        && tile_stats.heterozygous_count.len() == variant_count
        && tile_stats.homozygous_alternate_count.len() == variant_count
    {
        return Ok(());
    }
    Err(BgenError::Range(format!("Variant-major tile stats shape mismatch for {variant_count} variants.")))
}

#[allow(clippy::cast_possible_truncation, clippy::too_many_arguments, clippy::too_many_lines)]
fn decode_variant_dosages_into_row_major_matrix(
    mmap: &[u8],
    compression_type: CompressionType,
    sample_count: usize,
    sample_selection: &SampleSelection,
    variant_record: &VariantRecord,
    output_pointer_address: OutputBufferAddress,
    variant_index: usize,
    variant_count: usize,
    profiling_enabled: bool,
    trusted_no_missing_diploid: bool,
    collect_dosage_totals: bool,
    thread_scratch: &mut ThreadScratch,
) -> Result<VariantDecodeResult, BgenError> {
    let mut thread_local_profile_snapshot = ThreadLocalProfileSnapshot::default();
    let probability_block = read_probability_block(
        mmap,
        compression_type,
        variant_record,
        thread_scratch,
        &mut thread_local_profile_snapshot,
        profiling_enabled,
    )?;
    let block_bytes = probability_block;

    let mut cursor = 0;
    let stored_sample_count = u32_to_usize(read_u32_at(block_bytes, cursor)?)?;
    cursor += 4;
    if stored_sample_count != sample_count {
        return Err(BgenError::InvalidFormat(format!(
            "Variant '{}' stores {stored_sample_count} samples in its probability block, but the file header reports {sample_count}.",
            variant_record.resolved_variant_identifier,
        )));
    }

    let allele_count = read_u16_at(block_bytes, cursor)?;
    cursor += 2;
    if allele_count != 2 {
        return Err(BgenError::UnsupportedFormat(format!(
            "Variant '{}' is not biallelic.",
            variant_record.resolved_variant_identifier,
        )));
    }

    let minimum_ploidy = read_u8_at(block_bytes, cursor)?;
    cursor += 1;
    let maximum_ploidy = read_u8_at(block_bytes, cursor)?;
    cursor += 1;
    if minimum_ploidy != 2 || maximum_ploidy != 2 {
        return Err(BgenError::UnsupportedFormat(format!(
            "Variant '{}' uses ploidy bounds [{minimum_ploidy}, {maximum_ploidy}], but the native Rust reader currently supports diploid BGEN variants only.",
            variant_record.resolved_variant_identifier,
        )));
    }

    let sample_ploidy_and_missingness = read_exact_bytes(block_bytes, cursor, sample_count)?;
    cursor += sample_count;

    let phased_flag = read_u8_at(block_bytes, cursor)?;
    cursor += 1;
    let probability_bit_count = read_u8_at(block_bytes, cursor)?;
    cursor += 1;
    if !(1..=32).contains(&probability_bit_count) {
        return Err(BgenError::InvalidFormat(format!(
            "Variant '{}' uses {probability_bit_count} bits per probability, but BGEN Layout 2 requires a value between 1 and 32.",
            variant_record.resolved_variant_identifier,
        )));
    }

    if phased_flag == 0 && probability_bit_count == 8 {
        return decode_unphased_eight_bit_dosages_into_row_major_matrix(
            sample_ploidy_and_missingness,
            &block_bytes[cursor..],
            sample_selection,
            variant_record,
            output_pointer_address,
            variant_index,
            variant_count,
            profiling_enabled,
            trusted_no_missing_diploid,
            collect_dosage_totals,
            thread_local_profile_snapshot,
        );
    }

    let probability_scale_denominator =
        if probability_bit_count == 32 { f64::from(u32::MAX) } else { f64::from((1_u32 << probability_bit_count) - 1) };
    let probability_decode_start_time = profiling_enabled.then(Instant::now);
    let mut bit_reader = PackedProbabilityReader::new(&block_bytes[cursor..]);
    let mut output_matrix = unsafe {
        RowMajorOutputMatrix::<f32>::from_pointer_address(
            output_pointer_address,
            variant_count,
            "row-major BGEN dosage",
        )?
    };
    let mut output_column = output_matrix.column_mut(variant_index)?;
    let mut selected_dosage_total = 0.0_f32;
    if sample_selection.is_identity {
        let output_write_start_time = profiling_enabled.then(Instant::now);
        for (file_sample_index, ploidy_and_missingness) in sample_ploidy_and_missingness.iter().enumerate() {
            let observed_ploidy = ploidy_and_missingness & PLOIDY_MASK;
            if observed_ploidy != 2 {
                return Err(BgenError::UnsupportedFormat(format!(
                    "Variant '{}' contains a non-diploid sample at file sample index {file_sample_index}. Observed ploidy {observed_ploidy}.",
                    variant_record.resolved_variant_identifier,
                )));
            }
            let is_missing = (ploidy_and_missingness & MISSING_SAMPLE_FLAG_MASK) != 0;

            let dosage_value = match phased_flag {
                0 => {
                    let homozygous_reference_probability =
                        f64::from(bit_reader.read_probability(probability_bit_count)?) / probability_scale_denominator;
                    let heterozygous_probability =
                        f64::from(bit_reader.read_probability(probability_bit_count)?) / probability_scale_denominator;
                    if is_missing {
                        f32::NAN
                    } else {
                        let dosage_value =
                            2.0_f64 - ((2.0 * homozygous_reference_probability) + heterozygous_probability);
                        dosage_value as f32
                    }
                }
                1 => {
                    let first_haplotype_reference_probability =
                        f64::from(bit_reader.read_probability(probability_bit_count)?) / probability_scale_denominator;
                    let second_haplotype_reference_probability =
                        f64::from(bit_reader.read_probability(probability_bit_count)?) / probability_scale_denominator;
                    if is_missing {
                        f32::NAN
                    } else {
                        let dosage_value =
                            2.0_f64 - (first_haplotype_reference_probability + second_haplotype_reference_probability);
                        dosage_value as f32
                    }
                }
                unsupported_flag => {
                    return Err(BgenError::InvalidFormat(format!(
                        "Variant '{}' uses phased flag {unsupported_flag}, but BGEN Layout 2 requires 0 or 1.",
                        variant_record.resolved_variant_identifier,
                    )));
                }
            };

            unsafe {
                // Identity-aligned full-sample reads write one validated variant column across file-order rows.
                output_column.write_unchecked(file_sample_index, dosage_value);
            }
            if collect_dosage_totals && !dosage_value.is_nan() {
                selected_dosage_total += dosage_value;
            }
        }
        if let Some(output_write_start_time) = output_write_start_time {
            thread_local_profile_snapshot.output_write_ns += elapsed_nanoseconds(output_write_start_time);
            thread_local_profile_snapshot.output_write_count += 1;
            thread_local_profile_snapshot.output_byte_count +=
                u64::try_from(sample_ploidy_and_missingness.len().checked_mul(std::mem::size_of::<f32>()).ok_or_else(
                    || BgenError::Range("Integer overflow while profiling BGEN output bytes.".to_string()),
                )?)
                .unwrap_or(u64::MAX);
        }

        if let Some(probability_decode_start_time) = probability_decode_start_time {
            thread_local_profile_snapshot.probability_decode_ns += elapsed_nanoseconds(probability_decode_start_time);
            thread_local_profile_snapshot.probability_decode_count += 1;
        }
        record_variant_decode_if_enabled(&mut thread_local_profile_snapshot, profiling_enabled);

        return Ok(build_variant_decode_result(thread_local_profile_snapshot, selected_dosage_total));
    }

    let output_write_start_time = profiling_enabled.then(Instant::now);
    for (file_sample_index, ploidy_and_missingness) in sample_ploidy_and_missingness.iter().enumerate() {
        let observed_ploidy = ploidy_and_missingness & PLOIDY_MASK;
        if observed_ploidy != 2 {
            return Err(BgenError::UnsupportedFormat(format!(
                "Variant '{}' contains a non-diploid sample at file sample index {file_sample_index}. Observed ploidy {observed_ploidy}.",
                variant_record.resolved_variant_identifier,
            )));
        }
        let is_missing = (ploidy_and_missingness & MISSING_SAMPLE_FLAG_MASK) != 0;

        let dosage_value = match phased_flag {
            0 => {
                let homozygous_reference_probability =
                    f64::from(bit_reader.read_probability(probability_bit_count)?) / probability_scale_denominator;
                let heterozygous_probability =
                    f64::from(bit_reader.read_probability(probability_bit_count)?) / probability_scale_denominator;
                if is_missing {
                    f32::NAN
                } else {
                    let dosage_value = 2.0_f64 - ((2.0 * homozygous_reference_probability) + heterozygous_probability);
                    dosage_value as f32
                }
            }
            1 => {
                let first_haplotype_reference_probability =
                    f64::from(bit_reader.read_probability(probability_bit_count)?) / probability_scale_denominator;
                let second_haplotype_reference_probability =
                    f64::from(bit_reader.read_probability(probability_bit_count)?) / probability_scale_denominator;
                if is_missing {
                    f32::NAN
                } else {
                    let dosage_value =
                        2.0_f64 - (first_haplotype_reference_probability + second_haplotype_reference_probability);
                    dosage_value as f32
                }
            }
            unsupported_flag => {
                return Err(BgenError::InvalidFormat(format!(
                    "Variant '{}' uses phased flag {unsupported_flag}, but BGEN Layout 2 requires 0 or 1.",
                    variant_record.resolved_variant_identifier,
                )));
            }
        };

        let selected_index = sample_selection.file_to_selected_index[file_sample_index];
        if selected_index != usize::MAX {
            unsafe {
                // Selected indices are built by sample selection and map to valid output rows.
                output_column.write_unchecked(selected_index, dosage_value);
            }
            if collect_dosage_totals && !dosage_value.is_nan() {
                selected_dosage_total += dosage_value;
            }
        }
    }
    if let Some(output_write_start_time) = output_write_start_time {
        thread_local_profile_snapshot.output_write_ns += elapsed_nanoseconds(output_write_start_time);
        thread_local_profile_snapshot.output_write_count += 1;
        thread_local_profile_snapshot.output_byte_count += u64::try_from(
            sample_selection
                .selected_sample_count
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| BgenError::Range("Integer overflow while profiling BGEN output bytes.".to_string()))?,
        )
        .unwrap_or(u64::MAX);
    }

    if let Some(probability_decode_start_time) = probability_decode_start_time {
        thread_local_profile_snapshot.probability_decode_ns += elapsed_nanoseconds(probability_decode_start_time);
        thread_local_profile_snapshot.probability_decode_count += 1;
    }

    record_variant_decode_if_enabled(&mut thread_local_profile_snapshot, profiling_enabled);

    Ok(build_variant_decode_result(thread_local_profile_snapshot, selected_dosage_total))
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn decode_unphased_eight_bit_dosages_into_row_major_matrix(
    sample_ploidy_and_missingness: &[u8],
    packed_probability_bytes: &[u8],
    sample_selection: &SampleSelection,
    variant_record: &VariantRecord,
    output_pointer_address: OutputBufferAddress,
    variant_index: usize,
    variant_count: usize,
    profiling_enabled: bool,
    trusted_no_missing_diploid: bool,
    collect_dosage_totals: bool,
    mut thread_local_profile_snapshot: ThreadLocalProfileSnapshot,
) -> Result<VariantDecodeResult, BgenError> {
    let expected_probability_byte_count = sample_ploidy_and_missingness.len().checked_mul(2).ok_or_else(|| {
        BgenError::InvalidFormat("Integer overflow while decoding 8-bit BGEN probabilities.".to_string())
    })?;
    if packed_probability_bytes.len() < expected_probability_byte_count {
        return Err(BgenError::InvalidFormat(format!(
            "Variant '{}' ended before all 8-bit probabilities were decoded.",
            variant_record.resolved_variant_identifier,
        )));
    }

    let dosage_lookup = unphased_eight_bit_dosage_lookup();
    let probability_decode_start_time = profiling_enabled.then(Instant::now);
    let mut output_matrix = unsafe {
        RowMajorOutputMatrix::<f32>::from_pointer_address(
            output_pointer_address,
            variant_count,
            "row-major BGEN dosage",
        )?
    };
    let mut output_column = output_matrix.column_mut(variant_index)?;
    let probability_pairs =
        exact_eight_bit_probability_pairs(&packed_probability_bytes[..expected_probability_byte_count]);
    let all_samples_present =
        trusted_no_missing_diploid || trusted::all_samples_present_diploid(sample_ploidy_and_missingness);
    if trusted_no_missing_diploid {
        debug_assert!(
            trusted::all_samples_present_diploid(sample_ploidy_and_missingness),
            "trusted row-major decode skipped a ploidy scan before validation"
        );
    }
    if all_samples_present {
        let output_write_start_time = profiling_enabled.then(Instant::now);
        let selected_dosage_total = decode_all_present_unphased_eight_bit_row_major_selection(
            probability_pairs,
            sample_selection,
            &mut output_column,
            dosage_lookup,
            collect_dosage_totals,
        )?;
        if let Some(output_write_start_time) = output_write_start_time {
            thread_local_profile_snapshot.output_write_ns += elapsed_nanoseconds(output_write_start_time);
            thread_local_profile_snapshot.output_write_count += 1;
            thread_local_profile_snapshot.output_byte_count += u64::try_from(
                sample_selection.selected_sample_count.checked_mul(std::mem::size_of::<f32>()).ok_or_else(|| {
                    BgenError::Range("Integer overflow while profiling BGEN output bytes.".to_string())
                })?,
            )
            .unwrap_or(u64::MAX);
        }

        if let Some(probability_decode_start_time) = probability_decode_start_time {
            thread_local_profile_snapshot.probability_decode_ns += elapsed_nanoseconds(probability_decode_start_time);
            thread_local_profile_snapshot.probability_decode_count += 1;
        }
        record_variant_decode_if_enabled(&mut thread_local_profile_snapshot, profiling_enabled);

        return Ok(build_variant_decode_result(thread_local_profile_snapshot, selected_dosage_total));
    }

    let mut selected_dosage_total = 0.0_f32;
    if sample_selection.is_identity {
        let output_write_start_time = profiling_enabled.then(Instant::now);
        for (file_sample_index, (ploidy_and_missingness, probability_pair)) in
            sample_ploidy_and_missingness.iter().zip(probability_pairs.iter().copied()).enumerate()
        {
            let observed_ploidy = ploidy_and_missingness & PLOIDY_MASK;
            if observed_ploidy != 2 {
                return Err(BgenError::UnsupportedFormat(format!(
                    "Variant '{}' contains a non-diploid sample in an identity-aligned full-sample read. Observed ploidy {observed_ploidy}.",
                    variant_record.resolved_variant_identifier,
                )));
            }

            let packed_probability_index = packed_eight_bit_probability_index(probability_pair);

            let dosage_value = if (ploidy_and_missingness & MISSING_SAMPLE_FLAG_MASK) != 0 {
                f32::NAN
            } else {
                dosage_lookup[packed_probability_index]
            };
            unsafe {
                // Identity-aligned full-sample reads write file-order rows in the validated output column.
                output_column.write_unchecked(file_sample_index, dosage_value);
            }
            if collect_dosage_totals && !dosage_value.is_nan() {
                selected_dosage_total += dosage_value;
            }
        }
        if let Some(output_write_start_time) = output_write_start_time {
            thread_local_profile_snapshot.output_write_ns += elapsed_nanoseconds(output_write_start_time);
            thread_local_profile_snapshot.output_write_count += 1;
            thread_local_profile_snapshot.output_byte_count +=
                u64::try_from(sample_ploidy_and_missingness.len().checked_mul(std::mem::size_of::<f32>()).ok_or_else(
                    || BgenError::Range("Integer overflow while profiling BGEN output bytes.".to_string()),
                )?)
                .unwrap_or(u64::MAX);
        }

        if let Some(probability_decode_start_time) = probability_decode_start_time {
            thread_local_profile_snapshot.probability_decode_ns += elapsed_nanoseconds(probability_decode_start_time);
            thread_local_profile_snapshot.probability_decode_count += 1;
        }
        record_variant_decode_if_enabled(&mut thread_local_profile_snapshot, profiling_enabled);

        return Ok(build_variant_decode_result(thread_local_profile_snapshot, selected_dosage_total));
    }

    let output_write_start_time = profiling_enabled.then(Instant::now);
    for (file_sample_index, (ploidy_and_missingness, probability_pair)) in
        sample_ploidy_and_missingness.iter().zip(probability_pairs.iter().copied()).enumerate()
    {
        let observed_ploidy = ploidy_and_missingness & PLOIDY_MASK;
        if observed_ploidy != 2 {
            return Err(BgenError::UnsupportedFormat(format!(
                "Variant '{}' contains a non-diploid sample at file sample index {file_sample_index}. Observed ploidy {observed_ploidy}.",
                variant_record.resolved_variant_identifier,
            )));
        }

        let packed_probability_index = packed_eight_bit_probability_index(probability_pair);

        let dosage_value = if (ploidy_and_missingness & MISSING_SAMPLE_FLAG_MASK) != 0 {
            f32::NAN
        } else {
            dosage_lookup[packed_probability_index]
        };

        let selected_index = sample_selection.file_to_selected_index[file_sample_index];
        if selected_index != usize::MAX {
            unsafe {
                // Selected indices are built by sample selection and map to valid output rows.
                output_column.write_unchecked(selected_index, dosage_value);
            }
            if collect_dosage_totals && !dosage_value.is_nan() {
                selected_dosage_total += dosage_value;
            }
        }
    }
    if let Some(output_write_start_time) = output_write_start_time {
        thread_local_profile_snapshot.output_write_ns += elapsed_nanoseconds(output_write_start_time);
        thread_local_profile_snapshot.output_write_count += 1;
        thread_local_profile_snapshot.output_byte_count += u64::try_from(
            sample_selection
                .selected_sample_count
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| BgenError::Range("Integer overflow while profiling BGEN output bytes.".to_string()))?,
        )
        .unwrap_or(u64::MAX);
    }

    if let Some(probability_decode_start_time) = probability_decode_start_time {
        thread_local_profile_snapshot.probability_decode_ns += elapsed_nanoseconds(probability_decode_start_time);
        thread_local_profile_snapshot.probability_decode_count += 1;
    }
    record_variant_decode_if_enabled(&mut thread_local_profile_snapshot, profiling_enabled);

    Ok(build_variant_decode_result(thread_local_profile_snapshot, selected_dosage_total))
}

fn decode_all_present_unphased_eight_bit_row_major_selection(
    probability_pairs: &[[u8; 2]],
    sample_selection: &SampleSelection,
    output_column: &mut RowMajorOutputColumnMut<'_, f32>,
    dosage_lookup: &[f32],
    collect_dosage_totals: bool,
) -> Result<f32, BgenError> {
    let mut selected_dosage_total = 0.0_f32;
    if sample_selection.is_identity {
        for (file_sample_index, probability_pair) in probability_pairs.iter().copied().enumerate() {
            let dosage_value = decode_unphased_eight_bit_row_major_probability_pair(probability_pair, dosage_lookup);
            unsafe {
                // Identity-aligned full-sample reads write file-order rows in the validated output column.
                output_column.write_unchecked(file_sample_index, dosage_value);
            }
            if collect_dosage_totals {
                selected_dosage_total += dosage_value;
            }
        }
        return Ok(selected_dosage_total);
    }

    if let Some(contiguous_file_index_start) = sample_selection.contiguous_file_index_start {
        selected_dosage_total = decode_contiguous_all_present_unphased_eight_bit_row_major_selection(
            probability_pairs,
            contiguous_file_index_start,
            sample_selection.selected_sample_count,
            output_column,
            dosage_lookup,
            collect_dosage_totals,
        )?;
        return Ok(selected_dosage_total);
    }

    if row_major_selection_prefers_sparse_indices(sample_selection, probability_pairs.len()) {
        selected_dosage_total = decode_sparse_all_present_unphased_eight_bit_row_major_selection(
            probability_pairs,
            sample_selection,
            output_column,
            dosage_lookup,
            collect_dosage_totals,
        )?;
    } else {
        selected_dosage_total = decode_dense_all_present_unphased_eight_bit_row_major_selection(
            probability_pairs,
            sample_selection,
            output_column,
            dosage_lookup,
            collect_dosage_totals,
        );
    }
    Ok(selected_dosage_total)
}

fn decode_unphased_eight_bit_row_major_probability_pair(probability_pair: [u8; 2], dosage_lookup: &[f32]) -> f32 {
    dosage_lookup[packed_eight_bit_probability_index(probability_pair)]
}

fn decode_contiguous_all_present_unphased_eight_bit_row_major_selection(
    probability_pairs: &[[u8; 2]],
    contiguous_file_index_start: usize,
    selected_sample_count: usize,
    output_column: &mut RowMajorOutputColumnMut<'_, f32>,
    dosage_lookup: &[f32],
    collect_dosage_totals: bool,
) -> Result<f32, BgenError> {
    let contiguous_file_index_stop = contiguous_file_index_start
        .checked_add(selected_sample_count)
        .ok_or_else(|| BgenError::Range("Integer overflow while slicing contiguous BGEN samples.".to_string()))?;
    let selected_probability_pairs =
        probability_pairs.get(contiguous_file_index_start..contiguous_file_index_stop).ok_or_else(|| {
            BgenError::Range("Contiguous BGEN sample selection exceeds decoded probability pairs.".to_string())
        })?;
    let mut selected_dosage_total = 0.0_f32;
    for (selected_index, probability_pair) in selected_probability_pairs.iter().copied().enumerate() {
        let dosage_value = decode_unphased_eight_bit_row_major_probability_pair(probability_pair, dosage_lookup);
        unsafe {
            // Contiguous selections write selected-order rows to the validated output column.
            output_column.write_unchecked(selected_index, dosage_value);
        }
        if collect_dosage_totals {
            selected_dosage_total += dosage_value;
        }
    }
    Ok(selected_dosage_total)
}

fn decode_sparse_all_present_unphased_eight_bit_row_major_selection(
    probability_pairs: &[[u8; 2]],
    sample_selection: &SampleSelection,
    output_column: &mut RowMajorOutputColumnMut<'_, f32>,
    dosage_lookup: &[f32],
    collect_dosage_totals: bool,
) -> Result<f32, BgenError> {
    let mut selected_dosage_total = 0.0_f32;
    for (selected_index, file_sample_index) in sample_selection.selected_file_indices.iter().copied().enumerate() {
        let probability_pair = probability_pairs.get(file_sample_index).copied().ok_or_else(|| {
            BgenError::Range("Sparse BGEN sample selection exceeds decoded probability pairs.".to_string())
        })?;
        let dosage_value = decode_unphased_eight_bit_row_major_probability_pair(probability_pair, dosage_lookup);
        unsafe {
            // Sparse selected indices are validated by sample selection and map to output rows.
            output_column.write_unchecked(selected_index, dosage_value);
        }
        if collect_dosage_totals {
            selected_dosage_total += dosage_value;
        }
    }
    Ok(selected_dosage_total)
}

fn decode_dense_all_present_unphased_eight_bit_row_major_selection(
    probability_pairs: &[[u8; 2]],
    sample_selection: &SampleSelection,
    output_column: &mut RowMajorOutputColumnMut<'_, f32>,
    dosage_lookup: &[f32],
    collect_dosage_totals: bool,
) -> f32 {
    let mut selected_dosage_total = 0.0_f32;
    for (file_sample_index, probability_pair) in probability_pairs.iter().copied().enumerate() {
        let selected_index = sample_selection.file_to_selected_index[file_sample_index];
        if selected_index == usize::MAX {
            continue;
        }
        let dosage_value = decode_unphased_eight_bit_row_major_probability_pair(probability_pair, dosage_lookup);
        unsafe {
            // Dense-mask selected indices are validated by sample selection and map to output rows.
            output_column.write_unchecked(selected_index, dosage_value);
        }
        if collect_dosage_totals {
            selected_dosage_total += dosage_value;
        }
    }
    selected_dosage_total
}

fn row_major_selection_prefers_sparse_indices(sample_selection: &SampleSelection, file_sample_count: usize) -> bool {
    sample_selection.selected_sample_count.saturating_mul(2) <= file_sample_count
}

#[allow(clippy::cast_possible_truncation, clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
fn decode_variant_dosages_into_variant_major_matrix(
    mmap: &[u8],
    compression_type: CompressionType,
    sample_count: usize,
    sample_selection: &SampleSelection,
    variant_record: &VariantRecord,
    output_pointer_address: OutputBufferAddress,
    variant_index: usize,
    selected_sample_count: usize,
    profiling_enabled: bool,
    trusted_no_missing_diploid: bool,
    thread_scratch: &mut ThreadScratch,
) -> Result<VariantDecodeResult, BgenError> {
    let mut thread_local_profile_snapshot = ThreadLocalProfileSnapshot::default();
    let probability_block = read_probability_block(
        mmap,
        compression_type,
        variant_record,
        thread_scratch,
        &mut thread_local_profile_snapshot,
        profiling_enabled,
    )?;

    let mut cursor = 0;
    let stored_sample_count = u32_to_usize(read_u32_at(probability_block, cursor)?)?;
    cursor += 4;
    if stored_sample_count != sample_count {
        return Err(BgenError::InvalidFormat(format!(
            "Variant '{}' stores {stored_sample_count} samples in its probability block, but the file header reports {sample_count}.",
            variant_record.resolved_variant_identifier,
        )));
    }

    let allele_count = read_u16_at(probability_block, cursor)?;
    cursor += 2;
    if allele_count != 2 {
        return Err(BgenError::UnsupportedFormat(format!(
            "Variant '{}' is not biallelic.",
            variant_record.resolved_variant_identifier,
        )));
    }

    let minimum_ploidy = read_u8_at(probability_block, cursor)?;
    cursor += 1;
    let maximum_ploidy = read_u8_at(probability_block, cursor)?;
    cursor += 1;
    if minimum_ploidy != 2 || maximum_ploidy != 2 {
        return Err(BgenError::UnsupportedFormat(format!(
            "Variant '{}' uses ploidy bounds [{minimum_ploidy}, {maximum_ploidy}], but variant-major reads currently support diploid BGEN variants only.",
            variant_record.resolved_variant_identifier,
        )));
    }

    let sample_ploidy_and_missingness = read_exact_bytes(probability_block, cursor, sample_count)?;
    cursor += sample_count;

    let phased_flag = read_u8_at(probability_block, cursor)?;
    cursor += 1;
    let probability_bit_count = read_u8_at(probability_block, cursor)?;
    cursor += 1;
    if !(1..=32).contains(&probability_bit_count) {
        return Err(BgenError::InvalidFormat(format!(
            "Variant '{}' uses {probability_bit_count} bits per probability, but BGEN Layout 2 requires a value between 1 and 32.",
            variant_record.resolved_variant_identifier,
        )));
    }

    if phased_flag == 0 && probability_bit_count == 8 {
        return decode_unphased_eight_bit_dosages_into_variant_major_matrix(
            sample_ploidy_and_missingness,
            &probability_block[cursor..],
            sample_selection,
            variant_record,
            output_pointer_address,
            variant_index,
            selected_sample_count,
            profiling_enabled,
            trusted_no_missing_diploid,
            thread_local_profile_snapshot,
        );
    }

    let probability_scale_denominator =
        if probability_bit_count == 32 { f64::from(u32::MAX) } else { f64::from((1_u32 << probability_bit_count) - 1) };
    let probability_decode_start_time = profiling_enabled.then(Instant::now);
    let output_write_start_time = profiling_enabled.then(Instant::now);
    let mut bit_reader = PackedProbabilityReader::new(&probability_block[cursor..]);
    selected_sample_count_to_i32(selected_sample_count)?;
    let mut output_matrix = unsafe {
        VariantMajorOutputMatrix::<f32>::from_pointer_address(
            output_pointer_address,
            selected_sample_count,
            "variant-major BGEN",
        )?
    };
    let output_row = output_matrix.row_mut(variant_index)?;
    let all_samples_present =
        trusted_no_missing_diploid || trusted::all_samples_present_diploid(sample_ploidy_and_missingness);
    let mut selected_dosage_total = 0.0_f32;
    let mut selected_dosage_square_total = 0.0_f32;
    let mut selected_observation_count = 0_i32;
    let mut has_missing_values = false;
    let mut zero_count = 0_i32;
    let mut nonzero_count = 0_i32;
    let mut homozygous_reference_count = 0_i32;
    let mut heterozygous_count = 0_i32;
    let mut homozygous_alternate_count = 0_i32;

    for (file_sample_index, ploidy_and_missingness) in sample_ploidy_and_missingness.iter().enumerate() {
        let observed_ploidy = ploidy_and_missingness & PLOIDY_MASK;
        if observed_ploidy != 2 {
            return Err(BgenError::UnsupportedFormat(format!(
                "Variant '{}' contains a non-diploid sample at file sample index {file_sample_index}. Observed ploidy {observed_ploidy}.",
                variant_record.resolved_variant_identifier,
            )));
        }

        let dosage_value = match phased_flag {
            0 => {
                let homozygous_reference_probability =
                    f64::from(bit_reader.read_probability(probability_bit_count)?) / probability_scale_denominator;
                let heterozygous_probability =
                    f64::from(bit_reader.read_probability(probability_bit_count)?) / probability_scale_denominator;
                2.0_f64 - ((2.0 * homozygous_reference_probability) + heterozygous_probability)
            }
            1 => {
                let first_haplotype_reference_probability =
                    f64::from(bit_reader.read_probability(probability_bit_count)?) / probability_scale_denominator;
                let second_haplotype_reference_probability =
                    f64::from(bit_reader.read_probability(probability_bit_count)?) / probability_scale_denominator;
                2.0_f64 - (first_haplotype_reference_probability + second_haplotype_reference_probability)
            }
            unsupported_flag => {
                return Err(BgenError::InvalidFormat(format!(
                    "Variant '{}' uses phased flag {unsupported_flag}, but BGEN Layout 2 requires 0 or 1.",
                    variant_record.resolved_variant_identifier,
                )));
            }
        } as f32;

        let selected_index = if sample_selection.is_identity {
            file_sample_index
        } else {
            sample_selection.file_to_selected_index[file_sample_index]
        };
        if selected_index == usize::MAX {
            continue;
        }

        let is_missing = !all_samples_present && (ploidy_and_missingness & MISSING_SAMPLE_FLAG_MASK) != 0;
        let output_value = if is_missing { f32::NAN } else { dosage_value };
        output_row[selected_index] = output_value;
        if is_missing {
            has_missing_values = true;
            continue;
        }
        selected_dosage_total += dosage_value;
        selected_dosage_square_total += dosage_value * dosage_value;
        selected_observation_count += 1;
        preprocess::increment_dosage_summary_counts(
            dosage_value,
            &mut zero_count,
            &mut nonzero_count,
            &mut homozygous_reference_count,
            &mut heterozygous_count,
            &mut homozygous_alternate_count,
        );
    }

    impute_variant_major_row_if_needed(
        output_row,
        selected_dosage_total,
        selected_observation_count,
        has_missing_values,
    );
    record_variant_major_decode_profile(
        &mut thread_local_profile_snapshot,
        probability_decode_start_time,
        output_write_start_time,
        selected_sample_count,
    )?;

    Ok(VariantDecodeResult {
        profile_snapshot: thread_local_profile_snapshot,
        selected_dosage_total,
        selected_dosage_square_total,
        selected_observation_count,
        has_missing_values,
        zero_count,
        nonzero_count,
        homozygous_reference_count,
        heterozygous_count,
        homozygous_alternate_count,
    })
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn decode_unphased_eight_bit_dosages_into_variant_major_matrix(
    sample_ploidy_and_missingness: &[u8],
    packed_probability_bytes: &[u8],
    sample_selection: &SampleSelection,
    variant_record: &VariantRecord,
    output_pointer_address: OutputBufferAddress,
    variant_index: usize,
    selected_sample_count: usize,
    profiling_enabled: bool,
    trusted_no_missing_diploid: bool,
    mut thread_local_profile_snapshot: ThreadLocalProfileSnapshot,
) -> Result<VariantDecodeResult, BgenError> {
    let expected_probability_byte_count = sample_ploidy_and_missingness.len().checked_mul(2).ok_or_else(|| {
        BgenError::InvalidFormat("Integer overflow while decoding 8-bit BGEN probabilities.".to_string())
    })?;
    if packed_probability_bytes.len() < expected_probability_byte_count {
        return Err(BgenError::InvalidFormat(format!(
            "Variant '{}' ended before all 8-bit probabilities were decoded.",
            variant_record.resolved_variant_identifier,
        )));
    }

    let probability_decode_start_time = profiling_enabled.then(Instant::now);
    let output_write_start_time = profiling_enabled.then(Instant::now);
    selected_sample_count_to_i32(selected_sample_count)?;
    let mut output_matrix = unsafe {
        VariantMajorOutputMatrix::<f32>::from_pointer_address(
            output_pointer_address,
            selected_sample_count,
            "variant-major BGEN",
        )?
    };
    let output_row = output_matrix.row_mut(variant_index)?;
    let all_samples_present =
        trusted_no_missing_diploid || trusted::all_samples_present_diploid(sample_ploidy_and_missingness);

    if sample_selection.is_identity && all_samples_present {
        let decode_summary = simd::decode_unphased_eight_bit_identity_simd_or_scalar(
            &packed_probability_bytes[..expected_probability_byte_count],
            output_row,
        );
        record_variant_major_decode_profile(
            &mut thread_local_profile_snapshot,
            probability_decode_start_time,
            output_write_start_time,
            selected_sample_count,
        )?;

        return Ok(VariantDecodeResult {
            profile_snapshot: thread_local_profile_snapshot,
            selected_dosage_total: decode_summary.selected_dosage_total,
            selected_dosage_square_total: decode_summary.selected_dosage_square_total,
            selected_observation_count: decode_summary.selected_observation_count,
            has_missing_values: false,
            zero_count: decode_summary.zero_count,
            nonzero_count: decode_summary.nonzero_count,
            homozygous_reference_count: decode_summary.homozygous_reference_count,
            heterozygous_count: decode_summary.heterozygous_count,
            homozygous_alternate_count: decode_summary.homozygous_alternate_count,
        });
    }

    let mut selected_dosage_total = 0.0_f32;
    let mut selected_dosage_square_total = 0.0_f32;
    let mut selected_observation_count = 0_i32;
    let mut has_missing_values = false;
    let mut zero_count = 0_i32;
    let mut nonzero_count = 0_i32;
    let mut homozygous_reference_count = 0_i32;
    let mut heterozygous_count = 0_i32;
    let mut homozygous_alternate_count = 0_i32;

    let probability_pairs =
        exact_eight_bit_probability_pairs(&packed_probability_bytes[..expected_probability_byte_count]);
    if !sample_selection.is_identity && all_samples_present {
        if let Some(contiguous_file_index_start) = sample_selection.contiguous_file_index_start {
            let probability_offset = contiguous_file_index_start.checked_mul(2).ok_or_else(|| {
                BgenError::InvalidFormat("Integer overflow while indexing 8-bit BGEN probabilities.".to_string())
            })?;
            let selected_probability_byte_count = selected_sample_count.checked_mul(2).ok_or_else(|| {
                BgenError::InvalidFormat(
                    "Integer overflow while slicing selected 8-bit BGEN probabilities.".to_string(),
                )
            })?;
            let selected_probability_bytes = read_exact_bytes(
                &packed_probability_bytes[..expected_probability_byte_count],
                probability_offset,
                selected_probability_byte_count,
            )?;
            let decode_summary =
                simd::decode_unphased_eight_bit_identity_simd_or_scalar(selected_probability_bytes, output_row);
            record_variant_major_decode_profile(
                &mut thread_local_profile_snapshot,
                probability_decode_start_time,
                output_write_start_time,
                selected_sample_count,
            )?;

            return Ok(VariantDecodeResult {
                profile_snapshot: thread_local_profile_snapshot,
                selected_dosage_total: decode_summary.selected_dosage_total,
                selected_dosage_square_total: decode_summary.selected_dosage_square_total,
                selected_observation_count: decode_summary.selected_observation_count,
                has_missing_values: false,
                zero_count: decode_summary.zero_count,
                nonzero_count: decode_summary.nonzero_count,
                homozygous_reference_count: decode_summary.homozygous_reference_count,
                heterozygous_count: decode_summary.heterozygous_count,
                homozygous_alternate_count: decode_summary.homozygous_alternate_count,
            });
        }

        let dosage_lookup = unphased_eight_bit_dosage_lookup();
        for (selected_index, file_sample_index) in sample_selection.selected_file_indices.iter().copied().enumerate() {
            let probability_offset = file_sample_index.checked_mul(2).ok_or_else(|| {
                BgenError::InvalidFormat("Integer overflow while indexing 8-bit BGEN probabilities.".to_string())
            })?;
            let probability_pair = read_eight_bit_probability_pair(
                &packed_probability_bytes[..expected_probability_byte_count],
                probability_offset,
            )?;
            let packed_probability_index = packed_eight_bit_probability_index(probability_pair);
            let dosage_value = dosage_lookup[packed_probability_index];
            output_row[selected_index] = dosage_value;
            selected_dosage_total += dosage_value;
            selected_dosage_square_total += dosage_value * dosage_value;
            selected_observation_count += 1;
            preprocess::increment_dosage_summary_counts(
                dosage_value,
                &mut zero_count,
                &mut nonzero_count,
                &mut homozygous_reference_count,
                &mut heterozygous_count,
                &mut homozygous_alternate_count,
            );
        }
        record_variant_major_decode_profile(
            &mut thread_local_profile_snapshot,
            probability_decode_start_time,
            output_write_start_time,
            selected_sample_count,
        )?;

        return Ok(VariantDecodeResult {
            profile_snapshot: thread_local_profile_snapshot,
            selected_dosage_total,
            selected_dosage_square_total,
            selected_observation_count,
            has_missing_values: false,
            zero_count,
            nonzero_count,
            homozygous_reference_count,
            heterozygous_count,
            homozygous_alternate_count,
        });
    }

    let dosage_lookup = unphased_eight_bit_dosage_lookup();
    for (file_sample_index, (ploidy_and_missingness, probability_pair)) in
        sample_ploidy_and_missingness.iter().zip(probability_pairs.iter().copied()).enumerate()
    {
        let observed_ploidy = ploidy_and_missingness & PLOIDY_MASK;
        if observed_ploidy != 2 {
            return Err(BgenError::UnsupportedFormat(format!(
                "Variant '{}' contains a non-diploid sample at file sample index {file_sample_index}. Observed ploidy {observed_ploidy}.",
                variant_record.resolved_variant_identifier,
            )));
        }

        let selected_index = if sample_selection.is_identity {
            file_sample_index
        } else {
            sample_selection.file_to_selected_index[file_sample_index]
        };
        if selected_index == usize::MAX {
            continue;
        }

        let packed_probability_index = packed_eight_bit_probability_index(probability_pair);
        let dosage_value = dosage_lookup[packed_probability_index];
        let is_missing = !all_samples_present && (ploidy_and_missingness & MISSING_SAMPLE_FLAG_MASK) != 0;
        let output_value = if is_missing { f32::NAN } else { dosage_value };
        output_row[selected_index] = output_value;
        if is_missing {
            has_missing_values = true;
            continue;
        }
        selected_dosage_total += dosage_value;
        selected_dosage_square_total += dosage_value * dosage_value;
        selected_observation_count += 1;
        preprocess::increment_dosage_summary_counts(
            dosage_value,
            &mut zero_count,
            &mut nonzero_count,
            &mut homozygous_reference_count,
            &mut heterozygous_count,
            &mut homozygous_alternate_count,
        );
    }

    impute_variant_major_row_if_needed(
        output_row,
        selected_dosage_total,
        selected_observation_count,
        has_missing_values,
    );
    record_variant_major_decode_profile(
        &mut thread_local_profile_snapshot,
        probability_decode_start_time,
        output_write_start_time,
        selected_sample_count,
    )?;

    Ok(VariantDecodeResult {
        profile_snapshot: thread_local_profile_snapshot,
        selected_dosage_total,
        selected_dosage_square_total,
        selected_observation_count,
        has_missing_values,
        zero_count,
        nonzero_count,
        homozygous_reference_count,
        heterozygous_count,
        homozygous_alternate_count,
    })
}

#[allow(clippy::cast_precision_loss)]
fn impute_variant_major_row_if_needed(
    output_row: &mut [f32],
    selected_dosage_total: f32,
    selected_observation_count: i32,
    has_missing_values: bool,
) {
    if !has_missing_values {
        return;
    }
    let imputed_dosage_value = selected_dosage_total / selected_observation_count.max(1) as f32;
    for output_value in output_row {
        if output_value.is_nan() {
            *output_value = imputed_dosage_value;
        }
    }
}

fn record_variant_major_decode_profile(
    thread_local_profile_snapshot: &mut ThreadLocalProfileSnapshot,
    probability_decode_start_time: Option<Instant>,
    output_write_start_time: Option<Instant>,
    selected_sample_count: usize,
) -> Result<(), BgenError> {
    if probability_decode_start_time.is_none() && output_write_start_time.is_none() {
        return Ok(());
    }
    if let Some(output_write_start_time) = output_write_start_time {
        thread_local_profile_snapshot.output_write_ns += elapsed_nanoseconds(output_write_start_time);
        thread_local_profile_snapshot.output_write_count += 1;
        thread_local_profile_snapshot.output_byte_count += u64::try_from(
            selected_sample_count
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| BgenError::Range("Integer overflow while profiling BGEN output bytes.".to_string()))?,
        )
        .unwrap_or(u64::MAX);
    }
    if let Some(probability_decode_start_time) = probability_decode_start_time {
        thread_local_profile_snapshot.probability_decode_ns += elapsed_nanoseconds(probability_decode_start_time);
        thread_local_profile_snapshot.probability_decode_count += 1;
    }
    thread_local_profile_snapshot.variant_decode_count += 1;
    Ok(())
}

fn record_variant_decode_if_enabled(
    thread_local_profile_snapshot: &mut ThreadLocalProfileSnapshot,
    profiling_enabled: bool,
) {
    if profiling_enabled {
        thread_local_profile_snapshot.variant_decode_count += 1;
    }
}

pub(super) fn read_probability_block<'a>(
    mmap: &'a [u8],
    compression_type: CompressionType,
    variant_record: &VariantRecord,
    thread_scratch: &'a mut ThreadScratch,
    thread_local_profile_snapshot: &mut ThreadLocalProfileSnapshot,
    profiling_enabled: bool,
) -> Result<&'a [u8], BgenError> {
    let compressed_block_fetch_start_time = profiling_enabled.then(Instant::now);
    match compression_type {
        CompressionType::None => {
            let block_payload = read_exact_bytes(
                mmap,
                variant_record.probability_payload_offset,
                variant_record.probability_payload_length,
            )?;
            if let Some(compressed_block_fetch_start_time) = compressed_block_fetch_start_time {
                thread_local_profile_snapshot.compressed_block_fetch_ns +=
                    elapsed_nanoseconds(compressed_block_fetch_start_time);
                thread_local_profile_snapshot.compressed_block_fetch_count += 1;
                thread_local_profile_snapshot.compressed_byte_count +=
                    u64::try_from(variant_record.probability_payload_length).unwrap_or(u64::MAX);
            }
            if profiling_enabled {
                thread_local_profile_snapshot.uncompressed_byte_count +=
                    u64::try_from(variant_record.declared_uncompressed_block_length).unwrap_or(u64::MAX);
            }
            Ok(block_payload)
        }
        CompressionType::Zlib => {
            let compressed_payload = read_exact_bytes(
                mmap,
                variant_record.probability_payload_offset,
                variant_record.probability_payload_length,
            )?;
            if let Some(compressed_block_fetch_start_time) = compressed_block_fetch_start_time {
                thread_local_profile_snapshot.compressed_block_fetch_ns +=
                    elapsed_nanoseconds(compressed_block_fetch_start_time);
                thread_local_profile_snapshot.compressed_block_fetch_count += 1;
                thread_local_profile_snapshot.compressed_byte_count +=
                    u64::try_from(variant_record.probability_payload_length).unwrap_or(u64::MAX);
            }

            let decompression_start_time = profiling_enabled.then(Instant::now);
            decompress_zlib_block_into_scratch(
                compressed_payload,
                variant_record.declared_uncompressed_block_length,
                thread_scratch,
            )?;
            if let Some(decompression_start_time) = decompression_start_time {
                thread_local_profile_snapshot.decompression_ns += elapsed_nanoseconds(decompression_start_time);
                thread_local_profile_snapshot.decompression_count += 1;
            }
            if profiling_enabled {
                thread_local_profile_snapshot.uncompressed_byte_count +=
                    u64::try_from(variant_record.declared_uncompressed_block_length).unwrap_or(u64::MAX);
                thread_local_profile_snapshot.zlib_stream_count += 1;
            }
            Ok(thread_scratch.decompressed_probability_block.as_slice())
        }
    }
}

fn decompress_zlib_block_into_scratch(
    compressed_payload: &[u8],
    expected_length: usize,
    thread_scratch: &mut ThreadScratch,
) -> Result<(), BgenError> {
    thread_scratch.decompressed_probability_block.clear();
    if thread_scratch.decompressed_probability_block.capacity() < expected_length {
        thread_scratch
            .decompressed_probability_block
            .reserve(expected_length - thread_scratch.decompressed_probability_block.capacity());
    }
    thread_scratch.zlib_decompressor.reset(true);
    let total_output_before = thread_scratch.zlib_decompressor.total_out();
    let output_buffer: &mut [MaybeUninit<u8>] =
        &mut thread_scratch.decompressed_probability_block.spare_capacity_mut()[..expected_length];
    let status = thread_scratch
        .zlib_decompressor
        .decompress_uninit(compressed_payload, output_buffer, FlushDecompress::Finish)
        .map_err(std::io::Error::from)?;
    if status != Status::StreamEnd {
        return Err(BgenError::InvalidFormat(
            "Zlib-compressed BGEN block did not terminate at stream end.".to_string(),
        ));
    }
    let decompressed_length = usize::try_from(thread_scratch.zlib_decompressor.total_out() - total_output_before)
        .map_err(|_| BgenError::InvalidFormat("Decoded zlib block length does not fit into usize.".to_string()))?;
    if decompressed_length != expected_length {
        return Err(BgenError::InvalidFormat(format!(
            "Zlib-compressed BGEN block expanded to {decompressed_length} bytes, but the header declared {expected_length} bytes.",
        )));
    }
    unsafe {
        thread_scratch.decompressed_probability_block.set_len(decompressed_length);
    }
    Ok(())
}

struct PackedProbabilityReader<'a> {
    packed_probability_bytes: &'a [u8],
    byte_offset: usize,
    bit_buffer: u64,
    buffered_bit_count: u8,
}

impl<'a> PackedProbabilityReader<'a> {
    fn new(packed_probability_bytes: &'a [u8]) -> Self {
        Self { packed_probability_bytes, byte_offset: 0, bit_buffer: 0, buffered_bit_count: 0 }
    }

    fn read_probability(&mut self, bit_count: u8) -> Result<u32, BgenError> {
        while self.buffered_bit_count < bit_count {
            let next_probability_byte = self.packed_probability_bytes.get(self.byte_offset).ok_or_else(|| {
                BgenError::InvalidFormat(
                    "Packed BGEN probability stream ended before all probabilities were decoded.".to_string(),
                )
            })?;
            self.bit_buffer |= u64::from(*next_probability_byte) << self.buffered_bit_count;
            self.buffered_bit_count += 8;
            self.byte_offset += 1;
        }

        let mask = if bit_count == 32 { u64::from(u32::MAX) } else { (1_u64 << bit_count) - 1 };
        let probability_value =
            u32::try_from(self.bit_buffer & mask).expect("masked BGEN probability value should fit u32");
        self.bit_buffer >>= bit_count;
        self.buffered_bit_count -= bit_count;
        Ok(probability_value)
    }
}

pub(super) fn read_u8_at(buffer: &[u8], offset: usize) -> Result<u8, BgenError> {
    Ok(*read_exact_bytes(buffer, offset, 1)?
        .first()
        .ok_or_else(|| BgenError::InvalidFormat("Unexpected empty byte slice.".to_string()))?)
}

pub(super) fn read_u16_at(buffer: &[u8], offset: usize) -> Result<u16, BgenError> {
    let bytes = read_exact_bytes(buffer, offset, 2)?;
    let byte_array: [u8; 2] = bytes
        .try_into()
        .map_err(|_| BgenError::InvalidFormat("Failed to decode a two-byte integer from the BGEN file.".to_string()))?;
    Ok(u16::from_le_bytes(byte_array))
}

pub(super) fn read_u32_at(buffer: &[u8], offset: usize) -> Result<u32, BgenError> {
    let bytes = read_exact_bytes(buffer, offset, 4)?;
    let byte_array: [u8; 4] = bytes.try_into().map_err(|_| {
        BgenError::InvalidFormat("Failed to decode a four-byte integer from the BGEN file.".to_string())
    })?;
    Ok(u32::from_le_bytes(byte_array))
}

pub(super) fn read_exact_bytes(buffer: &[u8], offset: usize, length: usize) -> Result<&[u8], BgenError> {
    let stop = offset
        .checked_add(length)
        .ok_or_else(|| BgenError::InvalidFormat("Integer overflow while slicing BGEN file bytes.".to_string()))?;
    buffer
        .get(offset..stop)
        .ok_or_else(|| BgenError::InvalidFormat("Unexpected end of file while reading BGEN bytes.".to_string()))
}

pub(super) fn u32_to_usize(value: u32) -> Result<usize, BgenError> {
    usize::try_from(value).map_err(|_| {
        BgenError::InvalidFormat(format!(
            "BGEN integer value {value} does not fit into the native platform usize type.",
        ))
    })
}
