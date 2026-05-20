use std::mem::MaybeUninit;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use flate2::{Decompress, FlushDecompress, Status};

use super::metadata::VariantRecord;
use super::profile::{ThreadLocalProfileSnapshot, elapsed_nanoseconds};
use super::sample_selection::SampleSelection;
use super::trusted;
use super::{BgenError, CompressionType};
use crate::genotype::preprocess;

const MISSING_SAMPLE_FLAG_MASK: u8 = 0x80;
const PLOIDY_MASK: u8 = 0x3F;
const DEFAULT_DECODE_TILE_VARIANT_COUNT: usize = 64;
static DECODE_TILE_VARIANT_COUNT: AtomicUsize = AtomicUsize::new(DEFAULT_DECODE_TILE_VARIANT_COUNT);
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
    pub(super) selected_dosage_square_totals: Vec<f32>,
    pub(super) selected_observation_counts: Vec<i32>,
    pub(super) has_missing_values: bool,
    pub(super) zero_counts: Vec<i32>,
    pub(super) nonzero_counts: Vec<i32>,
    pub(super) homozygous_reference_counts: Vec<i32>,
    pub(super) heterozygous_counts: Vec<i32>,
    pub(super) homozygous_alternate_counts: Vec<i32>,
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

pub(super) fn decode_tile_variant_count() -> usize {
    DECODE_TILE_VARIANT_COUNT.load(Ordering::Relaxed)
}

#[allow(clippy::missing_errors_doc)]
pub fn set_decode_tile_variant_count(tile_variant_count: usize) -> Result<(), BgenError> {
    if tile_variant_count == 0 {
        return Err(BgenError::Range("BGEN decode tile variant count must be positive.".to_string()));
    }
    DECODE_TILE_VARIANT_COUNT.store(tile_variant_count, Ordering::Relaxed);
    Ok(())
}

pub(super) fn unphased_eight_bit_dosage_lookup() -> &'static [f32] {
    static UNPHASED_EIGHT_BIT_DOSAGE_LOOKUP: OnceLock<Vec<f32>> = OnceLock::new();
    UNPHASED_EIGHT_BIT_DOSAGE_LOOKUP.get_or_init(|| {
        let reciprocal_scale = 1.0_f32 / 255.0_f32;
        let mut dosage_lookup = Vec::with_capacity(usize::from(u16::MAX) + 1);
        for packed_probability_index in 0..=u16::MAX {
            let homozygous_reference_probability =
                f32::from((packed_probability_index & 0x00FF) as u8) * reciprocal_scale;
            let heterozygous_probability =
                f32::from(((packed_probability_index & 0xFF00) >> 8) as u8) * reciprocal_scale;
            dosage_lookup.push(2.0_f32 - ((2.0_f32 * homozygous_reference_probability) + heterozygous_probability));
        }
        dosage_lookup
    })
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
    output_pointer_address: usize,
    selected_variant_count: usize,
    tile_variant_start_index: usize,
    profiling_enabled: bool,
    trusted_no_missing_diploid: bool,
    collect_dosage_totals: bool,
    thread_scratch: &mut ThreadScratch,
) -> Result<DosageTileDecodeResult, BgenError> {
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

    let tile_pointer_address = thread_scratch.dosage_tile.as_mut_ptr() as usize;
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
        let variant_profile_snapshot = variant_decode_result.profile_snapshot;
        if collect_dosage_totals {
            selected_dosage_totals[tile_variant_index] = variant_decode_result.selected_dosage_total;
        }
        thread_local_profile_snapshot.compressed_block_fetch_ns += variant_profile_snapshot.compressed_block_fetch_ns;
        thread_local_profile_snapshot.compressed_block_fetch_count +=
            variant_profile_snapshot.compressed_block_fetch_count;
        thread_local_profile_snapshot.compressed_byte_count += variant_profile_snapshot.compressed_byte_count;
        thread_local_profile_snapshot.decompression_ns += variant_profile_snapshot.decompression_ns;
        thread_local_profile_snapshot.decompression_count += variant_profile_snapshot.decompression_count;
        thread_local_profile_snapshot.uncompressed_byte_count += variant_profile_snapshot.uncompressed_byte_count;
        thread_local_profile_snapshot.zlib_stream_count += variant_profile_snapshot.zlib_stream_count;
        thread_local_profile_snapshot.probability_decode_ns += variant_profile_snapshot.probability_decode_ns;
        thread_local_profile_snapshot.probability_decode_count += variant_profile_snapshot.probability_decode_count;
        thread_local_profile_snapshot.variant_decode_count += variant_profile_snapshot.variant_decode_count;
        thread_local_profile_snapshot.output_write_ns += variant_profile_snapshot.output_write_ns;
        thread_local_profile_snapshot.output_write_count += variant_profile_snapshot.output_write_count;
        thread_local_profile_snapshot.output_byte_count += variant_profile_snapshot.output_byte_count;
    }
    thread_local_profile_snapshot.decode_tile_count += 1;

    let copy_tile_start_time = profiling_enabled.then(Instant::now);
    let output_pointer = output_pointer_address as *mut f32;
    for selected_sample_index in 0..sample_selection.selected_sample_count {
        let tile_row_start = selected_sample_index * tile_variant_count;
        let output_row_start = (selected_sample_index * selected_variant_count) + tile_variant_start_index;
        unsafe {
            // Each parallel worker owns a disjoint contiguous variant span in every output row.
            std::ptr::copy_nonoverlapping(
                thread_scratch.dosage_tile.as_ptr().add(tile_row_start),
                output_pointer.add(output_row_start),
                tile_variant_count,
            );
        }
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

    Ok(DosageTileDecodeResult {
        profile_snapshot: thread_local_profile_snapshot,
        selected_dosage_totals,
        selected_dosage_square_totals: Vec::new(),
        selected_observation_counts: Vec::new(),
        has_missing_values: false,
        zero_counts: Vec::new(),
        nonzero_counts: Vec::new(),
        homozygous_reference_counts: Vec::new(),
        heterozygous_counts: Vec::new(),
        homozygous_alternate_counts: Vec::new(),
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn decode_variant_major_dosage_tile(
    mmap: &[u8],
    compression_type: CompressionType,
    sample_count: usize,
    sample_selection: &SampleSelection,
    variant_record_chunk: &[VariantRecord],
    output_pointer_address: usize,
    selected_sample_count: usize,
    tile_variant_start_index: usize,
    profiling_enabled: bool,
    trusted_no_missing_diploid: bool,
    thread_scratch: &mut ThreadScratch,
) -> Result<DosageTileDecodeResult, BgenError> {
    let mut thread_local_profile_snapshot = ThreadLocalProfileSnapshot::default();
    let mut selected_dosage_totals = vec![0.0_f32; variant_record_chunk.len()];
    let mut selected_dosage_square_totals = vec![0.0_f32; variant_record_chunk.len()];
    let mut selected_observation_counts = vec![0_i32; variant_record_chunk.len()];
    let mut zero_counts = vec![0_i32; variant_record_chunk.len()];
    let mut nonzero_counts = vec![0_i32; variant_record_chunk.len()];
    let mut homozygous_reference_counts = vec![0_i32; variant_record_chunk.len()];
    let mut heterozygous_counts = vec![0_i32; variant_record_chunk.len()];
    let mut homozygous_alternate_counts = vec![0_i32; variant_record_chunk.len()];
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
        let variant_profile_snapshot = variant_decode_result.profile_snapshot;
        selected_dosage_totals[tile_variant_index] = variant_decode_result.selected_dosage_total;
        selected_dosage_square_totals[tile_variant_index] = variant_decode_result.selected_dosage_square_total;
        selected_observation_counts[tile_variant_index] = variant_decode_result.selected_observation_count;
        has_missing_values |= variant_decode_result.has_missing_values;
        zero_counts[tile_variant_index] = variant_decode_result.zero_count;
        nonzero_counts[tile_variant_index] = variant_decode_result.nonzero_count;
        homozygous_reference_counts[tile_variant_index] = variant_decode_result.homozygous_reference_count;
        heterozygous_counts[tile_variant_index] = variant_decode_result.heterozygous_count;
        homozygous_alternate_counts[tile_variant_index] = variant_decode_result.homozygous_alternate_count;
        thread_local_profile_snapshot.compressed_block_fetch_ns += variant_profile_snapshot.compressed_block_fetch_ns;
        thread_local_profile_snapshot.compressed_block_fetch_count +=
            variant_profile_snapshot.compressed_block_fetch_count;
        thread_local_profile_snapshot.compressed_byte_count += variant_profile_snapshot.compressed_byte_count;
        thread_local_profile_snapshot.decompression_ns += variant_profile_snapshot.decompression_ns;
        thread_local_profile_snapshot.decompression_count += variant_profile_snapshot.decompression_count;
        thread_local_profile_snapshot.uncompressed_byte_count += variant_profile_snapshot.uncompressed_byte_count;
        thread_local_profile_snapshot.zlib_stream_count += variant_profile_snapshot.zlib_stream_count;
        thread_local_profile_snapshot.probability_decode_ns += variant_profile_snapshot.probability_decode_ns;
        thread_local_profile_snapshot.probability_decode_count += variant_profile_snapshot.probability_decode_count;
        thread_local_profile_snapshot.variant_decode_count += variant_profile_snapshot.variant_decode_count;
        thread_local_profile_snapshot.output_write_ns += variant_profile_snapshot.output_write_ns;
        thread_local_profile_snapshot.output_write_count += variant_profile_snapshot.output_write_count;
        thread_local_profile_snapshot.output_byte_count += variant_profile_snapshot.output_byte_count;
    }
    thread_local_profile_snapshot.decode_tile_count += 1;
    Ok(DosageTileDecodeResult {
        profile_snapshot: thread_local_profile_snapshot,
        selected_dosage_totals,
        selected_dosage_square_totals,
        selected_observation_counts,
        has_missing_values,
        zero_counts,
        nonzero_counts,
        homozygous_reference_counts,
        heterozygous_counts,
        homozygous_alternate_counts,
    })
}

#[allow(clippy::cast_possible_truncation, clippy::too_many_arguments, clippy::too_many_lines)]
fn decode_variant_dosages_into_row_major_matrix(
    mmap: &[u8],
    compression_type: CompressionType,
    sample_count: usize,
    sample_selection: &SampleSelection,
    variant_record: &VariantRecord,
    output_pointer_address: usize,
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
    let output_pointer = output_pointer_address as *mut f32;
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

            let output_offset = (file_sample_index * variant_count) + variant_index;
            unsafe {
                // Identity-aligned full-sample reads map file-order rows directly into output rows.
                output_pointer.add(output_offset).write(dosage_value);
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
        thread_local_profile_snapshot.variant_decode_count += 1;

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
            let output_offset = (selected_index * variant_count) + variant_index;
            unsafe {
                // Each parallel worker owns one distinct variant column, so these writes do not overlap.
                output_pointer.add(output_offset).write(dosage_value);
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

    thread_local_profile_snapshot.variant_decode_count += 1;

    Ok(build_variant_decode_result(thread_local_profile_snapshot, selected_dosage_total))
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn decode_unphased_eight_bit_dosages_into_row_major_matrix(
    sample_ploidy_and_missingness: &[u8],
    packed_probability_bytes: &[u8],
    sample_selection: &SampleSelection,
    variant_record: &VariantRecord,
    output_pointer_address: usize,
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
    let output_pointer = output_pointer_address as *mut f32;
    let probability_pairs = packed_probability_bytes[..expected_probability_byte_count].chunks_exact(2);
    let all_samples_present =
        trusted_no_missing_diploid || trusted::all_samples_present_diploid(sample_ploidy_and_missingness);
    let mut selected_dosage_total = 0.0_f32;
    if sample_selection.is_identity && all_samples_present {
        let output_write_start_time = profiling_enabled.then(Instant::now);
        let mut output_row_pointer = unsafe { output_pointer.add(variant_index) };
        for probability_pair in probability_pairs {
            let packed_probability_index = usize::from(probability_pair[0]) | (usize::from(probability_pair[1]) << 8);
            let dosage_value = dosage_lookup[packed_probability_index];
            unsafe {
                // Identity-aligned full-sample reads map file-order rows directly into output rows.
                output_row_pointer.write(dosage_value);
                output_row_pointer = output_row_pointer.add(variant_count);
            }
            if collect_dosage_totals {
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
        thread_local_profile_snapshot.variant_decode_count += 1;

        return Ok(build_variant_decode_result(thread_local_profile_snapshot, selected_dosage_total));
    }
    if sample_selection.is_identity {
        let output_write_start_time = profiling_enabled.then(Instant::now);
        let mut output_row_pointer = unsafe { output_pointer.add(variant_index) };
        for (ploidy_and_missingness, probability_pair) in sample_ploidy_and_missingness.iter().zip(probability_pairs) {
            let observed_ploidy = ploidy_and_missingness & PLOIDY_MASK;
            if observed_ploidy != 2 {
                return Err(BgenError::UnsupportedFormat(format!(
                    "Variant '{}' contains a non-diploid sample in an identity-aligned full-sample read. Observed ploidy {observed_ploidy}.",
                    variant_record.resolved_variant_identifier,
                )));
            }

            let packed_probability_index = usize::from(probability_pair[0]) | (usize::from(probability_pair[1]) << 8);

            let dosage_value = if (ploidy_and_missingness & MISSING_SAMPLE_FLAG_MASK) != 0 {
                f32::NAN
            } else {
                dosage_lookup[packed_probability_index]
            };
            unsafe {
                // Identity-aligned full-sample reads map file-order rows directly into output rows.
                output_row_pointer.write(dosage_value);
                output_row_pointer = output_row_pointer.add(variant_count);
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
        thread_local_profile_snapshot.variant_decode_count += 1;

        return Ok(build_variant_decode_result(thread_local_profile_snapshot, selected_dosage_total));
    }

    if all_samples_present {
        let output_write_start_time = profiling_enabled.then(Instant::now);
        for (file_sample_index, probability_pair) in probability_pairs.enumerate() {
            let packed_probability_index = usize::from(probability_pair[0]) | (usize::from(probability_pair[1]) << 8);
            let dosage_value = dosage_lookup[packed_probability_index];

            let selected_index = sample_selection.file_to_selected_index[file_sample_index];
            if selected_index != usize::MAX {
                let output_offset = (selected_index * variant_count) + variant_index;
                unsafe {
                    // Each parallel worker owns one distinct variant column, so these writes do not overlap.
                    output_pointer.add(output_offset).write(dosage_value);
                }
                if collect_dosage_totals {
                    selected_dosage_total += dosage_value;
                }
            }
        }
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
        thread_local_profile_snapshot.variant_decode_count += 1;

        return Ok(build_variant_decode_result(thread_local_profile_snapshot, selected_dosage_total));
    }

    let output_write_start_time = profiling_enabled.then(Instant::now);
    for (file_sample_index, (ploidy_and_missingness, probability_pair)) in
        sample_ploidy_and_missingness.iter().zip(probability_pairs).enumerate()
    {
        let observed_ploidy = ploidy_and_missingness & PLOIDY_MASK;
        if observed_ploidy != 2 {
            return Err(BgenError::UnsupportedFormat(format!(
                "Variant '{}' contains a non-diploid sample at file sample index {file_sample_index}. Observed ploidy {observed_ploidy}.",
                variant_record.resolved_variant_identifier,
            )));
        }

        let packed_probability_index = usize::from(probability_pair[0]) | (usize::from(probability_pair[1]) << 8);

        let dosage_value = if (ploidy_and_missingness & MISSING_SAMPLE_FLAG_MASK) != 0 {
            f32::NAN
        } else {
            dosage_lookup[packed_probability_index]
        };

        let selected_index = sample_selection.file_to_selected_index[file_sample_index];
        if selected_index != usize::MAX {
            let output_offset = (selected_index * variant_count) + variant_index;
            unsafe {
                // Each parallel worker owns one distinct variant column, so these writes do not overlap.
                output_pointer.add(output_offset).write(dosage_value);
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
    thread_local_profile_snapshot.variant_decode_count += 1;

    Ok(build_variant_decode_result(thread_local_profile_snapshot, selected_dosage_total))
}

#[allow(clippy::cast_possible_truncation, clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
fn decode_variant_dosages_into_variant_major_matrix(
    mmap: &[u8],
    compression_type: CompressionType,
    sample_count: usize,
    sample_selection: &SampleSelection,
    variant_record: &VariantRecord,
    output_pointer_address: usize,
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
    let output_pointer = output_pointer_address as *mut f32;
    let variant_row_offset = variant_index.checked_mul(selected_sample_count).ok_or_else(|| {
        BgenError::Range("Integer overflow while locating variant-major BGEN output row.".to_string())
    })?;
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
        unsafe {
            // Each parallel worker owns a distinct variant row in the variant-major output matrix.
            output_pointer.add(variant_row_offset + selected_index).write(output_value);
        }
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
        output_pointer,
        variant_row_offset,
        selected_sample_count,
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

#[allow(clippy::too_many_arguments)]
fn decode_unphased_eight_bit_dosages_into_variant_major_matrix(
    sample_ploidy_and_missingness: &[u8],
    packed_probability_bytes: &[u8],
    sample_selection: &SampleSelection,
    variant_record: &VariantRecord,
    output_pointer_address: usize,
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

    let dosage_lookup = unphased_eight_bit_dosage_lookup();
    let probability_decode_start_time = profiling_enabled.then(Instant::now);
    let output_write_start_time = profiling_enabled.then(Instant::now);
    let output_pointer = output_pointer_address as *mut f32;
    let variant_row_offset = variant_index.checked_mul(selected_sample_count).ok_or_else(|| {
        BgenError::Range("Integer overflow while locating variant-major BGEN output row.".to_string())
    })?;
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

    for (file_sample_index, (ploidy_and_missingness, probability_pair)) in sample_ploidy_and_missingness
        .iter()
        .zip(packed_probability_bytes[..expected_probability_byte_count].chunks_exact(2))
        .enumerate()
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

        let packed_probability_index = usize::from(probability_pair[0]) | (usize::from(probability_pair[1]) << 8);
        let dosage_value = dosage_lookup[packed_probability_index];
        let is_missing = !all_samples_present && (ploidy_and_missingness & MISSING_SAMPLE_FLAG_MASK) != 0;
        let output_value = if is_missing { f32::NAN } else { dosage_value };
        unsafe {
            // Each parallel worker owns a distinct variant row in the variant-major output matrix.
            output_pointer.add(variant_row_offset + selected_index).write(output_value);
        }
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
        output_pointer,
        variant_row_offset,
        selected_sample_count,
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
    output_pointer: *mut f32,
    variant_row_offset: usize,
    selected_sample_count: usize,
    selected_dosage_total: f32,
    selected_observation_count: i32,
    has_missing_values: bool,
) {
    if !has_missing_values {
        return;
    }
    let imputed_dosage_value = selected_dosage_total / selected_observation_count.max(1) as f32;
    for selected_sample_index in 0..selected_sample_count {
        let output_value = unsafe { output_pointer.add(variant_row_offset + selected_sample_index) };
        if unsafe { output_value.read().is_nan() } {
            unsafe {
                output_value.write(imputed_dosage_value);
            }
        }
    }
}

fn record_variant_major_decode_profile(
    thread_local_profile_snapshot: &mut ThreadLocalProfileSnapshot,
    probability_decode_start_time: Option<Instant>,
    output_write_start_time: Option<Instant>,
    selected_sample_count: usize,
) -> Result<(), BgenError> {
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
            thread_local_profile_snapshot.uncompressed_byte_count +=
                u64::try_from(variant_record.declared_uncompressed_block_length).unwrap_or(u64::MAX);
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
            thread_local_profile_snapshot.uncompressed_byte_count +=
                u64::try_from(variant_record.declared_uncompressed_block_length).unwrap_or(u64::MAX);
            thread_local_profile_snapshot.zlib_stream_count += 1;
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
    bit_offset: usize,
}

impl<'a> PackedProbabilityReader<'a> {
    fn new(packed_probability_bytes: &'a [u8]) -> Self {
        Self { packed_probability_bytes, bit_offset: 0 }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn read_probability(&mut self, bit_count: u8) -> Result<u32, BgenError> {
        let bit_count_usize = usize::from(bit_count);
        let byte_offset = self.bit_offset / 8;
        let bit_index_in_byte = self.bit_offset % 8;
        let last_required_bit = self.bit_offset + bit_count_usize;
        let last_required_byte = last_required_bit.div_ceil(8);
        if last_required_byte > self.packed_probability_bytes.len() {
            return Err(BgenError::InvalidFormat(
                "Packed BGEN probability stream ended before all probabilities were decoded.".to_string(),
            ));
        }

        let mut window = 0_u64;
        let bytes_to_copy = (self.packed_probability_bytes.len() - byte_offset).min(8);
        for copied_byte_index in 0..bytes_to_copy {
            window |=
                u64::from(self.packed_probability_bytes[byte_offset + copied_byte_index]) << (copied_byte_index * 8);
        }

        let mask = if bit_count == 32 { u64::from(u32::MAX) } else { (1_u64 << bit_count) - 1 };
        let probability_value = ((window >> bit_index_in_byte) & mask) as u32;
        self.bit_offset += bit_count_usize;
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
