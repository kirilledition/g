use std::fs::File;
use std::mem::MaybeUninit;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use flate2::{Decompress, FlushDecompress, Status};
use memmap2::{Mmap, MmapOptions};
use rayon::prelude::*;
use thiserror::Error;

use crate::genotype::common::{ChunkStats, GenotypeError, GenotypeReaderCore, VariantMetadataColumns};
use crate::genotype::preprocess;

mod index;
mod metadata;
mod profile;
mod sample_selection;
mod trusted;
pub use metadata::VariantMetadataLists;
use metadata::VariantRecord;
pub use profile::ReaderProfileSnapshot;
use profile::{ReaderProfiling, ThreadLocalProfileSnapshot, elapsed_nanoseconds};
use sample_selection::{SampleSelection, build_sample_selection};

const MISSING_SAMPLE_FLAG_MASK: u8 = 0x80;
const PLOIDY_MASK: u8 = 0x3F;
const VARIANT_IDENTIFIER_LENGTH_SIZE_IN_BYTES: usize = 2;
const ALLELE_LENGTH_SIZE_IN_BYTES: usize = 4;
const DEFAULT_DECODE_TILE_VARIANT_COUNT: usize = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompressionType {
    None,
    Zlib,
}

impl TryFrom<u32> for CompressionType {
    type Error = BgenError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::None),
            1 => Ok(Self::Zlib),
            unsupported_value => Err(BgenError::UnsupportedFormat(format!(
                "Unsupported BGEN compression flag {unsupported_value}. Only uncompressed and zlib-compressed blocks are supported.",
            ))),
        }
    }
}

#[derive(Debug)]
pub struct BgenReaderCore {
    bgen_path: PathBuf,
    mmap: Mmap,
    sample_count: usize,
    variant_count: usize,
    contains_embedded_samples: bool,
    sample_identifiers: Vec<String>,
    compression_type: CompressionType,
    trusted_no_missing_diploid: bool,
    variant_records: Vec<VariantRecord>,
    chromosome_boundary_indices: Vec<usize>,
    prepared_sample_selection: Mutex<Option<Arc<SampleSelection>>>,
    profiling: ReaderProfiling,
}

#[derive(Debug)]
struct VariantDecodeResult {
    profile_snapshot: ThreadLocalProfileSnapshot,
    selected_dosage_total: f32,
}

#[derive(Debug)]
struct DosageTileDecodeResult {
    profile_snapshot: ThreadLocalProfileSnapshot,
    selected_dosage_totals: Vec<f32>,
}

#[derive(Error, Debug)]
pub enum BgenError {
    #[error("{0}")]
    InvalidFormat(String),
    #[error("{0}")]
    UnsupportedFormat(String),
    #[error("{0}")]
    Range(String),
    #[error("I/O error while reading BGEN file: {0}")]
    Io(#[from] std::io::Error),
}

fn decode_tile_variant_count() -> usize {
    static DECODE_TILE_VARIANT_COUNT: OnceLock<usize> = OnceLock::new();
    *DECODE_TILE_VARIANT_COUNT.get_or_init(|| {
        std::env::var("G_BGEN_DECODE_TILE_VARIANT_COUNT")
            .ok()
            .and_then(|raw_value| raw_value.parse::<usize>().ok())
            .filter(|tile_variant_count| *tile_variant_count > 0)
            .unwrap_or(DEFAULT_DECODE_TILE_VARIANT_COUNT)
    })
}

fn unphased_eight_bit_dosage_lookup() -> &'static [f32] {
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

#[allow(clippy::missing_errors_doc)]
impl BgenReaderCore {
    pub fn open(bgen_path: &Path, trusted_no_missing_diploid: bool) -> Result<Self, BgenError> {
        let file = File::open(bgen_path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        let first_variant_offset = 4 + u32_to_usize(read_u32_at(&mmap, 0)?)?;
        let header_block_length = u32_to_usize(read_u32_at(&mmap, 4)?)?;
        if header_block_length < 20 {
            return Err(BgenError::InvalidFormat(format!(
                "BGEN header block length must be at least 20 bytes. Observed {header_block_length}.",
            )));
        }
        let variant_count = u32_to_usize(read_u32_at(&mmap, 8)?)?;
        let sample_count = u32_to_usize(read_u32_at(&mmap, 12)?)?;

        let magic_offset = 16;
        let magic_number = read_exact_bytes(&mmap, magic_offset, 4)?;
        if magic_number != b"bgen" && magic_number != [0_u8, 0, 0, 0] {
            return Err(BgenError::InvalidFormat(
                "BGEN header magic number must be `bgen` or four zero bytes.".to_string(),
            ));
        }

        let header_flags_offset = 4 + header_block_length - 4;
        let header_flags = read_u32_at(&mmap, header_flags_offset)?;
        let compression_type = CompressionType::try_from(header_flags & 0b11)?;
        let layout_identifier = (header_flags >> 2) & 0b1111;
        if layout_identifier != 2 {
            return Err(BgenError::UnsupportedFormat(format!(
                "Only BGEN Layout 2 is supported by the native Rust reader. Observed layout {layout_identifier}.",
            )));
        }
        let contains_embedded_samples = ((header_flags >> 31) & 1) == 1;

        let sample_block_offset = 4 + header_block_length;
        let sample_identifiers = if contains_embedded_samples {
            index::parse_sample_identifier_block(&mmap, sample_block_offset, first_variant_offset, sample_count)?
        } else {
            Vec::new()
        };

        let variant_records =
            index::parse_variant_records(&mmap, first_variant_offset, variant_count, sample_count, compression_type)?;
        let chromosome_boundary_indices = metadata::build_chromosome_boundary_indices(&variant_records);

        Ok(Self {
            bgen_path: bgen_path.to_path_buf(),
            mmap,
            sample_count,
            variant_count,
            contains_embedded_samples,
            sample_identifiers,
            compression_type,
            trusted_no_missing_diploid,
            variant_records,
            chromosome_boundary_indices,
            prepared_sample_selection: Mutex::new(None),
            profiling: ReaderProfiling::default(),
        })
    }

    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    pub fn variant_count(&self) -> usize {
        self.variant_count
    }

    pub fn contains_embedded_samples(&self) -> bool {
        self.contains_embedded_samples
    }

    pub fn sample_identifiers(&self) -> Vec<String> {
        self.sample_identifiers.clone()
    }

    pub fn chromosome_boundary_indices(&self) -> Vec<usize> {
        self.chromosome_boundary_indices.clone()
    }

    pub fn prepare_sample_selection(&self, sample_indices: &[i64]) -> Result<(), BgenError> {
        let sample_selection_start_time = Instant::now();
        let sample_selection = Arc::new(build_sample_selection(self.sample_count, sample_indices)?);
        self.profiling.record_sample_selection_prepare(elapsed_nanoseconds(sample_selection_start_time));
        let mut prepared_sample_selection = self
            .prepared_sample_selection
            .lock()
            .map_err(|_| BgenError::InvalidFormat("Prepared BGEN sample selection mutex was poisoned.".to_string()))?;
        *prepared_sample_selection = Some(sample_selection);
        Ok(())
    }

    pub fn clear_prepared_sample_selection(&self) -> Result<(), BgenError> {
        let mut prepared_sample_selection = self
            .prepared_sample_selection
            .lock()
            .map_err(|_| BgenError::InvalidFormat("Prepared BGEN sample selection mutex was poisoned.".to_string()))?;
        *prepared_sample_selection = None;
        Ok(())
    }

    pub fn reset_profile(&self) {
        self.profiling.reset();
    }

    pub fn profile_snapshot(&self) -> ReaderProfileSnapshot {
        self.profiling.snapshot()
    }

    pub fn validate_trusted_no_missing_diploid(&self) -> Result<(), BgenError> {
        let mut thread_scratch = ThreadScratch::default();
        let mut thread_local_profile_snapshot = ThreadLocalProfileSnapshot::default();
        for variant_record in &self.variant_records {
            trusted::validate_variant_compatible_with_trusted_no_missing_diploid(
                &self.mmap,
                self.compression_type,
                variant_record,
                self.sample_count,
                &mut thread_scratch,
                &mut thread_local_profile_snapshot,
            )?;
        }
        Ok(())
    }

    pub fn variant_metadata_slice(
        &self,
        variant_start: usize,
        variant_stop: usize,
    ) -> Result<VariantMetadataLists, BgenError> {
        let metadata_slice_start_time = Instant::now();
        validate_variant_bounds(variant_start, variant_stop, self.variant_count)?;

        let selected_variant_records = &self.variant_records[variant_start..variant_stop];
        let variant_metadata_lists = metadata::build_variant_metadata_lists(selected_variant_records);
        self.profiling.record_metadata_slice(elapsed_nanoseconds(metadata_slice_start_time));
        Ok(variant_metadata_lists)
    }

    pub fn read_dosage_f32(
        &self,
        sample_indices: &[i64],
        variant_start: usize,
        variant_stop: usize,
    ) -> Result<Vec<f32>, BgenError> {
        validate_variant_bounds(variant_start, variant_stop, self.variant_count)?;
        let sample_selection_start_time = Instant::now();
        let sample_selection = build_sample_selection(self.sample_count, sample_indices)?;
        self.profiling.record_sample_selection_prepare(elapsed_nanoseconds(sample_selection_start_time));
        let selected_sample_count = sample_selection.selected_sample_count;
        let selected_variant_count = variant_stop - variant_start;
        let mut row_major_dosage_values = vec![0.0_f32; selected_sample_count * selected_variant_count];
        self.read_dosage_f32_into_address_with_selection(
            &sample_selection,
            variant_start,
            variant_stop,
            row_major_dosage_values.as_mut_ptr() as usize,
            row_major_dosage_values.len(),
        )?;
        Ok(row_major_dosage_values)
    }

    pub fn read_dosage_f32_prepared(&self, variant_start: usize, variant_stop: usize) -> Result<Vec<f32>, BgenError> {
        validate_variant_bounds(variant_start, variant_stop, self.variant_count)?;
        let sample_selection = self.prepared_sample_selection_arc()?;
        let selected_sample_count = sample_selection.selected_sample_count;
        let selected_variant_count = variant_stop - variant_start;
        let mut row_major_dosage_values = vec![0.0_f32; selected_sample_count * selected_variant_count];
        self.read_dosage_f32_into_address_with_selection(
            sample_selection.as_ref(),
            variant_start,
            variant_stop,
            row_major_dosage_values.as_mut_ptr() as usize,
            row_major_dosage_values.len(),
        )?;
        Ok(row_major_dosage_values)
    }

    pub fn read_dosage_f32_into_address(
        &self,
        sample_indices: &[i64],
        variant_start: usize,
        variant_stop: usize,
        output_pointer_address: usize,
        output_value_count: usize,
    ) -> Result<(), BgenError> {
        validate_variant_bounds(variant_start, variant_stop, self.variant_count)?;
        let sample_selection_start_time = Instant::now();
        let sample_selection = build_sample_selection(self.sample_count, sample_indices)?;
        self.profiling.record_sample_selection_prepare(elapsed_nanoseconds(sample_selection_start_time));
        self.read_dosage_f32_into_address_with_selection(
            &sample_selection,
            variant_start,
            variant_stop,
            output_pointer_address,
            output_value_count,
        )
    }

    pub fn read_dosage_f32_into_address_prepared(
        &self,
        variant_start: usize,
        variant_stop: usize,
        output_pointer_address: usize,
        output_value_count: usize,
    ) -> Result<(), BgenError> {
        validate_variant_bounds(variant_start, variant_stop, self.variant_count)?;
        let sample_selection = self.prepared_sample_selection_arc()?;
        self.read_dosage_f32_into_address_with_selection(
            sample_selection.as_ref(),
            variant_start,
            variant_stop,
            output_pointer_address,
            output_value_count,
        )
    }

    pub fn read_preprocessed_dosage_f32_into_address_prepared(
        &self,
        variant_start: usize,
        variant_stop: usize,
        output_pointer_address: usize,
        output_value_count: usize,
    ) -> Result<ChunkStats, BgenError> {
        let sample_selection = self.prepared_sample_selection_arc()?;
        validate_variant_bounds(variant_start, variant_stop, self.variant_count)?;
        let selected_variant_count = variant_stop.saturating_sub(variant_start);
        if selected_variant_count == 0 {
            return Ok(preprocess::build_empty_chunk_stats(0, false));
        }
        let selected_sample_count = output_value_count.checked_div(selected_variant_count).ok_or_else(|| {
            BgenError::Range("Unable to resolve sample count for preprocessed BGEN dosage matrix.".to_string())
        })?;
        self.read_dosage_f32_into_address_with_selection(
            &sample_selection,
            variant_start,
            variant_stop,
            output_pointer_address,
            output_value_count,
        )?;
        let output_slice =
            unsafe { std::slice::from_raw_parts_mut(output_pointer_address as *mut f32, output_value_count) };
        preprocess::preprocess_row_major_dosage_matrix(output_slice, selected_sample_count, selected_variant_count)
            .map_err(|error| BgenError::Range(error.to_string()))
    }

    pub fn read_preprocessed_variant_major_dosage_f32_into_address_prepared(
        &self,
        variant_start: usize,
        variant_stop: usize,
        output_pointer_address: usize,
        output_value_count: usize,
    ) -> Result<ChunkStats, BgenError> {
        if !self.trusted_no_missing_diploid {
            return Err(BgenError::UnsupportedFormat(
                "Variant-major preprocessed BGEN reads require trusted_no_missing_diploid.".to_string(),
            ));
        }
        let sample_selection = self.prepared_sample_selection_arc()?;
        validate_variant_bounds(variant_start, variant_stop, self.variant_count)?;
        let selected_variant_count = variant_stop.saturating_sub(variant_start);
        let selected_sample_count = sample_selection.selected_sample_count;
        let expected_output_value_count =
            selected_sample_count.checked_mul(selected_variant_count).ok_or_else(|| {
                BgenError::Range("Integer overflow while validating variant-major BGEN output buffer size.".to_string())
            })?;
        if output_value_count != expected_output_value_count {
            return Err(BgenError::Range(format!(
                "Variant-major output buffer shape mismatch for BGEN dosage read. Expected {expected_output_value_count} float32 values, observed {output_value_count}.",
            )));
        }
        if selected_variant_count == 0 {
            return Ok(preprocess::build_empty_chunk_stats(0, false));
        }
        if selected_sample_count == 0 {
            return Ok(preprocess::build_empty_chunk_stats(selected_variant_count, false));
        }

        let profiling = &self.profiling;
        let profiling_enabled = profiling.is_enabled();
        profiling.record_selected_sample_count(selected_sample_count);
        let decode_tile_variant_count = decode_tile_variant_count();
        let decode_results = self.variant_records[variant_start..variant_stop]
            .par_chunks(decode_tile_variant_count)
            .enumerate()
            .map_init(ThreadScratch::default, |thread_scratch, (tile_index, variant_record_chunk)| {
                trusted::decode_trusted_variant_major_dosage_tile(
                    &self.mmap,
                    self.compression_type,
                    self.sample_count,
                    &sample_selection,
                    variant_record_chunk,
                    output_pointer_address,
                    selected_sample_count,
                    tile_index * decode_tile_variant_count,
                    profiling_enabled,
                    thread_scratch,
                )
            })
            .collect::<Result<Vec<DosageTileDecodeResult>, BgenError>>()?;
        for decode_result in decode_results {
            profiling.merge_thread_local_snapshot(&decode_result.profile_snapshot);
        }
        let output_slice =
            unsafe { std::slice::from_raw_parts(output_pointer_address as *const f32, output_value_count) };
        preprocess::summarize_variant_major_dosage_matrix(output_slice, selected_sample_count, selected_variant_count)
            .map_err(|error| BgenError::Range(error.to_string()))
    }

    pub fn bgen_path(&self) -> &Path {
        &self.bgen_path
    }

    fn prepared_sample_selection_arc(&self) -> Result<Arc<SampleSelection>, BgenError> {
        let prepared_sample_selection = self
            .prepared_sample_selection
            .lock()
            .map_err(|_| BgenError::InvalidFormat("Prepared BGEN sample selection mutex was poisoned.".to_string()))?;
        prepared_sample_selection.clone().ok_or_else(|| {
            BgenError::Range("Prepared BGEN sample selection was requested before binding aligned samples.".to_string())
        })
    }

    fn read_dosage_f32_into_address_with_selection(
        &self,
        sample_selection: &SampleSelection,
        variant_start: usize,
        variant_stop: usize,
        output_pointer_address: usize,
        output_value_count: usize,
    ) -> Result<(), BgenError> {
        self.read_dosage_f32_into_address_with_selection_and_optional_stats(
            sample_selection,
            variant_start,
            variant_stop,
            output_pointer_address,
            output_value_count,
            false,
        )
        .map(|_| ())
    }

    fn read_dosage_f32_into_address_with_selection_and_optional_stats(
        &self,
        sample_selection: &SampleSelection,
        variant_start: usize,
        variant_stop: usize,
        output_pointer_address: usize,
        output_value_count: usize,
        collect_dosage_totals: bool,
    ) -> Result<Option<Vec<f32>>, BgenError> {
        let selected_sample_count = sample_selection.selected_sample_count;
        let selected_variant_count = variant_stop - variant_start;
        let expected_output_value_count =
            selected_sample_count.checked_mul(selected_variant_count).ok_or_else(|| {
                BgenError::Range("Integer overflow while validating BGEN output buffer size.".to_string())
            })?;
        if output_value_count != expected_output_value_count {
            return Err(BgenError::Range(format!(
                "Output buffer shape mismatch for BGEN dosage read. Expected {expected_output_value_count} float32 values, observed {output_value_count}.",
            )));
        }
        if selected_sample_count == 0 || selected_variant_count == 0 {
            return Ok(collect_dosage_totals.then(|| vec![0.0_f32; selected_variant_count]));
        }

        let output_pointer = output_pointer_address;
        let profiling = &self.profiling;
        let profiling_enabled = profiling.is_enabled();
        profiling.record_selected_sample_count(selected_sample_count);
        let decode_tile_variant_count = decode_tile_variant_count();
        let decode_results = self.variant_records[variant_start..variant_stop]
            .par_chunks(decode_tile_variant_count)
            .enumerate()
            .map_init(ThreadScratch::default, |thread_scratch, (tile_index, variant_record_chunk)| {
                decode_variant_dosage_tile_into_row_major_matrix(
                    &self.mmap,
                    self.compression_type,
                    self.sample_count,
                    sample_selection,
                    variant_record_chunk,
                    output_pointer,
                    selected_variant_count,
                    tile_index * decode_tile_variant_count,
                    profiling_enabled,
                    self.trusted_no_missing_diploid,
                    collect_dosage_totals,
                    thread_scratch,
                )
            })
            .collect::<Result<Vec<DosageTileDecodeResult>, BgenError>>()?;
        let mut selected_dosage_totals = collect_dosage_totals.then(|| Vec::with_capacity(selected_variant_count));
        for decode_result in decode_results {
            profiling.merge_thread_local_snapshot(&decode_result.profile_snapshot);
            if let Some(totals) = &mut selected_dosage_totals {
                totals.extend(decode_result.selected_dosage_totals);
            }
        }
        Ok(selected_dosage_totals)
    }
}

impl GenotypeReaderCore for BgenReaderCore {
    fn sample_count(&self) -> usize {
        BgenReaderCore::sample_count(self)
    }

    fn variant_count(&self) -> usize {
        BgenReaderCore::variant_count(self)
    }

    fn sample_identifiers(&self) -> Vec<String> {
        BgenReaderCore::sample_identifiers(self)
    }

    fn chromosome_boundary_indices(&self) -> Vec<usize> {
        BgenReaderCore::chromosome_boundary_indices(self)
    }

    fn prepare_sample_selection(&self, sample_indices: &[i64]) -> Result<(), GenotypeError> {
        BgenReaderCore::prepare_sample_selection(self, sample_indices)
            .map_err(|error| convert_bgen_error_to_genotype_error(&error))
    }

    fn clear_prepared_sample_selection(&self) -> Result<(), GenotypeError> {
        BgenReaderCore::clear_prepared_sample_selection(self)
            .map_err(|error| convert_bgen_error_to_genotype_error(&error))
    }

    fn variant_metadata_slice(
        &self,
        variant_start: usize,
        variant_stop: usize,
    ) -> Result<VariantMetadataColumns, GenotypeError> {
        let (chromosome, variant_identifier, position, allele_one, allele_two) =
            BgenReaderCore::variant_metadata_slice(self, variant_start, variant_stop)
                .map_err(|error| convert_bgen_error_to_genotype_error(&error))?;
        Ok(VariantMetadataColumns { chromosome, variant_identifier, position, allele_one, allele_two })
    }

    fn read_preprocessed_dosage_f32_into_address_prepared(
        &self,
        variant_start: usize,
        variant_stop: usize,
        output_pointer_address: usize,
        output_value_count: usize,
    ) -> Result<ChunkStats, GenotypeError> {
        BgenReaderCore::read_preprocessed_dosage_f32_into_address_prepared(
            self,
            variant_start,
            variant_stop,
            output_pointer_address,
            output_value_count,
        )
        .map_err(|error| convert_bgen_error_to_genotype_error(&error))
    }
}

fn convert_bgen_error_to_genotype_error(error: &BgenError) -> GenotypeError {
    GenotypeError::Reader(error.to_string())
}

struct ThreadScratch {
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
fn decode_variant_dosage_tile_into_row_major_matrix(
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

    Ok(DosageTileDecodeResult { profile_snapshot: thread_local_profile_snapshot, selected_dosage_totals })
}

fn validate_variant_bounds(variant_start: usize, variant_stop: usize, variant_count: usize) -> Result<(), BgenError> {
    if variant_start > variant_stop || variant_stop > variant_count {
        return Err(BgenError::Range(format!(
            "Variant bounds must satisfy 0 <= start <= stop <= {variant_count}. Received start={variant_start}, stop={variant_stop}.",
        )));
    }
    Ok(())
}

fn validate_variant_probability_block(
    mmap: &[u8],
    compression_type: CompressionType,
    variant_record: &VariantRecord,
    sample_count: usize,
    variant_label: &str,
) -> Result<(), BgenError> {
    let mut thread_scratch = ThreadScratch::default();
    let mut thread_local_profile_snapshot = ThreadLocalProfileSnapshot::default();
    let probability_block = read_probability_block(
        mmap,
        compression_type,
        variant_record,
        &mut thread_scratch,
        &mut thread_local_profile_snapshot,
        false,
    )?;
    let observed_sample_count = u32_to_usize(read_u32_at(probability_block, 0)?)?;
    if observed_sample_count != sample_count {
        return Err(BgenError::InvalidFormat(format!(
            "The {variant_label} stores {observed_sample_count} samples in its probability block, but the file header reports {sample_count}.",
        )));
    }
    Ok(())
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

        return Ok(VariantDecodeResult { profile_snapshot: thread_local_profile_snapshot, selected_dosage_total });
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

    Ok(VariantDecodeResult { profile_snapshot: thread_local_profile_snapshot, selected_dosage_total })
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

        return Ok(VariantDecodeResult { profile_snapshot: thread_local_profile_snapshot, selected_dosage_total });
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

        return Ok(VariantDecodeResult { profile_snapshot: thread_local_profile_snapshot, selected_dosage_total });
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

        return Ok(VariantDecodeResult { profile_snapshot: thread_local_profile_snapshot, selected_dosage_total });
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

    Ok(VariantDecodeResult { profile_snapshot: thread_local_profile_snapshot, selected_dosage_total })
}

fn read_probability_block<'a>(
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

fn read_u8_at(buffer: &[u8], offset: usize) -> Result<u8, BgenError> {
    Ok(*read_exact_bytes(buffer, offset, 1)?
        .first()
        .ok_or_else(|| BgenError::InvalidFormat("Unexpected empty byte slice.".to_string()))?)
}

fn read_u16_at(buffer: &[u8], offset: usize) -> Result<u16, BgenError> {
    let bytes = read_exact_bytes(buffer, offset, 2)?;
    let byte_array: [u8; 2] = bytes
        .try_into()
        .map_err(|_| BgenError::InvalidFormat("Failed to decode a two-byte integer from the BGEN file.".to_string()))?;
    Ok(u16::from_le_bytes(byte_array))
}

fn read_u32_at(buffer: &[u8], offset: usize) -> Result<u32, BgenError> {
    let bytes = read_exact_bytes(buffer, offset, 4)?;
    let byte_array: [u8; 4] = bytes.try_into().map_err(|_| {
        BgenError::InvalidFormat("Failed to decode a four-byte integer from the BGEN file.".to_string())
    })?;
    Ok(u32::from_le_bytes(byte_array))
}

fn read_exact_bytes(buffer: &[u8], offset: usize, length: usize) -> Result<&[u8], BgenError> {
    let stop = offset
        .checked_add(length)
        .ok_or_else(|| BgenError::InvalidFormat("Integer overflow while slicing BGEN file bytes.".to_string()))?;
    buffer
        .get(offset..stop)
        .ok_or_else(|| BgenError::InvalidFormat("Unexpected end of file while reading BGEN bytes.".to_string()))
}

fn u32_to_usize(value: u32) -> Result<usize, BgenError> {
    usize::try_from(value).map_err(|_| {
        BgenError::InvalidFormat(format!(
            "BGEN integer value {value} does not fit into the native platform usize type.",
        ))
    })
}
