use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;

use memmap2::{Mmap, MmapOptions};
use rayon::prelude::*;
use thiserror::Error;

use crate::genotype::common::{ChunkStats, GenotypeError, GenotypeReaderCore, VariantMetadataColumns};
use crate::genotype::preprocess;

mod decode;
mod index;
mod metadata;
mod profile;
mod sample_selection;
mod trusted;
pub use decode::set_decode_tile_variant_count as set_bgen_decode_tile_variant_count;
use decode::{
    DosageTileDecodeResult, ThreadScratch, decode_tile_variant_count, decode_variant_dosage_tile_into_row_major_matrix,
    decode_variant_major_dosage_tile, read_exact_bytes, read_probability_block, read_u32_at, u32_to_usize,
};
pub use metadata::VariantMetadataLists;
use metadata::VariantRecord;
pub use profile::ReaderProfileSnapshot;
use profile::{ReaderProfiling, ThreadLocalProfileSnapshot, elapsed_nanoseconds};
use sample_selection::{SampleSelection, build_sample_selection};

const VARIANT_IDENTIFIER_LENGTH_SIZE_IN_BYTES: usize = 2;
const ALLELE_LENGTH_SIZE_IN_BYTES: usize = 4;

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
        let mut dosage_sum = Vec::with_capacity(selected_variant_count);
        let mut dosage_square_sum = Vec::with_capacity(selected_variant_count);
        let mut observation_count = Vec::with_capacity(selected_variant_count);
        let mut zero_count = Vec::with_capacity(selected_variant_count);
        let mut nonzero_count = Vec::with_capacity(selected_variant_count);
        let mut homozygous_reference_count = Vec::with_capacity(selected_variant_count);
        let mut heterozygous_count = Vec::with_capacity(selected_variant_count);
        let mut homozygous_alternate_count = Vec::with_capacity(selected_variant_count);
        let decode_results = self.variant_records[variant_start..variant_stop]
            .par_chunks(decode_tile_variant_count)
            .enumerate()
            .map_init(ThreadScratch::default, |thread_scratch, (tile_index, variant_record_chunk)| {
                if self.trusted_no_missing_diploid {
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
                } else {
                    decode_variant_major_dosage_tile(
                        &self.mmap,
                        self.compression_type,
                        self.sample_count,
                        &sample_selection,
                        variant_record_chunk,
                        output_pointer_address,
                        selected_sample_count,
                        tile_index * decode_tile_variant_count,
                        profiling_enabled,
                        self.trusted_no_missing_diploid,
                        thread_scratch,
                    )
                }
            })
            .collect::<Result<Vec<DosageTileDecodeResult>, BgenError>>()?;
        let mut has_missing_values = false;
        for decode_result in decode_results {
            profiling.merge_thread_local_snapshot(&decode_result.profile_snapshot);
            dosage_sum.extend(decode_result.selected_dosage_totals);
            dosage_square_sum.extend(decode_result.selected_dosage_square_totals);
            observation_count.extend(decode_result.selected_observation_counts);
            has_missing_values |= decode_result.has_missing_values;
            zero_count.extend(decode_result.zero_counts);
            nonzero_count.extend(decode_result.nonzero_counts);
            homozygous_reference_count.extend(decode_result.homozygous_reference_counts);
            heterozygous_count.extend(decode_result.heterozygous_counts);
            homozygous_alternate_count.extend(decode_result.homozygous_alternate_counts);
        }
        Ok(preprocess::build_chunk_stats_from_summaries(
            dosage_sum,
            dosage_square_sum,
            observation_count,
            zero_count,
            nonzero_count,
            homozygous_reference_count,
            heterozygous_count,
            homozygous_alternate_count,
            has_missing_values,
            selected_sample_count,
        ))
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
