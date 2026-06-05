use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use memmap2::{Mmap, MmapOptions};
use rayon::prelude::*;

use crate::genotype::common::{ChunkStats, GenotypeError, GenotypeReaderCore, VariantMetadataColumns};
use crate::genotype::preprocess;

use super::decode::{
    DosageTileDecodeResult, ThreadScratch, VariantMajorTileStatsMut, decode_tile_variant_count,
    decode_variant_dosage_tile_into_row_major_matrix, decode_variant_major_dosage_tile, read_exact_bytes, read_u32_at,
    u32_to_usize,
};
use super::error::{BgenError, convert_bgen_error_to_genotype_error};
use super::format::CompressionType;
use super::metadata::VariantRecord;
use super::profile::{ReaderProfileSnapshot, ReaderProfiling, ThreadLocalProfileSnapshot, elapsed_nanoseconds};
use super::sample_selection::{SampleSelection, build_sample_selection};
use super::{index, metadata, trusted};

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
    trusted_no_missing_diploid_validated: AtomicBool,
    variant_records: Vec<VariantRecord>,
    chromosome_boundary_indices: Vec<usize>,
    prepared_sample_selection: Mutex<Option<Arc<SampleSelection>>>,
    profiling: ReaderProfiling,
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
            trusted_no_missing_diploid_validated: AtomicBool::new(false),
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
        if self.trusted_no_missing_diploid {
            self.trusted_no_missing_diploid_validated.store(true, Ordering::Release);
        }
        Ok(())
    }

    pub fn mark_trusted_no_missing_diploid_validated(&self) -> Result<(), BgenError> {
        if !self.trusted_no_missing_diploid {
            return Err(BgenError::Range(
                "Trusted no-missing diploid validation cannot be marked on a non-trusted BGEN reader.".to_string(),
            ));
        }
        self.trusted_no_missing_diploid_validated.store(true, Ordering::Release);
        Ok(())
    }

    pub fn variant_metadata_slice(
        &self,
        variant_start: usize,
        variant_stop: usize,
    ) -> Result<VariantMetadataColumns, BgenError> {
        let metadata_slice_start_time = Instant::now();
        validate_variant_bounds(variant_start, variant_stop, self.variant_count)?;

        let selected_variant_records = &self.variant_records[variant_start..variant_stop];
        let variant_metadata_columns = metadata::build_variant_metadata_columns(selected_variant_records);
        self.profiling.record_metadata_slice(elapsed_nanoseconds(metadata_slice_start_time));
        Ok(variant_metadata_columns)
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

    #[allow(clippy::too_many_lines)]
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
        let trusted_decode_enabled =
            self.trusted_no_missing_diploid && self.trusted_no_missing_diploid_validated.load(Ordering::Acquire);
        let mut dosage_sum = vec![0.0_f32; selected_variant_count];
        let mut dosage_square_sum = vec![0.0_f32; selected_variant_count];
        let mut observation_count = vec![0_i32; selected_variant_count];
        let mut zero_count = vec![0_i32; selected_variant_count];
        let mut nonzero_count = vec![0_i32; selected_variant_count];
        let mut homozygous_reference_count = vec![0_i32; selected_variant_count];
        let mut heterozygous_count = vec![0_i32; selected_variant_count];
        let mut homozygous_alternate_count = vec![0_i32; selected_variant_count];
        let decode_results = self.variant_records[variant_start..variant_stop]
            .par_chunks(decode_tile_variant_count)
            .zip(dosage_sum.par_chunks_mut(decode_tile_variant_count))
            .zip(dosage_square_sum.par_chunks_mut(decode_tile_variant_count))
            .zip(observation_count.par_chunks_mut(decode_tile_variant_count))
            .zip(zero_count.par_chunks_mut(decode_tile_variant_count))
            .zip(nonzero_count.par_chunks_mut(decode_tile_variant_count))
            .zip(homozygous_reference_count.par_chunks_mut(decode_tile_variant_count))
            .zip(heterozygous_count.par_chunks_mut(decode_tile_variant_count))
            .zip(homozygous_alternate_count.par_chunks_mut(decode_tile_variant_count))
            .enumerate()
            .map_init(
                ThreadScratch::default,
                |thread_scratch,
                 (
                    tile_index,
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            ((variant_record_chunk, dosage_sum_chunk), dosage_square_sum_chunk),
                                            observation_count_chunk,
                                        ),
                                        zero_count_chunk,
                                    ),
                                    nonzero_count_chunk,
                                ),
                                homozygous_reference_count_chunk,
                            ),
                            heterozygous_count_chunk,
                        ),
                        homozygous_alternate_count_chunk,
                    ),
                )| {
                    let mut tile_stats = VariantMajorTileStatsMut {
                        dosage_sum: dosage_sum_chunk,
                        dosage_square_sum: dosage_square_sum_chunk,
                        observation_count: observation_count_chunk,
                        zero_count: zero_count_chunk,
                        nonzero_count: nonzero_count_chunk,
                        homozygous_reference_count: homozygous_reference_count_chunk,
                        heterozygous_count: heterozygous_count_chunk,
                        homozygous_alternate_count: homozygous_alternate_count_chunk,
                    };
                    if trusted_decode_enabled {
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
                            &mut tile_stats,
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
                            trusted_decode_enabled,
                            &mut tile_stats,
                            thread_scratch,
                        )
                    }
                },
            )
            .collect::<Result<Vec<_>, BgenError>>()?;
        let mut has_missing_values = false;
        for decode_result in decode_results {
            profiling.merge_thread_local_snapshot(&decode_result.profile_snapshot);
            has_missing_values |= decode_result.has_missing_values;
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

    #[allow(clippy::too_many_lines)]
    pub fn read_preprocessed_variant_major_packed8_probability_pairs_into_address_prepared(
        &self,
        variant_start: usize,
        variant_stop: usize,
        output_pointer_address: usize,
        output_value_count: usize,
    ) -> Result<ChunkStats, BgenError> {
        let sample_selection = self.prepared_sample_selection_arc()?;
        validate_variant_bounds(variant_start, variant_stop, self.variant_count)?;
        if !(self.trusted_no_missing_diploid && self.trusted_no_missing_diploid_validated.load(Ordering::Acquire)) {
            return Err(BgenError::UnsupportedFormat(
                "Packed8 BGEN probability-pair delivery requires trusted no-missing diploid validation.".to_string(),
            ));
        }
        let selected_variant_count = variant_stop.saturating_sub(variant_start);
        let selected_sample_count = sample_selection.selected_sample_count;
        let expected_output_value_count = selected_variant_count
            .checked_mul(selected_sample_count)
            .and_then(|value| value.checked_mul(2))
            .ok_or_else(|| {
                BgenError::Range(
                    "Integer overflow while validating packed8 BGEN probability-pair output buffer size.".to_string(),
                )
            })?;
        if output_value_count != expected_output_value_count {
            return Err(BgenError::Range(format!(
                "Variant-major packed8 output buffer shape mismatch for BGEN probability-pair read. Expected {expected_output_value_count} uint8 values, observed {output_value_count}.",
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
        let mut dosage_sum = vec![0.0_f32; selected_variant_count];
        let mut dosage_square_sum = vec![0.0_f32; selected_variant_count];
        let mut observation_count = vec![0_i32; selected_variant_count];
        let mut zero_count = vec![0_i32; selected_variant_count];
        let mut nonzero_count = vec![0_i32; selected_variant_count];
        let mut homozygous_reference_count = vec![0_i32; selected_variant_count];
        let mut heterozygous_count = vec![0_i32; selected_variant_count];
        let mut homozygous_alternate_count = vec![0_i32; selected_variant_count];
        let decode_results = self.variant_records[variant_start..variant_stop]
            .par_chunks(decode_tile_variant_count)
            .zip(dosage_sum.par_chunks_mut(decode_tile_variant_count))
            .zip(dosage_square_sum.par_chunks_mut(decode_tile_variant_count))
            .zip(observation_count.par_chunks_mut(decode_tile_variant_count))
            .zip(zero_count.par_chunks_mut(decode_tile_variant_count))
            .zip(nonzero_count.par_chunks_mut(decode_tile_variant_count))
            .zip(homozygous_reference_count.par_chunks_mut(decode_tile_variant_count))
            .zip(heterozygous_count.par_chunks_mut(decode_tile_variant_count))
            .zip(homozygous_alternate_count.par_chunks_mut(decode_tile_variant_count))
            .enumerate()
            .map_init(
                ThreadScratch::default,
                |thread_scratch,
                 (
                    tile_index,
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            ((variant_record_chunk, dosage_sum_chunk), dosage_square_sum_chunk),
                                            observation_count_chunk,
                                        ),
                                        zero_count_chunk,
                                    ),
                                    nonzero_count_chunk,
                                ),
                                homozygous_reference_count_chunk,
                            ),
                            heterozygous_count_chunk,
                        ),
                        homozygous_alternate_count_chunk,
                    ),
                )| {
                    let mut tile_stats = VariantMajorTileStatsMut {
                        dosage_sum: dosage_sum_chunk,
                        dosage_square_sum: dosage_square_sum_chunk,
                        observation_count: observation_count_chunk,
                        zero_count: zero_count_chunk,
                        nonzero_count: nonzero_count_chunk,
                        homozygous_reference_count: homozygous_reference_count_chunk,
                        heterozygous_count: heterozygous_count_chunk,
                        homozygous_alternate_count: homozygous_alternate_count_chunk,
                    };
                    trusted::decode_trusted_variant_major_packed8_probability_pair_tile(
                        &self.mmap,
                        self.compression_type,
                        self.sample_count,
                        &sample_selection,
                        variant_record_chunk,
                        output_pointer_address,
                        selected_sample_count,
                        tile_index * decode_tile_variant_count,
                        profiling_enabled,
                        &mut tile_stats,
                        thread_scratch,
                    )
                },
            )
            .collect::<Result<Vec<_>, BgenError>>()?;
        for decode_result in decode_results {
            profiling.merge_thread_local_snapshot(&decode_result.profile_snapshot);
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
            false,
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
        BgenReaderCore::variant_metadata_slice(self, variant_start, variant_stop)
            .map_err(|error| convert_bgen_error_to_genotype_error(&error))
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

fn validate_variant_bounds(variant_start: usize, variant_stop: usize, variant_count: usize) -> Result<(), BgenError> {
    if variant_start > variant_stop || variant_stop > variant_count {
        return Err(BgenError::Range(format!(
            "Variant bounds must satisfy 0 <= start <= stop <= {variant_count}. Received start={variant_start}, stop={variant_stop}.",
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn temporary_bgen_path(label: &str) -> PathBuf {
        let timestamp =
            SystemTime::now().duration_since(UNIX_EPOCH).expect("system time should be after unix epoch").as_nanos();
        std::env::temp_dir().join(format!("g-reader-{label}-{}-{timestamp}.bgen", std::process::id()))
    }

    fn minimal_bgen_header_bytes(variant_count: u32, sample_count: u32, flags: u32) -> Vec<u8> {
        let mut bytes = vec![0_u8; 24];
        bytes[0..4].copy_from_slice(&20_u32.to_le_bytes());
        bytes[4..8].copy_from_slice(&20_u32.to_le_bytes());
        bytes[8..12].copy_from_slice(&variant_count.to_le_bytes());
        bytes[12..16].copy_from_slice(&sample_count.to_le_bytes());
        bytes[16..20].copy_from_slice(b"bgen");
        bytes[20..24].copy_from_slice(&flags.to_le_bytes());
        bytes
    }

    fn append_bgen_string(bytes: &mut Vec<u8>, value: &str) {
        let value_length = u16::try_from(value.len()).expect("BGEN string length should fit u16");
        bytes.extend_from_slice(&value_length.to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn trusted_probability_block(probability_bytes: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        bytes.push(2);
        bytes.push(2);
        bytes.extend_from_slice(&[2, 2, 2]);
        bytes.push(0);
        bytes.push(8);
        bytes.extend_from_slice(probability_bytes);
        bytes
    }

    fn variant_payload(probability_block: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        append_bgen_string(&mut bytes, "var");
        append_bgen_string(&mut bytes, "rs");
        append_bgen_string(&mut bytes, "22");
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(b"A");
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(b"G");
        let block_length = u32::try_from(probability_block.len()).expect("probability block should fit u32");
        bytes.extend_from_slice(&block_length.to_le_bytes());
        bytes.extend_from_slice(probability_block);
        bytes
    }

    fn write_single_variant_bgen(path: &Path) {
        let probability_block = trusted_probability_block(&[0, 0, 255, 0, 0, 255]);
        let payload = variant_payload(&probability_block);
        let mut bytes = minimal_bgen_header_bytes(1, 3, 2 << 2);
        bytes.extend_from_slice(&payload);
        fs::write(path, bytes).expect("BGEN test fixture should be written");
    }

    #[test]
    fn private_reader_optional_stats_collects_row_major_dosage_totals() {
        let path = temporary_bgen_path("optional-stats");
        write_single_variant_bgen(&path);
        let reader = BgenReaderCore::open(&path, false).expect("BGEN reader should open");

        let empty_selection = build_sample_selection(reader.sample_count, &[]).expect("empty selection should build");
        let mut empty_output = Vec::<f32>::new();
        let empty_totals = reader
            .read_dosage_f32_into_address_with_selection_and_optional_stats(
                &empty_selection,
                0,
                1,
                empty_output.as_mut_ptr() as usize,
                0,
                true,
            )
            .expect("empty selected samples should return totals")
            .expect("totals should be collected");
        assert_eq!(empty_totals, vec![0.0]);

        let sample_selection =
            build_sample_selection(reader.sample_count, &[0, 2]).expect("non-contiguous selection should build");
        let mut output = vec![f32::NAN; 2];
        let totals = reader
            .read_dosage_f32_into_address_with_selection_and_optional_stats(
                &sample_selection,
                0,
                1,
                output.as_mut_ptr() as usize,
                output.len(),
                true,
            )
            .expect("row-major read should collect totals")
            .expect("totals should be present");
        assert_eq!(output, vec![2.0, 1.0]);
        assert_eq!(totals, vec![3.0]);

        let _ = fs::remove_file(path);
    }
}
