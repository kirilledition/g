use rayon::prelude::*;

use crate::bgen::decode::{
    ThreadScratch, VariantMajorSparseCandidateCountsMut, VariantMajorTileStatsMut, decode_tile_variant_count,
    decode_variant_major_dosage_tile,
};
use crate::bgen::error::BgenError;
use crate::bgen::sample_selection::SampleSelection;
use crate::bgen::trusted;
use crate::buffer::{OutputBufferAddress, OutputValueCount};
use crate::common::{ChunkStatisticsPolicy, ChunkStats};
use crate::preprocess;

use super::{BgenReaderCore, validate_variant_bounds};

#[derive(Clone, Copy, Debug)]
struct VariantMajorReadShape {
    selected_variant_count: usize,
    selected_sample_count: usize,
}

impl VariantMajorReadShape {
    fn from_selection(sample_selection: &SampleSelection, variant_start: usize, variant_stop: usize) -> Self {
        Self {
            selected_variant_count: variant_stop - variant_start,
            selected_sample_count: sample_selection.selected_sample_count(),
        }
    }

    fn empty_chunk_stats(self, statistics_policy: ChunkStatisticsPolicy) -> Option<ChunkStats> {
        if self.selected_variant_count == 0 {
            return Some(preprocess::build_empty_chunk_stats(0, statistics_policy));
        }
        if self.selected_sample_count == 0 {
            return Some(preprocess::build_empty_chunk_stats(self.selected_variant_count, statistics_policy));
        }
        None
    }
}

#[derive(Clone, Copy)]
struct VariantMajorDecodeRequest<'a> {
    sample_selection: &'a SampleSelection,
    variant_start: usize,
    output_pointer_address: OutputBufferAddress,
    shape: VariantMajorReadShape,
    decode_tile_variant_count: usize,
}

impl VariantMajorDecodeRequest<'_> {
    fn variant_stop(&self) -> usize {
        self.variant_start + self.shape.selected_variant_count
    }
}

struct VariantMajorStatsBuffers {
    dosage_sum: Vec<f32>,
    dosage_square_sum: Vec<f32>,
    observation_count: Vec<i32>,
    zero_count: Option<Vec<i32>>,
    homozygous_alternate_count: Option<Vec<i32>>,
}

impl VariantMajorStatsBuffers {
    fn new(selected_variant_count: usize, statistics_policy: ChunkStatisticsPolicy) -> Self {
        Self {
            dosage_sum: vec![0.0_f32; selected_variant_count],
            dosage_square_sum: vec![0.0_f32; selected_variant_count],
            observation_count: vec![0_i32; selected_variant_count],
            zero_count: statistics_policy.collect_sparse_candidate_mask.then(|| vec![0_i32; selected_variant_count]),
            homozygous_alternate_count: statistics_policy
                .collect_sparse_candidate_mask
                .then(|| vec![0_i32; selected_variant_count]),
        }
    }

    fn into_chunk_stats(
        self,
        selected_sample_count: usize,
        statistics_policy: ChunkStatisticsPolicy,
    ) -> Result<ChunkStats, BgenError> {
        preprocess::build_chunk_stats_from_summaries(
            self.dosage_sum,
            self.dosage_square_sum,
            self.observation_count,
            self.zero_count,
            self.homozygous_alternate_count,
            selected_sample_count,
            statistics_policy,
        )
        .map_err(|error| BgenError::Range(error.to_string()))
    }
}

impl BgenReaderCore {
    /// Read prepared variant-major dosages into a caller-provided f32 buffer.
    ///
    /// # Errors
    ///
    /// Returns an error when sample selection has not been prepared, variant
    /// bounds or output buffer size are invalid, or BGEN decoding fails.
    pub fn read_preprocessed_variant_major_dosage_f32_into_address_prepared(
        &self,
        variant_start: usize,
        variant_stop: usize,
        output_pointer_address: OutputBufferAddress,
        output_value_count: OutputValueCount,
        statistics_policy: ChunkStatisticsPolicy,
    ) -> Result<ChunkStats, BgenError> {
        let sample_selection = self.prepared_sample_selection_arc()?;
        validate_variant_bounds(variant_start, variant_stop, self.variant_count)?;
        let read_shape = VariantMajorReadShape::from_selection(&sample_selection, variant_start, variant_stop);
        validate_variant_major_dosage_output_value_count(read_shape, output_value_count)?;
        if let Some(empty_chunk_stats) = read_shape.empty_chunk_stats(statistics_policy) {
            return Ok(empty_chunk_stats);
        }

        let decode_request = VariantMajorDecodeRequest {
            sample_selection: &sample_selection,
            variant_start,
            output_pointer_address,
            shape: read_shape,
            decode_tile_variant_count: decode_tile_variant_count(),
        };
        let trusted_decode_enabled = self.trusted_no_missing_diploid_decode_enabled();
        let mut stats_buffers = VariantMajorStatsBuffers::new(read_shape.selected_variant_count, statistics_policy);
        self.decode_preprocessed_variant_major_dosage_tiles(
            decode_request,
            &mut stats_buffers,
            trusted_decode_enabled,
        )?;
        stats_buffers.into_chunk_stats(read_shape.selected_sample_count, statistics_policy)
    }

    /// Read validated trusted variant-major dosages without computing statistics.
    ///
    /// This output-only path is valid only after trusted no-missing diploid
    /// validation. It writes the same dosage values as the statistics-producing
    /// trusted path without allocating or reducing per-variant statistics.
    ///
    /// # Errors
    ///
    /// Returns an error when trusted validation is not active, sample selection
    /// has not been prepared, variant bounds or output buffer size are invalid,
    /// or BGEN decoding fails.
    pub fn read_trusted_variant_major_dosage_f32_into_address_prepared(
        &self,
        variant_start: usize,
        variant_stop: usize,
        output_pointer_address: OutputBufferAddress,
        output_value_count: OutputValueCount,
    ) -> Result<(), BgenError> {
        if !self.trusted_no_missing_diploid_decode_enabled() {
            return Err(BgenError::UnsupportedFormat(
                "Output-only trusted dosage decoding requires completed trusted no-missing diploid validation."
                    .to_string(),
            ));
        }
        let sample_selection = self.prepared_sample_selection_arc()?;
        validate_variant_bounds(variant_start, variant_stop, self.variant_count)?;
        let read_shape = VariantMajorReadShape::from_selection(&sample_selection, variant_start, variant_stop);
        validate_variant_major_dosage_output_value_count(read_shape, output_value_count)?;
        if read_shape.selected_variant_count == 0 || read_shape.selected_sample_count == 0 {
            return Ok(());
        }

        self.decode_trusted_variant_major_dosage_tiles_without_statistics(VariantMajorDecodeRequest {
            sample_selection: &sample_selection,
            variant_start,
            output_pointer_address,
            shape: read_shape,
            decode_tile_variant_count: decode_tile_variant_count(),
        })
    }

    /// Read prepared variant-major packed8 probability pairs into a caller-provided u8 buffer.
    ///
    /// # Errors
    ///
    /// Returns an error when sample selection has not been prepared, packed8
    /// preconditions fail, variant bounds or output buffer size are invalid, or
    /// BGEN decoding fails.
    pub fn read_preprocessed_variant_major_packed8_probability_pairs_into_address_prepared(
        &self,
        variant_start: usize,
        variant_stop: usize,
        output_pointer_address: OutputBufferAddress,
        output_value_count: OutputValueCount,
        statistics_policy: ChunkStatisticsPolicy,
    ) -> Result<ChunkStats, BgenError> {
        let sample_selection = self.prepared_sample_selection_arc()?;
        validate_variant_bounds(variant_start, variant_stop, self.variant_count)?;
        self.validate_packed8_probability_pair_preconditions()?;
        let read_shape = VariantMajorReadShape::from_selection(&sample_selection, variant_start, variant_stop);
        validate_variant_major_packed8_probability_pair_output_value_count(read_shape, output_value_count)?;
        if let Some(empty_chunk_stats) = read_shape.empty_chunk_stats(statistics_policy) {
            return Ok(empty_chunk_stats);
        }

        let decode_request = VariantMajorDecodeRequest {
            sample_selection: &sample_selection,
            variant_start,
            output_pointer_address,
            shape: read_shape,
            decode_tile_variant_count: decode_tile_variant_count(),
        };
        let mut stats_buffers = VariantMajorStatsBuffers::new(read_shape.selected_variant_count, statistics_policy);
        self.decode_preprocessed_variant_major_packed8_probability_pair_tiles(decode_request, &mut stats_buffers)?;
        stats_buffers.into_chunk_stats(read_shape.selected_sample_count, statistics_policy)
    }

    fn decode_preprocessed_variant_major_dosage_tiles(
        &self,
        request: VariantMajorDecodeRequest<'_>,
        stats_buffers: &mut VariantMajorStatsBuffers,
        trusted_decode_enabled: bool,
    ) -> Result<(), BgenError> {
        let decode_tile_variant_count = request.decode_tile_variant_count;
        let selected_variant_records = &self.variant_records[request.variant_start..request.variant_stop()];
        let decode_tile = |thread_scratch: &mut ThreadScratch,
                           tile_index: usize,
                           variant_record_chunk: &[crate::bgen::metadata::VariantRecord],
                           tile_stats: &mut VariantMajorTileStatsMut<'_>| {
            if trusted_decode_enabled {
                trusted::decode_trusted_variant_major_dosage_tile(
                    &self.mmap,
                    self.compression_type,
                    self.sample_count,
                    request.sample_selection,
                    variant_record_chunk,
                    request.output_pointer_address,
                    request.shape.selected_sample_count,
                    tile_index * decode_tile_variant_count,
                    false,
                    tile_stats,
                    thread_scratch,
                )
            } else {
                decode_variant_major_dosage_tile(
                    &self.mmap,
                    self.compression_type,
                    self.sample_count,
                    request.sample_selection,
                    variant_record_chunk,
                    request.output_pointer_address,
                    request.shape.selected_sample_count,
                    tile_index * decode_tile_variant_count,
                    false,
                    tile_stats,
                    thread_scratch,
                )
            }
        };
        let VariantMajorStatsBuffers {
            dosage_sum,
            dosage_square_sum,
            observation_count,
            zero_count,
            homozygous_alternate_count,
        } = stats_buffers;
        match (zero_count.as_mut(), homozygous_alternate_count.as_mut()) {
            (Some(zero_count), Some(homozygous_alternate_count)) => selected_variant_records
                .par_chunks(decode_tile_variant_count)
                .zip(dosage_sum.par_chunks_mut(decode_tile_variant_count))
                .zip(dosage_square_sum.par_chunks_mut(decode_tile_variant_count))
                .zip(observation_count.par_chunks_mut(decode_tile_variant_count))
                .zip(zero_count.par_chunks_mut(decode_tile_variant_count))
                .zip(homozygous_alternate_count.par_chunks_mut(decode_tile_variant_count))
                .enumerate()
                .try_for_each_init(
                    ThreadScratch::default,
                    |thread_scratch,
                     (
                        tile_index,
                        (
                            ((((variant_records, dosage_sum), dosage_square_sum), observation_count), zero_count),
                            homozygous_alternate_count,
                        ),
                    )| {
                        let mut tile_stats = variant_major_tile_stats_mut(
                            dosage_sum,
                            dosage_square_sum,
                            observation_count,
                            Some((zero_count, homozygous_alternate_count)),
                        );
                        decode_tile(thread_scratch, tile_index, variant_records, &mut tile_stats)
                    },
                ),
            (None, None) => selected_variant_records
                .par_chunks(decode_tile_variant_count)
                .zip(dosage_sum.par_chunks_mut(decode_tile_variant_count))
                .zip(dosage_square_sum.par_chunks_mut(decode_tile_variant_count))
                .zip(observation_count.par_chunks_mut(decode_tile_variant_count))
                .enumerate()
                .try_for_each_init(
                    ThreadScratch::default,
                    |thread_scratch, (tile_index, (((variant_records, dosage_sum), dosage_square_sum), observation_count))| {
                        let mut tile_stats =
                            variant_major_tile_stats_mut(dosage_sum, dosage_square_sum, observation_count, None);
                        decode_tile(thread_scratch, tile_index, variant_records, &mut tile_stats)
                    },
                ),
            _ => unreachable!("sparse statistic buffers are allocated together"),
        }
        .map_err(|failure| self.contextualize_variant_decode_failure(request.variant_start, failure))?;
        Ok(())
    }

    fn decode_trusted_variant_major_dosage_tiles_without_statistics(
        &self,
        request: VariantMajorDecodeRequest<'_>,
    ) -> Result<(), BgenError> {
        let decode_tile_variant_count = request.decode_tile_variant_count;
        self.variant_records[request.variant_start..request.variant_stop()]
            .par_chunks(decode_tile_variant_count)
            .enumerate()
            .try_for_each_init(ThreadScratch::default, |thread_scratch, (tile_index, variant_records)| {
                trusted::decode_trusted_variant_major_dosage_tile_without_statistics(
                    &self.mmap,
                    self.compression_type,
                    self.sample_count,
                    request.sample_selection,
                    variant_records,
                    request.output_pointer_address,
                    request.shape.selected_sample_count,
                    tile_index * decode_tile_variant_count,
                    false,
                    thread_scratch,
                )
            })
            .map_err(|failure| self.contextualize_variant_decode_failure(request.variant_start, failure))
    }

    fn decode_preprocessed_variant_major_packed8_probability_pair_tiles(
        &self,
        request: VariantMajorDecodeRequest<'_>,
        stats_buffers: &mut VariantMajorStatsBuffers,
    ) -> Result<(), BgenError> {
        let decode_tile_variant_count = request.decode_tile_variant_count;
        let selected_variant_records = &self.variant_records[request.variant_start..request.variant_stop()];
        let decode_tile = |thread_scratch: &mut ThreadScratch,
                           tile_index: usize,
                           variant_record_chunk: &[crate::bgen::metadata::VariantRecord],
                           tile_stats: &mut VariantMajorTileStatsMut<'_>| {
            trusted::decode_trusted_variant_major_packed8_probability_pair_tile(
                &self.mmap,
                self.compression_type,
                self.sample_count,
                request.sample_selection,
                variant_record_chunk,
                request.output_pointer_address,
                request.shape.selected_sample_count,
                tile_index * decode_tile_variant_count,
                false,
                tile_stats,
                thread_scratch,
            )
        };
        let VariantMajorStatsBuffers {
            dosage_sum,
            dosage_square_sum,
            observation_count,
            zero_count,
            homozygous_alternate_count,
        } = stats_buffers;
        match (zero_count.as_mut(), homozygous_alternate_count.as_mut()) {
            (Some(zero_count), Some(homozygous_alternate_count)) => selected_variant_records
                .par_chunks(decode_tile_variant_count)
                .zip(dosage_sum.par_chunks_mut(decode_tile_variant_count))
                .zip(dosage_square_sum.par_chunks_mut(decode_tile_variant_count))
                .zip(observation_count.par_chunks_mut(decode_tile_variant_count))
                .zip(zero_count.par_chunks_mut(decode_tile_variant_count))
                .zip(homozygous_alternate_count.par_chunks_mut(decode_tile_variant_count))
                .enumerate()
                .try_for_each_init(
                    ThreadScratch::default,
                    |thread_scratch,
                     (
                        tile_index,
                        (
                            ((((variant_records, dosage_sum), dosage_square_sum), observation_count), zero_count),
                            homozygous_alternate_count,
                        ),
                    )| {
                        let mut tile_stats = variant_major_tile_stats_mut(
                            dosage_sum,
                            dosage_square_sum,
                            observation_count,
                            Some((zero_count, homozygous_alternate_count)),
                        );
                        decode_tile(thread_scratch, tile_index, variant_records, &mut tile_stats)
                    },
                ),
            (None, None) => selected_variant_records
                .par_chunks(decode_tile_variant_count)
                .zip(dosage_sum.par_chunks_mut(decode_tile_variant_count))
                .zip(dosage_square_sum.par_chunks_mut(decode_tile_variant_count))
                .zip(observation_count.par_chunks_mut(decode_tile_variant_count))
                .enumerate()
                .try_for_each_init(
                    ThreadScratch::default,
                    |thread_scratch, (tile_index, (((variant_records, dosage_sum), dosage_square_sum), observation_count))| {
                        let mut tile_stats =
                            variant_major_tile_stats_mut(dosage_sum, dosage_square_sum, observation_count, None);
                        decode_tile(thread_scratch, tile_index, variant_records, &mut tile_stats)
                    },
                ),
            _ => unreachable!("sparse statistic buffers are allocated together"),
        }
        .map_err(|failure| self.contextualize_variant_decode_failure(request.variant_start, failure))?;
        Ok(())
    }
}

fn validate_variant_major_dosage_output_value_count(
    read_shape: VariantMajorReadShape,
    output_value_count: OutputValueCount,
) -> Result<(), BgenError> {
    let expected_output_value_count =
        read_shape.selected_sample_count.checked_mul(read_shape.selected_variant_count).ok_or_else(|| {
            BgenError::Range("Integer overflow while validating variant-major BGEN output buffer size.".to_string())
        })?;
    if output_value_count.get() != expected_output_value_count {
        return Err(BgenError::Range(format!(
            "Variant-major output buffer shape mismatch for BGEN dosage read. Expected {expected_output_value_count} float32 values, observed {}.",
            output_value_count.get(),
        )));
    }
    Ok(())
}

fn validate_variant_major_packed8_probability_pair_output_value_count(
    read_shape: VariantMajorReadShape,
    output_value_count: OutputValueCount,
) -> Result<(), BgenError> {
    let expected_output_value_count = read_shape
        .selected_variant_count
        .checked_mul(read_shape.selected_sample_count)
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| {
            BgenError::Range(
                "Integer overflow while validating packed8 BGEN probability-pair output buffer size.".to_string(),
            )
        })?;
    if output_value_count.get() != expected_output_value_count {
        return Err(BgenError::Range(format!(
            "Variant-major packed8 output buffer shape mismatch for BGEN probability-pair read. Expected {expected_output_value_count} uint8 values, observed {}.",
            output_value_count.get(),
        )));
    }
    Ok(())
}

fn variant_major_tile_stats_mut<'a>(
    dosage_sum: &'a mut [f32],
    dosage_square_sum: &'a mut [f32],
    observation_count: &'a mut [i32],
    sparse_candidate_counts: Option<(&'a mut [i32], &'a mut [i32])>,
) -> VariantMajorTileStatsMut<'a> {
    VariantMajorTileStatsMut {
        dosage_sum,
        dosage_square_sum,
        observation_count,
        sparse_candidate_counts: sparse_candidate_counts.map(|(zero_count, homozygous_alternate_count)| {
            VariantMajorSparseCandidateCountsMut { zero_count, homozygous_alternate_count }
        }),
    }
}
