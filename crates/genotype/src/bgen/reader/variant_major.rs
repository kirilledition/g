use std::mem::MaybeUninit;

use rayon::prelude::*;

use crate::bgen::decode::{
    ThreadScratch, VariantDecodeFailure, VariantMajorTileDecodeRequest, VariantMajorTileStatsMut,
    decode_variant_major_dosage_tile, with_worker_thread_scratch,
};
use crate::bgen::error::BgenError;
use crate::bgen::packed8;
use crate::bgen::sample_selection::SampleSelection;
use crate::bgen::source::coalesced_variant_window_stop;
use crate::common::{
    ChunkStatisticsPolicy, ChunkStats, GenotypeBatch, GenotypeBatchPayload, OwnedGenotypeBuffer, SessionBufferPool,
    SparseCandidateSummary,
};
use crate::preprocess;

use super::{BgenReadSession, BgenReaderCore, validate_variant_bounds};

const BGEN_DECODE_TILE_VARIANT_COUNT: usize = 32;

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

    fn dosage_value_count(self) -> Result<usize, BgenError> {
        self.selected_variant_count.checked_mul(self.selected_sample_count).ok_or_else(|| {
            BgenError::Range("Integer overflow while sizing variant-major BGEN dosage output.".to_string())
        })
    }

    fn packed8_value_count(self) -> Result<usize, BgenError> {
        self.dosage_value_count()?.checked_mul(2).ok_or_else(|| {
            BgenError::Range("Integer overflow while sizing variant-major packed8 BGEN output.".to_string())
        })
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
struct VariantMajorDecodeRequest<'selection> {
    sample_selection: &'selection SampleSelection,
    variant_start: usize,
    shape: VariantMajorReadShape,
}

impl VariantMajorDecodeRequest<'_> {
    fn variant_stop(self) -> usize {
        self.variant_start + self.shape.selected_variant_count
    }

    fn dosage_tile_value_count(self) -> Result<usize, BgenError> {
        BGEN_DECODE_TILE_VARIANT_COUNT
            .min(self.shape.selected_variant_count)
            .checked_mul(self.shape.selected_sample_count)
            .ok_or_else(|| BgenError::Range("Integer overflow while sizing a BGEN dosage decode tile.".to_string()))
    }

    fn packed8_tile_value_count(self) -> Result<usize, BgenError> {
        self.dosage_tile_value_count()?
            .checked_mul(2)
            .ok_or_else(|| BgenError::Range("Integer overflow while sizing a packed8 BGEN decode tile.".to_string()))
    }
}

struct VariantMajorStatsBuffers {
    dosage_sum: Vec<f32>,
    dosage_square_sum: Vec<f32>,
    observation_count: Vec<i32>,
    sparse_candidate_statistics: Option<Vec<SparseCandidateSummary>>,
}

struct OwnedVariantMajorDecode {
    genotypes: OwnedGenotypeBuffer,
    statistics: ChunkStats,
}

impl VariantMajorStatsBuffers {
    fn new(selected_variant_count: usize, statistics_policy: ChunkStatisticsPolicy) -> Self {
        Self {
            dosage_sum: vec![0.0_f32; selected_variant_count],
            dosage_square_sum: vec![0.0_f32; selected_variant_count],
            observation_count: vec![0_i32; selected_variant_count],
            sparse_candidate_statistics: statistics_policy
                .collect_sparse_candidate_mask
                .then(|| vec![SparseCandidateSummary::default(); selected_variant_count]),
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
            self.sparse_candidate_statistics,
            selected_sample_count,
            statistics_policy,
        )
        .map_err(|error| BgenError::Range(error.to_string()))
    }
}

impl BgenReadSession<'_> {
    /// Decode one variant-major batch into an exclusively owned output buffer.
    ///
    /// Batches may include compute-only tail variants so every JAX submission
    /// retains one shape. Dosage tails are zero-filled; packed8 tails use
    /// `[255, 0]`, the canonical zero-dosage probability pair.
    ///
    /// # Errors
    ///
    /// Returns an error when bounds or dimensions are invalid, packed8
    /// preconditions are unavailable, or BGEN decoding fails.
    pub fn decode_variant_major_batch(
        &self,
        variant_start: usize,
        variant_stop: usize,
        compute_variant_count: usize,
        use_packed8: bool,
        statistics_policy: ChunkStatisticsPolicy,
    ) -> Result<GenotypeBatch, BgenError> {
        validate_variant_bounds(variant_start, variant_stop, self.reader.variant_count())?;
        let read_shape = VariantMajorReadShape::from_selection(&self.sample_selection, variant_start, variant_stop);
        if compute_variant_count < read_shape.selected_variant_count {
            return Err(BgenError::Range(format!(
                "Compute variant count {compute_variant_count} is smaller than logical variant count {}.",
                read_shape.selected_variant_count,
            )));
        }
        let mut decoded = if use_packed8 {
            self.decode_owned_packed8_batch(read_shape, variant_start, compute_variant_count, statistics_policy)?
        } else {
            self.decode_owned_dosage_batch(read_shape, variant_start, compute_variant_count, statistics_policy)?
        };
        pad_compute_statistics(&mut decoded.statistics, compute_variant_count);
        self.reader.ensure_delivery_source_unchanged("BGEN source changed while a genotype batch was being read.")?;

        Ok(GenotypeBatch {
            variant_start_index: variant_start,
            logical_variant_count: read_shape.selected_variant_count,
            compute_variant_count,
            sample_count: read_shape.selected_sample_count,
            payload: GenotypeBatchPayload::Decoded { genotypes: decoded.genotypes, statistics: decoded.statistics },
        })
    }

    fn decode_owned_dosage_batch(
        &self,
        read_shape: VariantMajorReadShape,
        variant_start: usize,
        compute_variant_count: usize,
        statistics_policy: ChunkStatisticsPolicy,
    ) -> Result<OwnedVariantMajorDecode, BgenError> {
        let logical_output_value_count = read_shape.dosage_value_count()?;
        let compute_output_value_count = compute_variant_count
            .checked_mul(read_shape.selected_sample_count)
            .ok_or_else(|| BgenError::Range("Integer overflow while sizing compute dosage BGEN output.".to_string()))?;
        let mut output_values = Vec::<f32>::with_capacity(compute_output_value_count);
        let statistics = {
            let uninitialized_output = &mut output_values.spare_capacity_mut()[..compute_output_value_count];
            let (logical_output, compute_tail) = uninitialized_output.split_at_mut(logical_output_value_count);
            let statistics = self.reader.read_preprocessed_variant_major_dosage_f32_with_selection(
                &self.sample_selection,
                &self.positioned_source_window_pool,
                variant_start,
                variant_start + read_shape.selected_variant_count,
                logical_output,
                statistics_policy,
            )?;
            for dosage in compute_tail {
                dosage.write(0.0_f32);
            }
            statistics
        };
        unsafe {
            // The logical decoder and explicit zero tail cover the entire
            // compute allocation before success is published.
            output_values.set_len(compute_output_value_count);
        }
        Ok(OwnedVariantMajorDecode { genotypes: OwnedGenotypeBuffer::Dosage(output_values), statistics })
    }

    fn decode_owned_packed8_batch(
        &self,
        read_shape: VariantMajorReadShape,
        variant_start: usize,
        compute_variant_count: usize,
        statistics_policy: ChunkStatisticsPolicy,
    ) -> Result<OwnedVariantMajorDecode, BgenError> {
        let logical_output_value_count = read_shape.packed8_value_count()?;
        let compute_output_value_count = compute_variant_count
            .checked_mul(read_shape.selected_sample_count)
            .and_then(|value_count| value_count.checked_mul(2))
            .ok_or_else(|| {
                BgenError::Range("Integer overflow while sizing compute packed8 BGEN output.".to_string())
            })?;
        let mut output_values =
            crate::common::PooledPacked8Buffer::acquire(&self.packed8_buffer_pool, compute_output_value_count);
        let statistics = {
            let uninitialized_output = &mut output_values.values.spare_capacity_mut()[..compute_output_value_count];
            let (logical_output, compute_tail) = uninitialized_output.split_at_mut(logical_output_value_count);
            let statistics = self.reader.read_preprocessed_variant_major_packed8_probability_pairs_with_selection(
                &self.sample_selection,
                &self.positioned_source_window_pool,
                variant_start,
                variant_start + read_shape.selected_variant_count,
                logical_output,
                statistics_policy,
            )?;
            let (probability_pairs, remainder) = compute_tail.as_chunks_mut::<2>();
            if !remainder.is_empty() {
                return Err(BgenError::Range(
                    "Packed8 compute tail must contain complete probability pairs.".to_string(),
                ));
            }
            for probability_pair in probability_pairs {
                probability_pair[0].write(u8::MAX);
                probability_pair[1].write(0_u8);
            }
            statistics
        };
        unsafe {
            // The logical decoder and explicit `[255, 0]` tail initialization
            // cover the entire compute allocation before success is published.
            output_values.values.set_len(compute_output_value_count);
        }
        Ok(OwnedVariantMajorDecode { genotypes: OwnedGenotypeBuffer::Packed8(output_values), statistics })
    }
}

impl BgenReaderCore {
    fn read_preprocessed_variant_major_dosage_f32_with_selection(
        &self,
        sample_selection: &SampleSelection,
        positioned_source_window_pool: &SessionBufferPool<Vec<u8>>,
        variant_start: usize,
        variant_stop: usize,
        output_values: &mut [MaybeUninit<f32>],
        statistics_policy: ChunkStatisticsPolicy,
    ) -> Result<ChunkStats, BgenError> {
        validate_variant_bounds(variant_start, variant_stop, self.variant_count())?;
        let read_shape = VariantMajorReadShape::from_selection(sample_selection, variant_start, variant_stop);
        validate_output_value_count(read_shape.dosage_value_count()?, output_values.len(), "BGEN dosage")?;
        if let Some(empty_chunk_stats) = read_shape.empty_chunk_stats(statistics_policy) {
            return Ok(empty_chunk_stats);
        }

        let decode_request = VariantMajorDecodeRequest { sample_selection, variant_start, shape: read_shape };
        let mut stats_buffers = VariantMajorStatsBuffers::new(read_shape.selected_variant_count, statistics_policy);
        self.decode_preprocessed_variant_major_dosage_tiles(
            decode_request,
            positioned_source_window_pool,
            output_values,
            &mut stats_buffers,
        )?;
        stats_buffers.into_chunk_stats(read_shape.selected_sample_count, statistics_policy)
    }

    fn read_preprocessed_variant_major_packed8_probability_pairs_with_selection(
        &self,
        sample_selection: &SampleSelection,
        positioned_source_window_pool: &SessionBufferPool<Vec<u8>>,
        variant_start: usize,
        variant_stop: usize,
        output_values: &mut [MaybeUninit<u8>],
        statistics_policy: ChunkStatisticsPolicy,
    ) -> Result<ChunkStats, BgenError> {
        validate_variant_bounds(variant_start, variant_stop, self.variant_count())?;
        self.validate_packed8_probability_pair_preconditions()?;
        let read_shape = VariantMajorReadShape::from_selection(sample_selection, variant_start, variant_stop);
        validate_output_value_count(read_shape.packed8_value_count()?, output_values.len(), "packed8 BGEN")?;
        if let Some(empty_chunk_stats) = read_shape.empty_chunk_stats(statistics_policy) {
            return Ok(empty_chunk_stats);
        }

        let decode_request = VariantMajorDecodeRequest { sample_selection, variant_start, shape: read_shape };
        let mut stats_buffers = VariantMajorStatsBuffers::new(read_shape.selected_variant_count, statistics_policy);
        self.decode_preprocessed_variant_major_packed8_probability_pair_tiles(
            decode_request,
            positioned_source_window_pool,
            output_values,
            &mut stats_buffers,
        )?;
        stats_buffers.into_chunk_stats(read_shape.selected_sample_count, statistics_policy)
    }

    fn decode_preprocessed_variant_major_dosage_tiles(
        &self,
        request: VariantMajorDecodeRequest<'_>,
        positioned_source_window_pool: &SessionBufferPool<Vec<u8>>,
        output_values: &mut [MaybeUninit<f32>],
        stats_buffers: &mut VariantMajorStatsBuffers,
    ) -> Result<(), BgenError> {
        let output_tile_value_count = request.dosage_tile_value_count()?;
        let output_values_per_variant = request.shape.selected_sample_count;
        let selected_variant_records = &self.variant_records()[request.variant_start..request.variant_stop()];
        if let Some(source_window) = self.source.full_snapshot_window() {
            let decode_tile = |thread_scratch: &mut ThreadScratch,
                               tile_index: usize,
                               variant_record_chunk: &[crate::bgen::metadata::VariantRecord],
                               output_tile: &mut [MaybeUninit<f32>],
                               tile_stats: &mut VariantMajorTileStatsMut<'_>| {
                decode_variant_major_dosage_tile(
                    VariantMajorTileDecodeRequest {
                        source_window,
                        compression_type: self.compression_type(),
                        sample_count: self.sample_count(),
                        sample_selection: request.sample_selection,
                        variant_records: variant_record_chunk,
                        tile_variant_start_index: tile_index * BGEN_DECODE_TILE_VARIANT_COUNT,
                    },
                    output_tile,
                    tile_stats,
                    thread_scratch,
                )
            };
            decode_variant_major_tiles(
                selected_variant_records,
                output_values,
                output_tile_value_count,
                stats_buffers,
                0,
                decode_tile,
            )
            .map_err(|failure| self.contextualize_variant_decode_failure(request.variant_start, failure))?;
            return Ok(());
        }

        let mut source_window_buffer = positioned_source_window_pool.take_matching(|_buffer| true).unwrap_or_default();
        let decode_result = (|| {
            let mut window_variant_start = 0_usize;
            while window_variant_start < selected_variant_records.len() {
                let window_variant_stop = coalesced_variant_window_stop(selected_variant_records, window_variant_start)
                    .map_err(|error| {
                        self.contextualize_variant_error(request.variant_start + window_variant_start, error)
                    })?;
                let window_variant_records = &selected_variant_records[window_variant_start..window_variant_stop];
                let source_window =
                    self.source.read_variant_window(window_variant_records, &mut source_window_buffer).map_err(
                        |error| self.contextualize_variant_error(request.variant_start + window_variant_start, error),
                    )?;
                let output_value_range =
                    variant_value_range(window_variant_start, window_variant_stop, output_values_per_variant)?;
                let decode_tile = |thread_scratch: &mut ThreadScratch,
                                   tile_index: usize,
                                   variant_record_chunk: &[crate::bgen::metadata::VariantRecord],
                                   output_tile: &mut [MaybeUninit<f32>],
                                   tile_stats: &mut VariantMajorTileStatsMut<'_>| {
                    decode_variant_major_dosage_tile(
                        VariantMajorTileDecodeRequest {
                            source_window,
                            compression_type: self.compression_type(),
                            sample_count: self.sample_count(),
                            sample_selection: request.sample_selection,
                            variant_records: variant_record_chunk,
                            tile_variant_start_index: window_variant_start
                                + tile_index * BGEN_DECODE_TILE_VARIANT_COUNT,
                        },
                        output_tile,
                        tile_stats,
                        thread_scratch,
                    )
                };
                decode_variant_major_tiles(
                    window_variant_records,
                    &mut output_values[output_value_range],
                    output_tile_value_count,
                    stats_buffers,
                    window_variant_start,
                    decode_tile,
                )
                .map_err(|failure| self.contextualize_variant_decode_failure(request.variant_start, failure))?;
                window_variant_start = window_variant_stop;
            }
            Ok(())
        })();
        positioned_source_window_pool.release(source_window_buffer);
        decode_result
    }

    fn decode_preprocessed_variant_major_packed8_probability_pair_tiles(
        &self,
        request: VariantMajorDecodeRequest<'_>,
        positioned_source_window_pool: &SessionBufferPool<Vec<u8>>,
        output_values: &mut [MaybeUninit<u8>],
        stats_buffers: &mut VariantMajorStatsBuffers,
    ) -> Result<(), BgenError> {
        let output_tile_value_count = request.packed8_tile_value_count()?;
        let output_values_per_variant = request.shape.selected_sample_count.checked_mul(2).ok_or_else(|| {
            BgenError::Range("Integer overflow while sizing positioned packed8 BGEN output rows.".to_string())
        })?;
        let selected_variant_records = &self.variant_records()[request.variant_start..request.variant_stop()];
        if let Some(source_window) = self.source.full_snapshot_window() {
            let decode_tile = |thread_scratch: &mut ThreadScratch,
                               tile_index: usize,
                               variant_record_chunk: &[crate::bgen::metadata::VariantRecord],
                               output_tile: &mut [MaybeUninit<u8>],
                               tile_stats: &mut VariantMajorTileStatsMut<'_>| {
                packed8::decode_variant_major_probability_pair_tile(
                    source_window,
                    self.compression_type(),
                    self.sample_count(),
                    request.sample_selection,
                    variant_record_chunk,
                    output_tile,
                    tile_index * BGEN_DECODE_TILE_VARIANT_COUNT,
                    tile_stats,
                    thread_scratch,
                )
            };
            decode_variant_major_tiles(
                selected_variant_records,
                output_values,
                output_tile_value_count,
                stats_buffers,
                0,
                decode_tile,
            )
            .map_err(|failure| self.contextualize_variant_decode_failure(request.variant_start, failure))?;
            return Ok(());
        }

        let mut source_window_buffer = positioned_source_window_pool.take_matching(|_buffer| true).unwrap_or_default();
        let decode_result = (|| {
            let mut window_variant_start = 0_usize;
            while window_variant_start < selected_variant_records.len() {
                let window_variant_stop = coalesced_variant_window_stop(selected_variant_records, window_variant_start)
                    .map_err(|error| {
                        self.contextualize_variant_error(request.variant_start + window_variant_start, error)
                    })?;
                let window_variant_records = &selected_variant_records[window_variant_start..window_variant_stop];
                let source_window =
                    self.source.read_variant_window(window_variant_records, &mut source_window_buffer).map_err(
                        |error| self.contextualize_variant_error(request.variant_start + window_variant_start, error),
                    )?;
                let output_value_range =
                    variant_value_range(window_variant_start, window_variant_stop, output_values_per_variant)?;
                let decode_tile = |thread_scratch: &mut ThreadScratch,
                                   tile_index: usize,
                                   variant_record_chunk: &[crate::bgen::metadata::VariantRecord],
                                   output_tile: &mut [MaybeUninit<u8>],
                                   tile_stats: &mut VariantMajorTileStatsMut<'_>| {
                    packed8::decode_variant_major_probability_pair_tile(
                        source_window,
                        self.compression_type(),
                        self.sample_count(),
                        request.sample_selection,
                        variant_record_chunk,
                        output_tile,
                        window_variant_start + tile_index * BGEN_DECODE_TILE_VARIANT_COUNT,
                        tile_stats,
                        thread_scratch,
                    )
                };
                decode_variant_major_tiles(
                    window_variant_records,
                    &mut output_values[output_value_range],
                    output_tile_value_count,
                    stats_buffers,
                    window_variant_start,
                    decode_tile,
                )
                .map_err(|failure| self.contextualize_variant_decode_failure(request.variant_start, failure))?;
                window_variant_start = window_variant_stop;
            }
            Ok(())
        })();
        positioned_source_window_pool.release(source_window_buffer);
        decode_result
    }
}

fn variant_value_range(
    variant_start: usize,
    variant_stop: usize,
    values_per_variant: usize,
) -> Result<std::ops::Range<usize>, BgenError> {
    let value_start = variant_start
        .checked_mul(values_per_variant)
        .ok_or_else(|| BgenError::Range("Integer overflow while positioning a BGEN output window.".to_string()))?;
    let value_stop = variant_stop
        .checked_mul(values_per_variant)
        .ok_or_else(|| BgenError::Range("Integer overflow while sizing a BGEN output window.".to_string()))?;
    Ok(value_start..value_stop)
}

fn decode_variant_major_tiles<Value, DecodeTile>(
    selected_variant_records: &[crate::bgen::metadata::VariantRecord],
    output_values: &mut [MaybeUninit<Value>],
    output_tile_value_count: usize,
    stats_buffers: &mut VariantMajorStatsBuffers,
    stats_variant_start: usize,
    decode_tile: DecodeTile,
) -> Result<(), VariantDecodeFailure>
where
    Value: Send,
    DecodeTile: Fn(
            &mut ThreadScratch,
            usize,
            &[crate::bgen::metadata::VariantRecord],
            &mut [MaybeUninit<Value>],
            &mut VariantMajorTileStatsMut<'_>,
        ) -> Result<(), VariantDecodeFailure>
        + Sync,
{
    let VariantMajorStatsBuffers { dosage_sum, dosage_square_sum, observation_count, sparse_candidate_statistics } =
        stats_buffers;
    let stats_variant_stop =
        stats_variant_start.checked_add(selected_variant_records.len()).ok_or_else(|| VariantDecodeFailure {
            relative_variant_index: None,
            source: BgenError::Range("Integer overflow while slicing BGEN statistics windows.".to_string()),
        })?;
    let stats_variant_range = stats_variant_start..stats_variant_stop;
    let dosage_sum = &mut dosage_sum[stats_variant_range.clone()];
    let dosage_square_sum = &mut dosage_square_sum[stats_variant_range.clone()];
    let observation_count = &mut observation_count[stats_variant_range.clone()];
    let sparse_candidate_statistics =
        sparse_candidate_statistics.as_mut().map(|values| &mut values[stats_variant_range]);
    match sparse_candidate_statistics {
        Some(sparse_candidate_statistics) => selected_variant_records
            .par_chunks(BGEN_DECODE_TILE_VARIANT_COUNT)
            .zip(output_values.par_chunks_mut(output_tile_value_count))
            .zip(dosage_sum.par_chunks_mut(BGEN_DECODE_TILE_VARIANT_COUNT))
            .zip(dosage_square_sum.par_chunks_mut(BGEN_DECODE_TILE_VARIANT_COUNT))
            .zip(observation_count.par_chunks_mut(BGEN_DECODE_TILE_VARIANT_COUNT))
            .zip(sparse_candidate_statistics.par_chunks_mut(BGEN_DECODE_TILE_VARIANT_COUNT))
            .enumerate()
            .try_for_each(
                |(
                    tile_index,
                    (
                        ((((variant_records, output_tile), dosage_sum), dosage_square_sum), observation_count),
                        sparse_candidate_statistics,
                    ),
                )| {
                    let mut tile_stats = variant_major_tile_stats_mut(
                        dosage_sum,
                        dosage_square_sum,
                        observation_count,
                        Some(sparse_candidate_statistics),
                    );
                    with_worker_thread_scratch(|thread_scratch| {
                        decode_tile(thread_scratch, tile_index, variant_records, output_tile, &mut tile_stats)
                    })
                },
            ),
        None => selected_variant_records
            .par_chunks(BGEN_DECODE_TILE_VARIANT_COUNT)
            .zip(output_values.par_chunks_mut(output_tile_value_count))
            .zip(dosage_sum.par_chunks_mut(BGEN_DECODE_TILE_VARIANT_COUNT))
            .zip(dosage_square_sum.par_chunks_mut(BGEN_DECODE_TILE_VARIANT_COUNT))
            .zip(observation_count.par_chunks_mut(BGEN_DECODE_TILE_VARIANT_COUNT))
            .enumerate()
            .try_for_each(
                |(
                    tile_index,
                    ((((variant_records, output_tile), dosage_sum), dosage_square_sum), observation_count),
                )| {
                    let mut tile_stats =
                        variant_major_tile_stats_mut(dosage_sum, dosage_square_sum, observation_count, None);
                    with_worker_thread_scratch(|thread_scratch| {
                        decode_tile(thread_scratch, tile_index, variant_records, output_tile, &mut tile_stats)
                    })
                },
            ),
    }
}

fn validate_output_value_count(
    expected_output_value_count: usize,
    observed_output_value_count: usize,
    output_context: &'static str,
) -> Result<(), BgenError> {
    if observed_output_value_count != expected_output_value_count {
        return Err(BgenError::Range(format!(
            "Variant-major output buffer shape mismatch for {output_context}. Expected {expected_output_value_count} values, observed {observed_output_value_count}.",
        )));
    }
    Ok(())
}

fn pad_compute_statistics(statistics: &mut ChunkStats, compute_variant_count: usize) {
    statistics.compute.genotype_mean.resize(compute_variant_count, 0.0_f32);
    if let Some(imputed_dosage_square_sum) = statistics.compute.imputed_dosage_square_sum.as_mut() {
        imputed_dosage_square_sum.resize(compute_variant_count, 0.0_f32);
    }
    if let Some(sparse_candidate_mask) = statistics.compute.sparse_candidate_mask.as_mut() {
        sparse_candidate_mask.resize(compute_variant_count, false);
    }
}

fn variant_major_tile_stats_mut<'buffers>(
    dosage_sum: &'buffers mut [f32],
    dosage_square_sum: &'buffers mut [f32],
    observation_count: &'buffers mut [i32],
    sparse_candidate_statistics: Option<&'buffers mut [SparseCandidateSummary]>,
) -> VariantMajorTileStatsMut<'buffers> {
    VariantMajorTileStatsMut { dosage_sum, dosage_square_sum, observation_count, sparse_candidate_statistics }
}
