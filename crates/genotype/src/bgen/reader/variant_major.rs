use std::mem::MaybeUninit;

use rayon::prelude::*;

use crate::bgen::decode::{
    ThreadScratch, VariantMajorSparseCandidateCountsMut, VariantMajorTileStatsMut, decode_variant_major_dosage_tile,
    with_worker_thread_scratch,
};
use crate::bgen::error::BgenError;
use crate::bgen::packed8;
use crate::bgen::sample_selection::SampleSelection;
use crate::common::{ChunkStatisticsPolicy, ChunkStats, DecodedGenotypeBatch, OwnedGenotypeBuffer};
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
    zero_count: Option<Vec<i32>>,
    homozygous_alternate_count: Option<Vec<i32>>,
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
    ) -> Result<DecodedGenotypeBatch, BgenError> {
        validate_variant_bounds(variant_start, variant_stop, self.reader.variant_count)?;
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

        Ok(DecodedGenotypeBatch {
            variant_start_index: variant_start,
            logical_variant_count: read_shape.selected_variant_count,
            compute_variant_count,
            sample_count: read_shape.selected_sample_count,
            genotypes: decoded.genotypes,
            statistics: decoded.statistics,
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
    #[allow(clippy::too_many_arguments)]
    fn read_preprocessed_variant_major_dosage_f32_with_selection(
        &self,
        sample_selection: &SampleSelection,
        variant_start: usize,
        variant_stop: usize,
        output_values: &mut [MaybeUninit<f32>],
        statistics_policy: ChunkStatisticsPolicy,
    ) -> Result<ChunkStats, BgenError> {
        validate_variant_bounds(variant_start, variant_stop, self.variant_count)?;
        let read_shape = VariantMajorReadShape::from_selection(sample_selection, variant_start, variant_stop);
        validate_output_value_count(read_shape.dosage_value_count()?, output_values.len(), "BGEN dosage")?;
        if let Some(empty_chunk_stats) = read_shape.empty_chunk_stats(statistics_policy) {
            return Ok(empty_chunk_stats);
        }

        let decode_request = VariantMajorDecodeRequest { sample_selection, variant_start, shape: read_shape };
        let mut stats_buffers = VariantMajorStatsBuffers::new(read_shape.selected_variant_count, statistics_policy);
        self.decode_preprocessed_variant_major_dosage_tiles(decode_request, output_values, &mut stats_buffers)?;
        stats_buffers.into_chunk_stats(read_shape.selected_sample_count, statistics_policy)
    }

    #[allow(clippy::too_many_arguments)]
    fn read_preprocessed_variant_major_packed8_probability_pairs_with_selection(
        &self,
        sample_selection: &SampleSelection,
        variant_start: usize,
        variant_stop: usize,
        output_values: &mut [MaybeUninit<u8>],
        statistics_policy: ChunkStatisticsPolicy,
    ) -> Result<ChunkStats, BgenError> {
        validate_variant_bounds(variant_start, variant_stop, self.variant_count)?;
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
            output_values,
            &mut stats_buffers,
        )?;
        stats_buffers.into_chunk_stats(read_shape.selected_sample_count, statistics_policy)
    }

    fn decode_preprocessed_variant_major_dosage_tiles(
        &self,
        request: VariantMajorDecodeRequest<'_>,
        output_values: &mut [MaybeUninit<f32>],
        stats_buffers: &mut VariantMajorStatsBuffers,
    ) -> Result<(), BgenError> {
        let output_tile_value_count = request.dosage_tile_value_count()?;
        let selected_variant_records = &self.variant_records[request.variant_start..request.variant_stop()];
        let decode_tile = |thread_scratch: &mut ThreadScratch,
                           tile_index: usize,
                           variant_record_chunk: &[crate::bgen::metadata::VariantRecord],
                           output_tile: &mut [MaybeUninit<f32>],
                           tile_stats: &mut VariantMajorTileStatsMut<'_>| {
            decode_variant_major_dosage_tile(
                &self.mmap,
                self.compression_type,
                self.sample_count,
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
            decode_tile,
        )
        .map_err(|failure| self.contextualize_variant_decode_failure(request.variant_start, failure))?;
        Ok(())
    }

    fn decode_preprocessed_variant_major_packed8_probability_pair_tiles(
        &self,
        request: VariantMajorDecodeRequest<'_>,
        output_values: &mut [MaybeUninit<u8>],
        stats_buffers: &mut VariantMajorStatsBuffers,
    ) -> Result<(), BgenError> {
        let output_tile_value_count = request.packed8_tile_value_count()?;
        let selected_variant_records = &self.variant_records[request.variant_start..request.variant_stop()];
        let decode_tile = |thread_scratch: &mut ThreadScratch,
                           tile_index: usize,
                           variant_record_chunk: &[crate::bgen::metadata::VariantRecord],
                           output_tile: &mut [MaybeUninit<u8>],
                           tile_stats: &mut VariantMajorTileStatsMut<'_>| {
            packed8::decode_variant_major_probability_pair_tile(
                &self.mmap,
                self.compression_type,
                self.sample_count,
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
            decode_tile,
        )
        .map_err(|failure| self.contextualize_variant_decode_failure(request.variant_start, failure))?;
        Ok(())
    }
}

fn decode_variant_major_tiles<Value, DecodeTile>(
    selected_variant_records: &[crate::bgen::metadata::VariantRecord],
    output_values: &mut [MaybeUninit<Value>],
    output_tile_value_count: usize,
    stats_buffers: &mut VariantMajorStatsBuffers,
    decode_tile: DecodeTile,
) -> Result<(), crate::bgen::decode::VariantDecodeFailure>
where
    Value: Send,
    DecodeTile: Fn(
            &mut ThreadScratch,
            usize,
            &[crate::bgen::metadata::VariantRecord],
            &mut [MaybeUninit<Value>],
            &mut VariantMajorTileStatsMut<'_>,
        ) -> Result<(), crate::bgen::decode::VariantDecodeFailure>
        + Sync,
{
    let VariantMajorStatsBuffers {
        dosage_sum,
        dosage_square_sum,
        observation_count,
        zero_count,
        homozygous_alternate_count,
    } = stats_buffers;
    match (zero_count.as_mut(), homozygous_alternate_count.as_mut()) {
        (Some(zero_count), Some(homozygous_alternate_count)) => selected_variant_records
            .par_chunks(BGEN_DECODE_TILE_VARIANT_COUNT)
            .zip(output_values.par_chunks_mut(output_tile_value_count))
            .zip(dosage_sum.par_chunks_mut(BGEN_DECODE_TILE_VARIANT_COUNT))
            .zip(dosage_square_sum.par_chunks_mut(BGEN_DECODE_TILE_VARIANT_COUNT))
            .zip(observation_count.par_chunks_mut(BGEN_DECODE_TILE_VARIANT_COUNT))
            .zip(zero_count.par_chunks_mut(BGEN_DECODE_TILE_VARIANT_COUNT))
            .zip(homozygous_alternate_count.par_chunks_mut(BGEN_DECODE_TILE_VARIANT_COUNT))
            .enumerate()
            .try_for_each(
                |(
                    tile_index,
                    (
                        (
                            ((((variant_records, output_tile), dosage_sum), dosage_square_sum), observation_count),
                            zero_count,
                        ),
                        homozygous_alternate_count,
                    ),
                )| {
                    let mut tile_stats = variant_major_tile_stats_mut(
                        dosage_sum,
                        dosage_square_sum,
                        observation_count,
                        Some((zero_count, homozygous_alternate_count)),
                    );
                    with_worker_thread_scratch(|thread_scratch| {
                        decode_tile(thread_scratch, tile_index, variant_records, output_tile, &mut tile_stats)
                    })
                },
            ),
        (None, None) => selected_variant_records
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
        _ => unreachable!("sparse statistic buffers are allocated together"),
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
    sparse_candidate_counts: Option<(&'buffers mut [i32], &'buffers mut [i32])>,
) -> VariantMajorTileStatsMut<'buffers> {
    VariantMajorTileStatsMut {
        dosage_sum,
        dosage_square_sum,
        observation_count,
        sparse_candidate_counts: sparse_candidate_counts.map(|(zero_count, homozygous_alternate_count)| {
            VariantMajorSparseCandidateCountsMut { zero_count, homozygous_alternate_count }
        }),
    }
}
