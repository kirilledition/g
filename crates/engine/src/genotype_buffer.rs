//! Reusable host genotype buffers and grouped-sample projection.

use std::sync::Arc;

use g_genotype::{
    BgenError, BgenReaderCore, ChunkStatisticsPolicy, ChunkStats, GenotypeError, GenotypeResult, OutputBufferAddress,
    OutputValueCount,
};
use g_genotype_contracts::VariantMetadataColumns;

use crate::backend::OwnedGenotypeBuffer;

/// Reusable exact-size buffers for decoded association batches.
#[derive(Default)]
pub struct GenotypeBufferPool {
    dosage_buffers: Vec<Vec<f32>>,
    packed8_buffers: Vec<Vec<u8>>,
}

/// Allocate one logical-size genotype buffer with capacity for its compute shape.
///
/// # Errors
///
/// Returns an error when dimensions or the packed representation overflow.
pub(crate) fn allocate_genotype_buffer(
    logical_variant_count: usize,
    compute_variant_count: usize,
    sample_count: usize,
    use_packed8: bool,
) -> GenotypeResult<OwnedGenotypeBuffer> {
    if compute_variant_count < logical_variant_count {
        return Err(GenotypeError::InvalidInput(format!(
            "Compute variant count {compute_variant_count} is smaller than logical variant count {logical_variant_count}."
        )));
    }
    let value_count = logical_variant_count
        .checked_mul(sample_count)
        .ok_or_else(|| GenotypeError::InvalidInput("Logical genotype dimensions overflowed usize.".to_string()))?;
    let value_capacity = compute_variant_count
        .checked_mul(sample_count)
        .ok_or_else(|| GenotypeError::InvalidInput("Compute genotype dimensions overflowed usize.".to_string()))?;
    if use_packed8 {
        let packed_value_count = value_count
            .checked_mul(2)
            .ok_or_else(|| GenotypeError::InvalidInput("Packed8 genotype buffer size overflowed usize.".to_string()))?;
        let packed_value_capacity = value_capacity.checked_mul(2).ok_or_else(|| {
            GenotypeError::InvalidInput("Packed8 genotype buffer capacity overflowed usize.".to_string())
        })?;
        let mut values = Vec::with_capacity(packed_value_capacity);
        values.resize(packed_value_count, 0_u8);
        return Ok(OwnedGenotypeBuffer::Packed8(values));
    }
    let mut values = Vec::with_capacity(value_capacity);
    values.resize(value_count, 0.0_f32);
    Ok(OwnedGenotypeBuffer::Dosage(values))
}

impl GenotypeBufferPool {
    /// Acquire a zero-initialized buffer with the requested logical size.
    ///
    /// # Errors
    ///
    /// Returns an error when the packed representation size overflows.
    pub fn acquire(&mut self, value_count: usize, use_packed8: bool) -> GenotypeResult<OwnedGenotypeBuffer> {
        if use_packed8 {
            let packed_value_count = value_count.checked_mul(2).ok_or_else(|| {
                GenotypeError::InvalidInput("Packed8 genotype buffer size overflowed usize.".to_string())
            })?;
            let values = take_matching_buffer(&mut self.packed8_buffers, packed_value_count)
                .unwrap_or_else(|| vec![0_u8; packed_value_count]);
            return Ok(OwnedGenotypeBuffer::Packed8(values));
        }
        let values =
            take_matching_buffer(&mut self.dosage_buffers, value_count).unwrap_or_else(|| vec![0.0_f32; value_count]);
        Ok(OwnedGenotypeBuffer::Dosage(values))
    }

    /// Return a completed buffer for reuse.
    pub fn release(&mut self, buffer: OwnedGenotypeBuffer) {
        match buffer {
            OwnedGenotypeBuffer::Dosage(values) => self.dosage_buffers.push(values),
            OwnedGenotypeBuffer::Packed8(values) => self.packed8_buffers.push(values),
        }
    }
}

/// Decode one prepared BGEN interval and pad packed8 compute inputs in place.
///
/// # Errors
///
/// Returns an error when the reader cannot decode the requested interval.
pub(crate) fn decode_genotype_buffer(
    reader: &BgenReaderCore,
    variant_start_index: usize,
    variant_stop_index: usize,
    buffer: &mut OwnedGenotypeBuffer,
    statistics_policy: ChunkStatisticsPolicy,
    compute_variant_count: usize,
    sample_count: usize,
) -> Result<ChunkStats, BgenError> {
    let mut statistics = match buffer {
        OwnedGenotypeBuffer::Dosage(values) => reader.read_preprocessed_variant_major_dosage_f32_into_address_prepared(
            variant_start_index,
            variant_stop_index,
            OutputBufferAddress::from_mut_ptr(values.as_mut_ptr()),
            OutputValueCount::new(values.len()),
            statistics_policy,
        ),
        OwnedGenotypeBuffer::Packed8(values) => reader
            .read_preprocessed_variant_major_packed8_probability_pairs_into_address_prepared(
                variant_start_index,
                variant_stop_index,
                OutputBufferAddress::from_mut_ptr(values.as_mut_ptr()),
                OutputValueCount::new(values.len()),
                statistics_policy,
            ),
    }?;
    if let OwnedGenotypeBuffer::Packed8(values) = buffer {
        let logical_variant_count = statistics.compute.genotype_mean.len();
        if compute_variant_count != logical_variant_count {
            let logical_value_count = values.len();
            let compute_value_count = compute_variant_count
                .checked_mul(sample_count)
                .and_then(|value| value.checked_mul(2))
                .expect("compute dimensions were validated during allocation");
            debug_assert!(compute_variant_count > logical_variant_count);
            debug_assert!(values.capacity() >= compute_value_count);
            values.resize(compute_value_count, 0_u8);
            // Packed `[255, 0]` pairs decode to monomorphic zero dosage.
            for probability_pair in values[logical_value_count..].chunks_exact_mut(2) {
                probability_pair[0] = u8::MAX;
            }
            statistics.compute.genotype_mean.resize(compute_variant_count, 0.0_f32);
            if let Some(imputed_dosage_square_sum) = statistics.compute.imputed_dosage_square_sum.as_mut() {
                imputed_dosage_square_sum.resize(compute_variant_count, 0.0_f32);
            }
            if let Some(sparse_candidate_mask) = statistics.compute.sparse_candidate_mask.as_mut() {
                sparse_candidate_mask.resize(compute_variant_count, false);
            }
        }
    }
    Ok(statistics)
}

/// Project one variant-major union-sample matrix into a group sample order.
///
/// # Errors
///
/// Returns an error when dimensions overflow, buffer lengths disagree, or a
/// group sample position is outside the union.
pub(crate) fn project_variant_major_dosages(
    union_dosages: &[f32],
    union_sample_count: usize,
    variant_count: usize,
    group_sample_positions: &[usize],
    group_dosages: &mut [f32],
) -> GenotypeResult<()> {
    let expected_union_value_count = variant_count
        .checked_mul(union_sample_count)
        .ok_or_else(|| GenotypeError::InvalidInput("Union genotype dimensions overflowed usize.".to_string()))?;
    if union_dosages.len() != expected_union_value_count {
        return Err(GenotypeError::InvalidInput(format!(
            "Union genotype buffer contains {} values, expected {expected_union_value_count}.",
            union_dosages.len()
        )));
    }
    let expected_group_value_count = variant_count
        .checked_mul(group_sample_positions.len())
        .ok_or_else(|| GenotypeError::InvalidInput("Projected genotype dimensions overflowed usize.".to_string()))?;
    if group_dosages.len() != expected_group_value_count {
        return Err(GenotypeError::InvalidInput(format!(
            "Projected genotype buffer contains {} values, expected {expected_group_value_count}.",
            group_dosages.len()
        )));
    }
    for (union_row, group_row) in
        union_dosages.chunks_exact(union_sample_count).zip(group_dosages.chunks_exact_mut(group_sample_positions.len()))
    {
        for (output_value, sample_position) in group_row.iter_mut().zip(group_sample_positions) {
            *output_value = union_row[*sample_position];
        }
    }
    Ok(())
}

/// Resolve the chromosome represented by one planner-produced chunk.
///
/// # Errors
///
/// Returns an error for empty or inconsistent metadata. The chunk planner owns
/// the chromosome-homogeneity invariant.
pub(crate) fn homogeneous_chunk_chromosome(
    metadata: &VariantMetadataColumns,
    variant_count: usize,
) -> GenotypeResult<Arc<str>> {
    if variant_count == 0 {
        return Err(GenotypeError::InvalidInput("Association delivery received an empty BGEN chunk.".to_string()));
    }
    if metadata.len() != variant_count {
        return Err(GenotypeError::InvalidInput(format!(
            "Chromosome metadata contains {} values for a {variant_count}-variant chunk.",
            metadata.len()
        )));
    }
    let chromosome = metadata
        .shared_chromosome(0)
        .ok_or_else(|| GenotypeError::InvalidInput("Association delivery chunk has no chromosome.".to_string()))?;
    debug_assert!(metadata.chromosomes().all(|value| value == chromosome.as_ref()));
    Ok(chromosome)
}

fn take_matching_buffer<Buffer>(buffers: &mut Vec<Vec<Buffer>>, value_count: usize) -> Option<Vec<Buffer>> {
    let buffer_index = buffers.iter().position(|values| values.len() == value_count)?;
    Some(buffers.swap_remove(buffer_index))
}
