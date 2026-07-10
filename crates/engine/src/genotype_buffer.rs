//! Reusable host genotype buffers and grouped-sample projection.

use g_genotype::{
    BgenError, BgenReaderCore, ChunkStats, GenotypeError, GenotypeResult, OutputBufferAddress, OutputValueCount,
    VariantMetadataColumns,
};

use crate::association_scheduler::OwnedGenotypeBuffer;

/// Reusable exact-size buffers for decoded association batches.
#[derive(Default)]
pub struct GenotypeBufferPool {
    dosage_buffers: Vec<Vec<f32>>,
    packed8_buffers: Vec<Vec<u8>>,
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

/// Decode one prepared BGEN interval into an acquired scheduler buffer.
///
/// # Errors
///
/// Returns an error when the reader cannot decode the requested interval.
pub fn decode_genotype_buffer(
    reader: &BgenReaderCore,
    variant_start_index: usize,
    variant_stop_index: usize,
    buffer: &mut OwnedGenotypeBuffer,
) -> Result<ChunkStats, BgenError> {
    match buffer {
        OwnedGenotypeBuffer::Dosage(values) => reader.read_preprocessed_variant_major_dosage_f32_into_address_prepared(
            variant_start_index,
            variant_stop_index,
            OutputBufferAddress::from_mut_ptr(values.as_mut_ptr()),
            OutputValueCount::new(values.len()),
        ),
        OwnedGenotypeBuffer::Packed8(values) => reader
            .read_preprocessed_variant_major_packed8_probability_pairs_into_address_prepared(
                variant_start_index,
                variant_stop_index,
                OutputBufferAddress::from_mut_ptr(values.as_mut_ptr()),
                OutputValueCount::new(values.len()),
            ),
    }
}

/// Project one variant-major union-sample matrix into a group sample order.
///
/// # Errors
///
/// Returns an error when dimensions overflow, buffer lengths disagree, or a
/// group sample position is outside the union.
pub fn project_variant_major_dosages(
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
    if let Some(invalid_position) = group_sample_positions.iter().find(|position| **position >= union_sample_count) {
        return Err(GenotypeError::InvalidInput(format!(
            "Grouped-union sample position {invalid_position} is out of range for {union_sample_count} samples."
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

/// Resolve the single chromosome represented by one decoded chunk.
///
/// # Errors
///
/// Returns an error for empty, inconsistent, or cross-chromosome metadata.
pub fn homogeneous_chunk_chromosome(metadata: &VariantMetadataColumns, variant_count: usize) -> GenotypeResult<String> {
    if variant_count == 0 {
        return Err(GenotypeError::InvalidInput("Association delivery received an empty BGEN chunk.".to_string()));
    }
    if metadata.chromosome.len() != variant_count {
        return Err(GenotypeError::InvalidInput(format!(
            "Chromosome metadata contains {} values for a {variant_count}-variant chunk.",
            metadata.chromosome.len()
        )));
    }
    let chromosome = metadata
        .chromosome
        .first()
        .ok_or_else(|| GenotypeError::InvalidInput("Association delivery chunk has no chromosome.".to_string()))?;
    if metadata.chromosome.iter().any(|value| value != chromosome) {
        return Err(GenotypeError::InvalidInput(
            "Association delivery received a chunk spanning multiple chromosomes.".to_string(),
        ));
    }
    Ok(chromosome.clone())
}

fn take_matching_buffer<Buffer>(buffers: &mut Vec<Vec<Buffer>>, value_count: usize) -> Option<Vec<Buffer>> {
    let buffer_index = buffers.iter().position(|values| values.len() == value_count)?;
    Some(buffers.swap_remove(buffer_index))
}
