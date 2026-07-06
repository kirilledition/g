use g_genotype::ChunkSpec;

use super::ScheduleError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VariantMajorDosageBatchHandoffPlan {
    pub chunk_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkBatchPlan {
    pub chunk_batches: Vec<Vec<ChunkSpec>>,
}

impl ChunkBatchPlan {
    #[must_use]
    pub fn chunk_batch_count(&self) -> usize {
        self.chunk_batches.len()
    }

    #[must_use]
    pub fn chunk_count(&self) -> usize {
        self.chunk_batches.iter().map(Vec::len).sum()
    }

    #[must_use]
    pub fn into_chunk_batches(self) -> Vec<Vec<ChunkSpec>> {
        self.chunk_batches
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DosageWorkHandoffPlan {
    pub chunk_count: usize,
}

/// Plan a variant-major dosage batch handoff into the callback queue.
///
/// # Errors
///
/// Returns an error when the metadata, genotype matrix, and chunk-stat batches
/// have different lengths, or when the batch is empty.
pub fn plan_variant_major_dosage_batch_handoff(
    metadata_count: usize,
    genotype_matrix_by_variant_count: usize,
    chunk_stats_count: usize,
) -> Result<VariantMajorDosageBatchHandoffPlan, ScheduleError> {
    if metadata_count != genotype_matrix_by_variant_count || metadata_count != chunk_stats_count {
        return Err(ScheduleError::VariantMajorDosageBatchLengthMismatch);
    }
    if metadata_count == 0 {
        return Err(ScheduleError::EmptyVariantMajorDosageBatch);
    }
    let handoff_plan = plan_dosage_work_handoff(metadata_count)?;
    Ok(VariantMajorDosageBatchHandoffPlan { chunk_count: handoff_plan.chunk_count })
}

/// Partition planned genotype chunks into callback batches.
///
/// # Errors
///
/// Returns an error when the callback batch size is zero.
pub fn plan_chunk_batches(
    chunk_specs: &[ChunkSpec],
    callback_batch_size: usize,
) -> Result<ChunkBatchPlan, ScheduleError> {
    if callback_batch_size == 0 {
        return Err(ScheduleError::NonPositiveCallbackBatchSize);
    }
    let chunk_batches = chunk_specs.chunks(callback_batch_size).map(<[ChunkSpec]>::to_vec).collect();
    Ok(ChunkBatchPlan { chunk_batches })
}

/// Plan a dosage work handoff into the callback queue.
///
/// # Errors
///
/// Returns an error when the handoff contains no chunks.
pub fn plan_dosage_work_handoff(chunk_count: usize) -> Result<DosageWorkHandoffPlan, ScheduleError> {
    if chunk_count == 0 {
        return Err(ScheduleError::EmptyDosageWorkHandoff);
    }
    Ok(DosageWorkHandoffPlan { chunk_count })
}
