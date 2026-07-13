//! Association result delivery to native output sessions.

use std::sync::Arc;

use g_genotype_contracts::ChunkOutputStatistics;
use g_output::{
    NativeChunkHandle, NativeVariantMetadataHandle, OutputError, OutputWriterSession, Regenie2StatisticBatch,
};

/// Write one completed trait-major association batch.
///
/// # Errors
///
/// Returns an error when result shapes are inconsistent or an output session
/// rejects the chunk.
pub(crate) fn write_host_association_batch(
    writer_sessions: &[Arc<OutputWriterSession>],
    active_trait_indices: Option<&[usize]>,
    variant_start_index: usize,
    metadata: NativeVariantMetadataHandle,
    statistics: ChunkOutputStatistics,
    result: Regenie2StatisticBatch,
) -> Result<(), OutputError> {
    let chunk_identifier = i64::try_from(variant_start_index)
        .map_err(|_| OutputError::InvalidInput("Variant start index does not fit into int64 output.".to_string()))?;
    let chunk_handle = NativeChunkHandle::try_new(metadata, statistics, chunk_identifier)?;
    g_output::write_regenie2_multi_trait_chunk_f32(writer_sessions, active_trait_indices, &chunk_handle, result)
}
