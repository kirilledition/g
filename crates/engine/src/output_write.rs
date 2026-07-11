//! Association result delivery to native output sessions.

use std::sync::Arc;

use g_genotype_contracts::ChunkOutputStatistics;
use g_output::{
    NativeChunkHandle, NativeVariantMetadataHandle, OutputError, OutputWriterSession, Regenie2StatisticBatch,
};

use crate::backend::HostAssociationBatch;

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
    result: HostAssociationBatch,
) -> Result<(), OutputError> {
    let chunk_identifier = i64::try_from(variant_start_index)
        .map_err(|_| OutputError::InvalidInput("Variant start index does not fit into int64 output.".to_string()))?;
    let chunk_handle = NativeChunkHandle::try_new(metadata, statistics, chunk_identifier)?;
    let expected_trait_count = active_trait_indices.map_or(writer_sessions.len(), <[usize]>::len);
    let expected_variant_count = chunk_handle.row_count();
    validate_host_statistic_shape(
        result.trait_count,
        result.variant_count,
        expected_trait_count,
        expected_variant_count,
    )?;
    g_output::write_regenie2_multi_trait_chunk_f32(
        writer_sessions,
        active_trait_indices,
        &chunk_handle,
        Regenie2StatisticBatch {
            beta: result.beta,
            standard_error: result.standard_error,
            chi_squared: result.chi_squared,
            log10_p_value: result.log10_p_value,
            correction_code: result.correction_codes,
        },
    )
}

fn validate_host_statistic_shape(
    trait_count: usize,
    variant_count: usize,
    expected_trait_count: usize,
    expected_variant_count: usize,
) -> Result<(), OutputError> {
    if trait_count != expected_trait_count || variant_count != expected_variant_count {
        return Err(OutputError::InvalidInput(format!(
            "Materialized statistic shape ({trait_count}, {variant_count}) does not match expected ({expected_trait_count}, {expected_variant_count})."
        )));
    }
    Ok(())
}
