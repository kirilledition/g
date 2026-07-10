//! Association result delivery to native output sessions.

use std::sync::Arc;

use g_genotype::{ChunkStats, VariantMetadataColumns};
use g_output::{NativeChunkHandle, OutputError, OutputWriterSession, Regenie2StatisticBatch};

use crate::backend::HostAssociationBatch;

/// Write one completed trait-major association batch.
///
/// # Errors
///
/// Returns an error when result shapes are inconsistent or an output session
/// rejects the chunk.
pub fn write_host_association_batch(
    writer_sessions: &[Arc<OutputWriterSession>],
    active_trait_indices: &[usize],
    variant_start_index: usize,
    metadata: VariantMetadataColumns,
    statistics: ChunkStats,
    result: HostAssociationBatch,
) -> Result<(), OutputError> {
    let chunk_identifier = i64::try_from(variant_start_index)
        .map_err(|_| OutputError::InvalidInput("Variant start index does not fit into int64 output.".to_string()))?;
    let chunk_handle = NativeChunkHandle::new(metadata, statistics, chunk_identifier);
    let expected_trait_count = active_trait_indices.len();
    let expected_variant_count = chunk_handle.row_count();
    if let Some(correction_codes) = result.correction_codes.as_ref()
        && (correction_codes.trait_count != expected_trait_count
            || correction_codes.variant_count != expected_variant_count)
    {
        return Err(OutputError::InvalidInput(
            "Materialized correction-code shape does not match active traits and variant metadata.".to_string(),
        ));
    }
    let statistics = result.statistics;
    validate_host_statistic_shape(
        statistics.trait_count,
        statistics.variant_count,
        expected_trait_count,
        expected_variant_count,
    )?;
    g_output::write_regenie2_multi_trait_chunk_f32(
        writer_sessions,
        active_trait_indices,
        &chunk_handle,
        Regenie2StatisticBatch {
            beta: statistics.beta,
            standard_error: statistics.standard_error,
            chi_squared: statistics.chi_squared,
            log10_p_value: statistics.log10_p_value,
            correction_code: result.correction_codes.map(|matrix| matrix.values),
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
