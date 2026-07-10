//! Python error boundary backed by native `g-output` writer sessions.

use std::sync::Arc;

use g_output::{
    NativeChunkHandle, NativeChunkStats as NativeOutputChunkStats,
    VariantMetadataColumns as NativeOutputVariantMetadataColumns,
};
use g_runtime as native_run_events;
use pyo3::PyResult;
use pyo3::exceptions::PyValueError;

use crate::binding::{errors, run_events};

pub(crate) type OutputWriterSession = g_output::OutputWriterSession;

#[allow(clippy::too_many_arguments)]
pub(crate) fn create_output_writer_session_batch(
    run_directories: Vec<String>,
    chunks_directories: Vec<String>,
    association_mode: &str,
    writer_thread_count: usize,
    writer_queue_depth: usize,
    output_format: &str,
    output_statistic_dtype: &str,
    finalize_parquet: bool,
    chunks_per_arrow_file: usize,
    arrow_compression: &str,
    parquet_compression: &str,
    collect_stage_timings: bool,
) -> PyResult<Vec<Arc<OutputWriterSession>>> {
    if run_directories.len() != chunks_directories.len() {
        return Err(PyValueError::new_err(format!(
            "Output writer run directory count ({}) does not match chunks directory count ({}).",
            run_directories.len(),
            chunks_directories.len()
        )));
    }
    let writer_session_count = writer_session_count_as_i64(run_directories.len())?;
    record_pipeline_output_writer_sessions_create_started(association_mode, writer_session_count)?;
    g_output::create_output_writer_sessions(
        run_directories,
        chunks_directories,
        association_mode,
        writer_thread_count,
        writer_queue_depth,
        output_format,
        output_statistic_dtype,
        finalize_parquet,
        chunks_per_arrow_file,
        arrow_compression,
        parquet_compression,
        collect_stage_timings,
    )
    .map_err(|error| errors::convert_output_error("create_output_writer_sessions", error))
    .map(|sessions| sessions.into_iter().map(Arc::new).collect())
}

pub(crate) fn write_host_association_batch(
    writer_sessions: &[Arc<OutputWriterSession>],
    active_trait_indices: &[usize],
    variant_start_index: usize,
    metadata: &g_genotype::VariantMetadataColumns,
    chunk_stats: &g_genotype::ChunkStats,
    result: &g_engine::HostAssociationBatch,
) -> PyResult<()> {
    let chunk_identifier = i64::try_from(variant_start_index)
        .map_err(|_| PyValueError::new_err("Variant start index does not fit into int64 output."))?;
    let chunk_handle = NativeChunkHandle::new(
        Arc::new(convert_variant_metadata_to_output(metadata)),
        Arc::new(convert_chunk_stats_to_output(chunk_stats)),
        chunk_identifier,
    );
    let expected_trait_count = active_trait_indices.len();
    let expected_variant_count = chunk_handle.row_count();
    let extra_code_matrix = result.extra_codes.as_ref();
    if let Some(extra_codes) = extra_code_matrix
        && (extra_codes.trait_count != expected_trait_count || extra_codes.variant_count != expected_variant_count)
    {
        return Err(PyValueError::new_err(
            "Materialized extra-code shape does not match active traits and variant metadata.",
        ));
    }
    let native_writer_sessions = writer_sessions.iter().map(Arc::as_ref).collect::<Vec<_>>();
    match &result.statistics {
        g_engine::HostAssociationStatistics::Float32(statistics) => {
            validate_host_statistic_shape(
                statistics.trait_count,
                statistics.variant_count,
                expected_trait_count,
                expected_variant_count,
            )?;
            let rows = host_statistic_rows(statistics, extra_code_matrix)?;
            g_output::write_regenie2_multi_trait_chunk_f32(
                &native_writer_sessions,
                active_trait_indices,
                &chunk_handle,
                &rows,
            )
            .map_err(|error| errors::convert_output_error("write_host_association_batch", error))
        }
        g_engine::HostAssociationStatistics::Float64(statistics) => {
            validate_host_statistic_shape(
                statistics.trait_count,
                statistics.variant_count,
                expected_trait_count,
                expected_variant_count,
            )?;
            let rows = host_statistic_rows(statistics, extra_code_matrix)?;
            g_output::write_regenie2_multi_trait_chunk_f64(
                &native_writer_sessions,
                active_trait_indices,
                &chunk_handle,
                &rows,
            )
            .map_err(|error| errors::convert_output_error("write_host_association_batch", error))
        }
    }
}

pub(crate) fn finish_output_writer_sessions_for_delivery(
    writer_sessions: &[Arc<OutputWriterSession>],
    requested_thread_count: i64,
) -> PyResult<Vec<Option<String>>> {
    let result = (|| {
        let writer_session_count = writer_session_count_as_i64(writer_sessions.len())?;
        record_writer_sessions_finish_started(requested_thread_count, writer_session_count)?;
        let native_writer_sessions = writer_sessions.iter().map(Arc::as_ref).collect::<Vec<_>>();
        g_output::finish_output_writer_sessions_with_requested_threads(&native_writer_sessions, requested_thread_count)
            .map(optional_path_values_to_strings)
            .map_err(|error| errors::convert_output_error("finish_output_writer_sessions", error))
    })();
    if result.is_err() {
        abort_output_writer_sessions_for_delivery(writer_sessions);
    }
    result
}

pub(crate) fn finish_interrupted_output_writer_sessions_for_delivery(
    writer_sessions: &[Arc<OutputWriterSession>],
    requested_thread_count: i64,
    signal_exit_code: i64,
    signal_name: &str,
    signal_number: i64,
) -> PyResult<()> {
    let result = (|| {
        let writer_session_count = writer_session_count_as_i64(writer_sessions.len())?;
        record_writer_sessions_interrupted_flush_started(
            requested_thread_count,
            signal_exit_code,
            signal_name,
            signal_number,
            writer_session_count,
        )?;
        let native_writer_sessions = writer_sessions.iter().map(Arc::as_ref).collect::<Vec<_>>();
        g_output::finish_interrupted_output_writer_sessions_with_requested_threads(
            &native_writer_sessions,
            requested_thread_count,
            signal_name,
        )
        .map_err(|error| errors::convert_output_error("finish_interrupted_output_writer_sessions", error))
    })();
    if result.is_err() {
        abort_output_writer_sessions_for_delivery(writer_sessions);
    }
    result
}

pub(crate) fn abort_output_writer_sessions_for_delivery(writer_sessions: &[Arc<OutputWriterSession>]) {
    for writer_session in writer_sessions {
        let _ = writer_session.abort();
    }
}

fn validate_host_statistic_shape(
    trait_count: usize,
    variant_count: usize,
    expected_trait_count: usize,
    expected_variant_count: usize,
) -> PyResult<()> {
    if trait_count != expected_trait_count || variant_count != expected_variant_count {
        return Err(PyValueError::new_err(format!(
            "Materialized statistic shape ({trait_count}, {variant_count}) does not match expected ({expected_trait_count}, {expected_variant_count})."
        )));
    }
    Ok(())
}

fn host_statistic_rows<'values, Statistic>(
    statistics: &'values g_engine::HostAssociationStatisticMatrix<Statistic>,
    extra_codes: Option<&'values g_engine::HostExtraCodeMatrix>,
) -> PyResult<Vec<g_output::Regenie2StatisticSliceBundle<'values, Statistic>>> {
    let row_width = statistics.variant_count;
    let mut rows = Vec::with_capacity(statistics.trait_count);
    for row_index in 0..statistics.trait_count {
        let row_start = row_index
            .checked_mul(row_width)
            .ok_or_else(|| PyValueError::new_err("Materialized statistic row offset overflowed usize."))?;
        let row_stop = row_start
            .checked_add(row_width)
            .ok_or_else(|| PyValueError::new_err("Materialized statistic row bound overflowed usize."))?;
        rows.push(g_output::Regenie2StatisticSliceBundle {
            beta: &statistics.beta[row_start..row_stop],
            standard_error: &statistics.standard_error[row_start..row_stop],
            chi_squared: &statistics.chi_squared[row_start..row_stop],
            log10_p_value: &statistics.log10_p_value[row_start..row_stop],
            extra_code: extra_codes.map(|matrix| &matrix.values[row_start..row_stop]),
        });
    }
    Ok(rows)
}

fn record_writer_sessions_finish_started(requested_thread_count: i64, writer_session_count: i64) -> PyResult<()> {
    let payload = native_run_events::build_native_dispatch_writer_sessions_finish_started_diagnostic_payload(
        requested_thread_count,
        writer_session_count,
    );
    run_events::emit_run_diagnostic_event_payload(&payload)
}

fn record_writer_sessions_interrupted_flush_started(
    requested_thread_count: i64,
    signal_exit_code: i64,
    signal_name: &str,
    signal_number: i64,
    writer_session_count: i64,
) -> PyResult<()> {
    let payload = native_run_events::build_native_dispatch_writer_sessions_interrupted_flush_started_diagnostic_payload(
        requested_thread_count,
        signal_exit_code,
        signal_name,
        signal_number,
        writer_session_count,
    );
    run_events::emit_run_diagnostic_event_payload(&payload)
}

fn record_pipeline_output_writer_sessions_create_started(
    association_mode: &str,
    writer_session_count: i64,
) -> PyResult<()> {
    let payload = native_run_events::build_pipeline_output_writer_sessions_create_started_diagnostic_payload(
        association_mode,
        writer_session_count,
    );
    run_events::emit_run_diagnostic_event_payload(&payload)
}

fn writer_session_count_as_i64(writer_session_count: usize) -> PyResult<i64> {
    i64::try_from(writer_session_count)
        .map_err(|_| PyValueError::new_err("Writer session count exceeds native int64 capacity."))
}

fn optional_path_values_to_strings(paths: Vec<Option<std::path::PathBuf>>) -> Vec<Option<String>> {
    paths.into_iter().map(|maybe_path| maybe_path.map(|path| path.display().to_string())).collect()
}

fn convert_variant_metadata_to_output(
    metadata: &g_genotype::VariantMetadataColumns,
) -> NativeOutputVariantMetadataColumns {
    NativeOutputVariantMetadataColumns {
        chromosome: metadata.chromosome.clone(),
        variant_identifier: metadata.variant_identifier.clone(),
        position: metadata.position.clone(),
        allele_one: metadata.allele_one.clone(),
        allele_two: metadata.allele_two.clone(),
    }
}

fn convert_chunk_stats_to_output(chunk_stats: &g_genotype::ChunkStats) -> NativeOutputChunkStats {
    NativeOutputChunkStats {
        allele_one_frequency: chunk_stats.allele_one_frequency.clone(),
        observation_count: chunk_stats.observation_count.clone(),
        has_missing_values: chunk_stats.has_missing_values,
        dosage_sum: Arc::clone(&chunk_stats.dosage_sum),
        dosage_square_sum: chunk_stats.dosage_square_sum.clone(),
        imputed_dosage_square_sum: chunk_stats.imputed_dosage_square_sum.clone(),
        dosage_variance_numerator: chunk_stats.dosage_variance_numerator.clone(),
        info_score: chunk_stats.info_score.clone(),
        allele_count: Arc::clone(&chunk_stats.allele_count),
        minor_allele_count: chunk_stats.minor_allele_count.clone(),
        zero_count: chunk_stats.zero_count.clone(),
        nonzero_count: chunk_stats.nonzero_count.clone(),
        homozygous_reference_count: chunk_stats.homozygous_reference_count.clone(),
        heterozygous_count: chunk_stats.heterozygous_count.clone(),
        homozygous_alternate_count: chunk_stats.homozygous_alternate_count.clone(),
        is_sparse_candidate: chunk_stats.is_sparse_candidate.clone(),
        is_rare_sparse_firth_candidate: chunk_stats.is_rare_sparse_firth_candidate.clone(),
    }
}
