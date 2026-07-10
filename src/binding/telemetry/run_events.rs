//! Native run telemetry and diagnostic event adapters.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use g_engine as native_schedule;
use g_runtime as native_run_events;

use super::{logging, session};

pub(crate) type NativeRunArtifacts = native_run_events::RunArtifactsPayload;

pub(crate) fn run_failed_event_payload_from_error(
    error: &Bound<'_, PyAny>,
) -> PyResult<native_run_events::RunFailedEventPayload> {
    let error_type = error.get_type().name()?.to_string_lossy().into_owned();
    let error_message = error.str()?.to_string_lossy().into_owned();
    Ok(native_run_events::build_run_failed_event_payload(&error_type, &error_message))
}

pub(crate) fn record_execution_plan_prepared_events(
    telemetry_session: Option<&session::NativeTelemetryRunSession>,
    association_mode: &str,
    trait_type: &str,
    phenotype_count: i64,
    chunk_size: i64,
    variant_limit: Option<i64>,
    device: &str,
) -> PyResult<()> {
    if let Some(session) = telemetry_session {
        session.emit_execution_plan_prepared_event(
            association_mode,
            trait_type,
            phenotype_count,
            chunk_size,
            variant_limit,
            device,
        )?;
    }
    record_runner_execution_plan_prepared_diagnostic_event(
        association_mode,
        phenotype_count,
        chunk_size,
        variant_limit,
        device,
    )
}

pub(crate) fn record_phenotype_writer_finished_telemetry(
    telemetry_session: Option<&session::NativeTelemetryRunSession>,
    association_mode: &str,
    phenotype: &str,
    final_output_path: Option<String>,
) -> PyResult<()> {
    let Some(session) = telemetry_session else {
        return Ok(());
    };
    session.emit_phenotype_writer_finished_event(association_mode, phenotype, final_output_path)
}

pub(crate) fn record_multi_phenotype_writer_finished_telemetry(
    telemetry_session: Option<&session::NativeTelemetryRunSession>,
    association_mode: &str,
    phenotype_count: i64,
    final_output_paths: Vec<Option<String>>,
) -> PyResult<()> {
    let Some(session) = telemetry_session else {
        return Ok(());
    };
    session.emit_multi_phenotype_writer_finished_event(association_mode, phenotype_count, final_output_paths)
}

pub(crate) fn record_sample_alignment_completed_telemetry(
    telemetry_session: Option<&session::NativeTelemetryRunSession>,
    association_mode: &str,
    phenotype: Option<String>,
    phenotype_count: Option<i64>,
    sample_count: Option<i64>,
    covariate_count: Option<i64>,
    phenotype_group_count: Option<i64>,
) -> PyResult<()> {
    let Some(session) = telemetry_session else {
        return Ok(());
    };
    session.emit_sample_alignment_completed_event(
        association_mode,
        phenotype,
        phenotype_count,
        sample_count,
        covariate_count,
        phenotype_group_count,
    )
}

pub(crate) fn record_prediction_source_loaded_telemetry(
    telemetry_session: Option<&session::NativeTelemetryRunSession>,
    association_mode: &str,
    phenotype: Option<String>,
    phenotype_count: Option<i64>,
) -> PyResult<()> {
    let Some(session) = telemetry_session else {
        return Ok(());
    };
    session.emit_prediction_source_loaded_event(association_mode, phenotype, phenotype_count)
}

pub(crate) fn record_single_trait_preflight_completed_telemetry(
    telemetry_session: Option<&session::NativeTelemetryRunSession>,
    association_mode: &str,
    phenotype: &str,
    sample_count: i64,
    covariate_count: i64,
    chromosome_count: i64,
) -> PyResult<()> {
    let Some(session) = telemetry_session else {
        return Ok(());
    };
    session.emit_single_trait_preflight_completed_event(
        association_mode,
        phenotype,
        sample_count,
        covariate_count,
        chromosome_count,
    )
}

pub(crate) fn record_multi_phenotype_preflight_completed_telemetry(
    telemetry_session: Option<&session::NativeTelemetryRunSession>,
    association_mode: &str,
    phenotype_count: i64,
    sample_count: i64,
) -> PyResult<()> {
    let Some(session) = telemetry_session else {
        return Ok(());
    };
    session.emit_multi_phenotype_preflight_completed_event(association_mode, phenotype_count, sample_count)
}

pub(crate) fn record_multi_phenotype_sample_summary_telemetry(
    telemetry_session: Option<&session::NativeTelemetryRunSession>,
    association_mode: &str,
    multi_phenotype_sample_mode: &str,
    sample_counts: Vec<i64>,
    sample_set_fingerprints: Vec<Option<String>>,
    phenotype_group_count: i64,
) -> PyResult<()> {
    let Some(session) = telemetry_session else {
        return Ok(());
    };
    session.emit_multi_phenotype_sample_summary_event(
        association_mode,
        multi_phenotype_sample_mode,
        sample_counts,
        sample_set_fingerprints,
        phenotype_group_count,
    )
}

pub(crate) fn record_association_backend_selected_telemetry(
    telemetry_session: Option<&session::NativeTelemetryRunSession>,
    association_mode: &str,
    association_backend_kind: &str,
    device: &str,
    genotype_format: &str,
    phenotype: Option<String>,
    phenotype_count: Option<i64>,
) -> PyResult<()> {
    let Some(session) = telemetry_session else {
        return Ok(());
    };
    session.emit_association_backend_selected_event(
        association_mode,
        association_backend_kind,
        device,
        genotype_format,
        phenotype,
        phenotype_count,
    )
}

pub(crate) fn record_bgen_engine_opened_telemetry(
    telemetry_session: Option<&session::NativeTelemetryRunSession>,
    association_mode: &str,
    association_backend_kind: &str,
    sample_count: i64,
    variant_count: i64,
    phenotype: Option<String>,
    phenotype_count: Option<i64>,
) -> PyResult<()> {
    let Some(session) = telemetry_session else {
        return Ok(());
    };
    session.emit_bgen_engine_opened_event(
        association_mode,
        association_backend_kind,
        sample_count,
        variant_count,
        phenotype,
        phenotype_count,
    )
}

pub(crate) fn record_gpu_genotype_format_resolved_native_plan_events(
    telemetry_session: Option<&session::NativeTelemetryRunSession>,
    native_resolution_plan: &native_schedule::GpuGenotypeFormatResolutionPlan,
) -> PyResult<()> {
    if !native_resolution_plan.should_log_auto_resolution() {
        return Ok(());
    }
    let resolved_gpu_genotype_format = native_resolution_plan
        .resolved_gpu_genotype_format
        .as_deref()
        .ok_or_else(|| PyRuntimeError::new_err("Native GPU genotype-format resolution plan is not resolved."))?;
    let resolution_reason = native_resolution_plan.resolution_reason.as_deref().ok_or_else(|| {
        PyRuntimeError::new_err("Native GPU genotype-format resolution plan has no resolution reason.")
    })?;
    if let Some(session) = telemetry_session {
        session.emit_gpu_genotype_format_resolved_event(
            &native_resolution_plan.requested_gpu_genotype_format,
            resolved_gpu_genotype_format,
            resolution_reason,
            native_resolution_plan.fallback_error.clone(),
        )?;
    }
    record_pipeline_gpu_genotype_format_resolved_diagnostic_event(
        &native_resolution_plan.requested_gpu_genotype_format,
        resolved_gpu_genotype_format,
        resolution_reason,
        native_resolution_plan.fallback_error.as_deref(),
    )
}

pub(crate) fn record_runner_execution_plan_build_started_diagnostic_event() -> PyResult<()> {
    let payload = native_run_events::build_runner_execution_plan_build_started_diagnostic_payload();
    emit_run_diagnostic_event_payload(&payload)
}

fn record_runner_execution_plan_prepared_diagnostic_event(
    association_mode: &str,
    phenotype_count: i64,
    chunk_size: i64,
    variant_limit: Option<i64>,
    device: &str,
) -> PyResult<()> {
    let payload = native_run_events::build_runner_execution_plan_prepared_diagnostic_payload(
        association_mode,
        phenotype_count,
        chunk_size,
        variant_limit,
        device,
    );
    emit_run_diagnostic_event_payload(&payload)
}

pub(crate) fn record_runner_execution_plan_dispatch_started_diagnostic_event(
    phenotype_count: i64,
    association_mode: &str,
) -> PyResult<()> {
    let payload = native_run_events::build_runner_execution_plan_dispatch_started_diagnostic_payload(
        phenotype_count,
        association_mode,
    );
    emit_run_diagnostic_event_payload(&payload)
}

pub(crate) fn record_runner_execution_plan_finalization_started_diagnostic_event(
    phenotype_count: i64,
    association_mode: &str,
) -> PyResult<()> {
    let payload = native_run_events::build_runner_execution_plan_finalization_started_diagnostic_payload(
        phenotype_count,
        association_mode,
    );
    emit_run_diagnostic_event_payload(&payload)
}

pub(crate) fn record_runner_multi_phenotype_dispatch_started_diagnostic_event(
    phenotype_count: i64,
    association_mode: &str,
) -> PyResult<()> {
    let payload = native_run_events::build_runner_multi_phenotype_dispatch_started_diagnostic_payload(
        phenotype_count,
        association_mode,
    );
    emit_run_diagnostic_event_payload(&payload)
}

pub(crate) fn record_runner_single_phenotype_dispatch_started_diagnostic_event(
    association_mode: &str,
    phenotype: &str,
) -> PyResult<()> {
    let payload = native_run_events::build_runner_single_phenotype_dispatch_started_diagnostic_payload(
        association_mode,
        phenotype,
    );
    emit_run_diagnostic_event_payload(&payload)
}

pub(crate) fn record_runner_binary_engine_dispatch_started_diagnostic_event(phenotype: &str) -> PyResult<()> {
    let payload = native_run_events::build_runner_binary_engine_dispatch_started_diagnostic_payload(phenotype);
    emit_run_diagnostic_event_payload(&payload)
}

pub(crate) fn record_runner_linear_engine_dispatch_started_diagnostic_event(phenotype: &str) -> PyResult<()> {
    let payload = native_run_events::build_runner_linear_engine_dispatch_started_diagnostic_payload(phenotype);
    emit_run_diagnostic_event_payload(&payload)
}

pub(crate) fn record_runner_multi_phenotype_binary_engine_dispatch_started_diagnostic_event(
    phenotype_count: i64,
) -> PyResult<()> {
    let payload = native_run_events::build_runner_multi_phenotype_binary_engine_dispatch_started_diagnostic_payload(
        phenotype_count,
    );
    emit_run_diagnostic_event_payload(&payload)
}

pub(crate) fn record_runner_multi_phenotype_linear_engine_dispatch_started_diagnostic_event(
    phenotype_count: i64,
) -> PyResult<()> {
    let payload = native_run_events::build_runner_multi_phenotype_linear_engine_dispatch_started_diagnostic_payload(
        phenotype_count,
    );
    emit_run_diagnostic_event_payload(&payload)
}

pub(crate) fn record_native_runtime_knobs_configured_diagnostic_event(
    bgen_decode_tile_variant_count: i64,
    threads: Option<i64>,
) -> PyResult<()> {
    let payload = native_run_events::build_native_runtime_knobs_configured_diagnostic_payload(
        bgen_decode_tile_variant_count,
        threads,
    );
    emit_run_diagnostic_event_payload(&payload)
}

pub(crate) fn record_runner_metadata_artifacts_finalized_diagnostic_event(
    association_mode: &str,
    phenotype_count: i64,
) -> PyResult<()> {
    let payload = native_run_events::build_runner_metadata_artifacts_finalized_diagnostic_payload(
        association_mode,
        phenotype_count,
    );
    emit_run_diagnostic_event_payload(&payload)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn record_preflight_warning_diagnostic_events(
    messages: Vec<String>,
    chromosome_count: i64,
    covariate_count: i64,
    preflight_scope: &str,
    sample_count: i64,
    trusted_no_missing_diploid: bool,
) -> PyResult<()> {
    for (warning_index, message) in messages.into_iter().enumerate() {
        let warning_index_value = i64::try_from(warning_index)
            .map_err(|_| PyValueError::new_err("Preflight warning index exceeds native int64 capacity."))?;
        let payload = native_run_events::build_preflight_warning_diagnostic_payload(
            &message,
            chromosome_count,
            covariate_count,
            preflight_scope,
            sample_count,
            trusted_no_missing_diploid,
            warning_index_value,
        );
        emit_run_diagnostic_event_payload(&payload)?;
    }
    Ok(())
}

pub(crate) fn record_pipeline_bgen_engine_open_started_diagnostic_event(
    phenotype_count: Option<i64>,
    phenotype_name: Option<&str>,
    pipeline_label: &str,
    trusted_no_missing_diploid: bool,
    variant_limit: Option<i64>,
) -> PyResult<()> {
    let payload = native_run_events::build_pipeline_bgen_engine_open_started_diagnostic_payload(
        phenotype_count,
        phenotype_name,
        pipeline_label,
        trusted_no_missing_diploid,
        variant_limit,
    );
    emit_run_diagnostic_event_payload(&payload)
}

pub(crate) fn record_pipeline_bgen_engine_opened_diagnostic_event(
    phenotype_count: Option<i64>,
    phenotype_name: Option<&str>,
    pipeline_label: &str,
    sample_count: i64,
    variant_count: i64,
) -> PyResult<()> {
    let payload = native_run_events::build_pipeline_bgen_engine_opened_diagnostic_payload(
        phenotype_count,
        phenotype_name,
        pipeline_label,
        sample_count,
        variant_count,
    );
    emit_run_diagnostic_event_payload(&payload)
}

pub(crate) fn record_pipeline_prevalidated_bgen_engine_used_diagnostic_event(
    phenotype_count: Option<i64>,
    phenotype_name: Option<&str>,
    pipeline_label: &str,
) -> PyResult<()> {
    let payload = native_run_events::build_pipeline_prevalidated_bgen_engine_used_diagnostic_payload(
        phenotype_count,
        phenotype_name,
        pipeline_label,
    );
    emit_run_diagnostic_event_payload(&payload)
}

fn record_pipeline_gpu_genotype_format_resolved_diagnostic_event(
    requested_gpu_genotype_format: &str,
    resolved_gpu_genotype_format: &str,
    resolution_reason: &str,
    fallback_error: Option<&str>,
) -> PyResult<()> {
    let payload = native_run_events::build_pipeline_gpu_genotype_format_resolved_diagnostic_payload(
        requested_gpu_genotype_format,
        resolved_gpu_genotype_format,
        resolution_reason,
        fallback_error,
    );
    emit_run_diagnostic_event_payload(&payload)
}

pub(crate) fn record_pipeline_multi_phenotype_sample_summary_diagnostic_event(
    phenotype_count: i64,
    phenotype_group_count: i64,
    sample_counts_differ: bool,
    sample_mode: &str,
) -> PyResult<()> {
    let payload = native_run_events::build_pipeline_multi_phenotype_sample_summary_diagnostic_payload(
        phenotype_count,
        phenotype_group_count,
        sample_counts_differ,
        sample_mode,
    );
    emit_run_diagnostic_event_payload(&payload)
}

pub(crate) fn record_pipeline_multi_trait_started_diagnostic_event(
    association_mode: &str,
    phenotype_count: i64,
    sample_mode: &str,
) -> PyResult<()> {
    let payload = native_run_events::build_pipeline_multi_trait_started_diagnostic_payload(
        association_mode,
        phenotype_count,
        sample_mode,
    );
    emit_run_diagnostic_event_payload(&payload)
}

pub(crate) fn record_pipeline_multi_trait_input_load_started_diagnostic_event(phenotype_count: i64) -> PyResult<()> {
    let payload = native_run_events::build_pipeline_multi_trait_input_load_started_diagnostic_payload(phenotype_count);
    emit_run_diagnostic_event_payload(&payload)
}

pub(crate) fn record_pipeline_multi_trait_input_aligned_diagnostic_event(
    covariate_count: i64,
    phenotype_count: i64,
    sample_count: i64,
) -> PyResult<()> {
    let payload = native_run_events::build_pipeline_multi_trait_input_aligned_diagnostic_payload(
        covariate_count,
        phenotype_count,
        sample_count,
    );
    emit_run_diagnostic_event_payload(&payload)
}

pub(crate) fn record_pipeline_multi_trait_prediction_source_load_started_diagnostic_event(
    phenotype_count: i64,
) -> PyResult<()> {
    let payload = native_run_events::build_pipeline_multi_trait_prediction_source_load_started_diagnostic_payload(
        phenotype_count,
    );
    emit_run_diagnostic_event_payload(&payload)
}

pub(crate) fn record_pipeline_grouped_per_phenotype_started_diagnostic_event(
    association_mode: &str,
    phenotype_count: i64,
    sample_mode: &str,
) -> PyResult<()> {
    let payload = native_run_events::build_pipeline_grouped_per_phenotype_started_diagnostic_payload(
        association_mode,
        phenotype_count,
        sample_mode,
    );
    emit_run_diagnostic_event_payload(&payload)
}

pub(crate) fn record_pipeline_grouped_per_phenotype_groups_prepared_diagnostic_event(
    phenotype_count: i64,
    phenotype_group_count: i64,
) -> PyResult<()> {
    let payload = native_run_events::build_pipeline_grouped_per_phenotype_groups_prepared_diagnostic_payload(
        phenotype_count,
        phenotype_group_count,
    );
    emit_run_diagnostic_event_payload(&payload)
}

pub(crate) fn record_pipeline_grouped_union_delivery_selected_diagnostic_event(
    grouped_sample_count: i64,
    phenotype_group_count: i64,
    union_sample_count: i64,
) -> PyResult<()> {
    let payload = native_run_events::build_pipeline_grouped_union_delivery_selected_diagnostic_payload(
        grouped_sample_count,
        phenotype_group_count,
        union_sample_count,
    );
    emit_run_diagnostic_event_payload(&payload)
}

pub(crate) fn record_pipeline_multi_group_preflight_started_diagnostic_event(
    phenotype_count: i64,
    sample_count: i64,
    trusted_no_missing_diploid: bool,
    variant_limit: Option<i64>,
) -> PyResult<()> {
    let payload = native_run_events::build_pipeline_multi_group_preflight_started_diagnostic_payload(
        phenotype_count,
        sample_count,
        trusted_no_missing_diploid,
        variant_limit,
    );
    emit_run_diagnostic_event_payload(&payload)
}

pub(crate) fn record_pipeline_multi_group_preflight_completed_diagnostic_event(
    phenotype_count: i64,
    sample_count: i64,
    trusted_no_missing_diploid: bool,
    variant_limit: Option<i64>,
) -> PyResult<()> {
    let payload = native_run_events::build_pipeline_multi_group_preflight_completed_diagnostic_payload(
        phenotype_count,
        sample_count,
        trusted_no_missing_diploid,
        variant_limit,
    );
    emit_run_diagnostic_event_payload(&payload)
}

pub(crate) fn record_pipeline_single_trait_started_diagnostic_event(
    association_mode: &str,
    phenotype_name: &str,
    pipeline_label: &str,
) -> PyResult<()> {
    let payload = native_run_events::build_pipeline_single_trait_started_diagnostic_payload(
        association_mode,
        phenotype_name,
        pipeline_label,
    );
    emit_run_diagnostic_event_payload(&payload)
}

pub(crate) fn record_pipeline_single_trait_input_load_started_diagnostic_event(
    phenotype_name: &str,
    pipeline_label: &str,
) -> PyResult<()> {
    let payload = native_run_events::build_pipeline_single_trait_input_load_started_diagnostic_payload(
        phenotype_name,
        pipeline_label,
    );
    emit_run_diagnostic_event_payload(&payload)
}

pub(crate) fn record_pipeline_single_trait_input_aligned_diagnostic_event(
    covariate_count: i64,
    phenotype_name: &str,
    pipeline_label: &str,
    sample_count: i64,
) -> PyResult<()> {
    let payload = native_run_events::build_pipeline_single_trait_input_aligned_diagnostic_payload(
        covariate_count,
        phenotype_name,
        pipeline_label,
        sample_count,
    );
    emit_run_diagnostic_event_payload(&payload)
}

pub(crate) fn record_pipeline_single_trait_prediction_source_load_started_diagnostic_event(
    phenotype_name: &str,
    pipeline_label: &str,
) -> PyResult<()> {
    let payload = native_run_events::build_pipeline_single_trait_prediction_source_load_started_diagnostic_payload(
        phenotype_name,
        pipeline_label,
    );
    emit_run_diagnostic_event_payload(&payload)
}

pub(crate) fn record_pipeline_single_trait_preflight_started_diagnostic_event(
    phenotype_name: &str,
    pipeline_label: &str,
    trusted_no_missing_diploid: bool,
    variant_limit: Option<i64>,
) -> PyResult<()> {
    let payload = native_run_events::build_pipeline_single_trait_preflight_started_diagnostic_payload(
        phenotype_name,
        pipeline_label,
        trusted_no_missing_diploid,
        variant_limit,
    );
    emit_run_diagnostic_event_payload(&payload)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn record_pipeline_single_trait_preflight_completed_diagnostic_event(
    chromosome_count: i64,
    covariate_count: i64,
    phenotype_name: &str,
    pipeline_label: &str,
    sample_count: i64,
) -> PyResult<()> {
    let payload = native_run_events::build_pipeline_single_trait_preflight_completed_diagnostic_payload(
        chromosome_count,
        covariate_count,
        phenotype_name,
        pipeline_label,
        sample_count,
    );
    emit_run_diagnostic_event_payload(&payload)
}

pub(crate) fn record_native_dispatch_delivery_finished_diagnostic_event(
    pipeline_label: &str,
    processed_chunk_count: i64,
) -> PyResult<()> {
    let payload = native_run_events::build_native_dispatch_delivery_finished_diagnostic_payload(
        pipeline_label,
        processed_chunk_count,
    );
    emit_run_diagnostic_event_payload(&payload)
}

pub(crate) fn record_native_dispatch_pipeline_finished_diagnostic_event(
    final_parquet_path_count: i64,
    pipeline_label: &str,
) -> PyResult<()> {
    let payload = native_run_events::build_native_dispatch_pipeline_finished_diagnostic_payload(
        final_parquet_path_count,
        pipeline_label,
    );
    emit_run_diagnostic_event_payload(&payload)
}

pub(crate) fn emit_run_diagnostic_event_payload(event: &native_run_events::RunDiagnosticEventPayload) -> PyResult<()> {
    let fields_json = native_run_events::serialize_run_diagnostic_fields_json(&event.fields).map_err(|error| {
        pyo3::exceptions::PyValueError::new_err(format!("Failed to serialize diagnostic event fields: {error}"))
    })?;
    logging::emit_diagnostic_event(event.level, event.event_name, &event.message, Some(fields_json))
}
