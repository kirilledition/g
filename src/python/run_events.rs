//! PyO3 adapters for runtime-owned run lifecycle events.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule, PyTuple};

use g_engine as native_schedule;
use g_runtime as native_run_events;

use super::{
    callback_progress::{NativeCallbackProgressTelemetryEvent, NativeCallbackProgressUpdate},
    errors, logging, schedule,
};

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct NativeRunArtifacts {
    data: native_run_events::RunArtifactsPayload,
}

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct NativeRunArtifactPayload {
    data: native_run_events::RunArtifactPayload,
}

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct NativeRunCompletedEvent {
    data: native_run_events::RunCompletedEventPayload,
}

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct NativeRunInterruptedEvent {
    data: native_run_events::RunInterruptedEventPayload,
}

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct NativeRunFailedEvent {
    data: native_run_events::RunFailedEventPayload,
}

impl NativeRunArtifacts {
    pub(crate) fn new(data: native_run_events::RunArtifactsPayload) -> Self {
        Self { data }
    }
}

impl NativeRunArtifactPayload {
    fn new(data: native_run_events::RunArtifactPayload) -> Self {
        Self { data }
    }
}

impl NativeRunCompletedEvent {
    fn new(data: native_run_events::RunCompletedEventPayload) -> Self {
        Self { data }
    }
}

impl NativeRunInterruptedEvent {
    fn new(data: native_run_events::RunInterruptedEventPayload) -> Self {
        Self { data }
    }
}

impl NativeRunFailedEvent {
    pub(crate) fn new(data: native_run_events::RunFailedEventPayload) -> Self {
        Self { data }
    }
}

#[pymethods]
impl NativeRunArtifacts {
    #[getter]
    fn output_run_directory(&self) -> Option<&str> {
        self.data.output_run_directory.as_deref()
    }

    #[getter]
    fn final_dataset(&self) -> Option<&str> {
        self.data.final_dataset.as_deref()
    }

    #[getter]
    fn final_parquet(&self) -> Option<&str> {
        self.data.final_parquet.as_deref()
    }

    #[getter]
    fn final_regenie(&self) -> Option<&str> {
        self.data.final_regenie.as_deref()
    }

    #[getter]
    fn effective_config(&self) -> Option<&str> {
        self.data.effective_config.as_deref()
    }

    #[getter]
    fn phenotype_artifacts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let artifacts = self
            .data
            .phenotype_artifacts
            .iter()
            .cloned()
            .map(|artifact| Py::new(py, NativeRunArtifacts::new(artifact)))
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, &artifacts)
    }

    #[getter]
    fn phenotype_name(&self) -> Option<&str> {
        self.data.phenotype_name.as_deref()
    }

    #[getter]
    fn association_mode(&self) -> Option<&str> {
        self.data.association_mode.as_deref()
    }

    #[getter]
    fn phenotype_count(&self) -> Option<i64> {
        self.data.phenotype_count
    }

    #[getter]
    fn run_id(&self) -> Option<&str> {
        self.data.run_id.as_deref()
    }
}

#[pymethods]
impl NativeRunArtifactPayload {
    #[getter]
    fn phenotype_name(&self) -> Option<&str> {
        self.data.phenotype_name.as_deref()
    }

    #[getter]
    fn output_run_directory(&self) -> Option<&str> {
        self.data.output_run_directory.as_deref()
    }

    #[getter]
    fn final_dataset(&self) -> Option<&str> {
        self.data.final_dataset.as_deref()
    }

    #[getter]
    fn final_parquet(&self) -> Option<&str> {
        self.data.final_parquet.as_deref()
    }

    #[getter]
    fn final_regenie(&self) -> Option<&str> {
        self.data.final_regenie.as_deref()
    }

    #[getter]
    fn effective_config(&self) -> Option<&str> {
        self.data.effective_config.as_deref()
    }
}

#[pymethods]
impl NativeRunCompletedEvent {
    #[getter]
    fn run_id(&self) -> Option<&str> {
        self.data.run_id.as_deref()
    }

    #[getter]
    fn association_mode(&self) -> Option<&str> {
        self.data.association_mode.as_deref()
    }

    #[getter]
    fn phenotype_count(&self) -> Option<i64> {
        self.data.phenotype_count
    }

    #[getter]
    fn artifacts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let artifacts = self
            .data
            .artifacts
            .iter()
            .cloned()
            .map(|artifact| Py::new(py, NativeRunArtifactPayload::new(artifact)))
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, &artifacts)
    }
}

#[pymethods]
impl NativeRunInterruptedEvent {
    #[getter]
    fn signal_number(&self) -> i64 {
        self.data.signal_number
    }

    #[getter]
    fn signal_name(&self) -> &str {
        &self.data.signal_name
    }

    #[getter]
    fn exit_code(&self) -> i64 {
        self.data.exit_code
    }

    #[getter]
    fn flushed_for_resume(&self) -> bool {
        self.data.flushed_for_resume
    }
}

#[pymethods]
impl NativeRunFailedEvent {
    #[getter]
    fn error_type(&self) -> &str {
        &self.data.error_type
    }

    #[getter]
    fn error_message(&self) -> &str {
        &self.data.error_message
    }
}

fn record_runner_run_started_telemetry_event(
    telemetry_session: &Bound<'_, PyAny>,
    association_mode: &str,
    trait_type: &str,
    phenotype_count: i64,
    output_run_root: &str,
) -> PyResult<()> {
    let Some(native_session_handle) = optional_native_session_handle(telemetry_session)? else {
        return Ok(());
    };
    native_session_handle
        .call_method1("emit_run_started_event", (association_mode, trait_type, phenotype_count, output_run_root))
        .map(|_| ())
}

fn record_execution_plan_prepared_telemetry_event(
    telemetry_session: &Bound<'_, PyAny>,
    association_mode: &str,
    trait_type: &str,
    phenotype_count: i64,
    chunk_size: i64,
    variant_limit: Option<i64>,
    device: &str,
) -> PyResult<()> {
    let Some(native_session_handle) = optional_native_session_handle(telemetry_session)? else {
        return Ok(());
    };
    native_session_handle
        .call_method1(
            "emit_execution_plan_prepared_event",
            (association_mode, trait_type, phenotype_count, chunk_size, variant_limit, device),
        )
        .map(|_| ())
}

#[pyfunction]
pub fn record_runner_run_started_events(
    telemetry_session: &Bound<'_, PyAny>,
    association_mode: &str,
    trait_type: &str,
    phenotype_count: i64,
    output_run_root: &str,
) -> PyResult<()> {
    record_runner_run_started_telemetry_event(
        telemetry_session,
        association_mode,
        trait_type,
        phenotype_count,
        output_run_root,
    )?;
    record_runner_run_started_diagnostic_event(association_mode, trait_type, phenotype_count)
}

#[pyfunction]
pub fn record_runner_run_interrupted_events(
    telemetry_session: &Bound<'_, PyAny>,
    shutdown_request: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let event_payload = run_interrupted_event_payload_from_shutdown_request(shutdown_request)?;
    if let Some(native_session_handle) = optional_native_session_handle(telemetry_session)? {
        let event = Py::new(telemetry_session.py(), NativeRunInterruptedEvent::new(event_payload.clone()))?;
        native_session_handle.call_method1("emit_run_interrupted_event", (event,))?;
    }
    let payload = native_run_events::build_runner_run_interrupted_diagnostic_payload(&event_payload);
    emit_run_diagnostic_event_payload(&payload)
}

#[pyfunction]
pub fn record_runner_run_failed_events(telemetry_session: &Bound<'_, PyAny>, error: &Bound<'_, PyAny>) -> PyResult<()> {
    let event_payload = run_failed_event_payload_from_error(error)?;
    if let Some(native_session_handle) = optional_native_session_handle(telemetry_session)? {
        let event = Py::new(telemetry_session.py(), NativeRunFailedEvent::new(event_payload.clone()))?;
        native_session_handle.call_method1("emit_run_failed_event", (event,))?;
    }
    let payload = native_run_events::build_runner_run_failed_diagnostic_payload(&event_payload);
    emit_run_diagnostic_event_payload(&payload)
}

#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
pub fn attach_run_metadata_and_record_completed_events(
    telemetry_session: &Bound<'_, PyAny>,
    artifacts: &Bound<'_, PyAny>,
    run_id: Option<String>,
    association_mode: String,
    phenotype_count: i64,
) -> PyResult<NativeRunArtifacts> {
    let artifacts_payload = run_artifacts_payload_from_py(artifacts)?;
    let attached_artifacts = native_run_events::attach_run_metadata_to_artifacts(
        &artifacts_payload,
        run_id.as_deref(),
        &association_mode,
        phenotype_count,
    );
    let event_payload = native_run_events::build_run_completed_event_from_artifacts(&attached_artifacts);
    if let Some(native_session_handle) = optional_native_session_handle(telemetry_session)? {
        let event = Py::new(telemetry_session.py(), NativeRunCompletedEvent::new(event_payload.clone()))?;
        native_session_handle.call_method1("emit_run_completed_event", (event,))?;
    }
    let payload = native_run_events::build_runner_run_completed_diagnostic_payload(&event_payload);
    emit_run_diagnostic_event_payload(&payload)?;
    Ok(NativeRunArtifacts::new(attached_artifacts))
}

#[pyfunction]
pub fn record_execution_plan_prepared_events(
    telemetry_session: &Bound<'_, PyAny>,
    association_mode: &str,
    trait_type: &str,
    phenotype_count: i64,
    chunk_size: i64,
    variant_limit: Option<i64>,
    device: &str,
) -> PyResult<()> {
    record_execution_plan_prepared_telemetry_event(
        telemetry_session,
        association_mode,
        trait_type,
        phenotype_count,
        chunk_size,
        variant_limit,
        device,
    )?;
    record_runner_execution_plan_prepared_diagnostic_event(
        association_mode,
        phenotype_count,
        chunk_size,
        variant_limit,
        device,
    )
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub fn record_callback_progress_update_telemetry(
    py: Python<'_>,
    telemetry_session: Option<&Bound<'_, PyAny>>,
    progress_update: PyRef<'_, NativeCallbackProgressUpdate>,
) -> PyResult<()> {
    let Some(native_session_handle) = optional_native_session_handle_from_optional(telemetry_session)? else {
        return Ok(());
    };
    let telemetry_plan = progress_update.telemetry_plan_value();
    for progress_event in telemetry_plan.event_values() {
        emit_callback_progress_event(py, &native_session_handle, &progress_event)?;
    }
    let progress_record = telemetry_plan.progress_value();
    let progress_fields = PyDict::new(py);
    progress_fields.set_item("chromosome", progress_record.chromosome_value())?;
    progress_fields.set_item("chunk_identifier", progress_record.chunk_identifier_value())?;
    progress_fields.set_item("variant_start_index", progress_record.variant_start_index_value())?;
    progress_fields.set_item("variant_stop_index", progress_record.variant_stop_index_value())?;
    progress_fields.set_item("variant_count", progress_record.variant_count_value())?;
    native_session_handle
        .call_method1("emit_progress", (progress_record.processed_chunk_count_value(), progress_fields))?;
    Ok(())
}

#[pyfunction]
pub fn record_callback_progress_event_telemetry(
    py: Python<'_>,
    telemetry_session: Option<&Bound<'_, PyAny>>,
    progress_event: Option<PyRef<'_, NativeCallbackProgressTelemetryEvent>>,
) -> PyResult<()> {
    let Some(progress_event) = progress_event else {
        return Ok(());
    };
    let Some(native_session_handle) = optional_native_session_handle_from_optional(telemetry_session)? else {
        return Ok(());
    };
    emit_callback_progress_event(py, &native_session_handle, &progress_event)
}

#[pyfunction]
pub fn record_binary_correction_summary_telemetry(
    telemetry_session: Option<&Bound<'_, PyAny>>,
    summary_payload: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    let Some(summary_payload) = summary_payload else {
        return Ok(());
    };
    let Some(native_session_handle) = optional_native_session_handle_from_optional(telemetry_session)? else {
        return Ok(());
    };
    native_session_handle.call_method1("emit_binary_correction_summary_event", (summary_payload,))?;
    Ok(())
}

#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
pub fn record_phenotype_writer_finished_telemetry(
    telemetry_session: Option<&Bound<'_, PyAny>>,
    association_mode: &str,
    phenotype: &str,
    final_output_path: Option<String>,
) -> PyResult<()> {
    let Some(native_session_handle) = optional_native_session_handle_from_optional(telemetry_session)? else {
        return Ok(());
    };
    native_session_handle
        .call_method1("emit_phenotype_writer_finished_event", (association_mode, phenotype, final_output_path))?;
    Ok(())
}

#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
pub fn record_multi_phenotype_writer_finished_telemetry(
    telemetry_session: Option<&Bound<'_, PyAny>>,
    association_mode: &str,
    phenotype_count: i64,
    final_output_paths: Vec<Option<String>>,
) -> PyResult<()> {
    let Some(native_session_handle) = optional_native_session_handle_from_optional(telemetry_session)? else {
        return Ok(());
    };
    native_session_handle.call_method1(
        "emit_multi_phenotype_writer_finished_event",
        (association_mode, phenotype_count, final_output_paths),
    )?;
    Ok(())
}

#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
pub fn record_sample_alignment_completed_telemetry(
    telemetry_session: Option<&Bound<'_, PyAny>>,
    association_mode: &str,
    phenotype: Option<String>,
    phenotype_count: Option<i64>,
    sample_count: Option<i64>,
    covariate_count: Option<i64>,
    phenotype_group_count: Option<i64>,
) -> PyResult<()> {
    let Some(native_session_handle) = optional_native_session_handle_from_optional(telemetry_session)? else {
        return Ok(());
    };
    native_session_handle.call_method1(
        "emit_sample_alignment_completed_event",
        (association_mode, phenotype, phenotype_count, sample_count, covariate_count, phenotype_group_count),
    )?;
    Ok(())
}

#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
pub fn record_prediction_source_loaded_telemetry(
    telemetry_session: Option<&Bound<'_, PyAny>>,
    association_mode: &str,
    phenotype: Option<String>,
    phenotype_count: Option<i64>,
) -> PyResult<()> {
    let Some(native_session_handle) = optional_native_session_handle_from_optional(telemetry_session)? else {
        return Ok(());
    };
    native_session_handle
        .call_method1("emit_prediction_source_loaded_event", (association_mode, phenotype, phenotype_count))?;
    Ok(())
}

#[pyfunction]
pub fn record_single_trait_preflight_completed_telemetry(
    telemetry_session: Option<&Bound<'_, PyAny>>,
    association_mode: &str,
    phenotype: &str,
    sample_count: i64,
    covariate_count: i64,
    chromosome_count: i64,
) -> PyResult<()> {
    let Some(native_session_handle) = optional_native_session_handle_from_optional(telemetry_session)? else {
        return Ok(());
    };
    native_session_handle.call_method1(
        "emit_single_trait_preflight_completed_event",
        (association_mode, phenotype, sample_count, covariate_count, chromosome_count),
    )?;
    Ok(())
}

#[pyfunction]
pub fn record_multi_phenotype_preflight_completed_telemetry(
    telemetry_session: Option<&Bound<'_, PyAny>>,
    association_mode: &str,
    phenotype_count: i64,
    sample_count: i64,
) -> PyResult<()> {
    let Some(native_session_handle) = optional_native_session_handle_from_optional(telemetry_session)? else {
        return Ok(());
    };
    native_session_handle.call_method1(
        "emit_multi_phenotype_preflight_completed_event",
        (association_mode, phenotype_count, sample_count),
    )?;
    Ok(())
}

#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
pub fn record_multi_phenotype_sample_summary_telemetry(
    telemetry_session: Option<&Bound<'_, PyAny>>,
    association_mode: &str,
    multi_phenotype_sample_mode: &str,
    sample_counts: Vec<i64>,
    sample_set_fingerprints: Vec<Option<String>>,
    phenotype_group_count: i64,
) -> PyResult<()> {
    let Some(native_session_handle) = optional_native_session_handle_from_optional(telemetry_session)? else {
        return Ok(());
    };
    native_session_handle.call_method1(
        "emit_multi_phenotype_sample_summary_event",
        (association_mode, multi_phenotype_sample_mode, sample_counts, sample_set_fingerprints, phenotype_group_count),
    )?;
    Ok(())
}

#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
pub fn record_association_backend_selected_telemetry(
    telemetry_session: Option<&Bound<'_, PyAny>>,
    association_mode: &str,
    association_backend_kind: &str,
    device: &str,
    genotype_format: &str,
    phenotype: Option<String>,
    phenotype_count: Option<i64>,
) -> PyResult<()> {
    let Some(native_session_handle) = optional_native_session_handle_from_optional(telemetry_session)? else {
        return Ok(());
    };
    native_session_handle.call_method1(
        "emit_association_backend_selected_event",
        (association_mode, association_backend_kind, device, genotype_format, phenotype, phenotype_count),
    )?;
    Ok(())
}

#[allow(clippy::needless_pass_by_value)]
#[pyfunction]
pub fn record_bgen_engine_opened_telemetry(
    telemetry_session: Option<&Bound<'_, PyAny>>,
    association_mode: &str,
    association_backend_kind: &str,
    sample_count: i64,
    variant_count: i64,
    phenotype: Option<String>,
    phenotype_count: Option<i64>,
) -> PyResult<()> {
    let Some(native_session_handle) = optional_native_session_handle_from_optional(telemetry_session)? else {
        return Ok(());
    };
    native_session_handle.call_method1(
        "emit_bgen_engine_opened_event",
        (association_mode, association_backend_kind, sample_count, variant_count, phenotype, phenotype_count),
    )?;
    Ok(())
}

#[allow(clippy::needless_pass_by_value)]
fn record_gpu_genotype_format_resolved_telemetry_event(
    telemetry_session: &Bound<'_, PyAny>,
    requested_gpu_genotype_format: &str,
    resolved_gpu_genotype_format: &str,
    resolution_reason: &str,
    fallback_error: Option<String>,
) -> PyResult<()> {
    let Some(native_session_handle) = optional_native_session_handle(telemetry_session)? else {
        return Ok(());
    };
    native_session_handle
        .call_method1(
            "emit_gpu_genotype_format_resolved_event",
            (requested_gpu_genotype_format, resolved_gpu_genotype_format, resolution_reason, fallback_error),
        )
        .map(|_| ())
}

#[allow(clippy::needless_pass_by_value)]
fn record_gpu_genotype_format_resolved_events(
    telemetry_session: &Bound<'_, PyAny>,
    requested_gpu_genotype_format: &str,
    resolved_gpu_genotype_format: &str,
    resolution_reason: &str,
    fallback_error: Option<String>,
) -> PyResult<()> {
    record_gpu_genotype_format_resolved_telemetry_event(
        telemetry_session,
        requested_gpu_genotype_format,
        resolved_gpu_genotype_format,
        resolution_reason,
        fallback_error.clone(),
    )?;
    record_pipeline_gpu_genotype_format_resolved_diagnostic_event(
        requested_gpu_genotype_format,
        resolved_gpu_genotype_format,
        resolution_reason,
        fallback_error.as_deref(),
    )
}

fn record_gpu_genotype_format_resolved_native_plan_events(
    telemetry_session: &Bound<'_, PyAny>,
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
    record_gpu_genotype_format_resolved_events(
        telemetry_session,
        &native_resolution_plan.requested_gpu_genotype_format,
        resolved_gpu_genotype_format,
        resolution_reason,
        native_resolution_plan.fallback_error.clone(),
    )
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub fn plan_gpu_genotype_format_auto_to_dosage_and_record_events(
    telemetry_session: &Bound<'_, PyAny>,
    requested_gpu_genotype_format: String,
    resolution_reason: String,
) -> PyResult<schedule::NativeGpuGenotypeFormatResolutionPlan> {
    let native_resolution_plan =
        native_schedule::plan_gpu_genotype_format_auto_to_dosage(&requested_gpu_genotype_format, &resolution_reason)
            .map_err(|error| errors::convert_schedule_error(&error))?;
    record_gpu_genotype_format_resolved_native_plan_events(telemetry_session, &native_resolution_plan)?;
    Ok(native_resolution_plan.into())
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub fn plan_single_trait_binary_gpu_genotype_format_resolution_and_record_events(
    telemetry_session: &Bound<'_, PyAny>,
    requested_gpu_genotype_format: String,
    manifest_gpu_genotype_format: Option<String>,
    association_backend_genotype_format: Option<String>,
    resume: bool,
    jax_device: String,
) -> PyResult<schedule::NativeGpuGenotypeFormatResolutionPlan> {
    let native_resolution_plan = native_schedule::plan_single_trait_binary_gpu_genotype_format_resolution(
        &requested_gpu_genotype_format,
        manifest_gpu_genotype_format.as_deref(),
        association_backend_genotype_format.as_deref(),
        resume,
        &jax_device,
    )
    .map_err(|error| errors::convert_schedule_error(&error))?;
    record_gpu_genotype_format_resolved_native_plan_events(telemetry_session, &native_resolution_plan)?;
    Ok(native_resolution_plan.into())
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub fn plan_auto_gpu_genotype_format_after_trusted_validation_and_record_events(
    telemetry_session: &Bound<'_, PyAny>,
    fallback_error: Option<String>,
) -> PyResult<schedule::NativeGpuGenotypeFormatResolutionPlan> {
    let native_resolution_plan =
        native_schedule::plan_auto_gpu_genotype_format_after_trusted_validation(fallback_error.as_deref());
    record_gpu_genotype_format_resolved_native_plan_events(telemetry_session, &native_resolution_plan)?;
    Ok(native_resolution_plan.into())
}

fn record_runner_run_started_diagnostic_event(
    association_mode: &str,
    trait_type: &str,
    phenotype_count: i64,
) -> PyResult<()> {
    let payload =
        native_run_events::build_runner_run_started_diagnostic_payload(association_mode, trait_type, phenotype_count);
    emit_run_diagnostic_event_payload(&payload)
}

#[pyfunction]
pub fn record_runner_jax_runtime_configuration_started_diagnostic_event() -> PyResult<()> {
    let payload = native_run_events::build_runner_jax_runtime_configuration_started_diagnostic_payload();
    emit_run_diagnostic_event_payload(&payload)
}

#[pyfunction]
pub fn record_runner_execution_plan_build_started_diagnostic_event() -> PyResult<()> {
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

#[pyfunction]
pub fn record_runner_execution_plan_dispatch_started_diagnostic_event(
    phenotype_count: i64,
    association_mode: &str,
) -> PyResult<()> {
    let payload = native_run_events::build_runner_execution_plan_dispatch_started_diagnostic_payload(
        phenotype_count,
        association_mode,
    );
    emit_run_diagnostic_event_payload(&payload)
}

#[pyfunction]
pub fn record_runner_execution_plan_finalization_started_diagnostic_event(
    phenotype_count: i64,
    association_mode: &str,
) -> PyResult<()> {
    let payload = native_run_events::build_runner_execution_plan_finalization_started_diagnostic_payload(
        phenotype_count,
        association_mode,
    );
    emit_run_diagnostic_event_payload(&payload)
}

#[pyfunction]
pub fn record_runner_multi_phenotype_dispatch_started_diagnostic_event(
    phenotype_count: i64,
    association_mode: &str,
) -> PyResult<()> {
    let payload = native_run_events::build_runner_multi_phenotype_dispatch_started_diagnostic_payload(
        phenotype_count,
        association_mode,
    );
    emit_run_diagnostic_event_payload(&payload)
}

#[pyfunction]
pub fn record_runner_single_phenotype_dispatch_started_diagnostic_event(
    association_mode: &str,
    phenotype: &str,
) -> PyResult<()> {
    let payload = native_run_events::build_runner_single_phenotype_dispatch_started_diagnostic_payload(
        association_mode,
        phenotype,
    );
    emit_run_diagnostic_event_payload(&payload)
}

#[pyfunction]
pub fn record_runner_binary_engine_dispatch_started_diagnostic_event(phenotype: &str) -> PyResult<()> {
    let payload = native_run_events::build_runner_binary_engine_dispatch_started_diagnostic_payload(phenotype);
    emit_run_diagnostic_event_payload(&payload)
}

#[pyfunction]
pub fn record_runner_linear_engine_dispatch_started_diagnostic_event(phenotype: &str) -> PyResult<()> {
    let payload = native_run_events::build_runner_linear_engine_dispatch_started_diagnostic_payload(phenotype);
    emit_run_diagnostic_event_payload(&payload)
}

#[pyfunction]
pub fn record_runner_multi_phenotype_binary_engine_dispatch_started_diagnostic_event(
    phenotype_count: i64,
) -> PyResult<()> {
    let payload = native_run_events::build_runner_multi_phenotype_binary_engine_dispatch_started_diagnostic_payload(
        phenotype_count,
    );
    emit_run_diagnostic_event_payload(&payload)
}

#[pyfunction]
pub fn record_runner_multi_phenotype_linear_engine_dispatch_started_diagnostic_event(
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

#[pyfunction]
pub fn record_runner_metadata_artifacts_finalized_diagnostic_event(
    association_mode: &str,
    phenotype_count: i64,
) -> PyResult<()> {
    let payload = native_run_events::build_runner_metadata_artifacts_finalized_diagnostic_payload(
        association_mode,
        phenotype_count,
    );
    emit_run_diagnostic_event_payload(&payload)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn record_preflight_warning_diagnostic_events(
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

#[pyfunction]
pub fn record_pipeline_bgen_engine_open_started_diagnostic_event(
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

#[pyfunction]
pub fn record_pipeline_bgen_engine_opened_diagnostic_event(
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

#[pyfunction]
pub fn record_pipeline_prevalidated_bgen_engine_used_diagnostic_event(
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

#[pyfunction]
pub fn record_pipeline_multi_phenotype_sample_summary_diagnostic_event(
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

#[pyfunction]
pub fn record_pipeline_multi_trait_started_diagnostic_event(
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

#[pyfunction]
pub fn record_pipeline_multi_trait_input_load_started_diagnostic_event(phenotype_count: i64) -> PyResult<()> {
    let payload = native_run_events::build_pipeline_multi_trait_input_load_started_diagnostic_payload(phenotype_count);
    emit_run_diagnostic_event_payload(&payload)
}

#[pyfunction]
pub fn record_pipeline_multi_trait_input_aligned_diagnostic_event(
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

#[pyfunction]
pub fn record_pipeline_multi_trait_prediction_source_load_started_diagnostic_event(
    phenotype_count: i64,
) -> PyResult<()> {
    let payload = native_run_events::build_pipeline_multi_trait_prediction_source_load_started_diagnostic_payload(
        phenotype_count,
    );
    emit_run_diagnostic_event_payload(&payload)
}

#[pyfunction]
pub fn record_pipeline_grouped_per_phenotype_started_diagnostic_event(
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

#[pyfunction]
pub fn record_pipeline_grouped_per_phenotype_groups_prepared_diagnostic_event(
    phenotype_count: i64,
    phenotype_group_count: i64,
) -> PyResult<()> {
    let payload = native_run_events::build_pipeline_grouped_per_phenotype_groups_prepared_diagnostic_payload(
        phenotype_count,
        phenotype_group_count,
    );
    emit_run_diagnostic_event_payload(&payload)
}

#[pyfunction]
pub fn record_pipeline_grouped_union_delivery_selected_diagnostic_event(
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

#[pyfunction]
pub fn record_pipeline_multi_group_preflight_started_diagnostic_event(
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

#[pyfunction]
pub fn record_pipeline_multi_group_preflight_completed_diagnostic_event(
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

#[pyfunction]
pub fn record_pipeline_single_trait_started_diagnostic_event(
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

#[pyfunction]
pub fn record_pipeline_single_trait_input_load_started_diagnostic_event(
    phenotype_name: &str,
    pipeline_label: &str,
) -> PyResult<()> {
    let payload = native_run_events::build_pipeline_single_trait_input_load_started_diagnostic_payload(
        phenotype_name,
        pipeline_label,
    );
    emit_run_diagnostic_event_payload(&payload)
}

#[pyfunction]
pub fn record_pipeline_single_trait_input_aligned_diagnostic_event(
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

#[pyfunction]
pub fn record_pipeline_single_trait_prediction_source_load_started_diagnostic_event(
    phenotype_name: &str,
    pipeline_label: &str,
) -> PyResult<()> {
    let payload = native_run_events::build_pipeline_single_trait_prediction_source_load_started_diagnostic_payload(
        phenotype_name,
        pipeline_label,
    );
    emit_run_diagnostic_event_payload(&payload)
}

#[pyfunction]
pub fn record_pipeline_single_trait_preflight_started_diagnostic_event(
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

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn record_pipeline_single_trait_preflight_completed_diagnostic_event(
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

#[pyfunction]
pub fn record_native_dispatch_callback_drain_started_diagnostic_event() -> PyResult<()> {
    let payload = native_run_events::build_native_dispatch_callback_drain_started_diagnostic_payload();
    emit_run_diagnostic_event_payload(&payload)
}

#[pyfunction]
pub fn record_native_dispatch_delivery_started_diagnostic_event(
    committed_chunk_count: i64,
    pipeline_label: &str,
    variant_major_packed8_probability_pairs: bool,
) -> PyResult<()> {
    let payload = native_run_events::build_native_dispatch_delivery_started_diagnostic_payload(
        committed_chunk_count,
        pipeline_label,
        variant_major_packed8_probability_pairs,
    );
    emit_run_diagnostic_event_payload(&payload)
}

#[pyfunction]
pub fn record_native_dispatch_delivery_finished_diagnostic_event(
    pipeline_label: &str,
    processed_chunk_count: i64,
) -> PyResult<()> {
    let payload = native_run_events::build_native_dispatch_delivery_finished_diagnostic_payload(
        pipeline_label,
        processed_chunk_count,
    );
    emit_run_diagnostic_event_payload(&payload)
}

#[pyfunction]
pub fn record_native_dispatch_delivery_interrupted_diagnostic_event(
    pipeline_label: &str,
    signal_exit_code: i64,
    signal_name: &str,
    signal_number: i64,
) -> PyResult<()> {
    let payload = native_run_events::build_native_dispatch_delivery_interrupted_diagnostic_payload(
        pipeline_label,
        signal_exit_code,
        signal_name,
        signal_number,
    );
    emit_run_diagnostic_event_payload(&payload)
}

#[pyfunction]
pub fn record_native_dispatch_delivery_failed_diagnostic_event(
    exception_message: &str,
    exception_type: &str,
    pipeline_label: &str,
) -> PyResult<()> {
    let payload = native_run_events::build_native_dispatch_delivery_failed_diagnostic_payload(
        exception_message,
        exception_type,
        pipeline_label,
    );
    emit_run_diagnostic_event_payload(&payload)
}

#[pyfunction]
pub fn record_native_dispatch_pipeline_finished_diagnostic_event(
    final_parquet_path_count: i64,
    pipeline_label: &str,
) -> PyResult<()> {
    let payload = native_run_events::build_native_dispatch_pipeline_finished_diagnostic_payload(
        final_parquet_path_count,
        pipeline_label,
    );
    emit_run_diagnostic_event_payload(&payload)
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    register_run_lifecycle_exports(module)?;
    register_runner_diagnostic_exports(module)?;
    register_output_and_preflight_diagnostic_exports(module)?;
    register_pipeline_diagnostic_exports(module)?;
    register_native_dispatch_diagnostic_exports(module)?;
    Ok(())
}

fn register_run_lifecycle_exports(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeRunArtifacts>()?;
    module.add_class::<NativeRunArtifactPayload>()?;
    module.add_class::<NativeRunCompletedEvent>()?;
    module.add_class::<NativeRunInterruptedEvent>()?;
    module.add_class::<NativeRunFailedEvent>()?;
    module.add_function(wrap_pyfunction!(record_runner_run_started_events, module)?)?;
    module.add_function(wrap_pyfunction!(record_runner_run_interrupted_events, module)?)?;
    module.add_function(wrap_pyfunction!(record_runner_run_failed_events, module)?)?;
    module.add_function(wrap_pyfunction!(attach_run_metadata_and_record_completed_events, module)?)?;
    module.add_function(wrap_pyfunction!(record_execution_plan_prepared_events, module)?)?;
    module.add_function(wrap_pyfunction!(record_callback_progress_update_telemetry, module)?)?;
    module.add_function(wrap_pyfunction!(record_callback_progress_event_telemetry, module)?)?;
    module.add_function(wrap_pyfunction!(record_binary_correction_summary_telemetry, module)?)?;
    module.add_function(wrap_pyfunction!(record_phenotype_writer_finished_telemetry, module)?)?;
    module.add_function(wrap_pyfunction!(record_multi_phenotype_writer_finished_telemetry, module)?)?;
    module.add_function(wrap_pyfunction!(record_sample_alignment_completed_telemetry, module)?)?;
    module.add_function(wrap_pyfunction!(record_prediction_source_loaded_telemetry, module)?)?;
    module.add_function(wrap_pyfunction!(record_single_trait_preflight_completed_telemetry, module)?)?;
    module.add_function(wrap_pyfunction!(record_multi_phenotype_preflight_completed_telemetry, module)?)?;
    module.add_function(wrap_pyfunction!(record_multi_phenotype_sample_summary_telemetry, module)?)?;
    module.add_function(wrap_pyfunction!(record_association_backend_selected_telemetry, module)?)?;
    module.add_function(wrap_pyfunction!(record_bgen_engine_opened_telemetry, module)?)?;
    module.add_function(wrap_pyfunction!(plan_gpu_genotype_format_auto_to_dosage_and_record_events, module)?)?;
    module.add_function(wrap_pyfunction!(
        plan_single_trait_binary_gpu_genotype_format_resolution_and_record_events,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        plan_auto_gpu_genotype_format_after_trusted_validation_and_record_events,
        module
    )?)?;
    Ok(())
}

fn register_runner_diagnostic_exports(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(record_runner_jax_runtime_configuration_started_diagnostic_event, module)?)?;
    module.add_function(wrap_pyfunction!(record_runner_execution_plan_build_started_diagnostic_event, module)?)?;
    module.add_function(wrap_pyfunction!(record_runner_execution_plan_dispatch_started_diagnostic_event, module)?)?;
    module
        .add_function(wrap_pyfunction!(record_runner_execution_plan_finalization_started_diagnostic_event, module)?)?;
    module.add_function(wrap_pyfunction!(record_runner_multi_phenotype_dispatch_started_diagnostic_event, module)?)?;
    module.add_function(wrap_pyfunction!(record_runner_single_phenotype_dispatch_started_diagnostic_event, module)?)?;
    module.add_function(wrap_pyfunction!(record_runner_binary_engine_dispatch_started_diagnostic_event, module)?)?;
    module.add_function(wrap_pyfunction!(record_runner_linear_engine_dispatch_started_diagnostic_event, module)?)?;
    module.add_function(wrap_pyfunction!(
        record_runner_multi_phenotype_binary_engine_dispatch_started_diagnostic_event,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        record_runner_multi_phenotype_linear_engine_dispatch_started_diagnostic_event,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(record_runner_metadata_artifacts_finalized_diagnostic_event, module)?)?;
    Ok(())
}

fn register_output_and_preflight_diagnostic_exports(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(record_preflight_warning_diagnostic_events, module)?)?;
    Ok(())
}

fn register_pipeline_diagnostic_exports(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(record_pipeline_bgen_engine_open_started_diagnostic_event, module)?)?;
    module.add_function(wrap_pyfunction!(record_pipeline_bgen_engine_opened_diagnostic_event, module)?)?;
    module.add_function(wrap_pyfunction!(record_pipeline_prevalidated_bgen_engine_used_diagnostic_event, module)?)?;
    module.add_function(wrap_pyfunction!(record_pipeline_multi_phenotype_sample_summary_diagnostic_event, module)?)?;
    module.add_function(wrap_pyfunction!(record_pipeline_multi_trait_started_diagnostic_event, module)?)?;
    module.add_function(wrap_pyfunction!(record_pipeline_multi_trait_input_load_started_diagnostic_event, module)?)?;
    module.add_function(wrap_pyfunction!(record_pipeline_multi_trait_input_aligned_diagnostic_event, module)?)?;
    module.add_function(wrap_pyfunction!(
        record_pipeline_multi_trait_prediction_source_load_started_diagnostic_event,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(record_pipeline_grouped_per_phenotype_started_diagnostic_event, module)?)?;
    module.add_function(wrap_pyfunction!(
        record_pipeline_grouped_per_phenotype_groups_prepared_diagnostic_event,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(record_pipeline_grouped_union_delivery_selected_diagnostic_event, module)?)?;
    module.add_function(wrap_pyfunction!(record_pipeline_multi_group_preflight_started_diagnostic_event, module)?)?;
    module.add_function(wrap_pyfunction!(record_pipeline_multi_group_preflight_completed_diagnostic_event, module)?)?;
    module.add_function(wrap_pyfunction!(record_pipeline_single_trait_started_diagnostic_event, module)?)?;
    module.add_function(wrap_pyfunction!(record_pipeline_single_trait_input_load_started_diagnostic_event, module)?)?;
    module.add_function(wrap_pyfunction!(record_pipeline_single_trait_input_aligned_diagnostic_event, module)?)?;
    module.add_function(wrap_pyfunction!(
        record_pipeline_single_trait_prediction_source_load_started_diagnostic_event,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(record_pipeline_single_trait_preflight_started_diagnostic_event, module)?)?;
    module
        .add_function(wrap_pyfunction!(record_pipeline_single_trait_preflight_completed_diagnostic_event, module)?)?;
    Ok(())
}

fn register_native_dispatch_diagnostic_exports(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(record_native_dispatch_callback_drain_started_diagnostic_event, module)?)?;
    module.add_function(wrap_pyfunction!(record_native_dispatch_delivery_started_diagnostic_event, module)?)?;
    module.add_function(wrap_pyfunction!(record_native_dispatch_delivery_finished_diagnostic_event, module)?)?;
    module.add_function(wrap_pyfunction!(record_native_dispatch_delivery_interrupted_diagnostic_event, module)?)?;
    module.add_function(wrap_pyfunction!(record_native_dispatch_delivery_failed_diagnostic_event, module)?)?;
    module.add_function(wrap_pyfunction!(record_native_dispatch_pipeline_finished_diagnostic_event, module)?)?;
    Ok(())
}

pub(crate) fn run_completed_event_from_py(
    event: &Bound<'_, PyAny>,
) -> PyResult<native_run_events::RunCompletedEventPayload> {
    Ok(native_run_events::RunCompletedEventPayload {
        run_id: optional_string_attribute(event, "run_id")?,
        association_mode: optional_enum_value(event, "association_mode")?,
        phenotype_count: optional_i64_attribute(event, "phenotype_count")?,
        artifacts: artifact_payloads_from_py_event(event)?,
    })
}

pub(crate) fn run_interrupted_event_payload_from_shutdown_request(
    shutdown_request: &Bound<'_, PyAny>,
) -> PyResult<native_run_events::RunInterruptedEventPayload> {
    let shutdown_signal = shutdown_request.getattr("shutdown_signal")?;
    let signal_name = shutdown_signal.getattr("name")?.extract::<String>()?;
    Ok(native_run_events::build_run_interrupted_event_payload(
        shutdown_signal.getattr("number")?.extract::<i64>()?,
        &signal_name,
        shutdown_signal.getattr("exit_code")?.extract::<i64>()?,
        true,
    ))
}

pub(crate) fn run_failed_event_payload_from_error(
    error: &Bound<'_, PyAny>,
) -> PyResult<native_run_events::RunFailedEventPayload> {
    let error_type = error.get_type().name()?.to_string_lossy().into_owned();
    let error_message = error.str()?.to_string_lossy().into_owned();
    Ok(native_run_events::build_run_failed_event_payload(&error_type, &error_message))
}

pub(crate) fn run_interrupted_event_from_py(
    event: &Bound<'_, PyAny>,
) -> PyResult<native_run_events::RunInterruptedEventPayload> {
    Ok(native_run_events::RunInterruptedEventPayload {
        signal_number: event.getattr("signal_number")?.extract::<i64>()?,
        signal_name: event.getattr("signal_name")?.extract::<String>()?,
        exit_code: event.getattr("exit_code")?.extract::<i64>()?,
        flushed_for_resume: event.getattr("flushed_for_resume")?.extract::<bool>()?,
    })
}

pub(crate) fn run_failed_event_from_py(event: &Bound<'_, PyAny>) -> PyResult<native_run_events::RunFailedEventPayload> {
    Ok(native_run_events::RunFailedEventPayload {
        error_type: event.getattr("error_type")?.extract::<String>()?,
        error_message: event.getattr("error_message")?.extract::<String>()?,
    })
}

pub(crate) fn run_artifacts_payload_from_py(
    artifacts: &Bound<'_, PyAny>,
) -> PyResult<native_run_events::RunArtifactsPayload> {
    let phenotype_artifacts = artifacts.getattr("phenotype_artifacts")?;
    let mut artifact_payloads = Vec::new();
    for phenotype_artifact in phenotype_artifacts.try_iter()? {
        artifact_payloads.push(run_artifacts_payload_from_py(&phenotype_artifact?)?);
    }
    Ok(native_run_events::RunArtifactsPayload {
        output_run_directory: optional_path_string(artifacts, "output_run_directory")?,
        final_dataset: optional_path_string(artifacts, "final_dataset")?,
        final_parquet: optional_path_string(artifacts, "final_parquet")?,
        final_regenie: optional_path_string(artifacts, "final_regenie")?,
        effective_config: optional_path_string(artifacts, "effective_config")?,
        phenotype_artifacts: artifact_payloads,
        phenotype_name: optional_string_attribute(artifacts, "phenotype_name")?,
        association_mode: optional_enum_value(artifacts, "association_mode")?,
        phenotype_count: optional_i64_attribute(artifacts, "phenotype_count")?,
        run_id: optional_string_attribute(artifacts, "run_id")?,
    })
}

fn artifact_payloads_from_py_event(event: &Bound<'_, PyAny>) -> PyResult<Vec<native_run_events::RunArtifactPayload>> {
    let artifact_payloads = event.getattr("artifacts")?;
    let mut artifacts = Vec::new();
    for artifact in artifact_payloads.try_iter()? {
        artifacts.push(artifact_payload_from_py(&artifact?)?);
    }
    Ok(artifacts)
}

fn artifact_payload_from_py(artifact: &Bound<'_, PyAny>) -> PyResult<native_run_events::RunArtifactPayload> {
    Ok(native_run_events::RunArtifactPayload {
        phenotype_name: optional_string_attribute(artifact, "phenotype_name")?,
        output_run_directory: optional_path_string(artifact, "output_run_directory")?,
        final_dataset: optional_path_string(artifact, "final_dataset")?,
        final_parquet: optional_path_string(artifact, "final_parquet")?,
        final_regenie: optional_path_string(artifact, "final_regenie")?,
        effective_config: optional_path_string(artifact, "effective_config")?,
    })
}

pub(crate) fn emit_run_diagnostic_event_payload(event: &native_run_events::RunDiagnosticEventPayload) -> PyResult<()> {
    let fields_json = native_run_events::serialize_run_diagnostic_fields_json(&event.fields).map_err(|error| {
        pyo3::exceptions::PyValueError::new_err(format!("Failed to serialize diagnostic event fields: {error}"))
    })?;
    logging::emit_diagnostic_event(event.level, event.event_name, &event.message, Some(fields_json))
}

fn optional_native_session_handle<'py>(telemetry_session: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    if telemetry_session.is_none() {
        return Ok(None);
    }
    telemetry_session.getattr("native_session_handle").map(Some)
}

fn optional_native_session_handle_from_optional<'py>(
    telemetry_session: Option<&Bound<'py, PyAny>>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let Some(telemetry_session) = telemetry_session else {
        return Ok(None);
    };
    optional_native_session_handle(telemetry_session)
}

fn emit_callback_progress_event(
    py: Python<'_>,
    native_session_handle: &Bound<'_, PyAny>,
    progress_event: &NativeCallbackProgressTelemetryEvent,
) -> PyResult<()> {
    let progress_event_object = Py::new(py, progress_event.clone())?;
    native_session_handle.call_method1("emit_callback_progress_event", (progress_event_object,))?;
    Ok(())
}

pub(crate) fn run_completed_telemetry_fields_to_py_dict<'py>(
    py: Python<'py>,
    fields: &native_run_events::RunCompletedTelemetryFields,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("artifact_count", fields.artifact_count)?;
    let phenotype_artifacts = fields
        .phenotype_artifacts
        .iter()
        .map(|artifact| artifact_telemetry_fields_to_py_dict(py, artifact))
        .collect::<PyResult<Vec<_>>>()?;
    payload.set_item("phenotype_artifacts", PyTuple::new(py, &phenotype_artifacts)?)?;
    set_optional_string(&payload, "run_id", fields.run_id.as_deref())?;
    set_optional_string(&payload, "association_mode", fields.association_mode.as_deref())?;
    set_optional_i64(&payload, "phenotype_count", fields.phenotype_count)?;
    if let Some(single_artifact) = fields.single_artifact.as_ref() {
        copy_artifact_fields_to_py_dict(&payload, single_artifact)?;
    }
    Ok(payload)
}

pub(crate) fn run_interrupted_telemetry_fields_to_py_dict<'py>(
    py: Python<'py>,
    fields: &native_run_events::RunInterruptedTelemetryFields,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("failure_kind", fields.failure_kind)?;
    payload.set_item("signal_number", fields.signal_number)?;
    payload.set_item("signal_name", &fields.signal_name)?;
    payload.set_item("exit_code", fields.exit_code)?;
    payload.set_item("flushed_for_resume", fields.flushed_for_resume)?;
    Ok(payload)
}

pub(crate) fn run_failed_telemetry_fields_to_py_dict<'py>(
    py: Python<'py>,
    fields: &native_run_events::RunFailedTelemetryFields,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("failure_kind", fields.failure_kind)?;
    payload.set_item("error_type", &fields.error_type)?;
    payload.set_item("error_message", &fields.error_message)?;
    Ok(payload)
}

pub(crate) fn run_started_telemetry_fields_to_py_dict<'py>(
    py: Python<'py>,
    fields: &native_run_events::RunStartedTelemetryFields,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("association_mode", &fields.association_mode)?;
    payload.set_item("trait_type", &fields.trait_type)?;
    payload.set_item("phenotype_count", fields.phenotype_count)?;
    payload.set_item("output_run_root", &fields.output_run_root)?;
    Ok(payload)
}

pub(crate) fn execution_plan_prepared_telemetry_fields_to_py_dict<'py>(
    py: Python<'py>,
    fields: &native_run_events::ExecutionPlanPreparedTelemetryFields,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("association_mode", &fields.association_mode)?;
    payload.set_item("trait_type", &fields.trait_type)?;
    payload.set_item("phenotype_count", fields.phenotype_count)?;
    payload.set_item("chunk_size", fields.chunk_size)?;
    payload.set_item("variant_limit", fields.variant_limit)?;
    payload.set_item("device", &fields.device)?;
    Ok(payload)
}

pub(crate) fn effective_config_written_telemetry_fields_to_py_dict<'py>(
    py: Python<'py>,
    fields: &native_run_events::EffectiveConfigWrittenTelemetryFields,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("association_mode", &fields.association_mode)?;
    payload.set_item("phenotype", &fields.phenotype)?;
    payload.set_item("effective_config", &fields.effective_config)?;
    payload.set_item("output_run_directory", &fields.output_run_directory)?;
    Ok(payload)
}

pub(crate) fn phenotype_writer_finished_telemetry_fields_to_py_dict<'py>(
    py: Python<'py>,
    fields: &native_run_events::PhenotypeWriterFinishedTelemetryFields,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("association_mode", &fields.association_mode)?;
    payload.set_item("phenotype", &fields.phenotype)?;
    payload.set_item("final_output_path", &fields.final_output_path)?;
    Ok(payload)
}

pub(crate) fn multi_phenotype_writer_finished_telemetry_fields_to_py_dict<'py>(
    py: Python<'py>,
    fields: &native_run_events::MultiPhenotypeWriterFinishedTelemetryFields,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    let final_output_paths = fields.final_output_paths.iter().map(|path| path.as_deref()).collect::<Vec<_>>();
    payload.set_item("association_mode", &fields.association_mode)?;
    payload.set_item("phenotype_count", fields.phenotype_count)?;
    payload.set_item("final_output_paths", PyTuple::new(py, final_output_paths)?)?;
    Ok(payload)
}

pub(crate) fn single_trait_preflight_completed_telemetry_fields_to_py_dict<'py>(
    py: Python<'py>,
    fields: &native_run_events::SingleTraitPreflightCompletedTelemetryFields,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("association_mode", &fields.association_mode)?;
    payload.set_item("phenotype", &fields.phenotype)?;
    payload.set_item("sample_count", fields.sample_count)?;
    payload.set_item("covariate_count", fields.covariate_count)?;
    payload.set_item("chromosome_count", fields.chromosome_count)?;
    Ok(payload)
}

pub(crate) fn multi_phenotype_preflight_completed_telemetry_fields_to_py_dict<'py>(
    py: Python<'py>,
    fields: &native_run_events::MultiPhenotypePreflightCompletedTelemetryFields,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("association_mode", &fields.association_mode)?;
    payload.set_item("phenotype_count", fields.phenotype_count)?;
    payload.set_item("sample_count", fields.sample_count)?;
    Ok(payload)
}

pub(crate) fn sample_alignment_completed_telemetry_fields_to_py_dict<'py>(
    py: Python<'py>,
    fields: &native_run_events::SampleAlignmentCompletedTelemetryFields,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("association_mode", &fields.association_mode)?;
    set_optional_string(&payload, "phenotype", fields.phenotype.as_deref())?;
    set_optional_i64(&payload, "phenotype_count", fields.phenotype_count)?;
    set_optional_i64(&payload, "sample_count", fields.sample_count)?;
    set_optional_i64(&payload, "covariate_count", fields.covariate_count)?;
    set_optional_i64(&payload, "phenotype_group_count", fields.phenotype_group_count)?;
    Ok(payload)
}

pub(crate) fn prediction_source_loaded_telemetry_fields_to_py_dict<'py>(
    py: Python<'py>,
    fields: &native_run_events::PredictionSourceLoadedTelemetryFields,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("association_mode", &fields.association_mode)?;
    set_optional_string(&payload, "phenotype", fields.phenotype.as_deref())?;
    set_optional_i64(&payload, "phenotype_count", fields.phenotype_count)?;
    Ok(payload)
}

pub(crate) fn multi_phenotype_sample_summary_telemetry_fields_to_py_dict<'py>(
    py: Python<'py>,
    fields: &native_run_events::MultiPhenotypeSampleSummaryTelemetryFields,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("association_mode", &fields.association_mode)?;
    payload.set_item("multi_phenotype_sample_mode", &fields.multi_phenotype_sample_mode)?;
    payload.set_item("phenotype_count", fields.phenotype_count)?;
    payload.set_item("phenotype_group_count", fields.phenotype_group_count)?;
    payload.set_item("sample_counts", PyTuple::new(py, &fields.sample_counts)?)?;
    payload.set_item("sample_counts_differ", fields.sample_counts_differ)?;
    payload.set_item("shared_sample_set", fields.shared_sample_set)?;
    Ok(payload)
}

pub(crate) fn gpu_genotype_format_resolved_telemetry_fields_to_py_dict<'py>(
    py: Python<'py>,
    fields: &native_run_events::GpuGenotypeFormatResolvedTelemetryFields,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("requested_gpu_genotype_format", &fields.requested_gpu_genotype_format)?;
    payload.set_item("resolved_gpu_genotype_format", &fields.resolved_gpu_genotype_format)?;
    payload.set_item("resolution_reason", &fields.resolution_reason)?;
    set_optional_string(&payload, "fallback_error", fields.fallback_error.as_deref())?;
    Ok(payload)
}

pub(crate) fn association_backend_selected_telemetry_fields_to_py_dict<'py>(
    py: Python<'py>,
    fields: &native_run_events::AssociationBackendSelectedTelemetryFields,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("association_mode", &fields.association_mode)?;
    payload.set_item("association_backend_kind", &fields.association_backend_kind)?;
    payload.set_item("device", &fields.device)?;
    payload.set_item("genotype_format", &fields.genotype_format)?;
    set_optional_string(&payload, "phenotype", fields.phenotype.as_deref())?;
    set_optional_i64(&payload, "phenotype_count", fields.phenotype_count)?;
    Ok(payload)
}

pub(crate) fn bgen_engine_opened_telemetry_fields_to_py_dict<'py>(
    py: Python<'py>,
    fields: &native_run_events::BgenEngineOpenedTelemetryFields,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("association_mode", &fields.association_mode)?;
    payload.set_item("association_backend_kind", &fields.association_backend_kind)?;
    payload.set_item("sample_count", fields.sample_count)?;
    payload.set_item("variant_count", fields.variant_count)?;
    set_optional_string(&payload, "phenotype", fields.phenotype.as_deref())?;
    set_optional_i64(&payload, "phenotype_count", fields.phenotype_count)?;
    Ok(payload)
}

fn artifact_telemetry_fields_to_py_dict<'py>(
    py: Python<'py>,
    fields: &native_run_events::RunArtifactTelemetryFields,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    copy_artifact_fields_to_py_dict(&payload, fields)?;
    Ok(payload)
}

fn copy_artifact_fields_to_py_dict(
    payload: &Bound<'_, PyDict>,
    fields: &native_run_events::RunArtifactTelemetryFields,
) -> PyResult<()> {
    for field in &fields.fields {
        payload.set_item(field.key, &field.value)?;
    }
    Ok(())
}

fn set_optional_string(payload: &Bound<'_, PyDict>, payload_key: &str, value: Option<&str>) -> PyResult<()> {
    if let Some(value) = value {
        payload.set_item(payload_key, value)?;
    }
    Ok(())
}

fn set_optional_i64(payload: &Bound<'_, PyDict>, payload_key: &str, value: Option<i64>) -> PyResult<()> {
    if let Some(value) = value {
        payload.set_item(payload_key, value)?;
    }
    Ok(())
}

fn optional_string_attribute(source: &Bound<'_, PyAny>, attribute_name: &str) -> PyResult<Option<String>> {
    let value = source.getattr(attribute_name)?;
    if value.is_none() {
        return Ok(None);
    }
    Ok(Some(value.extract::<String>()?))
}

fn optional_i64_attribute(source: &Bound<'_, PyAny>, attribute_name: &str) -> PyResult<Option<i64>> {
    let value = source.getattr(attribute_name)?;
    if value.is_none() {
        return Ok(None);
    }
    Ok(Some(value.extract::<i64>()?))
}

fn optional_enum_value(source: &Bound<'_, PyAny>, attribute_name: &str) -> PyResult<Option<String>> {
    let value = source.getattr(attribute_name)?;
    if value.is_none() {
        return Ok(None);
    }
    if let Ok(value_text) = value.extract::<String>() {
        return Ok(Some(value_text));
    }
    Ok(Some(value.getattr("value")?.extract::<String>()?))
}

fn optional_path_string(source: &Bound<'_, PyAny>, attribute_name: &str) -> PyResult<Option<String>> {
    let value = source.getattr(attribute_name)?;
    if value.is_none() {
        return Ok(None);
    }
    Ok(Some(value.str()?.to_string_lossy().into_owned()))
}
