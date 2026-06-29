//! PyO3 adapters for runtime-owned run lifecycle events.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use g_runtime::run_events as native_run_events;

#[pyfunction]
pub fn build_run_completed_event_payload<'py>(
    py: Python<'py>,
    artifacts: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let artifacts_payload = run_artifacts_payload_from_py(artifacts)?;
    let event_payload = native_run_events::build_run_completed_event_from_artifacts(&artifacts_payload);
    run_completed_event_payload_to_py_dict(py, &event_payload)
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub fn attach_run_metadata_payload<'py>(
    py: Python<'py>,
    artifacts: &Bound<'py, PyAny>,
    run_id: Option<String>,
    association_mode: String,
    phenotype_count: i64,
) -> PyResult<Bound<'py, PyDict>> {
    let artifacts_payload = run_artifacts_payload_from_py(artifacts)?;
    let attached_artifacts = native_run_events::attach_run_metadata_to_artifacts(
        &artifacts_payload,
        run_id.as_deref(),
        &association_mode,
        phenotype_count,
    );
    run_artifacts_payload_to_py_dict(py, &attached_artifacts)
}

#[pyfunction]
pub fn build_run_interrupted_event_payload<'py>(
    py: Python<'py>,
    shutdown_request: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let shutdown_signal = shutdown_request.getattr("shutdown_signal")?;
    let signal_name = shutdown_signal.getattr("name")?.extract::<String>()?;
    let event_payload = native_run_events::build_run_interrupted_event_payload(
        shutdown_signal.getattr("number")?.extract::<i64>()?,
        &signal_name,
        shutdown_signal.getattr("exit_code")?.extract::<i64>()?,
        true,
    );
    run_interrupted_event_payload_to_py_dict(py, &event_payload)
}

#[pyfunction]
pub fn build_run_failed_event_payload<'py>(py: Python<'py>, error: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    let error_type = error.get_type().name()?.to_string_lossy().into_owned();
    let error_message = error.str()?.to_string_lossy().into_owned();
    let event_payload = native_run_events::build_run_failed_event_payload(&error_type, &error_message);
    run_failed_event_payload_to_py_dict(py, &event_payload)
}

#[pyfunction]
pub fn build_run_completed_telemetry_fields<'py>(
    py: Python<'py>,
    event: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let event_payload = run_completed_event_from_py(event)?;
    let fields = native_run_events::build_run_completed_telemetry_fields(&event_payload);
    run_completed_telemetry_fields_to_py_dict(py, &fields)
}

#[pyfunction]
pub fn build_run_interrupted_telemetry_fields<'py>(
    py: Python<'py>,
    event: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let event_payload = run_interrupted_event_from_py(event)?;
    let fields = native_run_events::build_run_interrupted_telemetry_fields(&event_payload);
    run_interrupted_telemetry_fields_to_py_dict(py, &fields)
}

#[pyfunction]
pub fn build_run_failed_telemetry_fields<'py>(
    py: Python<'py>,
    event: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let event_payload = run_failed_event_from_py(event)?;
    let fields = native_run_events::build_run_failed_telemetry_fields(&event_payload);
    run_failed_telemetry_fields_to_py_dict(py, &fields)
}

#[pyfunction]
pub fn render_run_completed_lines<'py>(py: Python<'py>, event: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyTuple>> {
    let event_payload = run_completed_event_from_py(event)?;
    PyTuple::new(py, native_run_events::render_run_completed_lines(&event_payload))
}

#[pyfunction]
pub fn render_run_interrupted_lines<'py>(py: Python<'py>, event: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyTuple>> {
    let event_payload = run_interrupted_event_from_py(event)?;
    PyTuple::new(py, native_run_events::render_run_interrupted_lines(&event_payload))
}

#[pyfunction]
pub fn render_run_failed_lines<'py>(py: Python<'py>, event: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyTuple>> {
    let event_payload = run_failed_event_from_py(event)?;
    PyTuple::new(py, native_run_events::render_run_failed_lines(&event_payload))
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

fn run_artifacts_payload_from_py(artifacts: &Bound<'_, PyAny>) -> PyResult<native_run_events::RunArtifactsPayload> {
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

fn run_artifacts_payload_to_py_dict<'py>(
    py: Python<'py>,
    artifacts: &native_run_events::RunArtifactsPayload,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("output_run_directory", &artifacts.output_run_directory)?;
    payload.set_item("final_dataset", &artifacts.final_dataset)?;
    payload.set_item("final_parquet", &artifacts.final_parquet)?;
    payload.set_item("final_regenie", &artifacts.final_regenie)?;
    payload.set_item("effective_config", &artifacts.effective_config)?;
    let phenotype_artifacts = artifacts
        .phenotype_artifacts
        .iter()
        .map(|phenotype_artifact| run_artifacts_payload_to_py_dict(py, phenotype_artifact))
        .collect::<PyResult<Vec<_>>>()?;
    payload.set_item("phenotype_artifacts", PyTuple::new(py, &phenotype_artifacts)?)?;
    payload.set_item("phenotype_name", &artifacts.phenotype_name)?;
    payload.set_item("association_mode", &artifacts.association_mode)?;
    payload.set_item("phenotype_count", artifacts.phenotype_count)?;
    payload.set_item("run_id", &artifacts.run_id)?;
    Ok(payload)
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

fn run_completed_event_payload_to_py_dict<'py>(
    py: Python<'py>,
    event: &native_run_events::RunCompletedEventPayload,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("run_id", &event.run_id)?;
    payload.set_item("association_mode", &event.association_mode)?;
    payload.set_item("phenotype_count", event.phenotype_count)?;
    let artifacts = event
        .artifacts
        .iter()
        .map(|artifact| run_artifact_payload_to_py_dict(py, artifact))
        .collect::<PyResult<Vec<_>>>()?;
    payload.set_item("artifacts", PyTuple::new(py, &artifacts)?)?;
    Ok(payload)
}

fn run_artifact_payload_to_py_dict<'py>(
    py: Python<'py>,
    artifact: &native_run_events::RunArtifactPayload,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("phenotype_name", &artifact.phenotype_name)?;
    payload.set_item("output_run_directory", &artifact.output_run_directory)?;
    payload.set_item("final_dataset", &artifact.final_dataset)?;
    payload.set_item("final_parquet", &artifact.final_parquet)?;
    payload.set_item("final_regenie", &artifact.final_regenie)?;
    payload.set_item("effective_config", &artifact.effective_config)?;
    Ok(payload)
}

fn run_interrupted_event_payload_to_py_dict<'py>(
    py: Python<'py>,
    event: &native_run_events::RunInterruptedEventPayload,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("signal_number", event.signal_number)?;
    payload.set_item("signal_name", &event.signal_name)?;
    payload.set_item("exit_code", event.exit_code)?;
    payload.set_item("flushed_for_resume", event.flushed_for_resume)?;
    Ok(payload)
}

fn run_failed_event_payload_to_py_dict<'py>(
    py: Python<'py>,
    event: &native_run_events::RunFailedEventPayload,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("error_type", &event.error_type)?;
    payload.set_item("error_message", &event.error_message)?;
    Ok(payload)
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
    Ok(Some(value.getattr("value")?.extract::<String>()?))
}

fn optional_path_string(source: &Bound<'_, PyAny>, attribute_name: &str) -> PyResult<Option<String>> {
    let value = source.getattr(attribute_name)?;
    if value.is_none() {
        return Ok(None);
    }
    Ok(Some(value.str()?.to_string_lossy().into_owned()))
}
