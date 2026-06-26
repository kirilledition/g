//! PyO3 adapters for runtime-owned run lifecycle events.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use g_runtime::run_events as native_run_events;

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

fn run_completed_event_from_py(event: &Bound<'_, PyAny>) -> PyResult<native_run_events::RunCompletedEventPayload> {
    Ok(native_run_events::RunCompletedEventPayload {
        run_id: optional_string_attribute(event, "run_id")?,
        association_mode: optional_enum_value(event, "association_mode")?,
        phenotype_count: optional_i64_attribute(event, "phenotype_count")?,
        artifacts: artifact_payloads_from_py_event(event)?,
    })
}

fn run_interrupted_event_from_py(event: &Bound<'_, PyAny>) -> PyResult<native_run_events::RunInterruptedEventPayload> {
    Ok(native_run_events::RunInterruptedEventPayload {
        signal_number: event.getattr("signal_number")?.extract::<i64>()?,
        signal_name: event.getattr("signal_name")?.extract::<String>()?,
        exit_code: event.getattr("exit_code")?.extract::<i64>()?,
        flushed_for_resume: event.getattr("flushed_for_resume")?.extract::<bool>()?,
    })
}

fn run_failed_event_from_py(event: &Bound<'_, PyAny>) -> PyResult<native_run_events::RunFailedEventPayload> {
    Ok(native_run_events::RunFailedEventPayload {
        error_type: event.getattr("error_type")?.extract::<String>()?,
        error_message: event.getattr("error_message")?.extract::<String>()?,
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

fn run_completed_telemetry_fields_to_py_dict<'py>(
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

fn run_interrupted_telemetry_fields_to_py_dict<'py>(
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

fn run_failed_telemetry_fields_to_py_dict<'py>(
    py: Python<'py>,
    fields: &native_run_events::RunFailedTelemetryFields,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("failure_kind", fields.failure_kind)?;
    payload.set_item("error_type", &fields.error_type)?;
    payload.set_item("error_message", &fields.error_message)?;
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
