//! Native helpers for run lifecycle telemetry payloads and terminal rendering.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

#[pyfunction]
pub fn build_run_completed_telemetry_fields<'py>(
    py: Python<'py>,
    event: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let artifact_mappings = artifact_mappings_from_event(py, event)?;
    let fields = PyDict::new(py);
    fields.set_item("artifact_count", artifact_mappings.len())?;
    fields.set_item("phenotype_artifacts", PyTuple::new(py, &artifact_mappings)?)?;
    set_optional_attribute(fields.as_any(), event, "run_id", "run_id")?;
    if let Some(association_mode) = optional_enum_value(event, "association_mode")? {
        fields.set_item("association_mode", association_mode)?;
    }
    set_optional_attribute(fields.as_any(), event, "phenotype_count", "phenotype_count")?;
    if artifact_mappings.len() == 1 {
        for (key, value) in &artifact_mappings[0] {
            fields.set_item(key, value)?;
        }
    }
    Ok(fields)
}

#[pyfunction]
pub fn build_run_interrupted_telemetry_fields<'py>(
    py: Python<'py>,
    event: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let fields = PyDict::new(py);
    fields.set_item("failure_kind", "graceful_shutdown")?;
    set_attribute(fields.as_any(), event, "signal_number", "signal_number")?;
    set_attribute(fields.as_any(), event, "signal_name", "signal_name")?;
    set_attribute(fields.as_any(), event, "exit_code", "exit_code")?;
    set_attribute(fields.as_any(), event, "flushed_for_resume", "flushed_for_resume")?;
    Ok(fields)
}

#[pyfunction]
pub fn build_run_failed_telemetry_fields<'py>(
    py: Python<'py>,
    event: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let fields = PyDict::new(py);
    fields.set_item("failure_kind", "exception")?;
    set_attribute(fields.as_any(), event, "error_type", "error_type")?;
    set_attribute(fields.as_any(), event, "error_message", "error_message")?;
    Ok(fields)
}

#[pyfunction]
pub fn render_run_completed_lines<'py>(py: Python<'py>, event: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyTuple>> {
    let artifact_payloads = event.getattr("artifacts")?;
    let mut lines = Vec::new();
    for artifact in artifact_payloads.try_iter()? {
        lines.extend(render_artifact_lines(&artifact?)?);
    }
    if lines.is_empty() {
        lines.push("Success. Run completed.".to_string());
    }
    PyTuple::new(py, lines)
}

#[pyfunction]
pub fn render_run_interrupted_lines<'py>(py: Python<'py>, event: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyTuple>> {
    let signal_name = event.getattr("signal_name")?.extract::<String>()?;
    PyTuple::new(
        py,
        [format!("Interrupted by {signal_name}. Flushed queued chunks and saved committed output for --resume.")],
    )
}

#[pyfunction]
pub fn render_run_failed_lines<'py>(py: Python<'py>, event: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyTuple>> {
    let error_message = event.getattr("error_message")?.extract::<String>()?;
    if error_message.is_empty() {
        let error_type = event.getattr("error_type")?.extract::<String>()?;
        return PyTuple::new(py, [format!("Error: {error_type}")]);
    }
    PyTuple::new(py, [format!("Error: {error_message}")])
}

fn artifact_mappings_from_event<'py>(py: Python<'py>, event: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let artifact_payloads = event.getattr("artifacts")?;
    let mut artifact_mappings = Vec::new();
    for artifact in artifact_payloads.try_iter()? {
        artifact_mappings.push(artifact_payload_to_mapping(py, &artifact?)?);
    }
    Ok(artifact_mappings)
}

fn artifact_payload_to_mapping<'py>(py: Python<'py>, artifact: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    set_optional_string_attribute(payload.as_any(), artifact, "phenotype", "phenotype_name")?;
    set_optional_path_attribute(payload.as_any(), artifact, "output_run_directory", "output_run_directory")?;
    set_optional_path_attribute(payload.as_any(), artifact, "final_dataset", "final_dataset")?;
    set_optional_path_attribute(payload.as_any(), artifact, "final_parquet", "final_parquet")?;
    set_optional_path_attribute(payload.as_any(), artifact, "final_regenie", "final_regenie")?;
    set_optional_path_attribute(payload.as_any(), artifact, "effective_config", "effective_config")?;
    Ok(payload)
}

fn render_artifact_lines(artifact: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let mut lines = Vec::new();
    if let Some(output_run_directory) = optional_path_string(artifact, "output_run_directory")? {
        lines.push(format!("Success. Chunked run saved to {output_run_directory}"));
    } else {
        lines.push("Success. Run completed.".to_string());
    }
    if let Some(final_dataset) = optional_path_string(artifact, "final_dataset")? {
        lines.push(format!("Parquet dataset saved to {final_dataset}"));
    }
    if let Some(final_parquet) = optional_path_string(artifact, "final_parquet")? {
        lines.push(format!("Finalized Parquet saved to {final_parquet}"));
    }
    if let Some(final_regenie) = optional_path_string(artifact, "final_regenie")? {
        lines.push(format!("REGENIE text output saved to {final_regenie}"));
    }
    Ok(lines)
}

fn set_attribute(
    payload: &Bound<'_, PyAny>,
    source: &Bound<'_, PyAny>,
    payload_key: &str,
    attribute_name: &str,
) -> PyResult<()> {
    payload.call_method1("__setitem__", (payload_key, source.getattr(attribute_name)?))?;
    Ok(())
}

fn set_optional_attribute(
    payload: &Bound<'_, PyAny>,
    source: &Bound<'_, PyAny>,
    payload_key: &str,
    attribute_name: &str,
) -> PyResult<()> {
    let value = source.getattr(attribute_name)?;
    if !value.is_none() {
        payload.call_method1("__setitem__", (payload_key, value))?;
    }
    Ok(())
}

fn set_optional_string_attribute(
    payload: &Bound<'_, PyAny>,
    source: &Bound<'_, PyAny>,
    payload_key: &str,
    attribute_name: &str,
) -> PyResult<()> {
    let value = source.getattr(attribute_name)?;
    if !value.is_none() {
        payload.call_method1("__setitem__", (payload_key, value.extract::<String>()?))?;
    }
    Ok(())
}

fn set_optional_path_attribute(
    payload: &Bound<'_, PyAny>,
    source: &Bound<'_, PyAny>,
    payload_key: &str,
    attribute_name: &str,
) -> PyResult<()> {
    if let Some(value) = optional_path_string(source, attribute_name)? {
        payload.call_method1("__setitem__", (payload_key, value))?;
    }
    Ok(())
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
