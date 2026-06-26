//! PyO3 adapters for telemetry path and counter policy helpers.

use std::path::Path;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use g_runtime::telemetry_policy as native_telemetry_policy;

#[pyfunction]
pub(crate) fn format_telemetry_timestamp_value(timestamp_seconds: f64) -> String {
    native_telemetry_policy::format_timestamp(timestamp_seconds)
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn resolve_telemetry_output_run_root_value(
    output_path: String,
    output_run_directory: Option<String>,
) -> String {
    native_telemetry_policy::resolve_output_run_root(
        Path::new(&output_path),
        output_run_directory.as_deref().map(Path::new),
    )
    .display()
    .to_string()
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn resolve_telemetry_paths_payload<'py>(
    py: Python<'py>,
    output_path: String,
    output_run_directory: Option<String>,
    telemetry_mode: String,
    log_dir: Option<String>,
    log_file: Option<String>,
    trace_file: Option<String>,
    profile_summary_json: Option<String>,
    stage_timings_json: Option<String>,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = native_telemetry_policy::resolve_telemetry_paths(
        Path::new(&output_path),
        output_run_directory.as_deref().map(Path::new),
        &telemetry_mode,
        log_dir.as_deref().map(Path::new),
        log_file.as_deref().map(Path::new),
        trace_file.as_deref().map(Path::new),
        profile_summary_json.as_deref().map(Path::new),
        stage_timings_json.as_deref().map(Path::new),
    )
    .map_err(PyValueError::new_err)?;
    let python_payload = PyDict::new(py);
    python_payload.set_item("log_dir", payload.log_dir)?;
    python_payload.set_item("stream_file", payload.stream_file)?;
    python_payload.set_item("profile_summary_json", payload.profile_summary_json)?;
    python_payload.set_item("stage_timings_json", payload.stage_timings_json)?;
    Ok(python_payload)
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn resolve_telemetry_stream_file_value(
    telemetry_mode: String,
    log_dir: Option<String>,
    log_file: Option<String>,
    trace_file: Option<String>,
) -> PyResult<Option<String>> {
    native_telemetry_policy::resolve_telemetry_stream_file(
        &telemetry_mode,
        log_dir.as_deref().map(Path::new),
        log_file.as_deref().map(Path::new),
        trace_file.as_deref().map(Path::new),
    )
    .map(|path| path.map(|value| value.display().to_string()))
    .map_err(PyValueError::new_err)
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn paths_refer_to_same_file_value(first_path: String, second_path: String) -> bool {
    native_telemetry_policy::paths_refer_to_same_file(Path::new(&first_path), Path::new(&second_path))
}

#[pyfunction]
pub(crate) fn build_empty_telemetry_writer_counters_payload<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let counters = native_telemetry_policy::build_empty_writer_counters();
    let payload = PyDict::new(py);
    payload.set_item("accepted_event_count", counters.accepted_event_count)?;
    payload.set_item("written_event_count", counters.written_event_count)?;
    payload.set_item("dropped_event_count", counters.dropped_event_count)?;
    payload.set_item("cap_dropped_event_count", counters.cap_dropped_event_count)?;
    payload.set_item("queue_dropped_event_count", counters.queue_dropped_event_count)?;
    payload.set_item("event_cap_exceeded", counters.event_cap_exceeded)?;
    payload.set_item("lossy", counters.lossy)?;
    payload.set_item("event_cap", counters.event_cap)?;
    payload.set_item("finish_flush_duration_seconds", counters.finish_flush_duration_seconds)?;
    Ok(payload)
}
