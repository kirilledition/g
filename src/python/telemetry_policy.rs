//! PyO3 adapters for telemetry path and counter policy helpers.

use std::path::Path;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

use g_runtime::telemetry_policy as native_telemetry_policy;

#[pyclass]
pub(crate) struct NativeTelemetrySessionPolicy {
    policy: native_telemetry_policy::TelemetrySessionPolicyPayload,
}

#[pymethods]
impl NativeTelemetrySessionPolicy {
    #[new]
    #[allow(clippy::needless_pass_by_value)]
    fn new(telemetry_mode: String, trace_event_cap: i64) -> Self {
        Self { policy: native_telemetry_policy::resolve_telemetry_session_policy(&telemetry_mode, trace_event_cap) }
    }

    #[getter]
    fn enabled(&self) -> bool {
        self.policy.enabled
    }

    #[getter]
    fn profile_enabled(&self) -> bool {
        self.policy.profile_enabled
    }

    #[getter]
    fn event_cap(&self) -> Option<i64> {
        self.policy.event_cap
    }
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

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeTelemetrySessionPolicy>()?;
    module.add_function(wrap_pyfunction!(resolve_telemetry_output_run_root_value, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_telemetry_paths_payload, module)?)?;
    Ok(())
}
