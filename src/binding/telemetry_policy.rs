//! PyO3 adapters for telemetry path and counter policy helpers.

use std::path::Path;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use g_runtime::debug as native_telemetry_policy;

#[pyclass(skip_from_py_object)]
pub(crate) struct NativeTelemetryPaths {
    payload: native_telemetry_policy::TelemetryPathsPayload,
}

#[pymethods]
impl NativeTelemetryPaths {
    #[getter]
    fn log_dir(&self) -> Option<&str> {
        self.payload.log_dir.as_deref()
    }

    #[getter]
    fn stream_file(&self) -> Option<&str> {
        self.payload.stream_file.as_deref()
    }

    #[getter]
    fn profile_summary_json(&self) -> Option<&str> {
        self.payload.profile_summary_json.as_deref()
    }

    #[getter]
    fn stage_timings_json(&self) -> Option<&str> {
        self.payload.stage_timings_json.as_deref()
    }
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn resolve_output_run_root_value(output_path: String, output_run_directory: Option<String>) -> String {
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
pub(crate) fn resolve_telemetry_paths(
    output_path: String,
    output_run_directory: Option<String>,
    telemetry_mode: String,
    log_dir: Option<String>,
    log_file: Option<String>,
    trace_file: Option<String>,
    profile_summary_json: Option<String>,
    stage_timings_json: Option<String>,
) -> PyResult<NativeTelemetryPaths> {
    let parsed_telemetry_mode = parse_telemetry_mode(&telemetry_mode)?;
    let payload = native_telemetry_policy::resolve_telemetry_paths(
        Path::new(&output_path),
        output_run_directory.as_deref().map(Path::new),
        parsed_telemetry_mode,
        log_dir.as_deref().map(Path::new),
        log_file.as_deref().map(Path::new),
        trace_file.as_deref().map(Path::new),
        profile_summary_json.as_deref().map(Path::new),
        stage_timings_json.as_deref().map(Path::new),
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(NativeTelemetryPaths { payload })
}

pub(crate) fn parse_telemetry_mode(telemetry_mode: &str) -> PyResult<native_telemetry_policy::TelemetryMode> {
    native_telemetry_policy::TelemetryMode::from_str_value(telemetry_mode).ok_or_else(|| {
        PyValueError::new_err(format!(
            "telemetry_mode must be one of {}, observed '{telemetry_mode}'.",
            native_telemetry_policy::TelemetryMode::accepted_values().join(", "),
        ))
    })
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeTelemetryPaths>()?;
    module.add_function(wrap_pyfunction!(resolve_output_run_root_value, module)?)?;
    module.add_function(wrap_pyfunction!(resolve_telemetry_paths, module)?)?;
    Ok(())
}
