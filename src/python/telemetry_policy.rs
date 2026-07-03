//! PyO3 adapters for telemetry path and counter policy helpers.

use std::path::Path;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

use g_runtime::telemetry_policy as native_telemetry_policy;

#[pyclass]
pub(crate) struct NativeTelemetrySessionPolicy {
    telemetry_mode: String,
    policy: native_telemetry_policy::TelemetrySessionPolicyPayload,
}

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct NativeTelemetryPaths {
    data: native_telemetry_policy::TelemetryPathsPayload,
}

#[pymethods]
impl NativeTelemetrySessionPolicy {
    #[new]
    #[allow(clippy::needless_pass_by_value)]
    fn new(telemetry_mode: String, trace_event_cap: i64) -> Self {
        Self {
            policy: native_telemetry_policy::resolve_telemetry_session_policy(&telemetry_mode, trace_event_cap),
            telemetry_mode,
        }
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

    #[allow(clippy::unused_self)]
    #[allow(clippy::needless_pass_by_value)]
    fn resolve_output_run_root_value(&self, output_path: String, output_run_directory: Option<String>) -> String {
        native_telemetry_policy::resolve_output_run_root(
            Path::new(&output_path),
            output_run_directory.as_deref().map(Path::new),
        )
        .display()
        .to_string()
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_value)]
    fn resolve_paths(
        &self,
        output_path: String,
        output_run_directory: Option<String>,
        log_dir: Option<String>,
        log_file: Option<String>,
        trace_file: Option<String>,
        profile_summary_json: Option<String>,
        stage_timings_json: Option<String>,
    ) -> PyResult<NativeTelemetryPaths> {
        resolve_telemetry_paths(
            &self.telemetry_mode,
            &output_path,
            output_run_directory.as_deref(),
            log_dir.as_deref(),
            log_file.as_deref(),
            trace_file.as_deref(),
            profile_summary_json.as_deref(),
            stage_timings_json.as_deref(),
        )
        .map(|data| NativeTelemetryPaths { data })
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_pass_by_value)]
    fn resolve_paths_payload<'py>(
        &self,
        py: Python<'py>,
        output_path: String,
        output_run_directory: Option<String>,
        log_dir: Option<String>,
        log_file: Option<String>,
        trace_file: Option<String>,
        profile_summary_json: Option<String>,
        stage_timings_json: Option<String>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let payload = resolve_telemetry_paths(
            &self.telemetry_mode,
            &output_path,
            output_run_directory.as_deref(),
            log_dir.as_deref(),
            log_file.as_deref(),
            trace_file.as_deref(),
            profile_summary_json.as_deref(),
            stage_timings_json.as_deref(),
        )?;
        let python_payload = PyDict::new(py);
        python_payload.set_item("log_dir", payload.log_dir)?;
        python_payload.set_item("stream_file", payload.stream_file)?;
        python_payload.set_item("profile_summary_json", payload.profile_summary_json)?;
        python_payload.set_item("stage_timings_json", payload.stage_timings_json)?;
        Ok(python_payload)
    }
}

#[pymethods]
impl NativeTelemetryPaths {
    #[getter]
    fn log_dir(&self) -> Option<String> {
        self.data.log_dir.clone()
    }

    #[getter]
    fn stream_file(&self) -> Option<String> {
        self.data.stream_file.clone()
    }

    #[getter]
    fn profile_summary_json(&self) -> Option<String> {
        self.data.profile_summary_json.clone()
    }

    #[getter]
    fn stage_timings_json(&self) -> Option<String> {
        self.data.stage_timings_json.clone()
    }
}

#[allow(clippy::too_many_arguments)]
fn resolve_telemetry_paths(
    telemetry_mode: &str,
    output_path: &str,
    output_run_directory: Option<&str>,
    log_dir: Option<&str>,
    log_file: Option<&str>,
    trace_file: Option<&str>,
    profile_summary_json: Option<&str>,
    stage_timings_json: Option<&str>,
) -> PyResult<native_telemetry_policy::TelemetryPathsPayload> {
    native_telemetry_policy::resolve_telemetry_paths(
        Path::new(output_path),
        output_run_directory.map(Path::new),
        telemetry_mode,
        log_dir.map(Path::new),
        log_file.map(Path::new),
        trace_file.map(Path::new),
        profile_summary_json.map(Path::new),
        stage_timings_json.map(Path::new),
    )
    .map_err(PyValueError::new_err)
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeTelemetrySessionPolicy>()?;
    module.add_class::<NativeTelemetryPaths>()?;
    Ok(())
}
