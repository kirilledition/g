//! Rust-owned CLI lifecycle driver for the temporary Python execution backend.

use std::path::Path;

use g_interface as interface;
use g_runtime as native_runtime;
use pyo3::exceptions::{PyAttributeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule, PyTuple};

use super::{config::RegenieConfig, logging, run_events, runtime, runtime_state, telemetry_policy};

#[pyclass]
pub(crate) struct NativeCliRunResult {
    exit_code: i32,
    stdout_chunks: Vec<String>,
    stderr_chunks: Vec<String>,
}

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct NativeCliTelemetryPaths {
    log_dir: Option<String>,
    stream_file: Option<String>,
    profile_summary_json: Option<String>,
    stage_timings_json: Option<String>,
}

#[pyclass]
pub(crate) struct NativeCliTelemetrySessionView {
    mode: String,
    paths: NativeCliTelemetryPaths,
    native_session_handle: Py<logging::NativeTelemetryRunSession>,
}

#[pyclass]
pub(crate) struct NativeCliRunContext {
    mode: String,
    paths: NativeCliTelemetryPaths,
    native_session_handle: Py<logging::NativeTelemetryRunSession>,
}

impl NativeCliRunResult {
    fn new(exit_code: i32, output_chunks: native_runtime::CliOutputChunks) -> Self {
        Self { exit_code, stdout_chunks: output_chunks.stdout_chunks, stderr_chunks: output_chunks.stderr_chunks }
    }
}

impl NativeCliTelemetryPaths {
    fn new(payload: native_runtime::TelemetryPathsPayload) -> Self {
        Self {
            log_dir: payload.log_dir,
            stream_file: payload.stream_file,
            profile_summary_json: payload.profile_summary_json,
            stage_timings_json: payload.stage_timings_json,
        }
    }
}

impl NativeCliRunContext {
    fn new(py: Python<'_>, config: &interface::RegenieConfigData) -> PyResult<Self> {
        let diagnostics_config = &config.g_diagnostics;
        let paths = NativeCliTelemetryPaths::new(resolve_cli_telemetry_paths(config)?);
        let native_session_handle = Py::new(
            py,
            logging::NativeTelemetryRunSession::new(
                diagnostics_config.telemetry.as_str(),
                paths.stream_file.clone(),
                f64::from(diagnostics_config.progress_interval_seconds),
                i64::from(diagnostics_config.progress_interval_chunks.get()),
                usize::try_from(diagnostics_config.log_queue_size.get())
                    .map_err(|_| PyValueError::new_err("log_queue_size does not fit into usize."))?,
                diagnostics_config.log_lossy,
                i64::from(diagnostics_config.trace_event_cap),
                None,
            )?,
        )?;
        Ok(Self { mode: diagnostics_config.telemetry.as_str().to_string(), paths, native_session_handle })
    }

    fn paths(&self) -> &NativeCliTelemetryPaths {
        &self.paths
    }

    fn telemetry_session_view_object(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let view = NativeCliTelemetrySessionView {
            mode: self.mode.clone(),
            paths: self.paths.clone(),
            native_session_handle: self.native_session_handle.clone_ref(py),
        };
        Ok(Py::new(py, view)?.into_any())
    }

    fn finish_telemetry(&self, py: Python<'_>) -> PyResult<()> {
        let native_session_handle = self.native_session_handle.bind(py);
        let has_native_telemetry_session =
            native_session_handle.getattr("has_native_telemetry_session")?.extract::<bool>()?;
        if has_native_telemetry_session {
            native_session_handle.call_method0("finish_with_current_close_event_metadata")?;
        }
        Ok(())
    }
}

#[pymethods]
impl NativeCliRunResult {
    #[getter]
    fn exit_code(&self) -> i32 {
        self.exit_code
    }

    #[getter]
    fn stdout_chunks<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.stdout_chunks)
    }

    #[getter]
    fn stderr_chunks<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.stderr_chunks)
    }
}

#[pymethods]
impl NativeCliTelemetryPaths {
    #[getter]
    fn log_dir(&self) -> Option<&str> {
        self.log_dir.as_deref()
    }

    #[getter]
    fn stream_file(&self) -> Option<&str> {
        self.stream_file.as_deref()
    }

    #[getter]
    fn profile_summary_json(&self) -> Option<&str> {
        self.profile_summary_json.as_deref()
    }

    #[getter]
    fn stage_timings_json(&self) -> Option<&str> {
        self.stage_timings_json.as_deref()
    }
}

#[pymethods]
impl NativeCliTelemetrySessionView {
    #[getter]
    fn mode(&self) -> &str {
        &self.mode
    }

    #[getter]
    fn paths(&self) -> NativeCliTelemetryPaths {
        self.paths.clone()
    }

    #[getter]
    fn enabled(&self, py: Python<'_>) -> PyResult<bool> {
        self.native_session_handle.bind(py).getattr("enabled")?.extract::<bool>()
    }

    #[getter]
    fn profile_enabled(&self, py: Python<'_>) -> PyResult<bool> {
        self.native_session_handle.bind(py).getattr("profile_enabled")?.extract::<bool>()
    }

    #[getter]
    fn run_id(&self, py: Python<'_>) -> PyResult<String> {
        self.native_session_handle.bind(py).getattr("run_id")?.extract::<String>()
    }

    #[getter]
    fn native_session_handle(&self, py: Python<'_>) -> Py<logging::NativeTelemetryRunSession> {
        self.native_session_handle.clone_ref(py)
    }

    #[getter]
    fn native_telemetry_session(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let native_session_handle = self.native_session_handle.bind(py);
        let has_native_telemetry_session =
            native_session_handle.getattr("has_native_telemetry_session")?.extract::<bool>()?;
        if has_native_telemetry_session {
            return Ok(self.native_session_handle.clone_ref(py).into_any());
        }
        Ok(py.None())
    }
}

#[pymethods]
impl NativeCliRunContext {
    fn telemetry_session_view(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.telemetry_session_view_object(py)
    }

    #[allow(clippy::unused_self)]
    fn native_artifacts_from_python_artifacts(
        &self,
        artifacts: &Bound<'_, PyAny>,
    ) -> PyResult<run_events::NativeRunArtifacts> {
        let artifacts_payload = run_events::run_artifacts_payload_from_py(artifacts)?;
        Ok(run_events::NativeRunArtifacts::new(artifacts_payload))
    }
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn run_with_python_backend(
    py: Python<'_>,
    arguments: Vec<String>,
    backend: Py<PyAny>,
) -> PyResult<NativeCliRunResult> {
    run_with_python_backend_impl(py, arguments, backend)
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn run_cli_with_python_backend(
    py: Python<'_>,
    arguments: Vec<String>,
    backend: Py<PyAny>,
) -> PyResult<NativeCliRunResult> {
    run_with_python_backend_impl(py, arguments, backend)
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeCliRunContext>()?;
    module.add_function(wrap_pyfunction!(run_with_python_backend, module)?)?;
    Ok(())
}

fn run_with_python_backend_impl(
    py: Python<'_>,
    arguments: Vec<String>,
    backend: Py<PyAny>,
) -> PyResult<NativeCliRunResult> {
    let interface::CliOutcomeData { exit_code, stdout, stderr, config } = interface::dispatch_cli(&arguments);
    let Some(config) = config else {
        let output_chunks = native_runtime::CliOutputBuffer::from_frontend_output(&stdout, &stderr).into_chunks();
        return Ok(NativeCliRunResult::new(exit_code, output_chunks));
    };

    let mut output_buffer = native_runtime::CliOutputBuffer::default();
    let mut lifecycle_state = native_runtime::CliRunLifecycleState::default();
    let regenie_config = Py::new(py, RegenieConfig::new(config.clone()))?;
    let context = match NativeCliRunContext::new(py, &config) {
        Ok(context) => Py::new(py, context)?,
        Err(error) => {
            let terminal_result = failed_terminal_result(py, &lifecycle_state, None, &error)?;
            let exit_code = output_buffer.append_terminal_result(terminal_result);
            return Ok(NativeCliRunResult::new(exit_code, output_buffer.into_chunks()));
        }
    };

    let backend_result = run_backend_lifecycle(
        py,
        &backend,
        regenie_config,
        context.clone_ref(py),
        &config,
        &stdout,
        &stderr,
        &mut lifecycle_state,
        &mut output_buffer,
    );
    let terminal_result = match backend_result {
        Ok(artifacts) => completed_terminal_result(artifacts.bind(py))?,
        Err(error) => terminal_result_from_error(py, &lifecycle_state, Some(&context), &error)?,
    };
    let exit_code = output_buffer.append_terminal_result(terminal_result);
    let close_result = finish_telemetry_result(py, &context, exit_code)?;
    let exit_code = output_buffer.append_terminal_result(close_result);
    Ok(NativeCliRunResult::new(exit_code, output_buffer.into_chunks()))
}

#[allow(clippy::too_many_arguments)]
fn run_backend_lifecycle(
    py: Python<'_>,
    backend: &Py<PyAny>,
    regenie_config: Py<RegenieConfig>,
    context: Py<NativeCliRunContext>,
    config: &interface::RegenieConfigData,
    stdout: &str,
    stderr: &str,
    lifecycle_state: &mut native_runtime::CliRunLifecycleState,
    output_buffer: &mut native_runtime::CliOutputBuffer,
) -> PyResult<Py<PyAny>> {
    {
        let context_reference = context.bind(py).borrow();
        initialize_cli_logging(py, config, context_reference.paths())?;
    }
    output_buffer.push_stdout_text(stdout);
    output_buffer.push_stderr_text(stderr);
    record_frontend_output(stdout, stderr)?;
    call_backend_with_shutdown_controller(py, backend, regenie_config, context, lifecycle_state)
}

fn call_backend_with_shutdown_controller(
    py: Python<'_>,
    backend: &Py<PyAny>,
    regenie_config: Py<RegenieConfig>,
    context: Py<NativeCliRunContext>,
    lifecycle_state: &mut native_runtime::CliRunLifecycleState,
) -> PyResult<Py<PyAny>> {
    let lifecycle_module = PyModule::import(py, "g.runner.lifecycle")?;
    let controller = lifecycle_module.getattr("GracefulShutdownController")?.call1((py.None(),))?;
    controller.call_method0("__enter__")?;
    lifecycle_state.mark_runner_started();
    let backend_result = backend.bind(py).call1((regenie_config, context)).map(Bound::unbind);
    let restore_result = controller.call_method1("__exit__", (py.None(), py.None(), py.None()));
    match (backend_result, restore_result) {
        (Ok(artifacts), Ok(_)) => Ok(artifacts),
        (_, Err(error)) | (Err(error), Ok(_)) => Err(error),
    }
}

fn initialize_cli_logging(
    py: Python<'_>,
    config: &interface::RegenieConfigData,
    paths: &NativeCliTelemetryPaths,
) -> PyResult<bool> {
    let diagnostics_config = &config.g_diagnostics;
    let telemetry_mode = telemetry_policy::parse_telemetry_mode(diagnostics_config.telemetry.as_str())?;
    let logging_policy = native_runtime::build_logging_runtime_policy(
        diagnostics_config.log_filter.clone(),
        diagnostics_config.log_file.clone(),
        diagnostics_config.log_stderr,
        i64::from(diagnostics_config.log_queue_size.get()),
        diagnostics_config.log_lossy,
        diagnostics_config.include_source_location,
        diagnostics_config.include_span_events,
        diagnostics_config.trace_file.clone(),
        diagnostics_config.trace_filter.clone(),
        Some(i64::from(diagnostics_config.trace_event_cap)),
        telemetry_mode,
        paths.stream_file.clone(),
    );
    runtime_state::initialize_process_logging_runtime_policy(py, logging_policy)
}

fn resolve_cli_telemetry_paths(
    config: &interface::RegenieConfigData,
) -> PyResult<native_runtime::TelemetryPathsPayload> {
    let output_path = config
        .g_output
        .out
        .as_deref()
        .ok_or_else(|| PyValueError::new_err("CLI run config must include an output path."))?;
    let diagnostics_config = &config.g_diagnostics;
    let telemetry_mode = telemetry_policy::parse_telemetry_mode(diagnostics_config.telemetry.as_str())?;
    native_runtime::resolve_telemetry_paths(
        Path::new(output_path),
        config.g_output.output_run_directory.as_deref().map(Path::new),
        telemetry_mode,
        diagnostics_config.log_dir.as_deref().map(Path::new),
        diagnostics_config.log_file.as_deref().map(Path::new),
        diagnostics_config.trace_file.as_deref().map(Path::new),
        diagnostics_config.profile_summary_json.as_deref().map(Path::new),
        diagnostics_config.stage_timings_json.as_deref().map(Path::new),
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

fn record_frontend_output(stdout_text: &str, stderr_text: &str) -> PyResult<()> {
    if !stdout_text.is_empty() {
        let payload = native_runtime::build_native_cli_stdout_diagnostic_payload(
            stdout_text,
            native_runtime::NATIVE_CLI_OUTPUT_LOG_LIMIT,
        );
        run_events::emit_run_diagnostic_event_payload(&payload)?;
    }
    if !stderr_text.is_empty() {
        let payload = native_runtime::build_native_cli_stderr_diagnostic_payload(
            stderr_text,
            native_runtime::NATIVE_CLI_OUTPUT_LOG_LIMIT,
        );
        run_events::emit_run_diagnostic_event_payload(&payload)?;
    }
    Ok(())
}

fn completed_terminal_result(artifacts: &Bound<'_, PyAny>) -> PyResult<native_runtime::CliTerminalResult> {
    let artifacts_payload = run_events::run_artifacts_payload_from_py(artifacts)?;
    let terminal_result = native_runtime::build_completed_cli_terminal_result(&artifacts_payload);
    runtime::record_completed_terminal_lines(&terminal_result.stdout_lines)?;
    Ok(terminal_result)
}

fn terminal_result_from_error(
    py: Python<'_>,
    lifecycle_state: &native_runtime::CliRunLifecycleState,
    context: Option<&Py<NativeCliRunContext>>,
    error: &PyErr,
) -> PyResult<native_runtime::CliTerminalResult> {
    if let Some(interrupted_event) = maybe_interrupted_event_from_pyerr(py, error)? {
        return interrupted_terminal_result(&interrupted_event);
    }
    let telemetry_session =
        context.map(|context| context.bind(py).borrow().telemetry_session_view_object(py)).transpose()?;
    let telemetry_session_bound = telemetry_session.as_ref().map(|session| session.bind(py));
    failed_terminal_result(py, lifecycle_state, telemetry_session_bound, error)
}

fn interrupted_terminal_result(
    interrupted_event: &native_runtime::RunInterruptedEventPayload,
) -> PyResult<native_runtime::CliTerminalResult> {
    let terminal_result = native_runtime::build_interrupted_cli_terminal_result(interrupted_event)
        .map_err(|_| PyValueError::new_err("shutdown exit code is outside the i32 range."))?;
    runtime::record_interrupted_terminal_lines(&terminal_result.stderr_lines)?;
    Ok(terminal_result)
}

fn failed_terminal_result(
    py: Python<'_>,
    lifecycle_state: &native_runtime::CliRunLifecycleState,
    telemetry_session: Option<&Bound<'_, PyAny>>,
    error: &PyErr,
) -> PyResult<native_runtime::CliTerminalResult> {
    let failed_event = run_events::run_failed_event_payload_from_error(error.value(py))?;
    runtime::emit_run_failed_telemetry_event_payload(lifecycle_state, py, telemetry_session, &failed_event)?;
    let terminal_result = native_runtime::build_failed_cli_terminal_result(&failed_event);
    runtime::record_failed_terminal_lines(&terminal_result.stderr_lines);
    Ok(terminal_result)
}

fn finish_telemetry_result(
    py: Python<'_>,
    context: &Py<NativeCliRunContext>,
    current_exit_code: i32,
) -> PyResult<native_runtime::CliTerminalResult> {
    let close_result = context.bind(py).borrow().finish_telemetry(py);
    match close_result {
        Ok(()) => Ok(native_runtime::CliTerminalResult::empty(current_exit_code)),
        Err(error) => telemetry_close_failure_result(py, current_exit_code, &error),
    }
}

fn telemetry_close_failure_result(
    py: Python<'_>,
    current_exit_code: i32,
    error: &PyErr,
) -> PyResult<native_runtime::CliTerminalResult> {
    let failed_event = run_events::run_failed_event_payload_from_error(error.value(py))?;
    let terminal_result =
        native_runtime::build_telemetry_close_failure_cli_terminal_result(current_exit_code, &failed_event);
    runtime::record_failed_terminal_lines(&terminal_result.stderr_lines);
    Ok(terminal_result)
}

fn maybe_interrupted_event_from_pyerr(
    py: Python<'_>,
    error: &PyErr,
) -> PyResult<Option<native_runtime::RunInterruptedEventPayload>> {
    match run_events::run_interrupted_event_payload_from_shutdown_request(error.value(py)) {
        Ok(interrupted_event) => Ok(Some(interrupted_event)),
        Err(interrupted_error) if interrupted_error.is_instance_of::<PyAttributeError>(py) => Ok(None),
        Err(interrupted_error) => Err(interrupted_error),
    }
}
