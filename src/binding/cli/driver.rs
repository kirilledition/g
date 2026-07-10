//! Rust-owned CLI lifecycle driver.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use g_interface as interface;
use g_runtime as native_runtime;
use pyo3::exceptions::{PyKeyboardInterrupt, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyTuple};

use crate::binding::engine::backend::{JaxBackendConfig, PyJaxBackend};
use crate::binding::engine::run_engine::NativeRunEngineSession;
use crate::binding::runtime::timing::NativeStageTimingRecorder;
use crate::binding::telemetry::session as telemetry_session;
use crate::binding::{logging, run_events, runtime, runtime_state, telemetry_policy};

#[pyclass]
pub(crate) struct NativeCliRunResult {
    exit_code: i32,
    stdout_chunks: Vec<String>,
    stderr_chunks: Vec<String>,
}

impl NativeCliRunResult {
    fn new(exit_code: i32, output_chunks: native_runtime::CliOutputChunks) -> Self {
        Self { exit_code, stdout_chunks: output_chunks.stdout_chunks, stderr_chunks: output_chunks.stderr_chunks }
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

/// Execute the native CLI, constructing the Python numerical backend only after
/// CLI validation and process runtime setup have completed.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn run(py: Python<'_>, arguments: Vec<String>) -> PyResult<NativeCliRunResult> {
    let interface::CliOutcomeData { exit_code, stdout, stderr, config } = interface::dispatch_cli(&arguments);
    let Some(config) = config else {
        let output_chunks = native_runtime::CliOutputBuffer::from_frontend_output(&stdout, &stderr).into_chunks();
        return Ok(NativeCliRunResult::new(exit_code, output_chunks));
    };

    let mut output_buffer = native_runtime::CliOutputBuffer::default();
    let _sigterm_scope = match native_runtime::begin_sigterm_shutdown_scope() {
        Ok(scope) => scope,
        Err(error) => {
            let error = PyValueError::new_err(error.to_string());
            let terminal_result = failed_terminal_result(py, None, &error)?;
            let exit_code = output_buffer.append_terminal_result(terminal_result);
            return Ok(NativeCliRunResult::new(exit_code, output_buffer.into_chunks()));
        }
    };
    let _logging_shutdown_guard = logging::LoggingShutdownGuard;
    let telemetry_paths = match resolve_cli_telemetry_paths(&config) {
        Ok(paths) => paths,
        Err(error) => {
            let terminal_result = failed_terminal_result(py, None, &error)?;
            let exit_code = output_buffer.append_terminal_result(terminal_result);
            return Ok(NativeCliRunResult::new(exit_code, output_buffer.into_chunks()));
        }
    };
    let run_id = native_runtime::generate_run_id();
    let timing_context = native_runtime::resolve_final_timing_output_context(
        config.g_diagnostics.stage_timings_json.as_deref(),
        telemetry_paths.stage_timings_json.as_deref(),
        telemetry_paths.profile_summary_json.as_deref(),
        Some(&run_id),
        telemetry_paths.profile_summary_json.is_some(),
        true,
    );
    let timing_recorder = NativeStageTimingRecorder::from_config(
        timing_context.stage_timing_path.is_some(),
        timing_context.force_stage_timing_recorder,
    );
    let telemetry_session = match create_telemetry_session(&config, &telemetry_paths, run_id) {
        Ok(session) => session,
        Err(error) => {
            let terminal_result = failed_terminal_result(py, None, &error)?;
            let exit_code = output_buffer.append_terminal_result(terminal_result);
            return Ok(NativeCliRunResult::new(exit_code, output_buffer.into_chunks()));
        }
    };

    let run_start_time = Instant::now();
    let mut execution_result = run_native_lifecycle(
        py,
        &config,
        &stdout,
        &stderr,
        &telemetry_paths,
        &telemetry_session,
        &mut output_buffer,
        timing_recorder.as_ref(),
    );
    if execution_result.is_ok()
        && let Err(error) = runtime::check_process_signals(py)
    {
        execution_result = Err(error);
    }
    let timing_result = finalize_timing_outputs(timing_recorder.as_ref(), &timing_context, run_start_time);
    if execution_result.is_ok()
        && let Err(error) = timing_result
    {
        execution_result = Err(error);
    }
    if execution_result.is_ok()
        && let Err(error) = runtime::check_process_signals(py)
    {
        execution_result = Err(error);
    }
    let mut terminal_result = match execution_result {
        Ok(artifacts) => completed_terminal_result(&artifacts)?,
        Err(error) => terminal_result_from_error(py, Some(&telemetry_session), &error)?,
    };
    let mut close_result = finish_telemetry_result(py, &telemetry_session, terminal_result.exit_code)?;
    if let Err(error) = runtime::check_process_signals(py) {
        terminal_result = terminal_result_from_error(py, None, &error)?;
        close_result = native_runtime::CliTerminalResult::empty(terminal_result.exit_code);
    }
    let _ = output_buffer.append_terminal_result(terminal_result);
    let exit_code = output_buffer.append_terminal_result(close_result);
    Ok(NativeCliRunResult::new(exit_code, output_buffer.into_chunks()))
}

#[allow(clippy::too_many_arguments)]
fn run_native_lifecycle(
    py: Python<'_>,
    config: &interface::RegenieConfigData,
    stdout: &str,
    stderr: &str,
    telemetry_paths: &native_runtime::TelemetryPathsPayload,
    telemetry_session: &telemetry_session::NativeTelemetryRunSession,
    output_buffer: &mut native_runtime::CliOutputBuffer,
    stage_timing_recorder: Option<&NativeStageTimingRecorder>,
) -> PyResult<run_events::NativeRunArtifacts> {
    let logging_policy = build_cli_logging_policy(config, telemetry_paths)?;
    runtime_state::initialize_process_logging_runtime_policy(py, logging_policy.clone())?;
    output_buffer.push_stdout_text(stdout);
    output_buffer.push_stderr_text(stderr);
    record_frontend_output(stdout, stderr)?;

    let runtime_start_time = Instant::now();
    runtime::configure_cli_process_runtime(py, config, &logging_policy, Some(telemetry_session))?;
    record_stage_duration(stage_timing_recorder, "jax_runtime_configuration", runtime_start_time)?;
    let backend_start_time = Instant::now();
    let backend_config =
        JaxBackendConfig::new(config.clone()).map_err(|error| PyValueError::new_err(error.to_string()))?;
    let backend_config = Py::new(py, backend_config)?;
    let backend =
        PyModule::import(py, "g.jax_backend")?.getattr("JaxAssociationBackend")?.call1((backend_config,))?.unbind();
    let backend = Arc::new(PyJaxBackend::new(backend));
    record_stage_duration(stage_timing_recorder, "jax_backend_initialization", backend_start_time)?;
    let engine_session = NativeRunEngineSession::from_config_internal(py, config)?;

    engine_session.run_with_backend_internal(py, backend, Some(telemetry_session), stage_timing_recorder)
}

fn create_telemetry_session(
    config: &interface::RegenieConfigData,
    paths: &native_runtime::TelemetryPathsPayload,
    run_id: String,
) -> PyResult<telemetry_session::NativeTelemetryRunSession> {
    let diagnostics_config = &config.g_diagnostics;
    let queue_size = usize::try_from(diagnostics_config.log_queue_size.get())
        .map_err(|_| PyValueError::new_err("log_queue_size does not fit into usize."))?;
    telemetry_session::NativeTelemetryRunSession::new(
        diagnostics_config.telemetry.as_str(),
        paths.stream_file.clone(),
        f64::from(diagnostics_config.progress_interval_seconds),
        i64::from(diagnostics_config.progress_interval_chunks.get()),
        queue_size,
        diagnostics_config.log_lossy,
        i64::from(diagnostics_config.trace_event_cap),
        Some(run_id),
    )
}

fn record_stage_duration(
    recorder: Option<&NativeStageTimingRecorder>,
    stage_name: &str,
    start_time: Instant,
) -> PyResult<()> {
    if let Some(recorder) = recorder {
        recorder.record_stage_duration(stage_name, start_time.elapsed().as_secs_f64())?;
    }
    Ok(())
}

fn finalize_timing_outputs(
    recorder: Option<&NativeStageTimingRecorder>,
    context: &native_runtime::FinalTimingOutputContext,
    run_start_time: Instant,
) -> PyResult<()> {
    let Some(recorder) = recorder else {
        return Ok(());
    };
    recorder.record_stage_duration("runner_total", run_start_time.elapsed().as_secs_f64())?;
    runtime::timing::record_final_timing_outputs_write_started(
        context.stage_timing_path.as_deref(),
        context.profile_summary_path.as_deref(),
        context.run_id.as_deref(),
    )?;
    recorder.write_final_timing_outputs(
        context.stage_timing_path.as_deref(),
        context.profile_summary_path.as_deref(),
        context.run_id.clone(),
    )
}

fn build_cli_logging_policy(
    config: &interface::RegenieConfigData,
    paths: &native_runtime::TelemetryPathsPayload,
) -> PyResult<native_runtime::LoggingRuntimePolicyPayload> {
    let diagnostics_config = &config.g_diagnostics;
    let telemetry_mode = telemetry_policy::parse_telemetry_mode(diagnostics_config.telemetry.as_str())?;
    Ok(native_runtime::build_logging_runtime_policy(
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
    ))
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

fn completed_terminal_result(
    artifacts: &run_events::NativeRunArtifacts,
) -> PyResult<native_runtime::CliTerminalResult> {
    let terminal_result = native_runtime::build_completed_cli_terminal_result(artifacts);
    runtime::record_completed_terminal_lines(&terminal_result.stdout_lines)?;
    Ok(terminal_result)
}

fn terminal_result_from_error(
    py: Python<'_>,
    telemetry_session: Option<&telemetry_session::NativeTelemetryRunSession>,
    error: &PyErr,
) -> PyResult<native_runtime::CliTerminalResult> {
    if let Some(interrupted_event) = maybe_interrupted_event_from_pyerr(py, error)? {
        return interrupted_terminal_result(&interrupted_event);
    }
    failed_terminal_result(py, telemetry_session, error)
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
    telemetry_session: Option<&telemetry_session::NativeTelemetryRunSession>,
    error: &PyErr,
) -> PyResult<native_runtime::CliTerminalResult> {
    let failed_event = run_events::run_failed_event_payload_from_error(error.value(py))?;
    if let Some(telemetry_session) = telemetry_session {
        let _ = telemetry_session.emit_run_failed_event(&failed_event);
    }
    let terminal_result = native_runtime::build_failed_cli_terminal_result(&failed_event);
    runtime::record_failed_terminal_lines(&terminal_result.stderr_lines);
    Ok(terminal_result)
}

fn finish_telemetry_result(
    py: Python<'_>,
    telemetry_session: &telemetry_session::NativeTelemetryRunSession,
    current_exit_code: i32,
) -> PyResult<native_runtime::CliTerminalResult> {
    match telemetry_session.finish_with_current_close_event() {
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
    if runtime::is_flushed_interrupt(error, py) {
        let shutdown_signal = native_runtime::build_shutdown_signal(2)
            .map_err(|shutdown_error| PyValueError::new_err(shutdown_error.to_string()))?;
        return Ok(Some(native_runtime::build_run_interrupted_event_payload(
            i64::from(shutdown_signal.number),
            &shutdown_signal.name,
            i64::from(shutdown_signal.exit_code),
            true,
        )));
    }
    if runtime::is_sigterm_request(error, py) {
        let shutdown_signal = native_runtime::build_shutdown_signal(15)
            .map_err(|shutdown_error| PyValueError::new_err(shutdown_error.to_string()))?;
        return Ok(Some(native_runtime::build_run_interrupted_event_payload(
            i64::from(shutdown_signal.number),
            &shutdown_signal.name,
            i64::from(shutdown_signal.exit_code),
            true,
        )));
    }
    if error.is_instance_of::<PyKeyboardInterrupt>(py) {
        let shutdown_signal = native_runtime::build_shutdown_signal(2)
            .map_err(|shutdown_error| PyValueError::new_err(shutdown_error.to_string()))?;
        return Ok(Some(native_runtime::build_run_interrupted_event_payload(
            i64::from(shutdown_signal.number),
            &shutdown_signal.name,
            i64::from(shutdown_signal.exit_code),
            false,
        )));
    }
    Ok(None)
}
