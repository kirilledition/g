//! Rust-owned CLI lifecycle driver.

use std::sync::Arc;
use std::time::Instant;

use g_interface as interface;
use g_runtime as native_runtime;
use pyo3::exceptions::{PyKeyboardInterrupt, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyTuple};

use crate::binding::engine::{JaxBackendConfig, PyJaxBackend};
use crate::binding::{errors, run_events, runtime, runtime_state};

#[pyclass]
pub(crate) struct NativeCliRunResult {
    exit_code: i32,
    stdout_chunks: Vec<String>,
    stderr_chunks: Vec<String>,
}

struct PythonRunHooks;

impl g_engine::RunHooks for PythonRunHooks {
    type Error = PyErr;

    fn check_interruption(&mut self) -> Result<(), Self::Error> {
        Python::attach(runtime::check_process_signals)
    }

    fn interruption_signal_name(error: &Self::Error) -> Option<&str> {
        Python::attach(|py| {
            if runtime::is_sigterm_request(error, py) {
                Some("SIGTERM")
            } else if error.is_instance_of::<PyKeyboardInterrupt>(py) {
                Some("SIGINT")
            } else {
                None
            }
        })
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
    match interface::dispatch_cli(&arguments) {
        interface::CliDispatch::Exit { exit_code, stdout, stderr } => {
            let output = native_runtime::CliOutputBuffer::from_frontend_output(&stdout, &stderr);
            Ok(NativeCliRunResult {
                exit_code,
                stdout_chunks: output.stdout_chunks,
                stderr_chunks: output.stderr_chunks,
            })
        }
        interface::CliDispatch::Run(compiled_run) => {
            execute_compiled_run(py, compiled_run.run_plan, compiled_run.effective_config_toml)
        }
    }
}

fn execute_compiled_run(
    py: Python<'_>,
    run_plan: g_plan::RunPlan,
    effective_config_toml: String,
) -> PyResult<NativeCliRunResult> {
    let mut output = native_runtime::CliOutputBuffer::default();
    let mut native_session = match native_runtime::NativeRunSession::new(&run_plan) {
        Ok(session) => session,
        Err(error) => {
            let error = PyRuntimeError::new_err(error.to_string());
            let terminal_result = failed_terminal_result(py, None, &error)?;
            let exit_code = output.append_terminal_result(terminal_result);
            return Ok(NativeCliRunResult {
                exit_code,
                stdout_chunks: output.stdout_chunks,
                stderr_chunks: output.stderr_chunks,
            });
        }
    };
    let thread_name = crate::binding::telemetry::current_python_thread_name()?;
    let mut execution_result = (|| {
        runtime_state::initialize_process_logging_runtime_policy(py, native_session.logging_policy.clone())?;
        let runtime_start_time = Instant::now();
        runtime::configure_cli_process_runtime(
            py,
            &run_plan,
            &native_session.logging_policy,
            Some(&native_session.telemetry_session),
        )?;
        native_session.record_stage_duration("jax_runtime_configuration", runtime_start_time);

        let backend_start_time = Instant::now();
        let backend_config = Py::new(py, JaxBackendConfig::new(&run_plan))?;
        let backend =
            PyModule::import(py, "g.jax_backend")?.getattr("JaxAssociationBackend")?.call1((backend_config,))?.unbind();
        let backend = Arc::new(PyJaxBackend::new(backend));
        native_session.record_stage_duration("jax_backend_initialization", backend_start_time);

        let telemetry_session = &native_session.telemetry_session;
        let stage_timing_recorder = native_session.stage_timing_recorder.as_mut();
        let thread_name_for_run = thread_name.as_str();
        let mut hooks = PythonRunHooks;
        py.detach(move || {
            g_engine::execute_coordinated_run(
                run_plan,
                effective_config_toml,
                backend,
                &mut hooks,
                telemetry_session,
                thread_name_for_run,
                stage_timing_recorder,
            )
        })
        .map_err(errors::convert_coordinated_run_error)
    })();
    if execution_result.is_ok()
        && let Err(error) = runtime::check_process_signals(py)
    {
        execution_result = Err(error);
    }
    let timing_result = native_session.finish_timing().map_err(|error| PyRuntimeError::new_err(error.to_string()));
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
        Err(error) => terminal_result_from_error(py, Some(&native_session.telemetry_session), &error)?,
    };
    let mut close_result =
        finish_telemetry_result(&native_session.telemetry_session, &thread_name, terminal_result.exit_code);
    if let Err(error) = runtime::check_process_signals(py) {
        terminal_result = terminal_result_from_error(py, None, &error)?;
        close_result = native_runtime::CliTerminalResult {
            exit_code: terminal_result.exit_code,
            stdout_lines: Vec::new(),
            stderr_lines: Vec::new(),
        };
    }
    let _ = output.append_terminal_result(terminal_result);
    let exit_code = output.append_terminal_result(close_result);
    Ok(NativeCliRunResult { exit_code, stdout_chunks: output.stdout_chunks, stderr_chunks: output.stderr_chunks })
}

fn completed_terminal_result(
    artifacts: &[native_runtime::PhenotypeRunArtifacts],
) -> PyResult<native_runtime::CliTerminalResult> {
    let terminal_result = native_runtime::CliTerminalResult {
        exit_code: 0,
        stdout_lines: native_runtime::render_run_completed_lines(artifacts),
        stderr_lines: Vec::new(),
    };
    runtime::record_completed_terminal_lines(&terminal_result.stdout_lines)?;
    Ok(terminal_result)
}

fn terminal_result_from_error(
    py: Python<'_>,
    telemetry_session: Option<&native_runtime::TelemetryRunSession>,
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
    let terminal_result = native_runtime::CliTerminalResult {
        exit_code: interrupted_event.exit_code,
        stdout_lines: Vec::new(),
        stderr_lines: native_runtime::render_run_interrupted_lines(interrupted_event),
    };
    runtime::record_interrupted_terminal_lines(&terminal_result.stderr_lines)?;
    Ok(terminal_result)
}

fn failed_terminal_result(
    py: Python<'_>,
    telemetry_session: Option<&native_runtime::TelemetryRunSession>,
    error: &PyErr,
) -> PyResult<native_runtime::CliTerminalResult> {
    let failed_event = run_events::run_failed_event_payload_from_error(error.value(py))?;
    if let Some(telemetry_session) = telemetry_session
        && let Ok(thread_name) = crate::binding::telemetry::current_python_thread_name()
    {
        let _ = telemetry_session.emit_run_failed_event(&thread_name, &failed_event);
    }
    let terminal_result = native_runtime::CliTerminalResult {
        exit_code: native_runtime::CLI_RUNTIME_FAILURE_EXIT_CODE,
        stdout_lines: Vec::new(),
        stderr_lines: native_runtime::render_run_failed_lines(&failed_event),
    };
    runtime::record_failed_terminal_lines(&terminal_result.stderr_lines);
    Ok(terminal_result)
}

fn finish_telemetry_result(
    telemetry_session: &native_runtime::TelemetryRunSession,
    thread_name: &str,
    current_exit_code: i32,
) -> native_runtime::CliTerminalResult {
    match telemetry_session.finish(thread_name) {
        Ok(()) => native_runtime::CliTerminalResult {
            exit_code: current_exit_code,
            stdout_lines: Vec::new(),
            stderr_lines: Vec::new(),
        },
        Err(error) => telemetry_close_failure_result(current_exit_code, &error),
    }
}

fn telemetry_close_failure_result(
    current_exit_code: i32,
    error: &native_runtime::TelemetryRunError,
) -> native_runtime::CliTerminalResult {
    let failed_event = native_runtime::build_run_failed_event_payload("TelemetryRunError", &error.to_string());
    let terminal_result = if current_exit_code == 0 {
        native_runtime::CliTerminalResult {
            exit_code: native_runtime::CLI_RUNTIME_FAILURE_EXIT_CODE,
            stdout_lines: Vec::new(),
            stderr_lines: native_runtime::render_run_failed_lines(&failed_event),
        }
    } else {
        native_runtime::CliTerminalResult {
            exit_code: current_exit_code,
            stdout_lines: Vec::new(),
            stderr_lines: Vec::new(),
        }
    };
    runtime::record_failed_terminal_lines(&terminal_result.stderr_lines);
    terminal_result
}

fn maybe_interrupted_event_from_pyerr(
    py: Python<'_>,
    error: &PyErr,
) -> PyResult<Option<native_runtime::RunInterruptedEventPayload>> {
    if runtime::is_flushed_interrupt(error, py) {
        let shutdown_signal = native_runtime::build_shutdown_signal(2)
            .map_err(|shutdown_error| PyValueError::new_err(shutdown_error.to_string()))?;
        return Ok(Some(native_runtime::build_run_interrupted_event_payload(
            &shutdown_signal.name,
            shutdown_signal.exit_code,
            true,
        )));
    }
    if runtime::is_sigterm_request(error, py) {
        let shutdown_signal = native_runtime::build_shutdown_signal(15)
            .map_err(|shutdown_error| PyValueError::new_err(shutdown_error.to_string()))?;
        return Ok(Some(native_runtime::build_run_interrupted_event_payload(
            &shutdown_signal.name,
            shutdown_signal.exit_code,
            true,
        )));
    }
    if error.is_instance_of::<PyKeyboardInterrupt>(py) {
        let shutdown_signal = native_runtime::build_shutdown_signal(2)
            .map_err(|shutdown_error| PyValueError::new_err(shutdown_error.to_string()))?;
        return Ok(Some(native_runtime::build_run_interrupted_event_payload(
            &shutdown_signal.name,
            shutdown_signal.exit_code,
            false,
        )));
    }
    Ok(None)
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeCliRunResult>()?;
    module.add_function(wrap_pyfunction!(run, module)?)?;
    Ok(())
}
