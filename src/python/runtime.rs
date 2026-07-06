use pyo3::exceptions::{PyAttributeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule, PyTuple};

use g_runtime as native_runtime;
use g_runtime::{
    CLI_RUNTIME_FAILURE_EXIT_CODE, CliRunLifecycleState, CliTelemetryCloseFailurePlan, CliTerminalResult,
    NATIVE_CLI_OUTPUT_LOG_LIMIT,
    plan_cli_run_failed_telemetry_emission as native_plan_cli_run_failed_telemetry_emission,
    plan_cli_telemetry_close_failure as native_plan_cli_telemetry_close_failure,
};

use super::run_events;

#[pyclass]
pub(super) struct NativeCliRunLifecycleState {
    state: CliRunLifecycleState,
}

#[pyclass]
pub(super) struct NativeCliTelemetryCloseFailurePlan {
    plan: CliTelemetryCloseFailurePlan,
}

#[pyclass]
pub(super) struct NativeCliTerminalResult {
    result: CliTerminalResult,
}

impl NativeCliTerminalResult {
    fn new(result: CliTerminalResult) -> Self {
        Self { result }
    }
}

#[pymethods]
impl NativeCliTerminalResult {
    #[getter]
    fn exit_code(&self) -> i32 {
        self.result.exit_code
    }

    #[getter]
    fn stdout_lines<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.result.stdout_lines)
    }

    #[getter]
    fn stderr_lines<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.result.stderr_lines)
    }
}

#[pymethods]
impl NativeCliTelemetryCloseFailurePlan {
    #[getter]
    fn should_report_failure(&self) -> bool {
        self.plan.should_report_failure
    }

    #[getter]
    fn exit_code(&self) -> i32 {
        self.plan.exit_code
    }
}

#[pymethods]
impl NativeCliRunLifecycleState {
    #[new]
    fn new() -> Self {
        Self { state: CliRunLifecycleState::default() }
    }

    #[getter]
    fn runner_started(&self) -> bool {
        self.state.runner_started()
    }

    fn mark_runner_started(&mut self) {
        self.state.mark_runner_started();
    }

    #[allow(clippy::unused_self)]
    fn record_frontend_output(&self, stdout_text: &str, stderr_text: &str) -> PyResult<()> {
        if !stdout_text.is_empty() {
            let payload =
                native_runtime::build_native_cli_stdout_diagnostic_payload(stdout_text, NATIVE_CLI_OUTPUT_LOG_LIMIT);
            run_events::emit_run_diagnostic_event_payload(&payload)?;
        }
        if !stderr_text.is_empty() {
            let payload =
                native_runtime::build_native_cli_stderr_diagnostic_payload(stderr_text, NATIVE_CLI_OUTPUT_LOG_LIMIT);
            run_events::emit_run_diagnostic_event_payload(&payload)?;
        }
        Ok(())
    }

    #[allow(clippy::unused_self)]
    fn completed_result(&self, artifacts: &Bound<'_, PyAny>) -> PyResult<NativeCliTerminalResult> {
        let artifacts_payload = run_events::run_artifacts_payload_from_py(artifacts)?;
        let completed_event = native_runtime::build_run_completed_event_from_artifacts(&artifacts_payload);
        let stdout_lines = native_runtime::render_run_completed_lines(&completed_event);
        record_completed_terminal_lines(&stdout_lines)?;
        Ok(NativeCliTerminalResult::new(CliTerminalResult::success(stdout_lines)))
    }

    #[allow(clippy::unused_self)]
    fn interrupted_result(&self, shutdown_request: &Bound<'_, PyAny>) -> PyResult<NativeCliTerminalResult> {
        let interrupted_event = run_events::run_interrupted_event_payload_from_shutdown_request(shutdown_request)?;
        let stderr_lines = native_runtime::render_run_interrupted_lines(&interrupted_event);
        record_interrupted_terminal_lines(&stderr_lines)?;
        Ok(NativeCliTerminalResult::new(CliTerminalResult::interrupted(
            i64_to_i32(interrupted_event.exit_code, "shutdown exit code")?,
            stderr_lines,
        )))
    }

    #[allow(clippy::missing_errors_doc)]
    fn failed_result<'py>(
        &self,
        py: Python<'py>,
        error: &Bound<'py, PyAny>,
        telemetry_session: &Bound<'py, PyAny>,
    ) -> PyResult<NativeCliTerminalResult> {
        let failed_event = run_events::run_failed_event_payload_from_error(error)?;
        emit_run_failed_telemetry_event_payload(&self.state, py, Some(telemetry_session), &failed_event)?;
        Ok(NativeCliTerminalResult::new(CliTerminalResult::failed(render_and_record_failed_terminal_lines(
            &failed_event,
        ))))
    }

    #[allow(clippy::missing_errors_doc)]
    fn finish_telemetry_result<'py>(
        &self,
        py: Python<'py>,
        current_exit_code: i32,
        telemetry_session: &Bound<'py, PyAny>,
    ) -> PyResult<NativeCliTerminalResult> {
        let close_result = optional_native_telemetry_session(py, telemetry_session).and_then(|native_session| {
            if let Some(active_native_session) = native_session {
                active_native_session.call_method0("finish_with_current_close_event_metadata")?;
            }
            Ok(())
        });
        match close_result {
            Ok(()) => Ok(NativeCliTerminalResult::new(CliTerminalResult::empty(current_exit_code))),
            Err(error) => telemetry_close_failure_result(py, current_exit_code, &error),
        }
    }

    #[allow(clippy::unused_self)]
    fn plan_telemetry_close_failure(
        &self,
        current_exit_code: i32,
        runtime_failure_exit_code: i32,
    ) -> NativeCliTelemetryCloseFailurePlan {
        NativeCliTelemetryCloseFailurePlan {
            plan: native_plan_cli_telemetry_close_failure(current_exit_code, runtime_failure_exit_code),
        }
    }

    #[allow(clippy::missing_errors_doc)]
    fn emit_run_failed_telemetry_event(
        &self,
        telemetry_session: &Bound<'_, PyAny>,
        failed_event: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let failed_event_payload = run_events::run_failed_event_from_py(failed_event)?;
        emit_run_failed_telemetry_event_payload(
            &self.state,
            failed_event.py(),
            Some(telemetry_session),
            &failed_event_payload,
        )
    }
}

fn record_completed_terminal_lines(lines: &[String]) -> PyResult<()> {
    for line in lines {
        let payload = native_runtime::build_native_cli_completed_line_diagnostic_payload(line);
        run_events::emit_run_diagnostic_event_payload(&payload)?;
    }
    Ok(())
}

fn record_interrupted_terminal_lines(lines: &[String]) -> PyResult<()> {
    for line in lines {
        let payload = native_runtime::build_native_cli_interrupted_line_diagnostic_payload(line);
        run_events::emit_run_diagnostic_event_payload(&payload)?;
    }
    Ok(())
}

fn render_and_record_failed_terminal_lines(event: &native_runtime::RunFailedEventPayload) -> Vec<String> {
    let lines = native_runtime::render_run_failed_lines(event);
    for line in &lines {
        let payload = native_runtime::build_native_cli_failed_line_diagnostic_payload(line);
        let _ = run_events::emit_run_diagnostic_event_payload(&payload);
    }
    lines
}

fn telemetry_close_failure_result(
    py: Python<'_>,
    current_exit_code: i32,
    error: &PyErr,
) -> PyResult<NativeCliTerminalResult> {
    let close_failure_plan = native_plan_cli_telemetry_close_failure(current_exit_code, CLI_RUNTIME_FAILURE_EXIT_CODE);
    if !close_failure_plan.should_report_failure {
        return Ok(NativeCliTerminalResult::new(CliTerminalResult::empty(close_failure_plan.exit_code)));
    }
    let error_value = error.value(py);
    let failed_event = run_events::run_failed_event_payload_from_error(&error_value)?;
    Ok(NativeCliTerminalResult::new(CliTerminalResult::new(
        close_failure_plan.exit_code,
        Vec::new(),
        render_and_record_failed_terminal_lines(&failed_event),
    )))
}

fn emit_run_failed_telemetry_event_payload<'py>(
    state: &CliRunLifecycleState,
    py: Python<'py>,
    telemetry_session: Option<&Bound<'py, PyAny>>,
    failed_event: &native_runtime::RunFailedEventPayload,
) -> PyResult<()> {
    let telemetry_plan = state.plan_run_failed_telemetry();
    if !telemetry_plan.should_log_run_failed_to_telemetry {
        return Ok(());
    }
    let Some(telemetry_session) = telemetry_session else {
        return Ok(());
    };
    if telemetry_session.is_none() {
        return Ok(());
    }
    let native_telemetry_session = match optional_native_telemetry_session(py, telemetry_session) {
        Ok(native_telemetry_session) => native_telemetry_session,
        Err(error) => {
            let emission_plan =
                native_plan_cli_run_failed_telemetry_emission(telemetry_plan.should_log_run_failed_to_telemetry, true);
            if emission_plan.should_suppress_errors {
                return Ok(());
            }
            return Err(error);
        }
    };
    let emission_plan = native_plan_cli_run_failed_telemetry_emission(
        telemetry_plan.should_log_run_failed_to_telemetry,
        native_telemetry_session.is_some(),
    );
    if !emission_plan.should_emit {
        return Ok(());
    }
    let Some(active_native_telemetry_session) = native_telemetry_session else {
        return Ok(());
    };
    let event_object = Py::new(py, run_events::NativeRunFailedEvent::new(failed_event.clone()))?;
    let emission_result = active_native_telemetry_session.call_method1("emit_run_failed_event", (event_object,));
    if emission_plan.should_suppress_errors {
        let _ = emission_result;
        return Ok(());
    }
    emission_result.map(|_| ())
}

fn i64_to_i32(value: i64, field_name: &str) -> PyResult<i32> {
    i32::try_from(value).map_err(|_| PyValueError::new_err(format!("{field_name} is outside the i32 range.")))
}

fn optional_native_telemetry_session<'py>(
    py: Python<'py>,
    telemetry_session: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    match telemetry_session.getattr("native_telemetry_session") {
        Ok(native_telemetry_session) if native_telemetry_session.is_none() => Ok(None),
        Ok(native_telemetry_session) => Ok(Some(native_telemetry_session)),
        Err(error) if error.is_instance_of::<PyAttributeError>(py) => Err(PyTypeError::new_err(
            "CLI run-failed telemetry requires a TelemetrySession with a native telemetry session handle.",
        )),
        Err(error) => Err(error),
    }
}

pub(super) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeCliRunLifecycleState>()?;
    module.add_class::<NativeCliTelemetryCloseFailurePlan>()?;
    module.add_class::<NativeCliTerminalResult>()?;
    Ok(())
}
