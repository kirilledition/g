use pyo3::exceptions::{PyAttributeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use g_runtime as native_runtime;
use g_runtime::{
    CliRunLifecycleState, plan_cli_run_failed_telemetry_emission as native_plan_cli_run_failed_telemetry_emission,
};

use super::run_events;

pub(super) fn record_completed_terminal_lines(lines: &[String]) -> PyResult<()> {
    for line in lines {
        let payload = native_runtime::build_native_cli_completed_line_diagnostic_payload(line);
        run_events::emit_run_diagnostic_event_payload(&payload)?;
    }
    Ok(())
}

pub(super) fn record_interrupted_terminal_lines(lines: &[String]) -> PyResult<()> {
    for line in lines {
        let payload = native_runtime::build_native_cli_interrupted_line_diagnostic_payload(line);
        run_events::emit_run_diagnostic_event_payload(&payload)?;
    }
    Ok(())
}

pub(super) fn render_and_record_failed_terminal_lines(event: &native_runtime::RunFailedEventPayload) -> Vec<String> {
    let lines = native_runtime::render_run_failed_lines(event);
    for line in &lines {
        let payload = native_runtime::build_native_cli_failed_line_diagnostic_payload(line);
        let _ = run_events::emit_run_diagnostic_event_payload(&payload);
    }
    lines
}

pub(super) fn emit_run_failed_telemetry_event_payload<'py>(
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
