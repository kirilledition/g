use pyo3::exceptions::PyException;
use pyo3::prelude::*;

use g_runtime as native_runtime;

pub(crate) mod jax_runtime;
pub(crate) mod runtime_state;
pub(crate) mod timing;

pub(crate) use crate::binding::errors;
pub(crate) use crate::binding::telemetry::{logging, run_events, session as telemetry_session};
pub(crate) use runtime_state::configure_cli_process_runtime;

pyo3::create_exception!(g, NativeSigtermRequested, PyException);
pyo3::create_exception!(g, NativeInterruptFlushed, PyException);

pub(crate) fn check_process_signals(py: Python<'_>) -> PyResult<()> {
    py.check_signals()?;
    if native_runtime::sigterm_shutdown_requested() {
        return Err(NativeSigtermRequested::new_err("SIGTERM requested graceful shutdown."));
    }
    Ok(())
}

pub(crate) fn is_sigterm_request(error: &PyErr, py: Python<'_>) -> bool {
    error.is_instance_of::<NativeSigtermRequested>(py)
}

pub(crate) fn is_flushed_interrupt(error: &PyErr, py: Python<'_>) -> bool {
    error.is_instance_of::<NativeInterruptFlushed>(py)
}

pub(crate) fn flushed_interrupt_error() -> PyErr {
    NativeInterruptFlushed::new_err("SIGINT interrupted the run after resumable output was flushed.")
}

pub(crate) fn record_completed_terminal_lines(lines: &[String]) -> PyResult<()> {
    for line in lines {
        let payload = native_runtime::build_native_cli_completed_line_diagnostic_payload(line);
        run_events::emit_run_diagnostic_event_payload(&payload)?;
    }
    Ok(())
}

pub(crate) fn record_interrupted_terminal_lines(lines: &[String]) -> PyResult<()> {
    for line in lines {
        let payload = native_runtime::build_native_cli_interrupted_line_diagnostic_payload(line);
        run_events::emit_run_diagnostic_event_payload(&payload)?;
    }
    Ok(())
}

pub(crate) fn record_failed_terminal_lines(lines: &[String]) {
    for line in lines {
        let payload = native_runtime::build_native_cli_failed_line_diagnostic_payload(line);
        let _ = run_events::emit_run_diagnostic_event_payload(&payload);
    }
}
