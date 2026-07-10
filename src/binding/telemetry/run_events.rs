//! Native run telemetry and diagnostic event adapters.

use pyo3::prelude::*;
use pyo3::types::PyAny;

use g_runtime as native_run_events;

pub(crate) fn run_failed_event_payload_from_error(
    error: &Bound<'_, PyAny>,
) -> PyResult<native_run_events::RunFailedEventPayload> {
    let error_type = error.get_type().name()?.to_string_lossy().into_owned();
    let error_message = error.str()?.to_string_lossy().into_owned();
    Ok(native_run_events::build_run_failed_event_payload(&error_type, &error_message))
}

pub(crate) fn emit_run_diagnostic_event_payload(event: &native_run_events::RunDiagnosticEventPayload) -> PyResult<()> {
    native_run_events::emit_run_diagnostic_event(event).map_err(|error| {
        pyo3::exceptions::PyValueError::new_err(format!("Failed to serialize diagnostic event fields: {error}"))
    })
}
