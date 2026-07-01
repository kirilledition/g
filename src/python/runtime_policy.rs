//! PyO3 adapters for deterministic process runtime policy helpers.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

use g_runtime::runtime_policy as native_runtime_policy;

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn build_logging_runtime_policy_payload<'py>(
    py: Python<'py>,
    log_filter: String,
    log_file: Option<String>,
    log_stderr: bool,
    log_queue_size: i64,
    log_lossy: bool,
    include_source_location: bool,
    include_span_events: bool,
    trace_file: Option<String>,
    trace_filter: String,
    trace_event_cap: Option<i64>,
    telemetry_mode: String,
    telemetry_stream_file: Option<String>,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = native_runtime_policy::build_logging_runtime_policy(
        log_filter,
        log_file,
        log_stderr,
        log_queue_size,
        log_lossy,
        include_source_location,
        include_span_events,
        trace_file,
        trace_filter,
        trace_event_cap,
        &telemetry_mode,
        telemetry_stream_file,
    );
    logging_runtime_policy_payload_to_dict(py, &payload)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn describe_logging_runtime_policy_value(
    log_filter: String,
    log_file: Option<String>,
    log_stderr: bool,
    log_queue_size: i64,
    log_lossy: bool,
    include_source_location: bool,
    include_span_events: bool,
    trace_file: Option<String>,
    trace_filter: String,
    trace_event_cap: Option<i64>,
) -> String {
    native_runtime_policy::describe_logging_runtime_policy(&native_runtime_policy::LoggingRuntimePolicyPayload {
        log_filter,
        log_file,
        log_stderr,
        log_queue_size,
        log_lossy,
        include_source_location,
        include_span_events,
        trace_file,
        trace_filter,
        trace_event_cap,
    })
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(build_logging_runtime_policy_payload, module)?)?;
    module.add_function(wrap_pyfunction!(describe_logging_runtime_policy_value, module)?)?;
    Ok(())
}

fn logging_runtime_policy_payload_to_dict<'py>(
    py: Python<'py>,
    payload: &native_runtime_policy::LoggingRuntimePolicyPayload,
) -> PyResult<Bound<'py, PyDict>> {
    let python_payload = PyDict::new(py);
    python_payload.set_item("log_filter", &payload.log_filter)?;
    python_payload.set_item("log_file", &payload.log_file)?;
    python_payload.set_item("log_stderr", payload.log_stderr)?;
    python_payload.set_item("log_queue_size", payload.log_queue_size)?;
    python_payload.set_item("log_lossy", payload.log_lossy)?;
    python_payload.set_item("include_source_location", payload.include_source_location)?;
    python_payload.set_item("include_span_events", payload.include_span_events)?;
    python_payload.set_item("trace_file", &payload.trace_file)?;
    python_payload.set_item("trace_filter", &payload.trace_filter)?;
    python_payload.set_item("trace_event_cap", payload.trace_event_cap)?;
    Ok(python_payload)
}
