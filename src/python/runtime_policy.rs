//! PyO3 adapters for deterministic process runtime policy helpers.

use pyo3::prelude::*;
use pyo3::types::PyModule;

use g_runtime::runtime_policy as native_runtime_policy;

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
    module.add_function(wrap_pyfunction!(describe_logging_runtime_policy_value, module)?)?;
    Ok(())
}
