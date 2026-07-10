//! Unified tracing setup for Rust and Python diagnostics.

#![allow(clippy::missing_errors_doc)]

use std::ffi::CString;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use g_runtime as native_logging_sink;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use super::errors;

const PYTHON_LOGGING_TARGET: &str = "g.python";

static PYTHON_LOGGING_INSTALLED: AtomicBool = AtomicBool::new(false);

#[expect(
    clippy::too_many_arguments,
    clippy::fn_params_excessive_bools,
    clippy::needless_pass_by_value,
    reason = "Runtime logging policy forwards concrete sink fields directly."
)]
pub(crate) fn initialize_logging(
    py: Python<'_>,
    log_filter: Option<String>,
    log_file: Option<String>,
    log_stderr: bool,
    log_queue_size: usize,
    log_lossy: bool,
    include_source_location: bool,
    include_span_events: bool,
    trace_file: Option<String>,
    trace_filter: Option<String>,
    trace_event_cap: Option<usize>,
) -> PyResult<bool> {
    let config = native_logging_sink::LoggingSinkConfig {
        log_filter: log_filter.as_deref(),
        log_file: log_file.as_deref().map(Path::new),
        log_stderr,
        log_queue_size,
        log_lossy,
        include_source_location,
        include_span_events,
        trace_file: trace_file.as_deref().map(Path::new),
        trace_filter: trace_filter.as_deref(),
        trace_event_cap,
    };
    native_logging_sink::initialize_logging_sinks(config, || setup_python_logging(py))
        .map_err(errors::convert_logging_sink_initialization_error)
}

fn setup_python_logging(py: Python<'_>) -> PyResult<()> {
    if PYTHON_LOGGING_INSTALLED
        .try_update(Ordering::AcqRel, Ordering::Acquire, |installed| (!installed).then_some(true))
        .is_err()
    {
        return Ok(());
    }

    if let Err(error) = install_python_logging(py) {
        PYTHON_LOGGING_INSTALLED.store(false, Ordering::Release);
        return Err(error);
    }
    Ok(())
}

fn install_python_logging(py: Python<'_>) -> PyResult<()> {
    pyo3_pylogger::setup_logging(py, PYTHON_LOGGING_TARGET)?;
    install_python_host_handler(py)
}

fn install_python_host_handler(py: Python<'_>) -> PyResult<()> {
    let logging = py.import("logging")?;
    let code = CString::new(
        r#"
root_logger = getLogger()
if not any(handler.__class__.__name__ == "HostHandler" for handler in root_logger.handlers):
    root_logger.addHandler(HostHandler())
root_logger.setLevel(NOTSET)
"#,
    )
    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    py.run(&code, Some(&logging.dict()), None)
}
