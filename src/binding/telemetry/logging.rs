//! Unified tracing setup for Rust and Python diagnostics.

#![allow(clippy::missing_errors_doc)]

use std::ffi::CString;
use std::sync::atomic::{AtomicBool, Ordering};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

const PYTHON_LOGGING_TARGET: &str = "g.python";

static PYTHON_LOGGING_INSTALLED: AtomicBool = AtomicBool::new(false);

pub(crate) fn install_python_logging(py: Python<'_>) -> PyResult<()> {
    if PYTHON_LOGGING_INSTALLED
        .try_update(Ordering::AcqRel, Ordering::Acquire, |installed| (!installed).then_some(true))
        .is_err()
    {
        return Ok(());
    }

    if let Err(error) = configure_python_logging(py) {
        PYTHON_LOGGING_INSTALLED.store(false, Ordering::Release);
        return Err(error);
    }
    Ok(())
}

fn configure_python_logging(py: Python<'_>) -> PyResult<()> {
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
