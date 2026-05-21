//! Unified tracing setup for Rust and Python diagnostics.

#![allow(clippy::missing_errors_doc)]

use std::ffi::CString;
use std::fs::{self, OpenOptions};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use tracing_appender::non_blocking::{NonBlocking, NonBlockingBuilder, WorkerGuard};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::prelude::*;

const DEFAULT_LOG_FILTER: &str = "info";
const PYTHON_LOGGING_TARGET: &str = "g.python";

static LOGGING_GUARDS: Mutex<Option<Vec<WorkerGuard>>> = Mutex::new(None);
static PYTHON_LOGGING_INSTALLED: AtomicBool = AtomicBool::new(false);

#[pyfunction]
#[pyo3(signature = (log_filter=None, log_file=None, log_stderr=true))]
pub fn initialize_logging(
    py: Python<'_>,
    log_filter: Option<String>,
    log_file: Option<String>,
    log_stderr: bool,
) -> PyResult<bool> {
    let mut logging_guards = lock_logging_guards()?;
    if logging_guards.is_some() {
        setup_python_logging(py)?;
        return Ok(false);
    }

    let resolved_log_filter = log_filter
        .filter(|candidate_filter| !candidate_filter.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_LOG_FILTER.to_string());
    let environment_filter = EnvFilter::try_new(&resolved_log_filter)
        .map_err(|error| PyValueError::new_err(format!("Invalid g log filter: {error}")))?;

    let mut worker_guards = Vec::new();
    let stderr_layer = if log_stderr {
        let (stderr_writer, stderr_guard) = build_non_blocking_writer(std::io::stderr(), "g-tracing-stderr");
        worker_guards.push(stderr_guard);
        Some(tracing_subscriber::fmt::layer().compact().with_writer(stderr_writer).with_ansi(true))
    } else {
        None
    };
    let file_layer = if let Some(log_file_path) = log_file {
        let (file_writer, file_guard) = build_log_file_writer(Path::new(&log_file_path))?;
        worker_guards.push(file_guard);
        Some(tracing_subscriber::fmt::layer().json().flatten_event(true).with_ansi(false).with_writer(file_writer))
    } else {
        None
    };

    let subscriber = tracing_subscriber::registry().with(environment_filter).with(stderr_layer).with(file_layer);
    if subscriber.try_init().is_err() {
        setup_python_logging(py)?;
        return Ok(false);
    }

    setup_python_logging(py)?;
    *logging_guards = Some(worker_guards);
    tracing::info!(target: "g.logging", "logging initialized");
    Ok(true)
}

#[pyfunction]
pub fn shutdown_logging() -> PyResult<()> {
    let mut logging_guards = lock_logging_guards()?;
    let _dropped_guards = logging_guards.take();
    Ok(())
}

fn lock_logging_guards() -> PyResult<std::sync::MutexGuard<'static, Option<Vec<WorkerGuard>>>> {
    LOGGING_GUARDS.lock().map_err(|_| PyRuntimeError::new_err("Logging guard mutex was poisoned."))
}

fn setup_python_logging(py: Python<'_>) -> PyResult<()> {
    if PYTHON_LOGGING_INSTALLED.load(Ordering::Acquire) {
        return Ok(());
    }
    pyo3_pylogger::setup_logging(py, PYTHON_LOGGING_TARGET)?;
    install_python_host_handler(py)?;
    register_shutdown_logging(py)?;
    PYTHON_LOGGING_INSTALLED.store(true, Ordering::Release);
    Ok(())
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

fn register_shutdown_logging(py: Python<'_>) -> PyResult<()> {
    let core_module = py.import("g._core")?;
    let shutdown_logging_function = core_module.getattr("shutdown_logging")?;
    let atexit = py.import("atexit")?;
    atexit.call_method1("register", (shutdown_logging_function,))?;
    Ok(())
}

fn build_log_file_writer(path: &Path) -> PyResult<(NonBlocking, WorkerGuard)> {
    if let Some(parent_directory) = path.parent().filter(|parent_directory| !parent_directory.as_os_str().is_empty()) {
        fs::create_dir_all(parent_directory).map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    }
    let log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    Ok(build_non_blocking_writer(log_file, "g-tracing-file"))
}

fn build_non_blocking_writer<Writer>(writer: Writer, thread_name: &str) -> (NonBlocking, WorkerGuard)
where
    Writer: std::io::Write + Send + 'static,
{
    NonBlockingBuilder::default().lossy(true).thread_name(thread_name).finish(writer)
}
