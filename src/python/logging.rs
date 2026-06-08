//! Unified tracing setup for Rust and Python diagnostics.

#![allow(clippy::missing_errors_doc)]

use std::ffi::CString;
use std::fs::{self, OpenOptions};
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use tracing_appender::non_blocking::{NonBlocking, NonBlockingBuilder, WorkerGuard};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::fmt::writer::MakeWriter;
use tracing_subscriber::prelude::*;

const DEFAULT_LOG_FILTER: &str = "info";
const PYTHON_LOGGING_TARGET: &str = "g.python";

static LOGGING_GUARDS: Mutex<Option<Vec<WorkerGuard>>> = Mutex::new(None);
static PYTHON_LOGGING_INSTALLED: AtomicBool = AtomicBool::new(false);
static TELEMETRY_WRITER: Mutex<Option<SharedTelemetryWriter>> = Mutex::new(None);

#[derive(Clone)]
struct SharedTelemetryWriter {
    path: PathBuf,
    writer: TelemetryWriterFactory,
}

#[derive(Clone)]
struct TelemetryWriterFactory {
    writer: NonBlocking,
    event_cap_state: Arc<TelemetryEventCapState>,
}

struct TelemetryLineWriter {
    writer: NonBlocking,
    event_cap_state: Arc<TelemetryEventCapState>,
    line_buffer: Vec<u8>,
}

struct TelemetryEventCapState {
    path: PathBuf,
    event_cap: Option<usize>,
    lossy: bool,
    written_event_count: AtomicUsize,
    dropped_event_count: AtomicUsize,
    exceeded: AtomicBool,
}

enum TelemetryCapAction {
    Write,
    Drop,
}

impl TelemetryWriterFactory {
    fn new(writer: NonBlocking, event_cap_state: TelemetryEventCapState) -> Self {
        Self { writer, event_cap_state: Arc::new(event_cap_state) }
    }

    fn write_json_line(&self, json_line: &str) -> io::Result<()> {
        let mut line_writer = self.make_writer();
        line_writer.write_all(json_line.as_bytes())?;
        if !json_line.ends_with('\n') {
            line_writer.write_all(b"\n")?;
        }
        line_writer.flush()
    }

    fn fail_if_lossless_cap_exceeded(&self) -> PyResult<()> {
        if self.event_cap_state.should_fail_for_cap_exceeded() {
            return Err(PyRuntimeError::new_err(self.event_cap_state.cap_exceeded_error_message()));
        }
        Ok(())
    }
}

impl<'a> MakeWriter<'a> for TelemetryWriterFactory {
    type Writer = TelemetryLineWriter;

    fn make_writer(&'a self) -> Self::Writer {
        TelemetryLineWriter {
            writer: self.writer.clone(),
            event_cap_state: Arc::clone(&self.event_cap_state),
            line_buffer: Vec::new(),
        }
    }
}

impl TelemetryLineWriter {
    fn write_complete_line(&mut self, line: &[u8]) -> io::Result<()> {
        match self.event_cap_state.reserve_event()? {
            TelemetryCapAction::Write => self.writer.write_all(line),
            TelemetryCapAction::Drop => Ok(()),
        }
    }
}

impl io::Write for TelemetryLineWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        if self.event_cap_state.event_cap.is_none() {
            self.writer.write_all(buffer)?;
            return Ok(buffer.len());
        }

        self.line_buffer.extend_from_slice(buffer);
        while let Some(newline_index) = self.line_buffer.iter().position(|byte| *byte == b'\n') {
            let complete_line = self.line_buffer.drain(..=newline_index).collect::<Vec<_>>();
            self.write_complete_line(&complete_line)?;
        }
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        if !self.line_buffer.is_empty() {
            let complete_line = std::mem::take(&mut self.line_buffer);
            self.write_complete_line(&complete_line)?;
        }
        self.writer.flush()
    }
}

impl TelemetryEventCapState {
    fn new(path: &Path, event_cap: Option<usize>, lossy: bool) -> Self {
        Self {
            path: path.to_path_buf(),
            event_cap,
            lossy,
            written_event_count: AtomicUsize::new(0),
            dropped_event_count: AtomicUsize::new(0),
            exceeded: AtomicBool::new(false),
        }
    }

    fn reserve_event(&self) -> io::Result<TelemetryCapAction> {
        let Some(event_cap) = self.event_cap else {
            self.written_event_count.fetch_add(1, Ordering::Relaxed);
            return Ok(TelemetryCapAction::Write);
        };

        loop {
            let written_event_count = self.written_event_count.load(Ordering::Acquire);
            if written_event_count >= event_cap {
                self.mark_exceeded();
                if self.lossy {
                    self.dropped_event_count.fetch_add(1, Ordering::Relaxed);
                    return Ok(TelemetryCapAction::Drop);
                }
                return Err(io::Error::other(self.cap_exceeded_error_message()));
            }
            if self
                .written_event_count
                .compare_exchange_weak(
                    written_event_count,
                    written_event_count + 1,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                return Ok(TelemetryCapAction::Write);
            }
        }
    }

    fn mark_exceeded(&self) {
        if !self.exceeded.swap(true, Ordering::AcqRel) && self.lossy {
            eprintln!("{}", self.cap_exceeded_drop_message());
        }
    }

    fn should_fail_for_cap_exceeded(&self) -> bool {
        self.exceeded.load(Ordering::Acquire) && !self.lossy
    }

    fn cap_exceeded_error_message(&self) -> String {
        let event_cap = self.event_cap.unwrap_or(0);
        format!(
            "Trace telemetry event cap exceeded at {event_cap} events for {}. \
             Increase --g-trace-event-cap or set --g-trace-event-cap 0 to disable the cap for intentional deep traces. \
             Use --g-log-lossy to drop events after the cap instead of failing.",
            self.path.display()
        )
    }

    fn cap_exceeded_drop_message(&self) -> String {
        let event_cap = self.event_cap.unwrap_or(0);
        format!(
            "Trace telemetry event cap reached at {event_cap} events for {}; dropping additional trace events because g-log-lossy is enabled.",
            self.path.display()
        )
    }
}

#[pyclass]
pub struct NativeTelemetrySession {
    path: PathBuf,
    writer: Mutex<Option<TelemetryWriterFactory>>,
    guard: Mutex<Option<WorkerGuard>>,
}

#[pymethods]
impl NativeTelemetrySession {
    #[new]
    #[pyo3(signature = (stream_file, queue_size=65536, lossy=true, event_cap=None))]
    pub fn new(stream_file: String, queue_size: usize, lossy: bool, event_cap: Option<usize>) -> PyResult<Self> {
        let path = PathBuf::from(stream_file);
        let (writer, guard) = build_telemetry_file_writer(&path, queue_size, lossy, normalize_event_cap(event_cap))?;
        replace_shared_telemetry_writer(path.clone(), writer.clone())?;
        Ok(Self { path, writer: Mutex::new(Some(writer)), guard: Mutex::new(Some(guard)) })
    }

    pub fn emit_json_line(&self, json_line: &str) -> PyResult<()> {
        let writer_guard =
            self.writer.lock().map_err(|_| PyRuntimeError::new_err("Telemetry writer mutex was poisoned."))?;
        let Some(writer) = writer_guard.as_ref() else {
            return Ok(());
        };
        writer.write_json_line(json_line).map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok(())
    }

    pub fn finish(&self) -> PyResult<()> {
        let mut writer_guard =
            self.writer.lock().map_err(|_| PyRuntimeError::new_err("Telemetry writer mutex was poisoned."))?;
        let dropped_writer = writer_guard.take();
        let mut guard =
            self.guard.lock().map_err(|_| PyRuntimeError::new_err("Telemetry guard mutex was poisoned."))?;
        let dropped_guard = guard.take();
        drop(dropped_guard);
        clear_shared_telemetry_writer(&self.path)?;
        if let Some(writer) = dropped_writer.as_ref() {
            writer.fail_if_lossless_cap_exceeded()?;
        }
        Ok(())
    }
}

#[pyfunction]
#[expect(
    clippy::too_many_arguments,
    clippy::fn_params_excessive_bools,
    reason = "PyO3 exposes documented Python logging keyword arguments directly."
)]
#[pyo3(signature = (
    log_filter=None,
    log_file=None,
    log_stderr=true,
    log_queue_size=65536,
    log_lossy=true,
    include_source_location=false,
    include_span_events=false,
    trace_file=None,
    trace_filter=None,
    trace_event_cap=None
))]
pub fn initialize_logging(
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
        let (stderr_writer, stderr_guard) =
            build_non_blocking_writer(std::io::stderr(), "g-tracing-stderr", log_queue_size, log_lossy);
        worker_guards.push(stderr_guard);
        let layer = tracing_subscriber::fmt::layer()
            .compact()
            .with_writer(stderr_writer)
            .with_ansi(true)
            .with_file(include_source_location)
            .with_line_number(include_source_location)
            .with_span_events(resolve_span_events(include_span_events));
        Some(layer.boxed())
    } else {
        None
    };
    let file_layer = if let Some(log_file_path) = log_file {
        let (file_writer, maybe_file_guard) =
            build_shared_or_log_file_writer(Path::new(&log_file_path), log_queue_size, log_lossy, None)?;
        if let Some(file_guard) = maybe_file_guard {
            worker_guards.push(file_guard);
        }
        let layer = tracing_subscriber::fmt::layer()
            .json()
            .flatten_event(true)
            .with_ansi(false)
            .with_writer(file_writer)
            .with_file(include_source_location)
            .with_line_number(include_source_location)
            .with_span_events(resolve_span_events(include_span_events));
        Some(layer.boxed())
    } else {
        None
    };
    let trace_layer = if let Some(trace_file_path) = trace_file {
        let (trace_writer, maybe_trace_guard) = build_shared_or_log_file_writer(
            Path::new(&trace_file_path),
            log_queue_size,
            log_lossy,
            normalize_event_cap(trace_event_cap),
        )?;
        if let Some(trace_guard) = maybe_trace_guard {
            worker_guards.push(trace_guard);
        }
        let resolved_trace_filter = trace_filter
            .filter(|candidate_filter| !candidate_filter.trim().is_empty())
            .unwrap_or_else(|| resolved_log_filter.clone());
        let trace_environment_filter = EnvFilter::try_new(&resolved_trace_filter)
            .map_err(|error| PyValueError::new_err(format!("Invalid g trace filter: {error}")))?;
        let layer = tracing_subscriber::fmt::layer()
            .json()
            .flatten_event(true)
            .with_ansi(false)
            .with_writer(trace_writer)
            .with_file(true)
            .with_line_number(true)
            .with_span_events(FmtSpan::FULL)
            .with_filter(trace_environment_filter);
        Some(layer.boxed())
    } else {
        None
    };

    let subscriber =
        tracing_subscriber::registry().with(environment_filter).with(stderr_layer).with(file_layer).with(trace_layer);
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

fn lock_telemetry_writer() -> PyResult<std::sync::MutexGuard<'static, Option<SharedTelemetryWriter>>> {
    TELEMETRY_WRITER.lock().map_err(|_| PyRuntimeError::new_err("Telemetry writer mutex was poisoned."))
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
    install_python_host_handler(py)?;
    register_shutdown_logging(py)?;
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

fn resolve_span_events(include_span_events: bool) -> FmtSpan {
    if include_span_events { FmtSpan::FULL } else { FmtSpan::NONE }
}

fn normalize_event_cap(event_cap: Option<usize>) -> Option<usize> {
    event_cap.filter(|cap| *cap > 0)
}

fn build_log_file_writer(path: &Path, log_queue_size: usize, log_lossy: bool) -> PyResult<(NonBlocking, WorkerGuard)> {
    if let Some(parent_directory) = path.parent().filter(|parent_directory| !parent_directory.as_os_str().is_empty()) {
        fs::create_dir_all(parent_directory).map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    }
    let log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    Ok(build_non_blocking_writer(log_file, "g-tracing-file", log_queue_size, log_lossy))
}

fn build_telemetry_file_writer(
    path: &Path,
    log_queue_size: usize,
    log_lossy: bool,
    event_cap: Option<usize>,
) -> PyResult<(TelemetryWriterFactory, WorkerGuard)> {
    let (writer, guard) = build_log_file_writer(path, log_queue_size, log_lossy)?;
    let event_cap_state = TelemetryEventCapState::new(path, event_cap, log_lossy);
    Ok((TelemetryWriterFactory::new(writer, event_cap_state), guard))
}

fn build_shared_or_log_file_writer(
    path: &Path,
    log_queue_size: usize,
    log_lossy: bool,
    event_cap: Option<usize>,
) -> PyResult<(TelemetryWriterFactory, Option<WorkerGuard>)> {
    if let Some(shared_writer) = shared_telemetry_writer_for_path(path)? {
        return Ok((shared_writer, None));
    }
    let (writer, guard) = build_telemetry_file_writer(path, log_queue_size, log_lossy, event_cap)?;
    Ok((writer, Some(guard)))
}

fn shared_telemetry_writer_for_path(path: &Path) -> PyResult<Option<TelemetryWriterFactory>> {
    let normalized_path = normalize_path_for_comparison(path);
    let telemetry_writer = lock_telemetry_writer()?;
    Ok(telemetry_writer
        .as_ref()
        .filter(|shared_writer| normalize_path_for_comparison(&shared_writer.path) == normalized_path)
        .map(|shared_writer| shared_writer.writer.clone()))
}

fn replace_shared_telemetry_writer(path: PathBuf, writer: TelemetryWriterFactory) -> PyResult<()> {
    let mut telemetry_writer = lock_telemetry_writer()?;
    *telemetry_writer = Some(SharedTelemetryWriter { path, writer });
    Ok(())
}

fn clear_shared_telemetry_writer(path: &Path) -> PyResult<()> {
    let normalized_path = normalize_path_for_comparison(path);
    let mut telemetry_writer = lock_telemetry_writer()?;
    if telemetry_writer
        .as_ref()
        .is_some_and(|shared_writer| normalize_path_for_comparison(&shared_writer.path) == normalized_path)
    {
        let _dropped_writer = telemetry_writer.take();
    }
    Ok(())
}

fn normalize_path_for_comparison(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

fn build_non_blocking_writer<Writer>(
    writer: Writer,
    thread_name: &str,
    log_queue_size: usize,
    log_lossy: bool,
) -> (NonBlocking, WorkerGuard)
where
    Writer: std::io::Write + Send + 'static,
{
    NonBlockingBuilder::default()
        .lossy(log_lossy)
        .buffered_lines_limit(log_queue_size)
        .thread_name(thread_name)
        .finish(writer)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    static NEXT_TEST_FILE_ID: AtomicUsize = AtomicUsize::new(0);

    fn telemetry_test_path(test_name: &str) -> PathBuf {
        let file_id = NEXT_TEST_FILE_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("g-{test_name}-{}-{file_id}.jsonl", std::process::id()))
    }

    #[test]
    fn telemetry_event_cap_fails_without_lossy_mode() {
        let path = telemetry_test_path("telemetry-cap-fails");
        let (telemetry_writer, guard) =
            build_telemetry_file_writer(&path, 32, false, Some(2)).expect("writer should build");

        telemetry_writer.write_json_line(r#"{"event":"first"}"#).expect("first event should write");
        telemetry_writer.write_json_line(r#"{"event":"second"}"#).expect("second event should write");
        let error =
            telemetry_writer.write_json_line(r#"{"event":"third"}"#).expect_err("third event should exceed cap");

        assert!(error.to_string().contains("Trace telemetry event cap exceeded at 2 events"));
        assert!(telemetry_writer.fail_if_lossless_cap_exceeded().is_err());
        drop(telemetry_writer);
        drop(guard);

        let line_count = fs::read_to_string(&path).expect("telemetry file should be readable").lines().count();
        assert_eq!(line_count, 2);
        fs::remove_file(path).expect("telemetry test file should be removed");
    }

    #[test]
    fn telemetry_event_cap_drops_with_lossy_mode() {
        let path = telemetry_test_path("telemetry-cap-drops");
        let (telemetry_writer, guard) =
            build_telemetry_file_writer(&path, 32, true, Some(1)).expect("writer should build");

        telemetry_writer.write_json_line(r#"{"event":"first"}"#).expect("first event should write");
        telemetry_writer.write_json_line(r#"{"event":"second"}"#).expect("second event should drop");
        telemetry_writer.write_json_line(r#"{"event":"third"}"#).expect("third event should drop");
        assert!(telemetry_writer.fail_if_lossless_cap_exceeded().is_ok());
        drop(telemetry_writer);
        drop(guard);

        let telemetry_text = fs::read_to_string(&path).expect("telemetry file should be readable");
        assert_eq!(telemetry_text.lines().count(), 1);
        assert!(telemetry_text.contains(r#""event":"first""#));
        fs::remove_file(path).expect("telemetry test file should be removed");
    }

    #[test]
    fn telemetry_event_cap_zero_disables_cap() {
        let path = telemetry_test_path("telemetry-cap-disabled");
        let (telemetry_writer, guard) =
            build_telemetry_file_writer(&path, 32, false, normalize_event_cap(Some(0))).expect("writer should build");

        telemetry_writer.write_json_line(r#"{"event":"first"}"#).expect("first event should write");
        telemetry_writer.write_json_line(r#"{"event":"second"}"#).expect("second event should write");
        telemetry_writer.write_json_line(r#"{"event":"third"}"#).expect("third event should write");
        assert!(telemetry_writer.fail_if_lossless_cap_exceeded().is_ok());
        drop(telemetry_writer);
        drop(guard);

        let line_count = fs::read_to_string(&path).expect("telemetry file should be readable").lines().count();
        assert_eq!(line_count, 3);
        fs::remove_file(path).expect("telemetry test file should be removed");
    }
}
