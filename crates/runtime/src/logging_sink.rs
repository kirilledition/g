//! Runtime-owned tracing subscriber setup and logging sink lifecycle.

use std::error::Error;
use std::fmt;
use std::path::Path;
use std::sync::Mutex;

use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::prelude::*;

use crate::telemetry_writer;

const DEFAULT_LOG_FILTER: &str = "info";

static LOGGING_GUARDS: Mutex<Option<Vec<telemetry_writer::TelemetryWriterGuard>>> = Mutex::new(None);

#[derive(Clone, Copy, Debug)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "Logging sink config mirrors the existing runtime policy booleans at the native ownership boundary."
)]
pub struct LoggingSinkConfig<'a> {
    pub log_filter: Option<&'a str>,
    pub log_file: Option<&'a Path>,
    pub log_stderr: bool,
    pub log_queue_size: usize,
    pub log_lossy: bool,
    pub include_source_location: bool,
    pub include_span_events: bool,
    pub trace_file: Option<&'a Path>,
    pub trace_filter: Option<&'a str>,
    pub trace_event_cap: Option<usize>,
}

#[derive(Debug)]
pub enum LoggingSinkError {
    InvalidLogFilter { message: String },
    InvalidTraceFilter { message: String },
    Writer(std::io::Error),
    LoggingGuardMutexPoisoned,
}

#[derive(Debug)]
pub enum LoggingSinkInitializationError<HostLoggingError> {
    Sink(LoggingSinkError),
    HostLogging(HostLoggingError),
}

/// Initialize Rust logging sinks and run the host logging bridge setup.
///
/// # Errors
///
/// Returns a sink error when filter parsing, writer setup, or guard locking
/// fails. Returns a host logging error when the provided bridge setup callback
/// fails.
pub fn initialize_logging_sinks<HostLoggingError>(
    config: LoggingSinkConfig<'_>,
    setup_host_logging: impl FnOnce() -> Result<(), HostLoggingError>,
) -> Result<bool, LoggingSinkInitializationError<HostLoggingError>> {
    let mut logging_guards = lock_logging_guards().map_err(LoggingSinkInitializationError::Sink)?;
    if logging_guards.is_some() {
        setup_host_logging().map_err(LoggingSinkInitializationError::HostLogging)?;
        return Ok(false);
    }

    let resolved_log_filter =
        config.log_filter.filter(|candidate_filter| !candidate_filter.trim().is_empty()).unwrap_or(DEFAULT_LOG_FILTER);
    let environment_filter = EnvFilter::try_new(resolved_log_filter).map_err(|error| {
        LoggingSinkInitializationError::Sink(LoggingSinkError::InvalidLogFilter { message: error.to_string() })
    })?;

    let mut worker_guards = Vec::new();
    let stderr_layer = if config.log_stderr {
        let (stderr_writer, stderr_guard) = telemetry_writer::build_non_blocking_writer(
            std::io::stderr(),
            "g-tracing-stderr",
            config.log_queue_size,
            config.log_lossy,
        );
        worker_guards.push(stderr_guard);
        let layer = tracing_subscriber::fmt::layer()
            .compact()
            .with_writer(stderr_writer)
            .with_ansi(true)
            .with_file(config.include_source_location)
            .with_line_number(config.include_source_location)
            .with_span_events(resolve_span_events(config.include_span_events));
        Some(layer.boxed())
    } else {
        None
    };
    let file_layer = if let Some(log_file_path) = config.log_file {
        let (file_writer, maybe_file_guard) = telemetry_writer::build_shared_or_log_file_writer(
            log_file_path,
            config.log_queue_size,
            config.log_lossy,
            None,
        )
        .map_err(|error| LoggingSinkInitializationError::Sink(LoggingSinkError::Writer(error)))?;
        if let Some(file_guard) = maybe_file_guard {
            worker_guards.push(file_guard);
        }
        let layer = tracing_subscriber::fmt::layer()
            .json()
            .flatten_event(true)
            .with_ansi(false)
            .with_writer(file_writer)
            .with_file(config.include_source_location)
            .with_line_number(config.include_source_location)
            .with_span_events(resolve_span_events(config.include_span_events));
        Some(layer.boxed())
    } else {
        None
    };
    let trace_layer = if let Some(trace_file_path) = config.trace_file {
        let (trace_writer, maybe_trace_guard) = telemetry_writer::build_shared_or_log_file_writer(
            trace_file_path,
            config.log_queue_size,
            config.log_lossy,
            telemetry_writer::normalize_event_cap(config.trace_event_cap),
        )
        .map_err(|error| LoggingSinkInitializationError::Sink(LoggingSinkError::Writer(error)))?;
        if let Some(trace_guard) = maybe_trace_guard {
            worker_guards.push(trace_guard);
        }
        let resolved_trace_filter = config
            .trace_filter
            .filter(|candidate_filter| !candidate_filter.trim().is_empty())
            .unwrap_or(resolved_log_filter);
        let trace_environment_filter = EnvFilter::try_new(resolved_trace_filter).map_err(|error| {
            LoggingSinkInitializationError::Sink(LoggingSinkError::InvalidTraceFilter { message: error.to_string() })
        })?;
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
        setup_host_logging().map_err(LoggingSinkInitializationError::HostLogging)?;
        return Ok(false);
    }

    setup_host_logging().map_err(LoggingSinkInitializationError::HostLogging)?;
    *logging_guards = Some(worker_guards);
    tracing::info!(target: "g.logging", "logging initialized");
    Ok(true)
}

/// Drop logging sink guards and flush non-blocking writers.
///
/// # Errors
///
/// Returns an error when the logging guard lock is poisoned.
pub fn shutdown_logging_sinks() -> Result<(), LoggingSinkError> {
    let mut logging_guards = lock_logging_guards()?;
    let _dropped_guards = logging_guards.take();
    Ok(())
}

impl fmt::Display for LoggingSinkError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLogFilter { message } => write!(formatter, "Invalid g log filter: {message}"),
            Self::InvalidTraceFilter { message } => write!(formatter, "Invalid g trace filter: {message}"),
            Self::Writer(error) => write!(formatter, "{error}"),
            Self::LoggingGuardMutexPoisoned => formatter.write_str("Logging guard mutex was poisoned."),
        }
    }
}

impl Error for LoggingSinkError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Writer(error) => Some(error),
            Self::InvalidLogFilter { .. } | Self::InvalidTraceFilter { .. } | Self::LoggingGuardMutexPoisoned => None,
        }
    }
}

fn lock_logging_guards()
-> Result<std::sync::MutexGuard<'static, Option<Vec<telemetry_writer::TelemetryWriterGuard>>>, LoggingSinkError> {
    LOGGING_GUARDS.lock().map_err(|_| LoggingSinkError::LoggingGuardMutexPoisoned)
}

const fn resolve_span_events(include_span_events: bool) -> FmtSpan {
    if include_span_events { FmtSpan::FULL } else { FmtSpan::NONE }
}
