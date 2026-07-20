//! Runtime-owned tracing subscriber setup and logging sink lifecycle.

use std::error::Error;
use std::fmt;
use std::sync::Mutex;

use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::prelude::*;

use crate::runtime_policy::NativeRunSessionPolicy;
use crate::telemetry_writer;

const DEFAULT_LOG_FILTER: &str = "info";

static LOGGING_SUBSCRIBER_STATE: Mutex<LoggingSubscriberState> = Mutex::new(LoggingSubscriberState::Uninitialized);

#[derive(Clone, Copy, Eq, PartialEq)]
enum LoggingSubscriberState {
    Uninitialized,
    Initialized,
}

pub(crate) struct RunLoggingSession {
    stderr_writer: Option<RunLoggingWriter>,
    file_writer: Option<RunLoggingWriter>,
}

struct RunLoggingWriter {
    kind: telemetry_writer::SharedLogWriterKind,
    writer: Option<tracing_appender::non_blocking::NonBlocking>,
    guard: Option<telemetry_writer::TelemetryWriterGuard>,
    is_registered: bool,
}

#[derive(Debug)]
pub enum LoggingSinkError {
    InvalidLogFilter { message: String },
    Writer(std::io::Error),
    LoggingStateMutexPoisoned,
}

/// Install the process-global Rust logging subscriber.
///
/// # Errors
///
/// Returns a sink error when filter parsing or subscriber-state locking fails.
pub(crate) fn initialize_logging_sinks(policy: &NativeRunSessionPolicy) -> Result<(), LoggingSinkError> {
    let mut subscriber_state = lock_subscriber_state()?;
    if *subscriber_state != LoggingSubscriberState::Uninitialized {
        return Ok(());
    }

    let resolved_log_filter = if policy.log_filter.trim().is_empty() { DEFAULT_LOG_FILTER } else { &policy.log_filter };
    let environment_filter = EnvFilter::try_new(resolved_log_filter)
        .map_err(|error| LoggingSinkError::InvalidLogFilter { message: error.to_string() })?;

    let stderr_layer = if policy.log_stderr {
        let stderr_writer =
            telemetry_writer::SharedLogWriterFactory::new(telemetry_writer::SharedLogWriterKind::Stderr);
        let layer = tracing_subscriber::fmt::layer()
            .compact()
            .with_writer(stderr_writer)
            .with_ansi(true)
            .with_file(policy.include_source_location)
            .with_line_number(policy.include_source_location)
            .with_span_events(resolve_span_events(policy.include_span_events));
        Some(layer.boxed())
    } else {
        None
    };
    let file_layer = if policy.log_file.is_some() {
        let file_writer = telemetry_writer::SharedLogWriterFactory::new(telemetry_writer::SharedLogWriterKind::File);
        let layer = tracing_subscriber::fmt::layer()
            .json()
            .flatten_event(true)
            .with_ansi(false)
            .with_writer(file_writer)
            .with_file(policy.include_source_location)
            .with_line_number(policy.include_source_location)
            .with_span_events(resolve_span_events(policy.include_span_events));
        Some(layer.boxed())
    } else {
        None
    };
    let structured_log_layer = if policy.telemetry_stream_file.is_some() {
        let structured_log_writer = telemetry_writer::SharedTelemetryWriterFactory;
        let structured_log_filter = EnvFilter::try_new(resolved_log_filter)
            .map_err(|error| LoggingSinkError::InvalidLogFilter { message: error.to_string() })?;
        let layer = tracing_subscriber::fmt::layer()
            .json()
            .flatten_event(true)
            .with_ansi(false)
            .with_writer(structured_log_writer)
            .with_file(true)
            .with_line_number(true)
            .with_span_events(FmtSpan::FULL)
            .with_filter(structured_log_filter);
        Some(layer.boxed())
    } else {
        None
    };

    let subscriber = tracing_subscriber::registry()
        .with(environment_filter)
        .with(stderr_layer)
        .with(file_layer)
        .with(structured_log_layer);
    let subscriber_was_installed = subscriber.try_init().is_ok();
    *subscriber_state = LoggingSubscriberState::Initialized;
    drop(subscriber_state);
    if subscriber_was_installed {
        tracing::info!(target: "g.logging", "logging initialized");
    }
    Ok(())
}

impl RunLoggingSession {
    /// Open and register the asynchronous process log writers for one run.
    ///
    /// # Errors
    ///
    /// Returns a sink error when a file cannot be opened or another run owns a writer.
    pub(crate) fn new(policy: &NativeRunSessionPolicy) -> Result<Self, LoggingSinkError> {
        let stderr_writer = if policy.log_stderr {
            let (writer, guard) = telemetry_writer::build_non_blocking_writer(
                std::io::stderr(),
                "g-tracing-stderr",
                policy.queue_size,
                policy.lossy,
            );
            Some(RunLoggingWriter::new(telemetry_writer::SharedLogWriterKind::Stderr, writer, guard)?)
        } else {
            None
        };
        let file_writer = if let Some(log_file_path) = policy.log_file.as_deref() {
            let (writer, guard) =
                telemetry_writer::build_log_file_writer(log_file_path, policy.queue_size, policy.lossy)
                    .map_err(LoggingSinkError::Writer)?;
            Some(RunLoggingWriter::new(telemetry_writer::SharedLogWriterKind::File, writer, guard)?)
        } else {
            None
        };
        Ok(Self { stderr_writer, file_writer })
    }

    /// Stop routing new records and flush every run-owned logging worker.
    ///
    /// # Errors
    ///
    /// Returns a sink error when a dynamic-writer registry is unavailable.
    pub(crate) fn finish(&mut self) -> Result<(), LoggingSinkError> {
        let stderr_result = self.stderr_writer.as_mut().map_or(Ok(()), RunLoggingWriter::finish);
        let file_result = self.file_writer.as_mut().map_or(Ok(()), RunLoggingWriter::finish);
        stderr_result.and(file_result)
    }
}

impl RunLoggingWriter {
    fn new(
        kind: telemetry_writer::SharedLogWriterKind,
        writer: tracing_appender::non_blocking::NonBlocking,
        guard: telemetry_writer::TelemetryWriterGuard,
    ) -> Result<Self, LoggingSinkError> {
        telemetry_writer::register_shared_log_writer(kind, writer.clone()).map_err(LoggingSinkError::Writer)?;
        Ok(Self { kind, writer: Some(writer), guard: Some(guard), is_registered: true })
    }

    fn finish(&mut self) -> Result<(), LoggingSinkError> {
        if self.is_registered {
            telemetry_writer::unregister_shared_log_writer(self.kind).map_err(LoggingSinkError::Writer)?;
            self.is_registered = false;
        }
        let writer = self.writer.take();
        let guard = self.guard.take();
        drop(writer);
        drop(guard);
        Ok(())
    }
}

impl Drop for RunLoggingWriter {
    fn drop(&mut self) {
        if self.is_registered {
            let _ = telemetry_writer::unregister_shared_log_writer(self.kind);
        }
    }
}

impl fmt::Display for LoggingSinkError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLogFilter { message } => write!(formatter, "Invalid g log filter: {message}"),
            Self::Writer(error) => write!(formatter, "{error}"),
            Self::LoggingStateMutexPoisoned => formatter.write_str("Logging subscriber-state mutex was poisoned."),
        }
    }
}

impl Error for LoggingSinkError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Writer(error) => Some(error),
            Self::InvalidLogFilter { .. } | Self::LoggingStateMutexPoisoned => None,
        }
    }
}

fn lock_subscriber_state() -> Result<std::sync::MutexGuard<'static, LoggingSubscriberState>, LoggingSinkError> {
    LOGGING_SUBSCRIBER_STATE.lock().map_err(|_| LoggingSinkError::LoggingStateMutexPoisoned)
}

const fn resolve_span_events(include_span_events: bool) -> FmtSpan {
    if include_span_events { FmtSpan::FULL } else { FmtSpan::NONE }
}

#[cfg(test)]
mod tests {
    use std::io::Write as _;

    use tracing_subscriber::fmt::writer::MakeWriter as _;

    use super::*;
    use crate::test_support::{RUNTIME_GLOBAL_TEST_MUTEX, TemporaryDirectory, disabled_session_policy};

    #[test]
    fn run_logging_session_routes_flushes_and_releases_file_writer() {
        let _global_guard = RUNTIME_GLOBAL_TEST_MUTEX.lock().expect("runtime global test mutex should be available");
        let temporary_directory = TemporaryDirectory::new("logging-session");
        let log_path = temporary_directory.path().join("nested/runtime.log");
        let mut policy = disabled_session_policy();
        policy.log_file = Some(log_path.clone());
        let mut session = RunLoggingSession::new(&policy).expect("logging session should open");

        let duplicate_error = RunLoggingSession::new(&policy).err().expect("duplicate writer should fail");
        assert!(matches!(&duplicate_error, LoggingSinkError::Writer(_)));
        assert!(duplicate_error.to_string().contains("already active"));
        assert!(duplicate_error.source().is_some());

        let factory = telemetry_writer::SharedLogWriterFactory::new(telemetry_writer::SharedLogWriterKind::File);
        let mut routed_writer = factory.make_writer();
        routed_writer.write_all(b"routed record\n").expect("shared route should accept record");
        drop(routed_writer);
        session.finish().expect("logging session should flush");
        session.finish().expect("logging finish should be idempotent");
        assert_eq!(std::fs::read(&log_path).expect("runtime log should be readable"), b"routed record\n");

        drop(session);
        let dropped_session = RunLoggingSession::new(&policy).expect("writer route should be reusable after finish");
        drop(dropped_session);
        let mut replacement = RunLoggingSession::new(&policy).expect("drop should unregister file writer");
        replacement.finish().expect("replacement session should finish");
    }

    #[test]
    fn logging_errors_and_span_policy_have_stable_diagnostics() {
        let invalid = LoggingSinkError::InvalidLogFilter { message: "bad directive".to_owned() };
        assert_eq!(invalid.to_string(), "Invalid g log filter: bad directive");
        assert!(invalid.source().is_none());
        let poisoned = LoggingSinkError::LoggingStateMutexPoisoned;
        assert_eq!(poisoned.to_string(), "Logging subscriber-state mutex was poisoned.");
        assert!(poisoned.source().is_none());
        assert_eq!(format!("{:?}", resolve_span_events(false)), "FmtSpan::NONE");
        assert_eq!(
            format!("{:?}", resolve_span_events(true)),
            "FmtSpan::NEW | FmtSpan::ENTER | FmtSpan::EXIT | FmtSpan::CLOSE"
        );
    }
}
