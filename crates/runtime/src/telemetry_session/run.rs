use std::fmt;
use std::io;
use std::path::Path;
use std::sync::{Arc, Mutex};

use serde::Serialize;

use crate::telemetry_writer::TelemetrySessionWriter;

use super::{build_current_telemetry_event_envelope, serialize_telemetry_event_json_line};

const TELEMETRY_SESSION_CLOSED_EVENT_NAME: &str = "telemetry_session_closed";
const TELEMETRY_SESSION_CLOSED_EVENT_LEVEL: &str = "debug";

#[derive(Debug)]
pub enum TelemetryRunError {
    WriterLockPoisoned,
    Io(io::Error),
    Serialization(serde_json::Error),
}

impl fmt::Display for TelemetryRunError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WriterLockPoisoned => formatter.write_str("Telemetry writer lock was poisoned."),
            Self::Io(error) => error.fmt(formatter),
            Self::Serialization(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for TelemetryRunError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Serialization(error) => Some(error),
            Self::WriterLockPoisoned => None,
        }
    }
}

impl From<io::Error> for TelemetryRunError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for TelemetryRunError {
    fn from(error: serde_json::Error) -> Self {
        Self::Serialization(error)
    }
}

struct EnabledTelemetryRunSession {
    run_id: Arc<str>,
    writer: Mutex<TelemetrySessionWriter>,
}

#[derive(Clone, Default)]
pub struct TelemetryRunSession {
    enabled: Option<Arc<EnabledTelemetryRunSession>>,
}

impl TelemetryRunSession {
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled.is_some()
    }

    /// Open an enabled telemetry writer with a shared run identifier.
    ///
    /// # Errors
    ///
    /// Returns an error when the configured writer cannot be opened.
    pub(crate) fn new(
        stream_file: &Path,
        queue_size: usize,
        lossy: bool,
        run_id: Arc<str>,
    ) -> Result<Self, TelemetryRunError> {
        let writer = TelemetrySessionWriter::new(stream_file, queue_size, lossy)?;
        Ok(Self { enabled: Some(Arc::new(EnabledTelemetryRunSession { run_id, writer: Mutex::new(writer) })) })
    }

    /// Serialize and emit one typed telemetry event.
    ///
    /// # Errors
    ///
    /// Returns an error when session state is unavailable, serialization
    /// fails, or the writer rejects the event.
    pub fn emit_current_event<Fields>(
        &self,
        thread_name: &str,
        event: &str,
        level: &str,
        fields: &Fields,
    ) -> Result<(), TelemetryRunError>
    where
        Fields: Serialize,
    {
        let Some(enabled) = self.enabled.as_ref() else {
            return Ok(());
        };
        let envelope = build_current_telemetry_event_envelope(enabled.run_id.as_ref(), event, level, thread_name);
        let json_line = serialize_telemetry_event_json_line(&envelope, fields)?;
        enabled.writer.lock().map_err(|_| TelemetryRunError::WriterLockPoisoned)?.write_json_line(&json_line)?;
        Ok(())
    }

    /// Emit close counters and flush the owned writer.
    ///
    /// # Errors
    ///
    /// Returns a telemetry state, serialization, or writer error.
    pub fn finish(&self, thread_name: &str) -> Result<(), TelemetryRunError> {
        let Some(enabled) = self.enabled.as_ref() else {
            return Ok(());
        };
        let mut writer = enabled.writer.lock().map_err(|_| TelemetryRunError::WriterLockPoisoned)?;
        writer.stop_shared_logging()?;
        let writer_counters = writer.counter_snapshot();
        let envelope = build_current_telemetry_event_envelope(
            enabled.run_id.as_ref(),
            TELEMETRY_SESSION_CLOSED_EVENT_NAME,
            TELEMETRY_SESSION_CLOSED_EVENT_LEVEL,
            thread_name,
        );
        let json_line = serialize_telemetry_event_json_line(&envelope, &writer_counters)?;
        writer.write_json_line(&json_line)?;
        writer.finish();
        Ok(())
    }
}
