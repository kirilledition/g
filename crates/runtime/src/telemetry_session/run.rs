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

#[cfg(test)]
mod tests {
    use std::error::Error as _;

    use serde::Serializer;
    use serde::ser::Error as _;

    use super::*;
    use crate::test_support::{RUNTIME_GLOBAL_TEST_MUTEX, TemporaryDirectory};

    struct SerializationFailure;

    impl Serialize for SerializationFailure {
        fn serialize<SerializerType>(
            &self,
            _serializer: SerializerType,
        ) -> Result<SerializerType::Ok, SerializerType::Error>
        where
            SerializerType: Serializer,
        {
            Err(SerializerType::Error::custom("intentional event serialization failure"))
        }
    }

    #[test]
    fn disabled_session_skips_serialization_and_io() {
        let session = TelemetryRunSession::default();
        assert!(!session.is_enabled());
        session
            .emit_current_event("main", "ignored", "info", &SerializationFailure)
            .expect("disabled session should not serialize fields");
        session.finish("main").expect("disabled session finish should be a no-op");
    }

    #[test]
    fn enabled_clones_share_run_id_writer_and_close_counters() {
        let _global_guard = RUNTIME_GLOBAL_TEST_MUTEX.lock().expect("runtime global test mutex should be available");
        let temporary_directory = TemporaryDirectory::new("telemetry-session");
        let stream_path = temporary_directory.path().join("nested/events.jsonl");
        let session = TelemetryRunSession::new(&stream_path, 16, false, Arc::from("run-123"))
            .expect("telemetry session should open");
        assert!(session.is_enabled());

        let serialization_error = session
            .emit_current_event("main", "invalid", "info", &SerializationFailure)
            .expect_err("enabled session should propagate serialization errors");
        assert!(matches!(&serialization_error, TelemetryRunError::Serialization(_)));
        assert!(serialization_error.source().is_some());

        session
            .emit_current_event("main", "run_started", "info", &serde_json::json!({"trait_count": 1}))
            .expect("first event should write");
        let cloned_session = session.clone();
        cloned_session
            .emit_current_event("worker", "chunk_finished", "debug", &serde_json::json!({"variant_count": 512}))
            .expect("cloned session event should write");
        session.finish("main").expect("telemetry session should finish");

        let stream_text = std::fs::read_to_string(&stream_path).expect("telemetry stream should be readable");
        let records: Vec<serde_json::Value> =
            stream_text.lines().map(|line| serde_json::from_str(line).expect("telemetry line should parse")).collect();
        assert_eq!(records.len(), 3);
        assert!(records.iter().all(|record| record["schema_version"] == 0));
        assert!(records.iter().all(|record| record["run_id"] == "run-123"));
        assert_eq!(records[0]["event"], "run_started");
        assert_eq!(records[0]["trait_count"], 1);
        assert_eq!(records[1]["event"], "chunk_finished");
        assert_eq!(records[1]["variant_count"], 512);
        assert_eq!(records[2]["event"], TELEMETRY_SESSION_CLOSED_EVENT_NAME);
        assert_eq!(records[2]["accepted_event_count"], 2);
        assert_eq!(records[2]["written_event_count"], 2);
        assert_eq!(records[2]["dropped_event_count"], 0);
        assert_eq!(records[2]["queue_dropped_event_count"], 0);
        assert_eq!(records[2]["lossy"], false);
    }

    #[test]
    fn poisoned_writer_lock_has_typed_error_and_drop_unregisters_writer() {
        let _global_guard = RUNTIME_GLOBAL_TEST_MUTEX.lock().expect("runtime global test mutex should be available");
        let temporary_directory = TemporaryDirectory::new("telemetry-poison");
        let stream_path = temporary_directory.path().join("events.jsonl");
        let session = TelemetryRunSession::new(&stream_path, 16, false, Arc::from("run-poison"))
            .expect("telemetry session should open");
        let enabled = Arc::clone(session.enabled.as_ref().expect("telemetry session should be enabled"));
        let _panic = std::thread::spawn(move || {
            let _guard = enabled.writer.lock().expect("writer lock should begin healthy");
            panic!("poison telemetry writer lock");
        })
        .join();

        let emit_error = session
            .emit_current_event("main", "event", "info", &serde_json::json!({}))
            .expect_err("poisoned writer should reject event");
        assert!(matches!(&emit_error, TelemetryRunError::WriterLockPoisoned));
        assert_eq!(emit_error.to_string(), "Telemetry writer lock was poisoned.");
        assert!(emit_error.source().is_none());
        let finish_error = session.finish("main").expect_err("poisoned writer should reject finish");
        assert!(matches!(finish_error, TelemetryRunError::WriterLockPoisoned));
    }

    #[test]
    fn telemetry_error_io_variant_exposes_source() {
        let error = TelemetryRunError::from(io::Error::new(io::ErrorKind::PermissionDenied, "denied"));
        assert_eq!(error.to_string(), "denied");
        assert!(error.source().is_some());
    }
}
