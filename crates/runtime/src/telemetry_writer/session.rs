use std::io;
use std::path::Path;

use crate::telemetry_session::TelemetryWriterCounterSnapshot;

use super::file::build_telemetry_file_writer;
use super::shared::{register_shared_telemetry_writer, unregister_shared_telemetry_writer};
use super::{TelemetryWriterFactory, TelemetryWriterGuard};

pub(crate) struct TelemetrySessionWriter {
    writer: Option<TelemetryWriterFactory>,
    guard: Option<TelemetryWriterGuard>,
    is_registered: bool,
}

impl TelemetrySessionWriter {
    /// Open a telemetry stream and register it for shared logging reuse.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when the telemetry file cannot be opened or the
    /// shared-writer registry cannot be updated.
    pub(crate) fn new(path: &Path, log_queue_size: usize, log_lossy: bool) -> io::Result<Self> {
        let (writer, guard) = build_telemetry_file_writer(path, log_queue_size, log_lossy)?;
        register_shared_telemetry_writer(writer.clone())?;
        Ok(Self { writer: Some(writer), guard: Some(guard), is_registered: true })
    }

    /// Write one JSON line to the telemetry stream when it is still open.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when the writer cannot append the line.
    pub(crate) fn write_json_line(&self, json_line: &str) -> io::Result<()> {
        let Some(writer) = self.writer.as_ref() else {
            return Ok(());
        };
        writer.write_json_line(json_line)
    }

    #[must_use]
    pub(crate) fn counter_snapshot(&self) -> TelemetryWriterCounterSnapshot {
        if let Some(writer) = self.writer.as_ref() {
            return writer.counter_snapshot();
        }
        TelemetryWriterCounterSnapshot::empty()
    }

    /// Stop routing process logging to this run and wait for active formatters.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when the shared-writer registry is unavailable.
    pub(crate) fn stop_shared_logging(&mut self) -> io::Result<()> {
        if self.is_registered {
            unregister_shared_telemetry_writer()?;
            self.is_registered = false;
        }
        Ok(())
    }

    /// Drop the run-owned worker guard after all records have been enqueued.
    pub(crate) fn finish(&mut self) {
        let writer = self.writer.take();
        let guard = self.guard.take();
        drop(writer);
        drop(guard);
    }
}

impl Drop for TelemetrySessionWriter {
    fn drop(&mut self) {
        if self.is_registered {
            let _ = unregister_shared_telemetry_writer();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{RUNTIME_GLOBAL_TEST_MUTEX, TemporaryDirectory};

    #[test]
    fn session_writer_enforces_single_owner_and_finishes_idempotently() {
        let _global_guard = RUNTIME_GLOBAL_TEST_MUTEX.lock().expect("runtime global test mutex should be available");
        let temporary_directory = TemporaryDirectory::new("telemetry-writer");
        let first_path = temporary_directory.path().join("first.jsonl");
        let second_path = temporary_directory.path().join("second.jsonl");
        let mut writer = TelemetrySessionWriter::new(&first_path, 16, false).expect("first writer should open");

        let duplicate_error =
            TelemetrySessionWriter::new(&second_path, 16, false).err().expect("second active writer should fail");
        assert_eq!(duplicate_error.kind(), io::ErrorKind::AlreadyExists);

        writer.write_json_line("{\"event\":\"one\"}\n").expect("event should enqueue");
        assert_eq!(writer.counter_snapshot().accepted_event_count, 1);
        writer.stop_shared_logging().expect("first unregister should succeed");
        writer.stop_shared_logging().expect("second unregister should be a no-op");
        writer.finish();
        writer.finish();
        assert_eq!(writer.counter_snapshot(), TelemetryWriterCounterSnapshot::empty());
        writer.write_json_line("{\"event\":\"ignored\"}\n").expect("finished writer should be a no-op");
        assert_eq!(std::fs::read(&first_path).expect("first stream should be readable"), b"{\"event\":\"one\"}\n");

        drop(writer);
        let dropped_writer =
            TelemetrySessionWriter::new(&second_path, 16, false).expect("registry should be reusable after finish");
        drop(dropped_writer);
        let mut replacement =
            TelemetrySessionWriter::new(&second_path, 16, false).expect("drop should unregister active writer");
        replacement.stop_shared_logging().expect("replacement should unregister");
        replacement.finish();
    }
}
