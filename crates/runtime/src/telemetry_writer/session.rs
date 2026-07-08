use std::io;
use std::path::PathBuf;
use std::time::Instant;

use crate::telemetry_session::{
    TelemetryCloseMetadataPayload, TelemetryWriterCounterSnapshot, build_telemetry_close_metadata,
};

use super::file::{build_telemetry_file_writer, normalize_event_cap};
use super::shared::{clear_shared_telemetry_writer, replace_shared_telemetry_writer};
use super::{TelemetryWriterFactory, TelemetryWriterGuard};

pub struct TelemetrySessionWriter {
    path: PathBuf,
    writer: Option<TelemetryWriterFactory>,
    guard: Option<TelemetryWriterGuard>,
    last_counter_snapshot: Option<TelemetryWriterCounterSnapshot>,
}

impl TelemetrySessionWriter {
    /// Open a telemetry stream and register it for shared logging reuse.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when the telemetry file cannot be opened or the
    /// shared-writer registry cannot be updated.
    pub fn new(path: PathBuf, log_queue_size: usize, log_lossy: bool, event_cap: Option<usize>) -> io::Result<Self> {
        let (writer, guard) =
            build_telemetry_file_writer(&path, log_queue_size, log_lossy, normalize_event_cap(event_cap))?;
        replace_shared_telemetry_writer(path.clone(), writer.clone())?;
        Ok(Self { path, writer: Some(writer), guard: Some(guard), last_counter_snapshot: None })
    }

    /// Write one JSON line to the telemetry stream when it is still open.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when the writer cannot append the line or when a
    /// lossless capped writer has reached its event cap.
    pub fn write_json_line(&self, json_line: &str) -> io::Result<()> {
        let Some(writer) = self.writer.as_ref() else {
            return Ok(());
        };
        writer.write_json_line(json_line)
    }

    #[must_use]
    pub fn counter_snapshot(&self) -> TelemetryWriterCounterSnapshot {
        if let Some(writer) = self.writer.as_ref() {
            return writer.counter_snapshot(None);
        }
        self.last_counter_snapshot.clone().unwrap_or_else(TelemetryWriterCounterSnapshot::empty)
    }

    #[must_use]
    pub fn close_metadata(&self) -> Option<TelemetryCloseMetadataPayload> {
        self.last_counter_snapshot.clone().map(build_telemetry_close_metadata)
    }

    /// Close the writer, flush buffered telemetry, and return final counters.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when the shared-writer registry cannot be cleared or
    /// when a lossless capped telemetry writer exceeded its event cap.
    pub fn finish_counter_snapshot(&mut self) -> io::Result<TelemetryWriterCounterSnapshot> {
        let finish_start_time = Instant::now();
        let dropped_writer = self.writer.take();
        let dropped_guard = self.guard.take();
        drop(dropped_guard);
        let finish_flush_duration_seconds = finish_start_time.elapsed().as_secs_f64();
        clear_shared_telemetry_writer(&self.path)?;
        let Some(writer) = dropped_writer.as_ref() else {
            return Ok(self.counter_snapshot());
        };

        let counter_snapshot = writer.counter_snapshot(Some(finish_flush_duration_seconds));
        self.last_counter_snapshot = Some(counter_snapshot.clone());
        writer.fail_if_lossless_cap_exceeded()?;
        Ok(counter_snapshot)
    }

    /// Close the writer and return telemetry close metadata.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when writer finalization fails.
    pub fn finish_close_metadata(&mut self) -> io::Result<TelemetryCloseMetadataPayload> {
        self.finish_counter_snapshot().map(build_telemetry_close_metadata)
    }
}
