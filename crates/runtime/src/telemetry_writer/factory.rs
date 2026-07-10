use std::io::{self, Write as _};
use std::sync::Arc;

use tracing_appender::non_blocking::NonBlocking;
use tracing_subscriber::fmt::writer::MakeWriter;

use crate::telemetry_session::{TelemetryEventCapState, TelemetryWriterCounterSnapshot};

use super::line::TelemetryLineWriter;

#[derive(Clone)]
pub struct TelemetryWriterFactory {
    writer: NonBlocking,
    event_cap_state: Arc<TelemetryEventCapState>,
}

impl TelemetryWriterFactory {
    #[must_use]
    pub fn new(writer: NonBlocking, event_cap_state: TelemetryEventCapState) -> Self {
        Self { writer, event_cap_state: Arc::new(event_cap_state) }
    }

    /// Write one JSON line to the telemetry stream.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when the writer cannot append the line or when a
    /// lossless capped writer has reached its event cap.
    pub fn write_json_line(&self, json_line: &str) -> io::Result<()> {
        let mut line_writer = self.make_writer();
        line_writer.write_all(json_line.as_bytes())?;
        if !json_line.ends_with('\n') {
            line_writer.write_all(b"\n")?;
        }
        line_writer.flush()
    }

    /// Return an error if a lossless capped writer exceeded its event cap.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when the event cap was exceeded and lossy mode is
    /// disabled.
    pub fn fail_if_lossless_cap_exceeded(&self) -> io::Result<()> {
        if self.event_cap_state.should_fail_for_cap_exceeded() {
            return Err(io::Error::other(self.event_cap_state.cap_exceeded_error_message()));
        }
        Ok(())
    }

    #[must_use]
    pub fn counter_snapshot(&self) -> TelemetryWriterCounterSnapshot {
        self.event_cap_state.counter_snapshot(self.writer.error_counter().dropped_lines())
    }
}

impl<'a> MakeWriter<'a> for TelemetryWriterFactory {
    type Writer = TelemetryLineWriter;

    fn make_writer(&'a self) -> Self::Writer {
        TelemetryLineWriter::new(self.writer.clone(), Arc::clone(&self.event_cap_state))
    }
}
