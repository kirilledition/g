use std::io::{self, Write as _};
use std::sync::Arc;

use tracing_appender::non_blocking::NonBlocking;
use tracing_subscriber::fmt::writer::MakeWriter;

use crate::telemetry_session::{TelemetryEventCounterState, TelemetryWriterCounterSnapshot};

use super::line::TelemetryLineWriter;

#[derive(Clone)]
pub struct TelemetryWriterFactory {
    writer: NonBlocking,
    event_counter_state: Arc<TelemetryEventCounterState>,
}

impl TelemetryWriterFactory {
    #[must_use]
    pub fn new(writer: NonBlocking, event_counter_state: TelemetryEventCounterState) -> Self {
        Self { writer, event_counter_state: Arc::new(event_counter_state) }
    }

    /// Write one JSON line to the telemetry stream.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when the writer cannot append the line.
    pub fn write_json_line(&self, json_line: &str) -> io::Result<()> {
        debug_assert!(json_line.ends_with('\n'), "serialized telemetry records must end with a newline");
        let mut line_writer = self.make_writer();
        line_writer.write_all(json_line.as_bytes())
    }

    #[must_use]
    pub fn counter_snapshot(&self) -> TelemetryWriterCounterSnapshot {
        self.event_counter_state.counter_snapshot(self.writer.error_counter().dropped_lines())
    }
}

impl<'a> MakeWriter<'a> for TelemetryWriterFactory {
    type Writer = TelemetryLineWriter;

    fn make_writer(&'a self) -> Self::Writer {
        TelemetryLineWriter { writer: self.writer.clone(), event_counter_state: Arc::clone(&self.event_counter_state) }
    }
}
