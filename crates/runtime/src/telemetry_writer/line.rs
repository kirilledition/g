use std::io::{self, Write as _};
use std::sync::Arc;

use tracing_appender::non_blocking::NonBlocking;

use crate::telemetry_session::{TelemetryCapAction, TelemetryEventCapState};

pub struct TelemetryLineWriter {
    writer: NonBlocking,
    event_cap_state: Arc<TelemetryEventCapState>,
    line_buffer: Vec<u8>,
}

impl TelemetryLineWriter {
    #[must_use]
    pub fn new(writer: NonBlocking, event_cap_state: Arc<TelemetryEventCapState>) -> Self {
        Self { writer, event_cap_state, line_buffer: Vec::new() }
    }

    fn write_complete_line(&mut self, line: &[u8]) -> io::Result<()> {
        match self.event_cap_state.reserve_event()? {
            TelemetryCapAction::Write => self.writer.write_all(line),
            TelemetryCapAction::Drop => Ok(()),
        }
    }
}

impl io::Write for TelemetryLineWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        if !self.event_cap_state.has_event_cap() {
            let event_count = memchr::memchr_iter(b'\n', buffer).count();
            self.event_cap_state.record_uncapped_event_count(event_count);
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
