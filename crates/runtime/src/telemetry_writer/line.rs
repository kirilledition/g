use std::io;
use std::sync::Arc;

use tracing_appender::non_blocking::NonBlocking;

use crate::telemetry_session::TelemetryEventCounterState;

pub(crate) struct TelemetryLineWriter {
    pub(super) writer: NonBlocking,
    pub(super) event_counter_state: Arc<TelemetryEventCounterState>,
}

impl io::Write for TelemetryLineWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.event_counter_state.record_event_count(memchr::memchr_iter(b'\n', buffer).count());
        self.writer.write_all(buffer)?;
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}
