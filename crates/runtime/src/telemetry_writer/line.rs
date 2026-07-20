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

#[cfg(test)]
mod tests {
    use std::io::Write as _;
    use std::sync::Mutex;

    use super::*;
    use crate::telemetry_writer::build_non_blocking_writer;

    #[derive(Clone)]
    struct SharedBufferWriter {
        bytes: Arc<Mutex<Vec<u8>>>,
    }

    impl io::Write for SharedBufferWriter {
        fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
            self.bytes.lock().expect("shared buffer should be available").extend_from_slice(buffer);
            Ok(buffer.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn line_writer_counts_complete_records_across_writes() {
        let bytes = Arc::new(Mutex::new(Vec::new()));
        let (writer, guard) = build_non_blocking_writer(
            SharedBufferWriter { bytes: Arc::clone(&bytes) },
            "g-runtime-line-test",
            16,
            false,
        );
        let event_counter_state = Arc::new(TelemetryEventCounterState::new(false));
        let mut line_writer = TelemetryLineWriter { writer, event_counter_state: Arc::clone(&event_counter_state) };

        assert_eq!(line_writer.write(b"one\npartial").expect("first buffer should write"), 11);
        assert_eq!(line_writer.write(b"\ntwo\n").expect("second buffer should write"), 5);
        line_writer.flush().expect("line writer should flush");
        drop(line_writer);
        drop(guard);

        assert_eq!(&*bytes.lock().expect("shared buffer should be available"), b"one\npartial\ntwo\n");
        assert_eq!(event_counter_state.counter_snapshot(0).accepted_event_count, 3);
    }
}
