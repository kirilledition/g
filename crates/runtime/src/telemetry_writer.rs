//! Runtime-owned telemetry file writer and shared stream state.

mod factory;
mod file;
mod line;
mod session;
mod shared;

pub use factory::TelemetryWriterFactory;
pub use file::{build_log_file_writer, build_non_blocking_writer, build_telemetry_file_writer, normalize_event_cap};
pub use line::TelemetryLineWriter;
pub use session::TelemetrySessionWriter;
pub use shared::{
    build_shared_or_log_file_writer, clear_shared_telemetry_writer, replace_shared_telemetry_writer,
    shared_telemetry_writer_for_path,
};

use tracing_appender::non_blocking::WorkerGuard;

pub type TelemetryWriterGuard = WorkerGuard;

#[cfg(test)]
use std::path::PathBuf;

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    static NEXT_TEST_FILE_ID: AtomicUsize = AtomicUsize::new(0);

    fn telemetry_test_path(test_name: &str) -> PathBuf {
        let file_id = NEXT_TEST_FILE_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("g-{test_name}-{}-{file_id}.jsonl", std::process::id()))
    }

    #[test]
    fn telemetry_event_cap_fails_without_lossy_mode() {
        let path = telemetry_test_path("telemetry-cap-fails");
        let (telemetry_writer, guard) =
            build_telemetry_file_writer(&path, 32, false, Some(2)).expect("writer should build");

        telemetry_writer.write_json_line(r#"{"event":"first"}"#).expect("first event should write");
        telemetry_writer.write_json_line(r#"{"event":"second"}"#).expect("second event should write");
        let error =
            telemetry_writer.write_json_line(r#"{"event":"third"}"#).expect_err("third event should exceed cap");

        assert!(error.to_string().contains("Trace telemetry event cap exceeded at 2 events"));
        assert!(telemetry_writer.fail_if_lossless_cap_exceeded().is_err());
        drop(telemetry_writer);
        drop(guard);

        let line_count = fs::read_to_string(&path).expect("telemetry file should be readable").lines().count();
        assert_eq!(line_count, 2);
        fs::remove_file(path).expect("telemetry test file should be removed");
    }

    #[test]
    fn telemetry_event_cap_drops_with_lossy_mode() {
        let path = telemetry_test_path("telemetry-cap-drops");
        let (telemetry_writer, guard) =
            build_telemetry_file_writer(&path, 32, true, Some(1)).expect("writer should build");

        telemetry_writer.write_json_line(r#"{"event":"first"}"#).expect("first event should write");
        telemetry_writer.write_json_line(r#"{"event":"second"}"#).expect("second event should drop");
        telemetry_writer.write_json_line(r#"{"event":"third"}"#).expect("third event should drop");
        assert!(telemetry_writer.fail_if_lossless_cap_exceeded().is_ok());
        drop(telemetry_writer);
        drop(guard);

        let telemetry_text = fs::read_to_string(&path).expect("telemetry file should be readable");
        assert_eq!(telemetry_text.lines().count(), 1);
        assert!(telemetry_text.contains(r#""event":"first""#));
        fs::remove_file(path).expect("telemetry test file should be removed");
    }

    #[test]
    fn telemetry_event_cap_zero_disables_cap() {
        let path = telemetry_test_path("telemetry-cap-disabled");
        let (telemetry_writer, guard) =
            build_telemetry_file_writer(&path, 32, false, normalize_event_cap(Some(0))).expect("writer should build");

        telemetry_writer.write_json_line(r#"{"event":"first"}"#).expect("first event should write");
        telemetry_writer.write_json_line(r#"{"event":"second"}"#).expect("second event should write");
        telemetry_writer.write_json_line(r#"{"event":"third"}"#).expect("third event should write");
        assert!(telemetry_writer.fail_if_lossless_cap_exceeded().is_ok());
        drop(telemetry_writer);
        drop(guard);

        let line_count = fs::read_to_string(&path).expect("telemetry file should be readable").lines().count();
        assert_eq!(line_count, 3);
        fs::remove_file(path).expect("telemetry test file should be removed");
    }

    #[test]
    fn telemetry_session_writer_finishes_and_clears_shared_writer() {
        let path = telemetry_test_path("telemetry-session-writer-finish");
        let mut telemetry_session_writer =
            TelemetrySessionWriter::new(path.clone(), 32, true, Some(2)).expect("session writer should build");

        assert!(shared_telemetry_writer_for_path(&path).expect("shared writer lookup should succeed").is_some());
        telemetry_session_writer.write_json_line(r#"{"event":"first"}"#).expect("first event should write");
        telemetry_session_writer.write_json_line(r#"{"event":"second"}"#).expect("second event should write");
        telemetry_session_writer.write_json_line(r#"{"event":"third"}"#).expect("third event should drop");

        let close_metadata = telemetry_session_writer.finish_close_metadata().expect("writer should finish");

        assert_eq!(telemetry_session_writer.close_metadata(), Some(close_metadata.clone()));
        assert!(shared_telemetry_writer_for_path(&path).expect("shared writer lookup should succeed").is_none());
        assert_eq!(close_metadata.writer_counters.accepted_event_count, 2);
        assert_eq!(close_metadata.writer_counters.cap_dropped_event_count, 1);
        assert!(close_metadata.writer_counters.finish_flush_duration_seconds.is_some());

        let telemetry_text = fs::read_to_string(&path).expect("telemetry file should be readable");
        assert_eq!(telemetry_text.lines().count(), 2);
        fs::remove_file(path).expect("telemetry test file should be removed");
    }
}
