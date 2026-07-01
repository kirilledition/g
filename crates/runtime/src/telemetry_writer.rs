//! Runtime-owned telemetry file writer and shared stream state.

use std::fs::{self, OpenOptions};
use std::io::{self, Write as _};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use tracing_appender::non_blocking::{NonBlocking, NonBlockingBuilder, WorkerGuard};
use tracing_subscriber::fmt::writer::MakeWriter;

use crate::telemetry_session::{
    TelemetryCapAction, TelemetryCloseMetadataPayload, TelemetryEventCapState, TelemetryWriterCounterSnapshot,
    build_telemetry_close_metadata,
};

static TELEMETRY_WRITER: Mutex<Option<SharedTelemetryWriter>> = Mutex::new(None);

pub type TelemetryWriterGuard = WorkerGuard;

#[derive(Clone)]
struct SharedTelemetryWriter {
    path: PathBuf,
    writer: TelemetryWriterFactory,
}

#[derive(Clone)]
pub struct TelemetryWriterFactory {
    writer: NonBlocking,
    event_cap_state: Arc<TelemetryEventCapState>,
}

pub struct TelemetryLineWriter {
    writer: NonBlocking,
    event_cap_state: Arc<TelemetryEventCapState>,
    line_buffer: Vec<u8>,
}

pub struct TelemetrySessionWriter {
    path: PathBuf,
    writer: Option<TelemetryWriterFactory>,
    guard: Option<TelemetryWriterGuard>,
    last_counter_snapshot: Option<TelemetryWriterCounterSnapshot>,
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
    pub fn counter_snapshot(&self, finish_flush_duration_seconds: Option<f64>) -> TelemetryWriterCounterSnapshot {
        self.event_cap_state
            .counter_snapshot(self.writer.error_counter().dropped_lines(), finish_flush_duration_seconds)
    }
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

impl<'a> MakeWriter<'a> for TelemetryWriterFactory {
    type Writer = TelemetryLineWriter;

    fn make_writer(&'a self) -> Self::Writer {
        TelemetryLineWriter {
            writer: self.writer.clone(),
            event_cap_state: Arc::clone(&self.event_cap_state),
            line_buffer: Vec::new(),
        }
    }
}

impl TelemetryLineWriter {
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

#[must_use]
pub fn normalize_event_cap(event_cap: Option<usize>) -> Option<usize> {
    event_cap.filter(|cap| *cap > 0)
}

pub fn build_non_blocking_writer<Writer>(
    writer: Writer,
    thread_name: &str,
    log_queue_size: usize,
    log_lossy: bool,
) -> (NonBlocking, TelemetryWriterGuard)
where
    Writer: std::io::Write + Send + 'static,
{
    NonBlockingBuilder::default()
        .lossy(log_lossy)
        .buffered_lines_limit(log_queue_size)
        .thread_name(thread_name)
        .finish(writer)
}

/// Open a log file and wrap it in a non-blocking writer.
///
/// # Errors
///
/// Returns an I/O error when parent directory creation or file opening fails.
pub fn build_log_file_writer(
    path: &Path,
    log_queue_size: usize,
    log_lossy: bool,
) -> io::Result<(NonBlocking, TelemetryWriterGuard)> {
    if let Some(parent_directory) = path.parent().filter(|parent_directory| !parent_directory.as_os_str().is_empty()) {
        fs::create_dir_all(parent_directory)?;
    }
    let log_file = OpenOptions::new().create(true).append(true).open(path)?;
    Ok(build_non_blocking_writer(log_file, "g-tracing-file", log_queue_size, log_lossy))
}

/// Open a capped telemetry file writer.
///
/// # Errors
///
/// Returns an I/O error when file creation fails.
pub fn build_telemetry_file_writer(
    path: &Path,
    log_queue_size: usize,
    log_lossy: bool,
    event_cap: Option<usize>,
) -> io::Result<(TelemetryWriterFactory, TelemetryWriterGuard)> {
    let (writer, guard) = build_log_file_writer(path, log_queue_size, log_lossy)?;
    let event_cap_state = TelemetryEventCapState::new(path, event_cap, log_lossy);
    Ok((TelemetryWriterFactory::new(writer, event_cap_state), guard))
}

/// Reuse the shared telemetry writer when logging and telemetry target the same file.
///
/// # Errors
///
/// Returns an I/O error when the shared writer lock is poisoned or file
/// creation fails.
pub fn build_shared_or_log_file_writer(
    path: &Path,
    log_queue_size: usize,
    log_lossy: bool,
    event_cap: Option<usize>,
) -> io::Result<(TelemetryWriterFactory, Option<TelemetryWriterGuard>)> {
    if let Some(shared_writer) = shared_telemetry_writer_for_path(path)? {
        return Ok((shared_writer, None));
    }
    let (writer, guard) = build_telemetry_file_writer(path, log_queue_size, log_lossy, event_cap)?;
    Ok((writer, Some(guard)))
}

/// Find the active shared telemetry writer for a path.
///
/// # Errors
///
/// Returns an I/O error when the shared writer lock is poisoned.
pub fn shared_telemetry_writer_for_path(path: &Path) -> io::Result<Option<TelemetryWriterFactory>> {
    let normalized_path = normalize_path_for_comparison(path);
    let telemetry_writer = lock_telemetry_writer()?;
    Ok(telemetry_writer
        .as_ref()
        .filter(|shared_writer| normalize_path_for_comparison(&shared_writer.path) == normalized_path)
        .map(|shared_writer| shared_writer.writer.clone()))
}

/// Replace the shared telemetry writer.
///
/// # Errors
///
/// Returns an I/O error when the shared writer lock is poisoned.
pub fn replace_shared_telemetry_writer(path: PathBuf, writer: TelemetryWriterFactory) -> io::Result<()> {
    let mut telemetry_writer = lock_telemetry_writer()?;
    *telemetry_writer = Some(SharedTelemetryWriter { path, writer });
    Ok(())
}

/// Clear the shared telemetry writer for a path.
///
/// # Errors
///
/// Returns an I/O error when the shared writer lock is poisoned.
pub fn clear_shared_telemetry_writer(path: &Path) -> io::Result<()> {
    let normalized_path = normalize_path_for_comparison(path);
    let mut telemetry_writer = lock_telemetry_writer()?;
    if telemetry_writer
        .as_ref()
        .is_some_and(|shared_writer| normalize_path_for_comparison(&shared_writer.path) == normalized_path)
    {
        let _dropped_writer = telemetry_writer.take();
    }
    Ok(())
}

fn lock_telemetry_writer() -> io::Result<std::sync::MutexGuard<'static, Option<SharedTelemetryWriter>>> {
    TELEMETRY_WRITER.lock().map_err(|_| io::Error::other("Telemetry writer mutex was poisoned."))
}

fn normalize_path_for_comparison(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

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
