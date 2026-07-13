use std::fs::{self, OpenOptions};
use std::io;
use std::path::Path;

use tracing_appender::non_blocking::{NonBlocking, NonBlockingBuilder};

use crate::telemetry_session::TelemetryEventCounterState;

use super::{TelemetryWriterFactory, TelemetryWriterGuard};

pub(crate) fn build_non_blocking_writer<Writer>(
    writer: Writer,
    thread_name: &str,
    log_queue_size: usize,
    log_lossy: bool,
) -> (NonBlocking, TelemetryWriterGuard)
where
    Writer: std::io::Write + Send + 'static,
{
    let (writer, guard) = NonBlockingBuilder::default()
        .lossy(log_lossy)
        .buffered_lines_limit(log_queue_size)
        .thread_name(thread_name)
        .finish(writer);
    (writer, guard)
}

/// Open a log file and wrap it in a non-blocking writer.
///
/// # Errors
///
/// Returns an I/O error when parent directory creation or file opening fails.
pub(crate) fn build_log_file_writer(
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

/// Open a counted telemetry file writer.
///
/// # Errors
///
/// Returns an I/O error when file creation fails.
pub(super) fn build_telemetry_file_writer(
    path: &Path,
    log_queue_size: usize,
    log_lossy: bool,
) -> io::Result<(TelemetryWriterFactory, TelemetryWriterGuard)> {
    let (writer, guard) = build_log_file_writer(path, log_queue_size, log_lossy)?;
    let event_counter_state = TelemetryEventCounterState::new(log_lossy);
    Ok((TelemetryWriterFactory::new(writer, event_counter_state), guard))
}
