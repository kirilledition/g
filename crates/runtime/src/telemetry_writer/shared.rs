use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use super::file::build_telemetry_file_writer;
use super::{TelemetryWriterFactory, TelemetryWriterGuard};

static TELEMETRY_WRITER: Mutex<Option<SharedTelemetryWriter>> = Mutex::new(None);

#[derive(Clone)]
struct SharedTelemetryWriter {
    path: PathBuf,
    writer: TelemetryWriterFactory,
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
