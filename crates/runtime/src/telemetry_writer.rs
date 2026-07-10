//! Runtime-owned telemetry file writer and shared stream state.

mod factory;
mod file;
mod line;
mod session;
mod shared;

pub(crate) use factory::TelemetryWriterFactory;
pub(crate) use file::{build_non_blocking_writer, normalize_event_cap};
pub(crate) use session::TelemetrySessionWriter;
pub(crate) use shared::build_shared_or_log_file_writer;

use tracing_appender::non_blocking::WorkerGuard;

pub(crate) type TelemetryWriterGuard = WorkerGuard;
