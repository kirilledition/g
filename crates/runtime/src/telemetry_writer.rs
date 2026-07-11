//! Runtime-owned telemetry file writer and shared stream state.

mod factory;
mod file;
mod line;
mod session;
mod shared;

pub(crate) use factory::TelemetryWriterFactory;
pub(crate) use file::{build_log_file_writer, build_non_blocking_writer};
pub(crate) use session::TelemetrySessionWriter;
pub(crate) use shared::{
    SharedLogWriterFactory, SharedLogWriterKind, SharedTelemetryWriterFactory, register_shared_log_writer,
    unregister_shared_log_writer,
};

use tracing_appender::non_blocking::WorkerGuard;

pub(crate) type TelemetryWriterGuard = WorkerGuard;
