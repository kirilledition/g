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
