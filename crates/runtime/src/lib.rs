#![warn(clippy::pedantic)]

#[cfg(not(target_pointer_width = "64"))]
compile_error!("g requires a 64-bit target.");

mod api;
mod diagnostics;
mod error;
mod logging_sink;
mod native_run_session;
mod rayon_runtime;
mod runtime_policy;
mod runtime_state;
mod shutdown;
mod telemetry_session;
mod telemetry_writer;
mod timing;

pub use api::*;
