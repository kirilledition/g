#![warn(clippy::pedantic)]

#[cfg(not(target_pointer_width = "64"))]
compile_error!("g requires a 64-bit target.");

mod api;
mod cli_runtime;
mod error;
mod jax_runtime;
mod logging_sink;
mod rayon_runtime;
mod run_events;
mod run_metadata;
mod runtime_paths;
mod runtime_policy;
mod runtime_state;
mod shutdown;
mod telemetry_policy;
mod telemetry_session;
mod telemetry_writer;
mod timing;
mod trusted_validation;

pub use api::*;
