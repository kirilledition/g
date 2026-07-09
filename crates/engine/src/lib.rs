#![warn(clippy::pedantic)]

#[cfg(not(target_pointer_width = "64"))]
compile_error!("g requires a 64-bit target.");

mod api;
mod backend;
mod callback_diagnostics;
mod callback_observation_schedule;
mod callback_progress;
mod callback_queue;
mod callback_summary;
mod callback_worker_schedule;
mod coordinator;
mod delivery_schedule;
mod effects;
mod error;
mod output_manifest;
mod output_schedule;
mod phase;
mod pipeline;
mod preflight;
mod preparation;
mod schedule;
mod trusted_validation;

pub use api::*;
