#![warn(clippy::pedantic)]

#[cfg(not(target_pointer_width = "64"))]
compile_error!("g requires a 64-bit target.");

mod api;
mod association_scheduler;
mod backend;
mod delivery_schedule;
mod null_logistic_policy;
mod output_manifest;
mod output_schedule;
mod pipeline;
mod preflight;
mod preparation;
mod schedule_error;
mod trusted_validation;

pub use api::*;
