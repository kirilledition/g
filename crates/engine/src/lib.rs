#![warn(clippy::pedantic)]

#[cfg(not(target_pointer_width = "64"))]
compile_error!("g requires a 64-bit target.");

mod api;
mod association_scheduler;
mod backend;
mod delivery;
mod delivery_execution;
mod genotype_buffer;
mod null_logistic_policy;
mod output_manifest;
mod output_schedule;
mod output_write;
mod pipeline;
mod preflight;
mod preparation;
mod progress;
mod run;
mod run_coordinator;
mod trusted_validation;

pub use api::*;
