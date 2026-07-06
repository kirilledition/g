#![warn(clippy::pedantic)]

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
#[cfg(any(test, feature = "test-support"))]
mod fake_backend;
#[cfg(any(test, feature = "test-support"))]
mod fake_effects;
mod output_manifest;
mod output_schedule;
mod phase;
mod pipeline;
mod preflight;
mod preparation;
mod schedule;
#[cfg(any(test, feature = "test-support"))]
pub mod test_support;
mod trusted_validation;

pub use api::*;
