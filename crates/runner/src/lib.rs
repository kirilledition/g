//! Python-free native CLI lifecycle coordination.

#![warn(clippy::pedantic)]

mod api;
mod backend_plan;
mod cli_output;
mod jax_runtime;
mod native_session_policy;
mod run;

pub use api::*;
