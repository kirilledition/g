//! Genotype reader contracts and format-specific implementations.

#[cfg(not(target_pointer_width = "64"))]
compile_error!("g requires a 64-bit target.");

mod api;
mod bgen;
mod common;
mod error;
mod planner;
mod preprocess;

pub use api::*;
