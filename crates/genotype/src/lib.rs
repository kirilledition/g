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

/// Maximum source size retained as an immutable owned BGEN snapshot.
#[cfg(feature = "benchmark-internals")]
#[doc(hidden)]
pub const BGEN_OWNED_SNAPSHOT_MAXIMUM_BYTE_COUNT: u64 = bgen::MAXIMUM_OWNED_SNAPSHOT_BYTE_COUNT;
