//! Canonical data-plane contracts shared by genotype producers and output consumers.

#[cfg(not(target_pointer_width = "64"))]
compile_error!("g requires a 64-bit target.");

mod api;
mod metadata;
mod statistics;

pub use api::*;
