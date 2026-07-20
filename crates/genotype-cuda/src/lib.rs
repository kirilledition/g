//! CUDA genotype delivery through the stable XLA foreign-function interface.

#[cfg(not(target_pointer_width = "64"))]
compile_error!("g requires a 64-bit target.");

mod api;

pub use api::*;
