#[cfg(not(target_pointer_width = "64"))]
compile_error!("g requires a 64-bit target.");

mod api;
mod error;
mod regenie;
mod sample;

pub use api::*;
