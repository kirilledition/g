#[cfg(not(target_pointer_width = "64"))]
compile_error!("g requires a 64-bit target.");

mod api;
mod enums;
mod host_policy;
mod numeric;
mod request;

/// Smallest total approximate-Firth budget whose floor split gives both phases two iterations.
pub const APPROXIMATE_FIRTH_MINIMUM_TOTAL_ITERATIONS: u32 = 4;

pub use api::*;
