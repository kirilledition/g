#[cfg(not(target_pointer_width = "64"))]
compile_error!("g requires a 64-bit target.");

mod api;
mod enums;
mod error;
mod host_policy;
mod numeric;
mod request;

pub use api::*;
