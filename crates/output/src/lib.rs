//! Native output persistence APIs.

#[cfg(not(target_pointer_width = "64"))]
compile_error!("g requires a 64-bit target.");

mod api;
mod chunk;
mod error;
mod manager;
mod manifest;
mod resume;
mod schema;
mod session;
mod timing;
mod write_plan;
mod writer;

pub use api::*;
