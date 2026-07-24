//! Native output persistence APIs.

#[cfg(not(target_pointer_width = "64"))]
compile_error!("g requires a 64-bit target.");

mod agreement;
mod api;
mod association_implementation;
mod chunk;
mod digest;
mod error;
mod genotype_delivery_execution;
mod manager;
mod manifest;
mod persistence;
mod schema;
mod session;
mod timing;
mod write_plan;
mod writer;

#[cfg(test)]
mod tests;

pub use api::*;

pub(crate) const CHUNKS_PER_PARQUET_FILE: usize = 8;
pub(crate) const WRITER_QUEUE_DEPTH: usize = 16;
