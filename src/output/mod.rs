//! Native output persistence APIs.

pub mod writer;

pub use writer::{NativeOutputWriterSession, NativeRegenieStep2Chunk, finalize_native_output_run_chunks};

#[cfg(feature = "python")]
pub use writer::{OutputWriterSession, finalize_output_run_chunks, scan_committed_chunk_identifiers};
