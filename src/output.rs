//! Native output persistence APIs.

pub mod writer;

pub use writer::{PyOutputWriterSession, finalize_output_run_chunks, scan_committed_chunk_identifiers};
