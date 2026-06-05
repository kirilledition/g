//! Native output persistence APIs.

mod finalization;
mod manifest;
mod resume;
mod schema;
mod session;
pub mod writer;

pub use finalization::finalize_output_run_chunks;
#[cfg(feature = "python")]
pub(crate) use resume::repair_strict_manifest_chunk_commits;
pub use resume::{scan_committed_chunk_identifiers, validate_strict_manifest_chunks};
pub(crate) use session::NativeChunkHandle;
pub use session::OutputWriterSession;
pub use writer::OutputWriterError;
