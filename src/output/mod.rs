//! Native output persistence APIs.

mod finalization;
mod manifest;
mod resume;
mod schema;
mod session;
pub mod writer;

pub use finalization::finalize_output_run_chunks;
pub(crate) use manifest::{
    CurrentRunManifestHeaderInput, OutputResumeMode, build_current_run_manifest_header_json, initialize_output_run,
    load_run_manifest_json, prepare_output_run, read_run_manifest_committed_chunk_identifiers_from_text,
    resolve_output_run_paths, validate_run_manifest_compatibility, write_run_manifest_json,
};
pub(crate) use resume::repair_strict_manifest_chunk_commits;
pub use resume::{scan_committed_chunk_identifiers, validate_strict_manifest_chunks};
pub(crate) use schema::OutputStatisticDtype;
pub(crate) use session::NativeChunkHandle;
pub use session::OutputWriterSession;
pub(crate) use writer::OutputFileFormat;
pub use writer::OutputWriterError;
