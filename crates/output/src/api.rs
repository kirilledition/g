//! Public output crate facade.

pub use crate::chunk::{NativeChunkHandle, NativeChunkStats, VariantMetadataColumns};
pub use crate::error::{OutputError, OutputResult};
pub use crate::finalization::finalize_output_run_chunks;
pub use crate::manifest::{
    InitializedOutputRun, OutputResumeMode, OutputRunPaths, PreparedOutputRun, initialize_output_run,
    prepare_output_run, validate_run_manifest_compatibility,
};
pub use crate::session::OutputWriterSession;
pub use crate::writer::OutputFileFormat;
