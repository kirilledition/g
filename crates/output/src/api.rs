//! Public output crate facade.

pub use crate::chunk::{NativeChunkHandle, NativeChunkStats, VariantMetadataColumns};
pub use crate::error::{OutputError, OutputResult};
pub use crate::manifest::{
    CurrentRunManifestHeaderInput, ManifestFileFingerprint, ManifestFileFingerprintCache, OutputResumeMode,
    OutputRunPaths, PreparedOutputRun, build_current_run_manifest_header_json_with_cache, extend_run_manifest_metadata,
    initialize_output_run, prepare_output_run, validate_output_run_resume_compatibility,
};
pub use crate::session::{
    OutputWriterSession, create_output_writer_sessions, finish_interrupted_output_writer_sessions,
    finish_output_writer_sessions,
};
pub use crate::write_plan::{
    Regenie2StatisticSliceBundle, finish_interrupted_output_writer_sessions_with_requested_threads,
    finish_output_writer_sessions_with_requested_threads, write_regenie2_multi_trait_chunk_f32,
    write_regenie2_multi_trait_chunk_f64,
};
pub use crate::writer::OutputFileFormat;
