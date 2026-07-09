//! Public output crate facade.

pub use crate::chunk::{NativeChunkHandle, NativeChunkStats, VariantMetadataColumns};
pub use crate::error::{OutputError, OutputResult};
pub use crate::finalization::finalize_output_run_chunks;
pub use crate::manifest::{
    CurrentRunManifestHeaderInput, InitializedOutputRun, ManifestFileFingerprint, ManifestFileFingerprintCache,
    OutputResumeMode, OutputRunPaths, PreparedOutputRun, RunManifestChunkCommit,
    build_current_run_manifest_header_json, build_current_run_manifest_header_json_with_cache,
    build_file_content_sha256, build_manifest_file_fingerprint, build_manifest_json_sha256,
    build_prepared_run_manifest_header_json, build_prepared_run_manifest_header_json_from_current_header_json,
    build_prepared_run_plan_from_current_header_json, build_prepared_run_plan_json_from_current_header_json,
    extend_run_manifest_metadata, initialize_output_run, load_run_manifest_json, prepare_output_run,
    read_run_manifest_committed_chunk_identifiers_from_text, resolve_output_run_paths,
    validate_output_run_resume_compatibility, validate_run_manifest_compatibility, write_run_manifest_json,
};
pub use crate::resume::{
    repair_strict_manifest_chunk_commits, scan_committed_chunk_identifiers, validate_strict_manifest_chunks,
};
pub use crate::session::{
    OutputWriterSession, create_output_writer_sessions, finish_interrupted_output_writer_sessions,
    finish_output_writer_sessions, validate_trait_major_statistic_shape,
};
pub use crate::write_plan::{
    MultiTraitOutputWritePlan, Regenie2StatisticSliceBundle, SingleTraitOutputWritePlan, WriterFinishExecutionPlan,
    finish_interrupted_output_writer_sessions_with_requested_threads,
    finish_output_writer_sessions_with_requested_threads, plan_multi_trait_output_write,
    plan_single_trait_output_write, plan_writer_finish_execution, resolve_writer_finish_thread_count,
    write_regenie2_multi_trait_chunk_f32, write_regenie2_multi_trait_chunk_f64,
};
pub use crate::writer::OutputFileFormat;
