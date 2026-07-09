#![allow(clippy::missing_errors_doc)]

mod chunks;
mod contract;
mod fingerprint;
mod header;
mod run;
mod validation;

pub use chunks::RunManifestChunkCommit;
pub use contract::{
    build_prepared_run_manifest_header_json, build_prepared_run_manifest_header_json_from_current_header_json,
    build_prepared_run_plan_from_current_header_json, build_prepared_run_plan_json_from_current_header_json,
};
pub use fingerprint::{
    ManifestFileFingerprint, ManifestFileFingerprintCache, build_file_content_sha256, build_manifest_file_fingerprint,
    build_manifest_json_sha256,
};
pub(crate) use fingerprint::{build_manifest_value_sha256, manifest_file_fingerprint_to_value};
pub use header::{
    CurrentRunManifestHeaderInput, build_current_run_manifest_header_json,
    build_current_run_manifest_header_json_with_cache,
};
pub use run::{
    InitializedOutputRun, OutputResumeMode, OutputRunPaths, PreparedOutputRun, extend_run_manifest_metadata,
    initialize_output_run, load_run_manifest_json, prepare_output_run,
    read_run_manifest_committed_chunk_identifiers_from_text, resolve_output_run_paths,
    validate_output_run_resume_compatibility, validate_run_manifest_compatibility, write_run_manifest_json,
};
pub(crate) use run::{
    mark_run_manifest_finalized, mark_run_manifest_finalized_output, mark_run_manifest_interrupted,
    read_run_manifest_chunk_commits, read_run_manifest_chunk_commits_from_text, record_run_manifest_chunk_commits,
};

const RUN_MANIFEST_FILE_NAME: &str = "run_manifest.json";
const RUN_MANIFEST_SCHEMA_VERSION: i64 = 9;
const OUTPUT_SCHEMA_VERSION: i64 = 2;
const JAX_MATMUL_PRECISION_WHEN_UNSET: &str = "float32";
const RESUME_POLICY: &str = "manifest_committed_chunks";
