#![allow(clippy::missing_errors_doc)]

mod chunks;
mod fingerprint;
mod header;
mod run;
mod validation;

pub(crate) use chunks::RunManifestChunkCommit;
pub use fingerprint::{ManifestFileFingerprint, ManifestFileFingerprintCache};
pub(crate) use fingerprint::{build_manifest_value_sha256, manifest_file_fingerprint_to_value};
pub use header::{CurrentRunManifestHeaderInput, build_current_run_manifest_header_json_with_cache};
pub use run::{
    OutputResumeMode, OutputRunPaths, PreparedOutputRun, extend_run_manifest_metadata, initialize_output_run,
    prepare_output_run, validate_output_run_resume_compatibility,
};
pub(crate) use run::{
    mark_run_manifest_finalized_output, mark_run_manifest_interrupted, read_run_manifest_chunk_commits,
    read_run_manifest_chunk_commits_from_text, record_run_manifest_chunk_commits,
};

const RUN_MANIFEST_FILE_NAME: &str = "run_manifest.json";
const RUN_MANIFEST_SCHEMA_VERSION: i64 = 9;
const OUTPUT_SCHEMA_VERSION: i64 = 2;
const JAX_MATMUL_PRECISION_WHEN_UNSET: &str = "float32";
const RESUME_POLICY: &str = "manifest_committed_chunks";
