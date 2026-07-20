mod chunks;
mod fingerprint;
mod header;
mod run;
mod validation;

pub(crate) use chunks::{RunManifestChunkCommit, read_chunk_commits_from_text};
pub(crate) use fingerprint::ManifestFileFingerprint;
pub use fingerprint::ManifestFileFingerprintCache;
pub(crate) use fingerprint::{build_manifest_value_sha256, manifest_file_fingerprint_to_value};
pub(crate) use header::build_current_run_manifest_header_value_with_cache;
pub use header::{CurrentRunManifestHeaderInput, PredictionLocoFileFingerprint};
pub(crate) use run::{
    OutputRunPaths, PreparedOutputRun, extend_run_manifest_metadata, initialize_output_run, prepare_output_run,
    reconcile_output_run_resume,
};
pub(crate) use run::{
    mark_run_manifest_completed, mark_run_manifest_interrupted, read_run_manifest_chunk_commits_from_text,
    read_run_manifest_gpu_genotype_format_from_text, record_run_manifest_chunk_commits,
};

const RUN_MANIFEST_FILE_NAME: &str = "run_manifest.json";
const RUN_MANIFEST_SCHEMA_VERSION: i64 = 0;
const OUTPUT_SCHEMA_VERSION: i64 = 0;
const RESUME_POLICY: &str = "manifest_committed_chunks";
