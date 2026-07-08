//! Administrative output manifest and resume exports.

pub use crate::manifest::{
    CurrentRunManifestHeaderInput, ManifestFileFingerprint, ManifestFileFingerprintCache, RunManifestChunkCommit,
    build_current_run_manifest_header_json, build_current_run_manifest_header_json_with_cache,
    build_file_content_sha256, build_manifest_file_fingerprint, build_manifest_json_sha256,
    build_prepared_run_manifest_header_json, build_prepared_run_manifest_header_json_from_current_header_json,
    build_prepared_run_plan_from_current_header_json, build_prepared_run_plan_json_from_current_header_json,
    extend_run_manifest_metadata, load_run_manifest_json, read_run_manifest_committed_chunk_identifiers_from_text,
    resolve_output_run_paths, write_run_manifest_json,
};
pub use crate::resume::{
    repair_strict_manifest_chunk_commits, scan_committed_chunk_identifiers, validate_strict_manifest_chunks,
};
