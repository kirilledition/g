use serde_json::json;

use crate::manifest;

use super::{OutputResult, RegenieStep2ChunkWriteBatch};

pub(super) fn build_run_manifest_chunk_commits(
    job: &RegenieStep2ChunkWriteBatch,
) -> OutputResult<Vec<manifest::RunManifestChunkCommit>> {
    job.chunks
        .iter()
        .map(|chunk_job| {
            let variant_stop_index = chunk_job.chunk_handle.variant_stop_index()?;
            let row_count = i64::try_from(chunk_job.chunk_handle.row_count()).map_err(|_| {
                crate::error::OutputError::InvalidInput(
                    "Output chunk row count exceeds the signed manifest count range.".to_string(),
                )
            })?;
            Ok(manifest::RunManifestChunkCommit {
                chunk_identifier: chunk_job.chunk_handle.chunk_identifier,
                variant_start_index: chunk_job.chunk_handle.chunk_identifier,
                variant_stop_index,
                row_count,
                chunk_file_name: job.chunk_file_name.clone(),
            })
        })
        .collect()
}

pub(super) fn build_chunk_commit_metadata_text(
    chunk_commits: &[manifest::RunManifestChunkCommit],
) -> OutputResult<String> {
    let chunk_commit_values = chunk_commits
        .iter()
        .map(|chunk_commit| {
            json!({
                "chunk_identifier": chunk_commit.chunk_identifier,
                "variant_start_index": chunk_commit.variant_start_index,
                "variant_stop_index": chunk_commit.variant_stop_index,
                "row_count": chunk_commit.row_count,
                "chunk_file_name": chunk_commit.chunk_file_name,
            })
        })
        .collect::<Vec<_>>();
    serde_json::to_string(&chunk_commit_values).map_err(crate::error::OutputError::runtime)
}

pub(crate) fn build_part_file_name(first_chunk_identifier: i64, last_chunk_identifier: i64) -> String {
    if first_chunk_identifier == last_chunk_identifier {
        return format!("part_{first_chunk_identifier:09}.parquet");
    }
    format!("part_{first_chunk_identifier:09}_{last_chunk_identifier:09}.parquet")
}
