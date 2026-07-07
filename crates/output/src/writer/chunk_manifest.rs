use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::datatypes::Schema;
use serde_json::json;

use crate::manifest;
use crate::schema;
use crate::schema::OutputStatisticDtype;

use super::{OutputFileFormat, OutputWriterResult, RegenieStep2ChunkWriteBatch};

pub(super) fn build_run_manifest_chunk_commits(
    job: &RegenieStep2ChunkWriteBatch,
    output_format: OutputFileFormat,
    compression: &str,
) -> OutputWriterResult<Vec<manifest::RunManifestChunkCommit>> {
    job.chunks
        .iter()
        .map(|chunk_job| {
            let variant_stop_index = chunk_job.chunk_handle.variant_stop_index()?;
            Ok(manifest::RunManifestChunkCommit {
                chunk_identifier: chunk_job.chunk_handle.chunk_identifier,
                output_format: output_format.value().to_string(),
                compression: compression.to_string(),
                variant_start_index: chunk_job.chunk_handle.variant_start_index(),
                variant_stop_index,
                row_count: chunk_job.chunk_handle.row_count(),
                chunk_file_name: job.chunk_file_name.clone(),
            })
        })
        .collect()
}

pub(super) fn build_regenie_step2_chunk_file_schema(
    chunk_commits: &[manifest::RunManifestChunkCommit],
    output_statistic_dtype: OutputStatisticDtype,
) -> OutputWriterResult<Arc<Schema>> {
    let mut metadata = HashMap::new();
    metadata.insert(schema::CHUNK_COMMITS_METADATA_KEY.to_string(), build_chunk_commit_metadata_text(chunk_commits)?);
    Ok(Arc::new(
        schema::get_regenie_step2_chunk_schema(output_statistic_dtype).as_ref().clone().with_metadata(metadata),
    ))
}

pub(super) fn build_chunk_commit_metadata_text(
    chunk_commits: &[manifest::RunManifestChunkCommit],
) -> OutputWriterResult<String> {
    let chunk_commit_values = chunk_commits
        .iter()
        .map(|chunk_commit| {
            json!({
                "chunk_identifier": chunk_commit.chunk_identifier,
                "output_format": chunk_commit.output_format,
                "compression": chunk_commit.compression,
                "variant_start_index": chunk_commit.variant_start_index,
                "variant_stop_index": chunk_commit.variant_stop_index,
                "row_count": chunk_commit.row_count,
                "chunk_file_name": chunk_commit.chunk_file_name,
            })
        })
        .collect::<Vec<_>>();
    serde_json::to_string(&chunk_commit_values).map_err(crate::error::OutputError::runtime)
}

pub(super) fn build_chunk_file_name(first_chunk_identifier: i64, last_chunk_identifier: i64) -> String {
    if first_chunk_identifier == last_chunk_identifier {
        return format!("chunk_{first_chunk_identifier:09}.arrow");
    }
    format!("chunk_{first_chunk_identifier:09}_{last_chunk_identifier:09}.arrow")
}

pub(super) fn build_part_file_name(first_chunk_identifier: i64, last_chunk_identifier: i64) -> String {
    if first_chunk_identifier == last_chunk_identifier {
        return format!("part_{first_chunk_identifier:09}.parquet");
    }
    format!("part_{first_chunk_identifier:09}_{last_chunk_identifier:09}.parquet")
}

pub(super) fn build_regenie_text_part_file_name(first_chunk_identifier: i64, last_chunk_identifier: i64) -> String {
    if first_chunk_identifier == last_chunk_identifier {
        return format!("part_{first_chunk_identifier:09}.regenie");
    }
    format!("part_{first_chunk_identifier:09}_{last_chunk_identifier:09}.regenie")
}

pub(crate) fn build_regenie_text_metadata_sidecar_path(chunk_file_path: &Path) -> PathBuf {
    chunk_file_path.with_extension("regenie.json")
}

pub(crate) fn build_output_file_name(
    output_format: OutputFileFormat,
    first_chunk_identifier: i64,
    last_chunk_identifier: i64,
) -> String {
    match output_format {
        OutputFileFormat::Arrow => build_chunk_file_name(first_chunk_identifier, last_chunk_identifier),
        OutputFileFormat::Parquet => build_part_file_name(first_chunk_identifier, last_chunk_identifier),
        OutputFileFormat::Regenie => build_regenie_text_part_file_name(first_chunk_identifier, last_chunk_identifier),
    }
}
