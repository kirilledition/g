use std::collections::BTreeMap;

use serde_json::{Value, json};

use crate::error::{OutputError, OutputResult};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RunManifestChunkCommit {
    pub chunk_identifier: i64,
    pub output_format: String,
    pub compression: String,
    pub variant_start_index: i64,
    pub variant_stop_index: i64,
    pub row_count: i64,
    pub chunk_file_name: String,
}

pub(super) fn insert_or_validate_chunk_commit(
    committed_chunks_by_identifier: &mut BTreeMap<i64, RunManifestChunkCommit>,
    chunk_commit: RunManifestChunkCommit,
) -> OutputResult<()> {
    match committed_chunks_by_identifier.get(&chunk_commit.chunk_identifier) {
        Some(existing_commit) if existing_commit != &chunk_commit => Err(OutputError::InvalidInput(format!(
            "Run manifest has conflicting commit metadata for chunk {}.",
            chunk_commit.chunk_identifier
        ))),
        Some(_) => Ok(()),
        None => {
            committed_chunks_by_identifier.insert(chunk_commit.chunk_identifier, chunk_commit);
            Ok(())
        }
    }
}

pub(super) fn chunk_commit_to_value(chunk_commit: &RunManifestChunkCommit) -> Value {
    json!({
        "chunk_identifier": chunk_commit.chunk_identifier,
        "output_format": chunk_commit.output_format,
        "compression": chunk_commit.compression,
        "variant_start_index": chunk_commit.variant_start_index,
        "variant_stop_index": chunk_commit.variant_stop_index,
        "row_count": chunk_commit.row_count,
        "chunk_file_name": chunk_commit.chunk_file_name,
    })
}

pub(super) fn read_run_manifest_chunk_commit(committed_chunk: &Value) -> OutputResult<RunManifestChunkCommit> {
    let chunk_file_name = committed_chunk.get("chunk_file_name").and_then(Value::as_str).ok_or_else(|| {
        OutputError::InvalidInput("Run manifest committed chunk entry is missing chunk_file_name.".to_string())
    })?;
    Ok(RunManifestChunkCommit {
        chunk_identifier: read_manifest_integer(committed_chunk, "chunk_identifier")?,
        output_format: read_optional_manifest_string(committed_chunk, "output_format")
            .unwrap_or_else(|| infer_output_format_from_file_name(chunk_file_name).to_string()),
        compression: read_optional_manifest_string(committed_chunk, "compression")
            .unwrap_or_else(|| "none".to_string()),
        variant_start_index: read_manifest_integer(committed_chunk, "variant_start_index")?,
        variant_stop_index: read_manifest_integer(committed_chunk, "variant_stop_index")?,
        row_count: read_manifest_non_negative_integer(committed_chunk, "row_count")?,
        chunk_file_name: chunk_file_name.to_string(),
    })
}

fn read_optional_manifest_string(committed_chunk: &Value, field_name: &str) -> Option<String> {
    committed_chunk.get(field_name).and_then(Value::as_str).map(str::to_string)
}

fn infer_output_format_from_file_name(chunk_file_name: &str) -> &'static str {
    if chunk_file_name.ends_with(".regenie") {
        return "regenie";
    }
    if chunk_file_name.ends_with(".parquet") {
        return "parquet";
    }
    "arrow"
}

fn read_manifest_integer(committed_chunk: &Value, field_name: &str) -> OutputResult<i64> {
    committed_chunk.get(field_name).and_then(Value::as_i64).ok_or_else(|| {
        OutputError::InvalidInput(format!("Run manifest committed chunk entry is missing {field_name}."))
    })
}

fn read_manifest_non_negative_integer(committed_chunk: &Value, field_name: &str) -> OutputResult<i64> {
    let value = read_manifest_integer(committed_chunk, field_name)?;
    if value < 0 {
        return Err(OutputError::InvalidInput(format!(
            "Run manifest committed chunk entry {field_name} must be non-negative."
        )));
    }
    Ok(value)
}
