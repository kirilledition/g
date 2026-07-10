use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

use arrow::datatypes::Schema;
use arrow::ipc::reader::FileReader as ArrowFileReader;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::reader::{FileReader as ParquetFileReader, SerializedFileReader};
use serde_json::Value;

use crate::error::OutputError;
use crate::manifest;
use crate::schema;
use crate::writer;

#[derive(Clone)]
struct ChunkCommitObservation {
    chunk_identifier: i64,
    output_format: String,
    compression: String,
    variant_start_index: i64,
    variant_stop_index: i64,
    row_count: i64,
}

pub(super) struct ChunkFileCommitObservation {
    pub(super) schema: Arc<Schema>,
    pub(super) chunk_commits: Vec<manifest::RunManifestChunkCommit>,
}

pub(super) fn scan_committed_chunk_commits(
    chunks_directory: &Path,
) -> Result<Vec<manifest::RunManifestChunkCommit>, OutputError> {
    if !chunks_directory.exists() {
        return Ok(Vec::new());
    }
    let mut chunk_file_paths = std::fs::read_dir(chunks_directory)
        .map_err(OutputError::runtime)?
        .filter_map(|directory_entry| directory_entry.ok().map(|entry| entry.path()))
        .filter(|chunk_file_path| {
            chunk_file_path
                .extension()
                .is_some_and(|extension| extension == "arrow" || extension == "parquet" || extension == "regenie")
        })
        .collect::<Vec<_>>();
    chunk_file_paths.sort();
    let mut chunk_commits = BTreeMap::new();
    let mut expected_schema: Option<Arc<Schema>> = None;
    for chunk_file_path in chunk_file_paths {
        let chunk_file_observation = inspect_chunk_file_commits(&chunk_file_path)?;
        match expected_schema.as_ref() {
            Some(expected_schema) if expected_schema.fields() != chunk_file_observation.schema.fields() => {
                return Err(OutputError::InvalidInput(format!(
                    "Strict resume found incompatible Arrow schema in {}.",
                    chunk_file_path.display()
                )));
            }
            None => expected_schema = Some(Arc::clone(&chunk_file_observation.schema)),
            Some(_) => {}
        }
        for chunk_commit in chunk_file_observation.chunk_commits {
            if chunk_commits.insert(chunk_commit.chunk_identifier, chunk_commit).is_some() {
                return Err(OutputError::InvalidInput(
                    "Strict resume found duplicate Arrow commit metadata for a chunk.".to_string(),
                ));
            }
        }
    }
    Ok(chunk_commits.into_values().collect())
}

pub(super) fn inspect_chunk_file_commits(chunk_file_path: &Path) -> Result<ChunkFileCommitObservation, OutputError> {
    if chunk_file_path.extension().is_some_and(|extension| extension == "parquet") {
        return inspect_parquet_chunk_file_commits(chunk_file_path);
    }
    if chunk_file_path.extension().is_some_and(|extension| extension == "regenie") {
        return inspect_regenie_text_chunk_file_commits(chunk_file_path);
    }
    let input_file = File::open(chunk_file_path).map_err(OutputError::runtime)?;
    let file_reader = ArrowFileReader::try_new(input_file, None).map_err(OutputError::runtime)?;
    let schema = file_reader.schema();
    let chunk_file_name = chunk_file_path
        .file_name()
        .and_then(|file_name| file_name.to_str())
        .ok_or_else(|| OutputError::Runtime("Rust output writer chunk file name is not UTF-8.".to_string()))?
        .to_string();
    let Some(chunk_commits) = read_schema_chunk_commits(schema.as_ref())? else {
        return Err(OutputError::InvalidInput(format!(
            "Strict resume Arrow chunk is missing chunk commit metadata: {}",
            chunk_file_path.display()
        )));
    };
    let chunk_commits = inspect_metadata_chunk_file_commits(file_reader, chunk_commits, &chunk_file_name)?;
    Ok(ChunkFileCommitObservation { schema, chunk_commits })
}

fn inspect_metadata_chunk_file_commits(
    file_reader: ArrowFileReader<File>,
    chunk_commits: Vec<ChunkCommitObservation>,
    chunk_file_name: &str,
) -> Result<Vec<manifest::RunManifestChunkCommit>, OutputError> {
    let mut batch_row_counts = Vec::with_capacity(chunk_commits.len());
    for maybe_batch in file_reader {
        let batch = maybe_batch.map_err(OutputError::runtime)?;
        batch_row_counts.push(i64::try_from(batch.num_rows()).map_err(OutputError::runtime)?);
    }
    if batch_row_counts.len() != chunk_commits.len() {
        return Err(OutputError::InvalidInput(format!(
            "Strict resume batch count mismatch for chunk file {chunk_file_name}."
        )));
    }
    let mut manifest_commits = Vec::with_capacity(chunk_commits.len());
    for (observed_row_count, chunk_commit) in batch_row_counts.iter().zip(chunk_commits) {
        if *observed_row_count != chunk_commit.row_count {
            return Err(OutputError::InvalidInput(format!(
                "Strict resume row count mismatch for chunk {}.",
                chunk_commit.chunk_identifier
            )));
        }
        manifest_commits.push(manifest::RunManifestChunkCommit {
            chunk_identifier: chunk_commit.chunk_identifier,
            output_format: chunk_commit.output_format,
            compression: chunk_commit.compression,
            variant_start_index: chunk_commit.variant_start_index,
            variant_stop_index: chunk_commit.variant_stop_index,
            row_count: chunk_commit.row_count,
            chunk_file_name: chunk_file_name.to_string(),
        });
    }
    Ok(manifest_commits)
}

fn inspect_parquet_chunk_file_commits(chunk_file_path: &Path) -> Result<ChunkFileCommitObservation, OutputError> {
    let schema = read_parquet_arrow_schema(chunk_file_path)?;
    let chunk_file_name = chunk_file_path
        .file_name()
        .and_then(|file_name| file_name.to_str())
        .ok_or_else(|| OutputError::Runtime("Rust output writer part file name is not UTF-8.".to_string()))?
        .to_string();
    let input_file = File::open(chunk_file_path).map_err(OutputError::runtime)?;
    let parquet_reader = SerializedFileReader::new(input_file).map_err(OutputError::runtime)?;
    let file_metadata = parquet_reader.metadata().file_metadata();
    let observed_row_count = file_metadata.num_rows();
    let chunk_commit_text = file_metadata
        .key_value_metadata()
        .and_then(|metadata| metadata.iter().find(|entry| entry.key == schema::CHUNK_COMMITS_METADATA_KEY))
        .and_then(|entry| entry.value.as_deref())
        .ok_or_else(|| {
            OutputError::InvalidInput(format!(
                "Strict resume Parquet part is missing chunk commit metadata: {}",
                chunk_file_path.display()
            ))
        })?;
    let chunk_commits = read_chunk_commit_observations_text(chunk_commit_text)?;
    let summed_row_count = chunk_commits
        .iter()
        .try_fold(0_i64, |total, chunk_commit| total.checked_add(chunk_commit.row_count).ok_or(()))
        .map_err(|()| OutputError::Runtime("Rust output writer Parquet row count overflowed.".to_string()))?;
    if summed_row_count != observed_row_count {
        return Err(OutputError::InvalidInput(format!(
            "Strict resume Parquet row count mismatch for part {chunk_file_name}."
        )));
    }
    let mut manifest_commits = Vec::with_capacity(chunk_commits.len());
    for chunk_commit in chunk_commits {
        if chunk_commit.output_format != "parquet" {
            return Err(OutputError::InvalidInput(format!(
                "Strict resume Parquet part has non-Parquet commit metadata for chunk {}.",
                chunk_commit.chunk_identifier
            )));
        }
        manifest_commits.push(manifest::RunManifestChunkCommit {
            chunk_identifier: chunk_commit.chunk_identifier,
            output_format: chunk_commit.output_format,
            compression: chunk_commit.compression,
            variant_start_index: chunk_commit.variant_start_index,
            variant_stop_index: chunk_commit.variant_stop_index,
            row_count: chunk_commit.row_count,
            chunk_file_name: chunk_file_name.clone(),
        });
    }
    Ok(ChunkFileCommitObservation { schema, chunk_commits: manifest_commits })
}

fn inspect_regenie_text_chunk_file_commits(chunk_file_path: &Path) -> Result<ChunkFileCommitObservation, OutputError> {
    let schema = Arc::clone(schema::get_regenie_step2_final_schema(schema::OutputStatisticDtype::Float32));
    let chunk_file_name = chunk_file_path
        .file_name()
        .and_then(|file_name| file_name.to_str())
        .ok_or_else(|| OutputError::Runtime("Rust output writer text part file name is not UTF-8.".to_string()))?
        .to_string();
    let sidecar_path = writer::build_regenie_text_metadata_sidecar_path(chunk_file_path);
    let chunk_commit_text = std::fs::read_to_string(&sidecar_path).map_err(|error| {
        OutputError::InvalidInput(format!(
            "Strict resume REGENIE text part is missing chunk commit metadata: {} ({error})",
            sidecar_path.display()
        ))
    })?;
    let chunk_commits = read_chunk_commit_observations_text(&chunk_commit_text)?;
    let observed_row_count = count_regenie_text_rows(chunk_file_path)?;
    let summed_row_count = chunk_commits
        .iter()
        .try_fold(0_i64, |total, chunk_commit| total.checked_add(chunk_commit.row_count).ok_or(()))
        .map_err(|()| OutputError::Runtime("Rust output writer REGENIE text row count overflowed.".to_string()))?;
    if summed_row_count != observed_row_count {
        return Err(OutputError::InvalidInput(format!(
            "Strict resume REGENIE text row count mismatch for part {chunk_file_name}."
        )));
    }
    let mut manifest_commits = Vec::with_capacity(chunk_commits.len());
    for chunk_commit in chunk_commits {
        if chunk_commit.output_format != "regenie" {
            return Err(OutputError::InvalidInput(format!(
                "Strict resume REGENIE text part has non-REGENIE commit metadata for chunk {}.",
                chunk_commit.chunk_identifier
            )));
        }
        manifest_commits.push(manifest::RunManifestChunkCommit {
            chunk_identifier: chunk_commit.chunk_identifier,
            output_format: chunk_commit.output_format,
            compression: chunk_commit.compression,
            variant_start_index: chunk_commit.variant_start_index,
            variant_stop_index: chunk_commit.variant_stop_index,
            row_count: chunk_commit.row_count,
            chunk_file_name: chunk_file_name.clone(),
        });
    }
    Ok(ChunkFileCommitObservation { schema, chunk_commits: manifest_commits })
}

fn count_regenie_text_rows(chunk_file_path: &Path) -> Result<i64, OutputError> {
    let input_file = File::open(chunk_file_path).map_err(OutputError::runtime)?;
    let mut input_reader = BufReader::new(input_file);
    let mut header_line = String::new();
    input_reader.read_line(&mut header_line).map_err(OutputError::runtime)?;
    let observed_header = header_line.trim_end_matches(['\r', '\n']);
    let expected_header = writer::REGENIE_STEP2_TEXT_HEADER.trim_end_matches('\n');
    if observed_header != expected_header {
        return Err(OutputError::InvalidInput(format!(
            "Strict resume REGENIE text part has an unexpected header: {}",
            chunk_file_path.display()
        )));
    }
    let mut row_count = 0_i64;
    let mut row_line = String::new();
    let expected_column_count = writer::REGENIE_STEP2_TEXT_HEADER.trim_end_matches('\n').split('\t').count();
    loop {
        row_line.clear();
        let read_byte_count = input_reader.read_line(&mut row_line).map_err(OutputError::runtime)?;
        if read_byte_count == 0 {
            break;
        }
        let row = row_line.trim_end_matches(['\r', '\n']);
        if row.split('\t').count() != expected_column_count {
            return Err(OutputError::InvalidInput(format!(
                "Strict resume REGENIE text part has a row with an unexpected column count: {}",
                chunk_file_path.display()
            )));
        }
        row_count = row_count
            .checked_add(1)
            .ok_or_else(|| OutputError::Runtime("Rust output writer REGENIE text row count overflowed.".to_string()))?;
    }
    Ok(row_count)
}

fn read_parquet_arrow_schema(chunk_file_path: &Path) -> Result<Arc<Schema>, OutputError> {
    let input_file = File::open(chunk_file_path).map_err(OutputError::runtime)?;
    let parquet_reader = ParquetRecordBatchReaderBuilder::try_new(input_file).map_err(OutputError::runtime)?;
    Ok(parquet_reader.schema().clone())
}

fn read_schema_chunk_commits(chunk_schema: &Schema) -> Result<Option<Vec<ChunkCommitObservation>>, OutputError> {
    let Some(chunk_commits_text) = chunk_schema.metadata().get(schema::CHUNK_COMMITS_METADATA_KEY) else {
        return Ok(None);
    };
    Ok(Some(read_chunk_commit_observations_text(chunk_commits_text)?))
}

fn read_chunk_commit_observations_text(chunk_commits_text: &str) -> Result<Vec<ChunkCommitObservation>, OutputError> {
    let chunk_commit_values = serde_json::from_str::<Value>(chunk_commits_text).map_err(OutputError::runtime)?;
    let chunk_commit_array = chunk_commit_values
        .as_array()
        .ok_or_else(|| OutputError::Runtime("Rust output writer chunk commit metadata must be a list.".to_string()))?;
    let mut chunk_commits = Vec::with_capacity(chunk_commit_array.len());
    for chunk_commit_value in chunk_commit_array {
        chunk_commits.push(ChunkCommitObservation {
            chunk_identifier: read_manifest_integer(chunk_commit_value, "chunk_identifier")?,
            output_format: read_optional_manifest_string(chunk_commit_value, "output_format")
                .unwrap_or_else(|| "arrow".to_string()),
            compression: read_optional_manifest_string(chunk_commit_value, "compression")
                .unwrap_or_else(|| "none".to_string()),
            variant_start_index: read_manifest_integer(chunk_commit_value, "variant_start_index")?,
            variant_stop_index: read_manifest_integer(chunk_commit_value, "variant_stop_index")?,
            row_count: read_manifest_integer(chunk_commit_value, "row_count")?,
        });
    }
    Ok(chunk_commits)
}

fn read_optional_manifest_string(committed_chunk: &Value, field_name: &str) -> Option<String> {
    committed_chunk.get(field_name).and_then(Value::as_str).map(str::to_string)
}

fn read_manifest_integer(committed_chunk: &Value, field_name: &str) -> Result<i64, OutputError> {
    committed_chunk.get(field_name).and_then(Value::as_i64).ok_or_else(|| {
        OutputError::InvalidInput(format!("Run manifest committed chunk entry is missing {field_name}."))
    })
}
