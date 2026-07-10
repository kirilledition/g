#![allow(clippy::missing_errors_doc)]

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::datatypes::Schema;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::error::OutputError;
use crate::{manifest, schema};

pub(crate) fn repair_strict_manifest_chunk_commits(
    parts_directory: &Path,
    manifest_json: &str,
) -> Result<Vec<manifest::RunManifestChunkCommit>, OutputError> {
    let mut repaired_commits = manifest::read_run_manifest_chunk_commits_from_text(manifest_json)?
        .into_iter()
        .map(|chunk_commit| (chunk_commit.chunk_identifier, chunk_commit))
        .collect::<BTreeMap<_, _>>();
    let scanned_commits = scan_committed_chunk_commits(parts_directory)?
        .into_iter()
        .map(|chunk_commit| (chunk_commit.chunk_identifier, chunk_commit))
        .collect::<BTreeMap<_, _>>();
    for existing_commit in repaired_commits.values() {
        let chunk_file_path = parts_directory.join(&existing_commit.chunk_file_name);
        if !chunk_file_path.exists() {
            return Err(OutputError::InvalidInput(format!(
                "Strict resume manifest references missing chunk file: {}",
                chunk_file_path.display()
            )));
        }
        match scanned_commits.get(&existing_commit.chunk_identifier) {
            Some(scanned_commit) if scanned_commit == existing_commit => {}
            Some(_) => {
                return Err(OutputError::InvalidInput(format!(
                    "Strict resume found conflicting commit metadata for chunk {}.",
                    existing_commit.chunk_identifier
                )));
            }
            None => {
                return Err(OutputError::InvalidInput(format!(
                    "Strict resume manifest references unobserved commit metadata for chunk {}.",
                    existing_commit.chunk_identifier
                )));
            }
        }
    }
    for (chunk_identifier, chunk_commit) in scanned_commits {
        if let Some(existing_commit) = repaired_commits.get(&chunk_identifier) {
            if existing_commit != &chunk_commit {
                return Err(OutputError::InvalidInput(format!(
                    "Strict resume found conflicting commit metadata for chunk {chunk_identifier}."
                )));
            }
        } else {
            repaired_commits.insert(chunk_identifier, chunk_commit);
        }
    }
    Ok(repaired_commits.into_values().collect())
}

struct PartCommitObservation {
    schema: Arc<Schema>,
    chunk_commits: Vec<manifest::RunManifestChunkCommit>,
}

fn scan_committed_chunk_commits(parts_directory: &Path) -> Result<Vec<manifest::RunManifestChunkCommit>, OutputError> {
    if !parts_directory.exists() {
        return Ok(Vec::new());
    }
    let mut part_paths = Vec::new();
    for directory_entry in std::fs::read_dir(parts_directory).map_err(OutputError::runtime)? {
        let part_path = directory_entry.map_err(OutputError::runtime)?.path();
        if part_path.extension().is_some_and(|extension| extension == "parquet") {
            part_paths.push(part_path);
        }
    }
    part_paths.sort();

    let mut chunk_commits = BTreeMap::new();
    let mut expected_schema: Option<Arc<Schema>> = None;
    for part_path in part_paths {
        let observation = inspect_parquet_part(&part_path)?;
        match expected_schema.as_ref() {
            Some(expected_schema) if expected_schema.fields() != observation.schema.fields() => {
                return Err(OutputError::InvalidInput(format!(
                    "Strict resume found an incompatible schema in {}.",
                    part_path.display()
                )));
            }
            None => expected_schema = Some(Arc::clone(&observation.schema)),
            Some(_) => {}
        }
        for chunk_commit in observation.chunk_commits {
            if chunk_commits.insert(chunk_commit.chunk_identifier, chunk_commit).is_some() {
                return Err(OutputError::InvalidInput(
                    "Strict resume found duplicate commit metadata for a chunk.".to_string(),
                ));
            }
        }
    }
    Ok(chunk_commits.into_values().collect())
}

fn inspect_parquet_part(part_path: &Path) -> Result<PartCommitObservation, OutputError> {
    let input_file = File::open(part_path).map_err(OutputError::runtime)?;
    let parquet_arrow_reader = ParquetRecordBatchReaderBuilder::try_new(input_file).map_err(OutputError::runtime)?;
    let part_schema = parquet_arrow_reader.schema().clone();
    let file_metadata = parquet_arrow_reader.metadata().file_metadata();

    let part_file_name = part_path
        .file_name()
        .and_then(|file_name| file_name.to_str())
        .ok_or_else(|| OutputError::Runtime("Parquet part file name is not UTF-8.".to_string()))?
        .to_string();
    let chunk_commit_text = file_metadata
        .key_value_metadata()
        .and_then(|metadata| metadata.iter().find(|entry| entry.key == schema::CHUNK_COMMITS_METADATA_KEY))
        .and_then(|entry| entry.value.as_deref())
        .ok_or_else(|| {
            OutputError::InvalidInput(format!(
                "Strict resume Parquet part is missing chunk commit metadata: {}",
                part_path.display()
            ))
        })?;
    let chunk_commits = manifest::read_chunk_commits_from_text(chunk_commit_text, &part_file_name)?;
    let committed_row_count = chunk_commits
        .iter()
        .try_fold(0_i64, |total, chunk_commit| total.checked_add(chunk_commit.row_count))
        .ok_or_else(|| OutputError::Runtime("Parquet committed row count overflowed.".to_string()))?;
    if committed_row_count != file_metadata.num_rows() {
        return Err(OutputError::InvalidInput(format!(
            "Strict resume row count mismatch for Parquet part {part_file_name}."
        )));
    }
    Ok(PartCommitObservation { schema: part_schema, chunk_commits })
}
