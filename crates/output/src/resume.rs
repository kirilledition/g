#![allow(clippy::missing_errors_doc)]

use std::collections::BTreeMap;
use std::fs::File;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use arrow::datatypes::Schema;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::error::OutputError;
use crate::{manifest, schema};

pub(crate) fn repair_strict_manifest_chunk_commits(
    parts_directory: &Path,
    manifest_json: &str,
    planned_chunk_ranges: &[Range<usize>],
) -> Result<Vec<manifest::RunManifestChunkCommit>, OutputError> {
    let planned_chunk_stops_by_start = build_planned_chunk_stops_by_start(planned_chunk_ranges)?;
    let mut repaired_commits = BTreeMap::new();
    for chunk_commit in manifest::read_run_manifest_chunk_commits_from_text(manifest_json)? {
        validate_chunk_commit_geometry(&chunk_commit, &planned_chunk_stops_by_start)?;
        repaired_commits.insert(chunk_commit.chunk_identifier, chunk_commit);
    }
    let scanned_commits = scan_committed_chunk_commits(parts_directory, &planned_chunk_stops_by_start)?
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

fn scan_committed_chunk_commits(
    parts_directory: &Path,
    planned_chunk_stops_by_start: &BTreeMap<usize, usize>,
) -> Result<Vec<manifest::RunManifestChunkCommit>, OutputError> {
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
    for part_path in part_paths {
        let observation = inspect_parquet_part(&part_path)?;
        if observation.schema.as_ref() != schema::REGENIE_STEP2_CHUNK_SCHEMA.as_ref() {
            return Err(OutputError::InvalidInput(format!(
                "Strict resume found an incompatible schema in {}.",
                part_path.display()
            )));
        }
        for chunk_commit in observation.chunk_commits {
            validate_chunk_commit_geometry(&chunk_commit, planned_chunk_stops_by_start)?;
            if chunk_commits.insert(chunk_commit.chunk_identifier, chunk_commit).is_some() {
                return Err(OutputError::InvalidInput(
                    "Strict resume found duplicate commit metadata for a chunk.".to_string(),
                ));
            }
        }
    }
    Ok(chunk_commits.into_values().collect())
}

fn build_planned_chunk_stops_by_start(
    planned_chunk_ranges: &[Range<usize>],
) -> Result<BTreeMap<usize, usize>, OutputError> {
    let mut planned_chunk_stops_by_start = BTreeMap::new();
    for chunk_range in planned_chunk_ranges {
        if chunk_range.start >= chunk_range.end {
            return Err(OutputError::InvalidInput(format!(
                "Planned output chunk range {}..{} is empty or reversed.",
                chunk_range.start, chunk_range.end
            )));
        }
        if planned_chunk_stops_by_start.insert(chunk_range.start, chunk_range.end).is_some() {
            return Err(OutputError::InvalidInput(format!(
                "Planned output chunk geometry has duplicate start index {}.",
                chunk_range.start
            )));
        }
    }
    Ok(planned_chunk_stops_by_start)
}

fn validate_chunk_commit_geometry(
    chunk_commit: &manifest::RunManifestChunkCommit,
    planned_chunk_stops_by_start: &BTreeMap<usize, usize>,
) -> Result<(), OutputError> {
    if chunk_commit.chunk_identifier != chunk_commit.variant_start_index {
        return Err(OutputError::InvalidInput(format!(
            "Strict resume chunk {} does not identify its variant start index {}.",
            chunk_commit.chunk_identifier, chunk_commit.variant_start_index
        )));
    }
    let chunk_start = usize::try_from(chunk_commit.variant_start_index).map_err(|_| {
        OutputError::InvalidInput(format!(
            "Strict resume chunk {} has an out-of-bounds start index.",
            chunk_commit.chunk_identifier
        ))
    })?;
    let chunk_stop = usize::try_from(chunk_commit.variant_stop_index).map_err(|_| {
        OutputError::InvalidInput(format!(
            "Strict resume chunk {} has an out-of-bounds stop index.",
            chunk_commit.chunk_identifier
        ))
    })?;
    let row_count = usize::try_from(chunk_commit.row_count).map_err(|_| {
        OutputError::InvalidInput(format!(
            "Strict resume chunk {} has an out-of-bounds row count.",
            chunk_commit.chunk_identifier
        ))
    })?;
    let Some(expected_chunk_stop) = planned_chunk_stops_by_start.get(&chunk_start).copied() else {
        return Err(OutputError::InvalidInput(format!(
            "Strict resume chunk {} is not present in the current BGEN chunk plan.",
            chunk_commit.chunk_identifier
        )));
    };
    let expected_row_count = expected_chunk_stop.checked_sub(chunk_start).ok_or_else(|| {
        OutputError::InvalidInput(format!(
            "Planned output chunk range {chunk_start}..{expected_chunk_stop} is reversed."
        ))
    })?;
    if chunk_stop != expected_chunk_stop || row_count != expected_row_count {
        return Err(OutputError::InvalidInput(format!(
            "Strict resume chunk {} geometry does not match the current BGEN chunk plan.",
            chunk_commit.chunk_identifier
        )));
    }
    Ok(())
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
