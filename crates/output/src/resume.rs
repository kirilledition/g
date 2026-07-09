#![allow(clippy::missing_errors_doc)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

use arrow::datatypes::Schema;

use crate::error::OutputError;
use crate::manifest;

mod chunk_files;

use chunk_files::scan_committed_chunk_commits;

pub fn scan_committed_chunk_identifiers(chunks_directory: &Path) -> Result<Vec<i64>, OutputError> {
    Ok(scan_committed_chunk_commits(chunks_directory)?
        .into_iter()
        .map(|chunk_commit| chunk_commit.chunk_identifier)
        .collect())
}

pub fn validate_strict_manifest_chunks(chunks_directory: &Path, manifest_json: &str) -> Result<Vec<i64>, OutputError> {
    let manifest_commits = manifest::read_run_manifest_chunk_commits_from_text(manifest_json)?;
    let mut committed_identifiers = BTreeSet::new();
    let mut expected_schema: Option<Arc<Schema>> = None;
    for (chunk_file_name, chunk_commits) in group_manifest_commits_by_file(manifest_commits) {
        let chunk_file_path = chunks_directory.join(&chunk_file_name);
        if !chunk_file_path.exists() {
            return Err(OutputError::InvalidInput(format!(
                "Strict resume manifest references missing chunk file: {}",
                chunk_file_path.display()
            )));
        }
        validate_manifest_chunk_file_commits(
            &chunk_file_path,
            &chunk_commits,
            &mut expected_schema,
            &mut committed_identifiers,
        )?;
    }
    Ok(committed_identifiers.into_iter().collect())
}

pub fn repair_strict_manifest_chunk_commits(
    chunks_directory: &Path,
    manifest_json: &str,
) -> Result<Vec<manifest::RunManifestChunkCommit>, OutputError> {
    let mut repaired_commits = manifest::read_run_manifest_chunk_commits_from_text(manifest_json)?
        .into_iter()
        .map(|chunk_commit| (chunk_commit.chunk_identifier, chunk_commit))
        .collect::<BTreeMap<_, _>>();
    let scanned_commits = scan_committed_chunk_commits(chunks_directory)?
        .into_iter()
        .map(|chunk_commit| (chunk_commit.chunk_identifier, chunk_commit))
        .collect::<BTreeMap<_, _>>();
    for existing_commit in repaired_commits.values() {
        let chunk_file_path = chunks_directory.join(&existing_commit.chunk_file_name);
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

fn group_manifest_commits_by_file(
    manifest_commits: Vec<manifest::RunManifestChunkCommit>,
) -> BTreeMap<String, Vec<manifest::RunManifestChunkCommit>> {
    let mut chunk_commits_by_file = BTreeMap::<String, Vec<manifest::RunManifestChunkCommit>>::new();
    for chunk_commit in manifest_commits {
        chunk_commits_by_file.entry(chunk_commit.chunk_file_name.clone()).or_default().push(chunk_commit);
    }
    chunk_commits_by_file
}

fn validate_manifest_chunk_file_commits(
    chunk_file_path: &Path,
    expected_commits: &[manifest::RunManifestChunkCommit],
    expected_schema: &mut Option<Arc<Schema>>,
    committed_identifiers: &mut BTreeSet<i64>,
) -> Result<(), OutputError> {
    let chunk_file_observation = chunk_files::inspect_chunk_file_commits(chunk_file_path)?;
    match expected_schema.as_ref() {
        Some(expected_schema) if expected_schema.fields() != chunk_file_observation.schema.fields() => {
            return Err(OutputError::InvalidInput(format!(
                "Strict resume found incompatible Arrow schema in {}.",
                chunk_file_path.display()
            )));
        }
        None => *expected_schema = Some(Arc::clone(&chunk_file_observation.schema)),
        Some(_) => {}
    }
    let observed_commits = collect_chunk_commits_by_identifier(chunk_file_observation.chunk_commits)?;
    let expected_commit_identifiers =
        expected_commits.iter().map(|chunk_commit| chunk_commit.chunk_identifier).collect::<BTreeSet<_>>();
    let observed_commit_identifiers = observed_commits.keys().copied().collect::<BTreeSet<_>>();
    for expected_commit in expected_commits {
        let Some(observed_commit) = observed_commits.get(&expected_commit.chunk_identifier) else {
            return Err(OutputError::InvalidInput(format!(
                "Strict resume manifest commit set does not match chunk file {}.",
                chunk_file_path.display()
            )));
        };
        validate_manifest_chunk_commit(expected_commit, observed_commit)?;
        committed_identifiers.insert(expected_commit.chunk_identifier);
    }
    if observed_commit_identifiers != expected_commit_identifiers {
        return Err(OutputError::InvalidInput(format!(
            "Strict resume manifest commit set does not match chunk file {}.",
            chunk_file_path.display()
        )));
    }
    Ok(())
}

fn validate_manifest_chunk_commit(
    expected_commit: &manifest::RunManifestChunkCommit,
    observed_commit: &manifest::RunManifestChunkCommit,
) -> Result<(), OutputError> {
    if observed_commit.variant_start_index != expected_commit.variant_start_index
        || observed_commit.variant_stop_index != expected_commit.variant_stop_index
    {
        return Err(OutputError::InvalidInput(format!(
            "Strict resume variant range mismatch for chunk {}.",
            expected_commit.chunk_identifier
        )));
    }
    if observed_commit.row_count != expected_commit.row_count {
        return Err(OutputError::InvalidInput(format!(
            "Strict resume row count mismatch for chunk {}.",
            expected_commit.chunk_identifier
        )));
    }
    if observed_commit != expected_commit {
        return Err(OutputError::InvalidInput(format!(
            "Strict resume found conflicting commit metadata for chunk {}.",
            expected_commit.chunk_identifier
        )));
    }
    Ok(())
}

fn collect_chunk_commits_by_identifier(
    chunk_commits: Vec<manifest::RunManifestChunkCommit>,
) -> Result<BTreeMap<i64, manifest::RunManifestChunkCommit>, OutputError> {
    let mut chunk_commits_by_identifier = BTreeMap::new();
    for chunk_commit in chunk_commits {
        if chunk_commits_by_identifier.insert(chunk_commit.chunk_identifier, chunk_commit).is_some() {
            return Err(OutputError::InvalidInput(
                "Strict resume found duplicate Arrow commit metadata for a chunk.".to_string(),
            ));
        }
    }
    Ok(chunk_commits_by_identifier)
}
