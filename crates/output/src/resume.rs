#![allow(clippy::missing_errors_doc)]

use std::collections::BTreeMap;
use std::path::Path;

use crate::error::OutputError;
use crate::manifest;

mod chunk_files;

use chunk_files::scan_committed_chunk_commits;

pub(crate) fn repair_strict_manifest_chunk_commits(
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
