//! Atomically publish verified chunk commits together with the terminal state.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::{Map, Value};

use super::chunks::{self, RunManifestChunkCommit};
use super::run;
use crate::error::{OutputError, OutputResult};

#[derive(Clone, Copy)]
pub(crate) enum TerminalRunState<'signal> {
    Completed,
    Interrupted { signal_name: &'signal str },
}

struct PreviousLifecycleValues {
    status: Option<Value>,
    interrupted_signal: Option<Value>,
}

pub(crate) fn finalize_run_manifest(
    run_directory: &Path,
    chunk_commits: Vec<RunManifestChunkCommit>,
    terminal_state: TerminalRunState<'_>,
) -> OutputResult<()> {
    finalize_run_manifest_with_writer(
        run_directory,
        chunk_commits,
        terminal_state,
        run::write_run_manifest_value_atomic,
    )
}

fn finalize_run_manifest_with_writer(
    run_directory: &Path,
    chunk_commits: Vec<RunManifestChunkCommit>,
    terminal_state: TerminalRunState<'_>,
    mut write_manifest: impl FnMut(&Path, &Value) -> OutputResult<()>,
) -> OutputResult<()> {
    let has_new_commits = !chunk_commits.is_empty();
    run::with_locked_run_manifest(run_directory, |manifest_path, manifest| {
        let manifest_object = manifest
            .as_object_mut()
            .ok_or_else(|| OutputError::InvalidInput("Run manifest must contain a JSON object.".to_string()))?;
        merge_chunk_commits(manifest_object, chunk_commits)?;
        let previous_lifecycle = apply_terminal_state(manifest_object, terminal_state);
        let Err(terminal_error) = write_manifest(manifest_path, manifest) else {
            return Ok(());
        };
        if has_new_commits {
            // Preserve completed worker progress even when terminal publication
            // fails. The same lock covers this fallback; the raw atomic writer
            // must not try to acquire the manifest lock recursively.
            previous_lifecycle
                .restore(manifest.as_object_mut().expect("terminal updates preserve the validated manifest object"));
            if let Err(commit_error) = write_manifest(manifest_path, manifest) {
                return Err(OutputError::Runtime(format!(
                    "Run manifest terminal update failed: {terminal_error}; preserving chunk commits also failed: {commit_error}"
                )));
            }
        }
        Err(terminal_error)
    })
}

fn merge_chunk_commits(
    manifest_object: &mut Map<String, Value>,
    chunk_commits: Vec<RunManifestChunkCommit>,
) -> OutputResult<()> {
    if chunk_commits.is_empty() {
        return Ok(());
    }
    let committed_chunks = manifest_object
        .entry("committed_chunks".to_string())
        .or_insert_with(|| Value::Array(Vec::new()))
        .as_array_mut()
        .ok_or_else(|| OutputError::InvalidInput("Run manifest committed_chunks field must be a list.".to_string()))?;
    let mut committed_chunks_by_identifier = BTreeMap::new();
    for committed_chunk in committed_chunks.iter() {
        let existing_commit = chunks::read_run_manifest_chunk_commit(committed_chunk)?;
        chunks::insert_or_validate_chunk_commit(&mut committed_chunks_by_identifier, existing_commit)?;
    }
    for chunk_commit in chunk_commits {
        chunks::insert_or_validate_chunk_commit(&mut committed_chunks_by_identifier, chunk_commit)?;
    }
    *committed_chunks = committed_chunks_by_identifier.values().map(chunks::chunk_commit_to_value).collect();
    Ok(())
}

fn apply_terminal_state(
    manifest: &mut Map<String, Value>,
    terminal_state: TerminalRunState<'_>,
) -> PreviousLifecycleValues {
    let previous_lifecycle = PreviousLifecycleValues {
        status: manifest.remove("status"),
        interrupted_signal: manifest.remove("interrupted_signal"),
    };
    let status = match terminal_state {
        TerminalRunState::Completed => "completed",
        TerminalRunState::Interrupted { signal_name } => {
            manifest.insert("interrupted_signal".to_string(), Value::String(signal_name.to_string()));
            "interrupted"
        }
    };
    manifest.insert("status".to_string(), Value::String(status.to_string()));
    previous_lifecycle
}

impl PreviousLifecycleValues {
    fn restore(self, manifest: &mut Map<String, Value>) {
        for (field_name, previous_value) in [("status", self.status), ("interrupted_signal", self.interrupted_signal)] {
            if let Some(value) = previous_value {
                manifest.insert(field_name.to_string(), value);
            } else {
                manifest.remove(field_name);
            }
        }
    }
}

#[cfg(test)]
mod tests;
