use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use serde_json::{Map, Value, json};

use crate::error::OutputError;
use crate::resume;
use crate::writer::OutputFileFormat;

use super::chunks::{RunManifestChunkCommit, chunk_commit_to_value, insert_or_validate_chunk_commit};
use super::{RUN_MANIFEST_FILE_NAME, chunks, validation};

type ManifestResult<T> = Result<T, OutputError>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OutputRunPaths {
    pub run_directory: PathBuf,
    pub chunks_directory: PathBuf,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedOutputRun {
    pub output_run_paths: OutputRunPaths,
    pub existing_manifest_json: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InitializedOutputRun {
    pub committed_chunk_identifiers: Vec<i64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutputResumeMode {
    Fast,
    Strict,
}

impl OutputResumeMode {
    pub fn parse(resume_mode: &str) -> Result<Self, OutputError> {
        match resume_mode {
            "fast" => Ok(Self::Fast),
            "strict" => Ok(Self::Strict),
            unsupported_resume_mode => Err(OutputError::InvalidInput(format!(
                "Resume mode must be 'fast' or 'strict', observed '{unsupported_resume_mode}'."
            ))),
        }
    }
}

#[must_use]
pub fn resolve_output_run_paths(
    output_root: &Path,
    association_mode: &str,
    output_format: OutputFileFormat,
) -> OutputRunPaths {
    let run_directory = if output_root.extension().is_some_and(|extension| extension == "run") {
        output_root.to_path_buf()
    } else {
        PathBuf::from(format!("{}.{}.run", output_root.display(), association_mode))
    };
    let output_directory_name = match output_format {
        OutputFileFormat::Arrow => "chunks",
        OutputFileFormat::Parquet => "parts",
        OutputFileFormat::Regenie => "regenie",
    };
    OutputRunPaths { chunks_directory: run_directory.join(output_directory_name), run_directory }
}

pub fn prepare_output_run(
    output_root: &Path,
    association_mode: &str,
    output_format: OutputFileFormat,
    resume: bool,
) -> Result<PreparedOutputRun, OutputError> {
    let output_run_paths = resolve_output_run_paths(output_root, association_mode, output_format);
    if !resume && directory_exists_and_is_non_empty(&output_run_paths.run_directory)? {
        return Err(OutputError::InvalidInput(format!(
            "Output run directory '{}' already exists and is not empty. Use --resume or choose a new output path.",
            output_run_paths.run_directory.display()
        )));
    }
    std::fs::create_dir_all(&output_run_paths.chunks_directory).map_err(OutputError::runtime)?;
    let existing_manifest_json = load_run_manifest_json(&output_run_paths.run_directory)?;
    if resume && existing_manifest_json.is_none() {
        return Err(OutputError::InvalidInput("Resume requires run_manifest.json.".to_string()));
    }
    Ok(PreparedOutputRun { output_run_paths, existing_manifest_json })
}

pub fn load_run_manifest_json(run_directory: &Path) -> Result<Option<String>, OutputError> {
    let manifest_path = run_directory.join(RUN_MANIFEST_FILE_NAME);
    if !manifest_path.exists() {
        return Ok(None);
    }
    let manifest_json = std::fs::read_to_string(&manifest_path).map_err(OutputError::runtime)?;
    parse_run_manifest_text(&manifest_json, Some(&manifest_path))?;
    Ok(Some(manifest_json))
}

pub fn write_run_manifest_json(run_directory: &Path, manifest_json: &str) -> Result<(), OutputError> {
    let manifest = parse_run_manifest_text(manifest_json, None)?;
    write_run_manifest_value(run_directory, &manifest)
}

pub fn extend_run_manifest_metadata(run_directory: &Path, command: Value, runtime: Value) -> Result<(), OutputError> {
    upsert_run_manifest(run_directory, |manifest| {
        let manifest_object = manifest
            .as_object_mut()
            .ok_or_else(|| OutputError::InvalidInput("Run manifest must contain a JSON object.".to_string()))?;
        manifest_object.insert("command".to_string(), command);
        manifest_object.insert("runtime".to_string(), runtime);
        Ok(())
    })
}

pub fn validate_run_manifest_compatibility(manifest_json: &str, current_header_json: &str) -> Result<(), OutputError> {
    let manifest = parse_run_manifest_text(manifest_json, None)?;
    let current_header = parse_current_header_text(current_header_json)?;
    validation::validate_manifest_compatibility_values(&manifest, &current_header)
}

pub fn read_run_manifest_committed_chunk_identifiers_from_text(manifest_json: &str) -> Result<Vec<i64>, OutputError> {
    let manifest = parse_run_manifest_text(manifest_json, None)?;
    read_run_manifest_committed_chunk_identifiers(&manifest)
}

pub fn initialize_output_run(
    run_directory: &Path,
    chunks_directory: &Path,
    existing_manifest_json: Option<&str>,
    current_header_json: &str,
    resume: bool,
    resume_mode: OutputResumeMode,
) -> Result<InitializedOutputRun, OutputError> {
    let current_header = parse_current_header_text(current_header_json)?;
    let (mut manifest, committed_chunks, committed_chunk_identifiers) = if let Some(existing_manifest_text) =
        existing_manifest_json
    {
        let existing_manifest = parse_run_manifest_text(existing_manifest_text, None)?;
        validation::validate_manifest_compatibility_values(&existing_manifest, &current_header)?;
        let manifest_committed_chunks = read_run_manifest_committed_chunks(&existing_manifest)?;
        let _validated_committed_chunk_identifiers = read_run_manifest_committed_chunk_identifiers(&existing_manifest)?;
        if resume {
            match resume_mode {
                OutputResumeMode::Fast => {
                    let committed_chunk_identifiers =
                        read_run_manifest_committed_chunk_identifiers(&existing_manifest)?;
                    (existing_manifest, manifest_committed_chunks, committed_chunk_identifiers)
                }
                OutputResumeMode::Strict => {
                    let repaired_commits =
                        resume::repair_strict_manifest_chunk_commits(chunks_directory, existing_manifest_text)?;
                    let committed_chunk_identifiers =
                        repaired_commits.iter().map(|chunk_commit| chunk_commit.chunk_identifier).collect();
                    let committed_chunks = repaired_commits.iter().map(chunk_commit_to_value).collect::<Vec<_>>();
                    (existing_manifest, committed_chunks, committed_chunk_identifiers)
                }
            }
        } else {
            (existing_manifest, manifest_committed_chunks, Vec::new())
        }
    } else {
        if resume {
            return Err(OutputError::InvalidInput("Resume requires run_manifest.json.".to_string()));
        }
        let manifest = load_run_manifest_value(run_directory)?.unwrap_or_else(|| Value::Object(Map::new()));
        (manifest, Vec::new(), Vec::new())
    };
    merge_manifest_header(&mut manifest, &current_header)?;
    let manifest_object = manifest
        .as_object_mut()
        .ok_or_else(|| OutputError::InvalidInput("Run manifest must contain a JSON object.".to_string()))?;
    manifest_object.insert("committed_chunks".to_string(), Value::Array(committed_chunks));
    manifest_object.entry("finalized".to_string()).or_insert(Value::Bool(false));
    write_run_manifest_value(run_directory, &manifest)?;
    Ok(InitializedOutputRun { committed_chunk_identifiers })
}

pub(crate) fn read_run_manifest_chunk_commits(run_directory: &Path) -> ManifestResult<Vec<RunManifestChunkCommit>> {
    let manifest_path = run_directory.join(RUN_MANIFEST_FILE_NAME);
    let manifest_text = std::fs::read_to_string(&manifest_path).map_err(OutputError::runtime)?;
    read_run_manifest_chunk_commits_from_text(&manifest_text)
}

pub(crate) fn read_run_manifest_chunk_commits_from_text(
    manifest_json: &str,
) -> ManifestResult<Vec<RunManifestChunkCommit>> {
    let manifest = parse_run_manifest_text(manifest_json, None)?;
    let committed_chunks = manifest
        .get("committed_chunks")
        .and_then(Value::as_array)
        .ok_or_else(|| OutputError::InvalidInput("Run manifest committed_chunks field must be a list.".to_string()))?;
    let mut committed_chunks_by_identifier = BTreeMap::new();
    for committed_chunk in committed_chunks {
        insert_or_validate_chunk_commit(
            &mut committed_chunks_by_identifier,
            chunks::read_run_manifest_chunk_commit(committed_chunk)?,
        )?;
    }
    Ok(committed_chunks_by_identifier.into_values().collect())
}

pub(crate) fn record_run_manifest_chunk_commits(
    run_directory: &Path,
    chunk_commits: Vec<RunManifestChunkCommit>,
) -> ManifestResult<()> {
    if chunk_commits.is_empty() {
        return Ok(());
    }
    update_run_manifest(run_directory, |manifest| {
        let manifest_object = manifest
            .as_object_mut()
            .ok_or_else(|| OutputError::InvalidInput("Run manifest must contain a JSON object.".to_string()))?;
        let committed_chunks = manifest_object
            .entry("committed_chunks".to_string())
            .or_insert_with(|| Value::Array(Vec::new()))
            .as_array_mut()
            .ok_or_else(|| {
                OutputError::InvalidInput("Run manifest committed_chunks field must be a list.".to_string())
            })?;
        let mut committed_chunks_by_identifier = BTreeMap::new();
        for committed_chunk in committed_chunks.iter() {
            let existing_commit = chunks::read_run_manifest_chunk_commit(committed_chunk)?;
            insert_or_validate_chunk_commit(&mut committed_chunks_by_identifier, existing_commit)?;
        }
        for chunk_commit in chunk_commits {
            insert_or_validate_chunk_commit(&mut committed_chunks_by_identifier, chunk_commit)?;
        }
        *committed_chunks = committed_chunks_by_identifier.values().map(chunk_commit_to_value).collect();
        Ok(())
    })
}

pub(crate) fn mark_run_manifest_finalized(
    final_parquet_path: &Path,
    row_count: usize,
    chunk_file_count: usize,
) -> ManifestResult<()> {
    mark_run_manifest_finalized_output(final_parquet_path, row_count, chunk_file_count, "parquet")
}

pub(crate) fn mark_run_manifest_finalized_output(
    final_output_path: &Path,
    row_count: usize,
    chunk_file_count: usize,
    output_format: &str,
) -> ManifestResult<()> {
    let Some(run_directory) = final_output_path.parent() else {
        return Ok(());
    };
    update_run_manifest(run_directory, |manifest| {
        let manifest_object = manifest
            .as_object_mut()
            .ok_or_else(|| OutputError::InvalidInput("Run manifest must contain a JSON object.".to_string()))?;
        manifest_object.insert("finalized".to_string(), Value::Bool(true));
        manifest_object.insert("final_output".to_string(), Value::String(final_output_path.display().to_string()));
        manifest_object.insert("final_output_format".to_string(), Value::String(output_format.to_string()));
        match output_format {
            "parquet" => {
                manifest_object
                    .insert("final_parquet".to_string(), Value::String(final_output_path.display().to_string()));
                manifest_object.remove("final_regenie");
            }
            "regenie" => {
                manifest_object
                    .insert("final_regenie".to_string(), Value::String(final_output_path.display().to_string()));
                manifest_object.remove("final_parquet");
            }
            _ => {
                manifest_object.remove("final_parquet");
                manifest_object.remove("final_regenie");
            }
        }
        manifest_object.insert("final_row_count".to_string(), json!(row_count));
        manifest_object.insert("final_chunk_file_count".to_string(), json!(chunk_file_count));
        manifest_object.remove("interrupted");
        manifest_object.remove("interrupted_signal");
        Ok(())
    })
}

pub(crate) fn mark_run_manifest_interrupted(run_directory: &Path, signal_name: &str) -> ManifestResult<()> {
    update_run_manifest(run_directory, |manifest| {
        let manifest_object = manifest
            .as_object_mut()
            .ok_or_else(|| OutputError::InvalidInput("Run manifest must contain a JSON object.".to_string()))?;
        manifest_object.insert("finalized".to_string(), Value::Bool(false));
        manifest_object.insert("interrupted".to_string(), Value::Bool(true));
        manifest_object.insert("interrupted_signal".to_string(), Value::String(signal_name.to_string()));
        manifest_object.remove("final_parquet");
        manifest_object.remove("final_regenie");
        manifest_object.remove("final_output");
        manifest_object.remove("final_output_format");
        manifest_object.remove("final_row_count");
        manifest_object.remove("final_chunk_file_count");
        Ok(())
    })
}

fn directory_exists_and_is_non_empty(directory_path: &Path) -> Result<bool, OutputError> {
    if !directory_path.exists() {
        return Ok(false);
    }
    let mut directory_entries = std::fs::read_dir(directory_path).map_err(OutputError::runtime)?;
    match directory_entries.next() {
        Some(Ok(_directory_entry)) => Ok(true),
        Some(Err(error)) => Err(OutputError::runtime(error)),
        None => Ok(false),
    }
}

fn load_run_manifest_value(run_directory: &Path) -> Result<Option<Value>, OutputError> {
    let Some(manifest_json) = load_run_manifest_json(run_directory)? else {
        return Ok(None);
    };
    parse_run_manifest_text(&manifest_json, Some(&run_directory.join(RUN_MANIFEST_FILE_NAME))).map(Some)
}

fn parse_run_manifest_text(manifest_json: &str, manifest_path: Option<&Path>) -> Result<Value, OutputError> {
    let manifest =
        serde_json::from_str::<Value>(manifest_json).map_err(|error| OutputError::InvalidInput(error.to_string()))?;
    if manifest.is_object() {
        return Ok(manifest);
    }
    let message = match manifest_path {
        Some(path) => format!("Run manifest '{}' must contain a JSON object.", path.display()),
        None => "Run manifest must contain a JSON object.".to_string(),
    };
    Err(OutputError::InvalidInput(message))
}

fn parse_current_header_text(current_header_json: &str) -> Result<Value, OutputError> {
    let current_header = serde_json::from_str::<Value>(current_header_json)
        .map_err(|error| OutputError::InvalidInput(error.to_string()))?;
    if current_header.is_object() {
        return Ok(current_header);
    }
    Err(OutputError::InvalidInput("Current run manifest header must contain a JSON object.".to_string()))
}

fn read_run_manifest_committed_chunks(manifest: &Value) -> Result<Vec<Value>, OutputError> {
    let Some(committed_chunks) = manifest.get("committed_chunks") else {
        return Ok(Vec::new());
    };
    let committed_chunks_array = committed_chunks
        .as_array()
        .ok_or_else(|| OutputError::InvalidInput("Run manifest committed_chunks field must be a list.".to_string()))?;
    for committed_chunk in committed_chunks_array {
        if !committed_chunk.is_object() {
            return Err(OutputError::InvalidInput("Run manifest committed chunk entries must be objects.".to_string()));
        }
    }
    Ok(committed_chunks_array.clone())
}

fn read_run_manifest_committed_chunk_identifiers(manifest: &Value) -> Result<Vec<i64>, OutputError> {
    let committed_chunks = read_run_manifest_committed_chunks(manifest)?;
    let mut committed_chunk_identifiers = BTreeSet::new();
    for committed_chunk in committed_chunks {
        let Some(chunk_identifier) = committed_chunk.get("chunk_identifier").and_then(Value::as_i64) else {
            return Err(OutputError::InvalidInput(
                "Run manifest committed chunk entry is missing chunk_identifier.".to_string(),
            ));
        };
        committed_chunk_identifiers.insert(chunk_identifier);
    }
    Ok(committed_chunk_identifiers.into_iter().collect())
}

fn merge_manifest_header(manifest: &mut Value, current_header: &Value) -> Result<(), OutputError> {
    let manifest_object = manifest
        .as_object_mut()
        .ok_or_else(|| OutputError::InvalidInput("Run manifest must contain a JSON object.".to_string()))?;
    let current_header_object = current_header.as_object().ok_or_else(|| {
        OutputError::InvalidInput("Current run manifest header must contain a JSON object.".to_string())
    })?;
    for (field_name, field_value) in current_header_object {
        manifest_object.insert(field_name.clone(), field_value.clone());
    }
    Ok(())
}

fn write_run_manifest_value(run_directory: &Path, manifest: &Value) -> ManifestResult<()> {
    let manifest_path = run_directory.join(RUN_MANIFEST_FILE_NAME);
    let manifest_lock = get_run_manifest_update_lock();
    let _manifest_guard =
        manifest_lock.lock().map_err(|_| OutputError::runtime("Run manifest update lock was poisoned."))?;
    write_run_manifest_value_atomic(&manifest_path, manifest)
}

fn update_run_manifest(
    run_directory: &Path,
    update_manifest: impl FnOnce(&mut Value) -> ManifestResult<()>,
) -> ManifestResult<()> {
    let manifest_path = run_directory.join(RUN_MANIFEST_FILE_NAME);
    if !manifest_path.exists() {
        return Ok(());
    }
    let manifest_lock = get_run_manifest_update_lock();
    let _manifest_guard =
        manifest_lock.lock().map_err(|_| OutputError::runtime("Run manifest update lock was poisoned."))?;
    let manifest_text = std::fs::read_to_string(&manifest_path).map_err(OutputError::runtime)?;
    let mut manifest = parse_run_manifest_text(&manifest_text, Some(&manifest_path))?;
    update_manifest(&mut manifest)?;
    write_run_manifest_value_atomic(&manifest_path, &manifest)
}

fn upsert_run_manifest(
    run_directory: &Path,
    update_manifest: impl FnOnce(&mut Value) -> ManifestResult<()>,
) -> ManifestResult<()> {
    let manifest_path = run_directory.join(RUN_MANIFEST_FILE_NAME);
    let manifest_lock = get_run_manifest_update_lock();
    let _manifest_guard =
        manifest_lock.lock().map_err(|_| OutputError::runtime("Run manifest update lock was poisoned."))?;
    let mut manifest = if manifest_path.exists() {
        let manifest_text = std::fs::read_to_string(&manifest_path).map_err(OutputError::runtime)?;
        parse_run_manifest_text(&manifest_text, Some(&manifest_path))?
    } else {
        Value::Object(Map::new())
    };
    update_manifest(&mut manifest)?;
    write_run_manifest_value_atomic(&manifest_path, &manifest)
}

fn write_run_manifest_value_atomic(manifest_path: &Path, manifest: &Value) -> ManifestResult<()> {
    let temporary_manifest_path = manifest_path.with_extension("json.tmp");
    let mut temporary_manifest_file = File::create(&temporary_manifest_path).map_err(OutputError::runtime)?;
    let manifest_bytes = serde_json::to_vec_pretty(manifest).map_err(OutputError::runtime)?;
    temporary_manifest_file.write_all(&manifest_bytes).map_err(OutputError::runtime)?;
    temporary_manifest_file.write_all(b"\n").map_err(OutputError::runtime)?;
    temporary_manifest_file.sync_all().map_err(OutputError::runtime)?;
    std::fs::rename(&temporary_manifest_path, manifest_path).map_err(OutputError::runtime)
}

fn get_run_manifest_update_lock() -> &'static Mutex<()> {
    static RUN_MANIFEST_UPDATE_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    RUN_MANIFEST_UPDATE_LOCK.get_or_init(|| Mutex::new(()))
}
