use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use serde_json::{Map, Value};

use super::chunks::{RunManifestChunkCommit, chunk_commit_to_value, insert_or_validate_chunk_commit};
use super::{RUN_MANIFEST_FILE_NAME, chunks, validation};
use crate::error::{OutputError, OutputResult};
use crate::resume;

#[derive(Debug, Eq, PartialEq)]
pub struct OutputRunPaths {
    pub run_directory: PathBuf,
    pub parts_directory: PathBuf,
}

#[derive(Debug, Eq, PartialEq)]
pub struct PreparedOutputRun {
    pub output_run_paths: OutputRunPaths,
    pub existing_manifest_json: Option<String>,
}

#[must_use]
fn resolve_output_run_paths(output_root: &Path, association_mode: &str) -> OutputRunPaths {
    let run_directory = if output_root.extension().is_some_and(|extension| extension == "run") {
        output_root.to_path_buf()
    } else {
        PathBuf::from(format!("{}.{}.run", output_root.display(), association_mode))
    };
    OutputRunPaths { parts_directory: run_directory.join("parts"), run_directory }
}

pub fn prepare_output_run(
    output_root: &Path,
    association_mode: &str,
    resume: bool,
) -> Result<PreparedOutputRun, OutputError> {
    let output_run_paths = resolve_output_run_paths(output_root, association_mode);
    if !resume && directory_exists_and_is_non_empty(&output_run_paths.run_directory)? {
        return Err(OutputError::InvalidInput(format!(
            "Output run directory '{}' already exists and is not empty. Enable [output].resume or choose a new output path.",
            output_run_paths.run_directory.display()
        )));
    }
    std::fs::create_dir_all(&output_run_paths.parts_directory).map_err(OutputError::runtime)?;
    let existing_manifest_json = load_run_manifest_json(&output_run_paths.run_directory)?;
    if resume && existing_manifest_json.is_none() {
        return Err(OutputError::InvalidInput("Resume requires run_manifest.json.".to_string()));
    }
    Ok(PreparedOutputRun { output_run_paths, existing_manifest_json })
}

pub(crate) fn load_run_manifest_json(run_directory: &Path) -> Result<Option<String>, OutputError> {
    let manifest_path = run_directory.join(RUN_MANIFEST_FILE_NAME);
    if !manifest_path.exists() {
        return Ok(None);
    }
    let manifest_json = std::fs::read_to_string(&manifest_path).map_err(OutputError::runtime)?;
    parse_run_manifest_text(&manifest_json, Some(&manifest_path))?;
    Ok(Some(manifest_json))
}

pub(crate) fn read_run_manifest_gpu_genotype_format_from_text(
    manifest_json: &str,
) -> Result<g_plan::GpuGenotypeFormat, OutputError> {
    let manifest = parse_run_manifest_text(manifest_json, None)?;
    let genotype_format = manifest
        .pointer("/execution_plan/association_backend/genotype_format")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            OutputError::InvalidInput(
                "Run manifest execution_plan.association_backend.genotype_format is missing.".to_string(),
            )
        })?;
    match genotype_format {
        "dosage" => Ok(g_plan::GpuGenotypeFormat::Dosage),
        "packed8" => Ok(g_plan::GpuGenotypeFormat::Packed8),
        unsupported_format => Err(OutputError::InvalidInput(format!(
            "Run manifest has unsupported execution-plan GPU genotype format '{unsupported_format}'."
        ))),
    }
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

fn validate_run_manifest_compatibility(manifest_json: &str, current_header: &Value) -> Result<(), OutputError> {
    let manifest = parse_run_manifest_text(manifest_json, None)?;
    validation::validate_manifest_compatibility_values(&manifest, current_header)
}

pub fn validate_output_run_resume_compatibility(
    parts_directory: &Path,
    manifest_json: &str,
    current_header: &Value,
    resume_mode: g_plan::ResumeMode,
) -> Result<(), OutputError> {
    validate_run_manifest_compatibility(manifest_json, current_header)?;
    if resume_mode == g_plan::ResumeMode::Strict {
        resume::repair_strict_manifest_chunk_commits(parts_directory, manifest_json)?;
    }
    Ok(())
}

pub fn initialize_output_run(
    run_directory: &Path,
    parts_directory: &Path,
    existing_manifest_json: Option<&str>,
    current_header: &Value,
    resume: bool,
    resume_mode: g_plan::ResumeMode,
) -> Result<Vec<i64>, OutputError> {
    let (mut manifest, committed_chunks, committed_chunk_identifiers) = if let Some(existing_manifest_text) =
        existing_manifest_json
    {
        let existing_manifest = parse_run_manifest_text(existing_manifest_text, None)?;
        validation::validate_manifest_compatibility_values(&existing_manifest, current_header)?;
        let manifest_committed_chunks = read_run_manifest_committed_chunks(&existing_manifest)?;
        let _validated_committed_chunk_identifiers = read_run_manifest_committed_chunk_identifiers(&existing_manifest)?;
        if resume {
            match resume_mode {
                g_plan::ResumeMode::Fast => {
                    let committed_chunk_identifiers =
                        read_run_manifest_committed_chunk_identifiers(&existing_manifest)?;
                    (existing_manifest, manifest_committed_chunks, committed_chunk_identifiers)
                }
                g_plan::ResumeMode::Strict => {
                    let repaired_commits =
                        resume::repair_strict_manifest_chunk_commits(parts_directory, existing_manifest_text)?;
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
    merge_manifest_header(&mut manifest, current_header)?;
    let manifest_object = manifest
        .as_object_mut()
        .ok_or_else(|| OutputError::InvalidInput("Run manifest must contain a JSON object.".to_string()))?;
    manifest_object.insert("committed_chunks".to_string(), Value::Array(committed_chunks));
    manifest_object.insert("status".to_string(), Value::String("running".to_string()));
    manifest_object.remove("interrupted_signal");
    write_run_manifest_value(run_directory, &manifest)?;
    Ok(committed_chunk_identifiers)
}

pub(crate) fn read_run_manifest_chunk_commits_from_text(
    manifest_json: &str,
) -> OutputResult<Vec<RunManifestChunkCommit>> {
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
) -> OutputResult<()> {
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

pub(crate) fn mark_run_manifest_completed(run_directory: &Path) -> OutputResult<()> {
    update_run_manifest(run_directory, |manifest| {
        let manifest_object = manifest
            .as_object_mut()
            .ok_or_else(|| OutputError::InvalidInput("Run manifest must contain a JSON object.".to_string()))?;
        manifest_object.insert("status".to_string(), Value::String("completed".to_string()));
        manifest_object.remove("interrupted_signal");
        Ok(())
    })
}

pub(crate) fn mark_run_manifest_interrupted(run_directory: &Path, signal_name: &str) -> OutputResult<()> {
    update_run_manifest(run_directory, |manifest| {
        let manifest_object = manifest
            .as_object_mut()
            .ok_or_else(|| OutputError::InvalidInput("Run manifest must contain a JSON object.".to_string()))?;
        manifest_object.insert("status".to_string(), Value::String("interrupted".to_string()));
        manifest_object.insert("interrupted_signal".to_string(), Value::String(signal_name.to_string()));
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

fn write_run_manifest_value(run_directory: &Path, manifest: &Value) -> OutputResult<()> {
    let manifest_path = run_directory.join(RUN_MANIFEST_FILE_NAME);
    let manifest_lock = get_run_manifest_update_lock();
    let _manifest_guard =
        manifest_lock.lock().map_err(|_| OutputError::runtime("Run manifest update lock was poisoned."))?;
    write_run_manifest_value_atomic(&manifest_path, manifest)
}

fn update_run_manifest(
    run_directory: &Path,
    update_manifest: impl FnOnce(&mut Value) -> OutputResult<()>,
) -> OutputResult<()> {
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
    update_manifest: impl FnOnce(&mut Value) -> OutputResult<()>,
) -> OutputResult<()> {
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

fn write_run_manifest_value_atomic(manifest_path: &Path, manifest: &Value) -> OutputResult<()> {
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
