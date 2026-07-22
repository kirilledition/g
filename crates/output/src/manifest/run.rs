use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Write;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use serde_json::{Map, Value};

use super::chunks::{RunManifestChunkCommit, chunk_commit_to_value, insert_or_validate_chunk_commit};
use super::{RUN_MANIFEST_FILE_NAME, chunks, validation};
use crate::error::{OutputError, OutputResult};
use crate::resume;

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct OutputRunPaths {
    pub run_directory: PathBuf,
    pub parts_directory: PathBuf,
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct PreparedOutputRun {
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

pub(crate) fn prepare_output_run(
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

pub(crate) fn extend_run_manifest_metadata(
    run_directory: &Path,
    command: Value,
    runtime: Value,
) -> Result<(), OutputError> {
    update_run_manifest(run_directory, |manifest| {
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

pub(crate) fn reconcile_output_run_resume(
    parts_directory: &Path,
    manifest_json: &str,
    current_header: &Value,
    planned_chunk_ranges: &[Range<usize>],
) -> Result<Vec<RunManifestChunkCommit>, OutputError> {
    validate_run_manifest_compatibility(manifest_json, current_header)?;
    resume::repair_strict_manifest_chunk_commits(parts_directory, manifest_json, planned_chunk_ranges)
}

pub(crate) fn initialize_output_run(
    run_directory: &Path,
    existing_manifest_json: Option<&str>,
    current_header: &Value,
    resumed_chunk_commits: Option<Vec<RunManifestChunkCommit>>,
) -> Result<Vec<i64>, OutputError> {
    let (mut manifest, committed_chunks, committed_chunk_identifiers) =
        if let Some(existing_manifest_text) = existing_manifest_json {
            let existing_manifest = parse_run_manifest_text(existing_manifest_text, None)?;
            if let Some(resumed_chunk_commits) = resumed_chunk_commits {
                let committed_chunk_identifiers =
                    resumed_chunk_commits.iter().map(|chunk_commit| chunk_commit.chunk_identifier).collect();
                let committed_chunks = resumed_chunk_commits.iter().map(chunk_commit_to_value).collect::<Vec<_>>();
                (existing_manifest, committed_chunks, committed_chunk_identifiers)
            } else {
                validation::validate_manifest_compatibility_values(&existing_manifest, current_header)?;
                let manifest_committed_chunks = read_run_manifest_committed_chunks(&existing_manifest)?;
                let _validated_committed_chunk_identifiers =
                    read_run_manifest_committed_chunk_identifiers(&existing_manifest)?;
                (existing_manifest, manifest_committed_chunks, Vec::new())
            }
        } else {
            if resumed_chunk_commits.is_some() {
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
        let chunk_commit = chunks::read_run_manifest_chunk_commit(committed_chunk)?;
        if committed_chunks_by_identifier.insert(chunk_commit.chunk_identifier, chunk_commit).is_some() {
            return Err(OutputError::InvalidInput(
                "Strict resume manifest contains duplicate chunk identifiers.".to_string(),
            ));
        }
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
    let manifest_lock = get_run_manifest_update_lock();
    let _manifest_guard =
        manifest_lock.lock().map_err(|_| OutputError::runtime("Run manifest update lock was poisoned."))?;
    let manifest_text = match std::fs::read_to_string(&manifest_path) {
        Ok(manifest_text) => manifest_text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(OutputError::MissingRunManifest { manifest_path });
        }
        Err(error) => return Err(OutputError::runtime(error)),
    };
    let mut manifest = parse_run_manifest_text(&manifest_text, Some(&manifest_path))?;
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

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use serde_json::{Value, json};

    use crate::error::OutputError;

    use super::{
        RUN_MANIFEST_FILE_NAME, RunManifestChunkCommit, extend_run_manifest_metadata, initialize_output_run,
        load_run_manifest_json, mark_run_manifest_completed, mark_run_manifest_interrupted, prepare_output_run,
        read_run_manifest_chunk_commits_from_text, read_run_manifest_gpu_genotype_format_from_text,
        record_run_manifest_chunk_commits, resolve_output_run_paths,
    };

    struct TestDirectory {
        path: PathBuf,
    }

    impl TestDirectory {
        fn new(label: &str) -> Self {
            static DIRECTORY_COUNTER: AtomicU64 = AtomicU64::new(0);
            let sequence = DIRECTORY_COUNTER.fetch_add(1, Ordering::Relaxed);
            let timestamp =
                SystemTime::now().duration_since(UNIX_EPOCH).expect("test time is after Unix epoch").as_nanos();
            let path = std::env::temp_dir()
                .join(format!("g-output-manifest-{label}-{}-{timestamp}-{sequence}", std::process::id()));
            std::fs::create_dir_all(&path).expect("test directory is created");
            Self { path }
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn read_manifest(run_directory: &Path) -> Value {
        serde_json::from_str(
            &std::fs::read_to_string(run_directory.join(RUN_MANIFEST_FILE_NAME)).expect("manifest reads"),
        )
        .expect("manifest is valid JSON")
    }

    fn chunk_commit(chunk_identifier: i64, file_name: &str) -> RunManifestChunkCommit {
        RunManifestChunkCommit {
            chunk_identifier,
            variant_start_index: chunk_identifier,
            variant_stop_index: chunk_identifier + 2,
            row_count: 2,
            chunk_file_name: file_name.to_string(),
        }
    }

    #[test]
    fn output_paths_accept_explicit_run_directory_or_add_mode_suffix() {
        let explicit = resolve_output_run_paths(Path::new("results.run"), "regenie2_binary");
        assert_eq!(explicit.run_directory, Path::new("results.run"));
        assert_eq!(explicit.parts_directory, Path::new("results.run/parts"));

        let derived = resolve_output_run_paths(Path::new("results"), "regenie2_binary");
        assert_eq!(derived.run_directory, Path::new("results.regenie2_binary.run"));
        assert_eq!(derived.parts_directory, Path::new("results.regenie2_binary.run/parts"));
    }

    #[test]
    fn prepare_output_run_enforces_fresh_and_resume_preconditions() {
        let directory = TestDirectory::new("prepare");
        let root = directory.path.join("fresh");
        let prepared = prepare_output_run(&root, "regenie2_binary", false).expect("fresh run prepares");
        assert!(prepared.output_run_paths.parts_directory.is_dir());
        assert_eq!(prepared.existing_manifest_json, None);

        std::fs::write(prepared.output_run_paths.run_directory.join("sentinel"), b"occupied").expect("sentinel writes");
        let error = prepare_output_run(&root, "regenie2_binary", false).expect_err("nonempty fresh run is rejected");
        assert!(error.to_string().contains("already exists and is not empty"));

        let resume_root = directory.path.join("resume");
        let error =
            prepare_output_run(&resume_root, "regenie2_binary", true).expect_err("resume without manifest is rejected");
        assert!(error.to_string().contains("Resume requires run_manifest.json"));
    }

    #[test]
    fn manifest_loader_rejects_invalid_json_and_non_object_roots() {
        let directory = TestDirectory::new("load");
        assert_eq!(load_run_manifest_json(&directory.path).expect("missing manifest is valid"), None);
        for malformed in ["not-json", "[]"] {
            std::fs::write(directory.path.join(RUN_MANIFEST_FILE_NAME), malformed).expect("fixture writes");
            assert!(load_run_manifest_json(&directory.path).is_err());
        }
    }

    #[test]
    fn manifest_gpu_format_reader_accepts_public_formats_and_rejects_unknown_or_missing() {
        for (format, expected) in
            [("dosage", g_plan::GpuGenotypeFormat::Dosage), ("packed8", g_plan::GpuGenotypeFormat::Packed8)]
        {
            let manifest = json!({"execution_plan": {"association_backend": {"genotype_format": format}}});
            assert_eq!(
                read_run_manifest_gpu_genotype_format_from_text(&manifest.to_string()).expect("format reads"),
                expected
            );
        }

        let unsupported = json!({"execution_plan": {"association_backend": {"genotype_format": "packed16"}}});
        let error = read_run_manifest_gpu_genotype_format_from_text(&unsupported.to_string())
            .expect_err("unsupported format is rejected");
        assert!(error.to_string().contains("unsupported execution-plan GPU genotype format 'packed16'"));
        assert!(read_run_manifest_gpu_genotype_format_from_text("{}").is_err());
    }

    #[test]
    fn manifest_lifecycle_is_atomic_sorted_and_idempotent() {
        let directory = TestDirectory::new("lifecycle");
        let header = json!({"schema_version": 0, "output_schema_version": 0, "execution_plan": {"name": "test"}});
        let identifiers =
            initialize_output_run(&directory.path, None, &header, None).expect("new manifest initializes");
        assert!(identifiers.is_empty());
        assert_eq!(read_manifest(&directory.path)["status"], "running");

        record_run_manifest_chunk_commits(
            &directory.path,
            vec![chunk_commit(4, "part-4.parquet"), chunk_commit(0, "part-0.parquet")],
        )
        .expect("commits record");
        record_run_manifest_chunk_commits(&directory.path, vec![chunk_commit(0, "part-0.parquet")])
            .expect("identical commit replay is idempotent");
        let manifest = read_manifest(&directory.path);
        let identifiers = manifest["committed_chunks"]
            .as_array()
            .expect("commits are a list")
            .iter()
            .map(|commit| commit["chunk_identifier"].as_i64().expect("identifier is int64"))
            .collect::<Vec<_>>();
        assert_eq!(identifiers, [0, 4]);

        let error = record_run_manifest_chunk_commits(&directory.path, vec![chunk_commit(0, "conflict.parquet")])
            .expect_err("conflicting replay is rejected");
        assert!(error.to_string().contains("conflicting commit metadata"));

        mark_run_manifest_interrupted(&directory.path, "SIGTERM").expect("manifest marks interrupted");
        let interrupted = read_manifest(&directory.path);
        assert_eq!(interrupted["status"], "interrupted");
        assert_eq!(interrupted["interrupted_signal"], "SIGTERM");
        mark_run_manifest_completed(&directory.path).expect("manifest marks complete");
        let completed = read_manifest(&directory.path);
        assert_eq!(completed["status"], "completed");
        assert!(completed.get("interrupted_signal").is_none());

        extend_run_manifest_metadata(&directory.path, json!({"name": "g"}), json!({"gpu": "test"}))
            .expect("metadata extends");
        let extended = read_manifest(&directory.path);
        assert_eq!(extended["command"]["name"], "g");
        assert_eq!(extended["runtime"]["gpu"], "test");
    }

    #[test]
    fn lifecycle_updates_reject_a_missing_manifest_without_recreating_it() {
        let directory = TestDirectory::new("missing-manifest");
        let manifest_path = directory.path.join(RUN_MANIFEST_FILE_NAME);
        let update_errors = [
            mark_run_manifest_completed(&directory.path).expect_err("completion requires a manifest"),
            mark_run_manifest_interrupted(&directory.path, "SIGINT").expect_err("interruption requires a manifest"),
            record_run_manifest_chunk_commits(&directory.path, vec![chunk_commit(0, "part-0.parquet")])
                .expect_err("chunk commits require a manifest"),
            extend_run_manifest_metadata(&directory.path, json!(["g", "run"]), json!({"device": "gpu"}))
                .expect_err("metadata extension requires a manifest"),
        ];
        for error in update_errors {
            assert!(
                matches!(
                    error,
                    OutputError::MissingRunManifest { manifest_path: ref observed_path }
                        if observed_path == &manifest_path
                ),
                "unexpected missing-manifest error: {error}"
            );
        }
        assert!(!manifest_path.exists());
    }

    #[test]
    fn resumed_initialization_replaces_commits_and_clears_interruption() {
        let directory = TestDirectory::new("resume");
        let header = json!({"schema_version": 0, "execution_plan": {"name": "test"}});
        let existing = json!({
            "schema_version": 0,
            "execution_plan": {"name": "test"},
            "status": "interrupted",
            "interrupted_signal": "SIGTERM",
            "committed_chunks": [],
        });
        let commits = vec![chunk_commit(4, "part-4.parquet"), chunk_commit(0, "part-0.parquet")];
        let identifiers = initialize_output_run(&directory.path, Some(&existing.to_string()), &header, Some(commits))
            .expect("resumed manifest initializes");
        assert_eq!(identifiers, [4, 0]);
        let manifest = read_manifest(&directory.path);
        assert_eq!(manifest["status"], "running");
        assert!(manifest.get("interrupted_signal").is_none());
        assert_eq!(manifest["committed_chunks"].as_array().expect("commits are a list").len(), 2);

        let error = initialize_output_run(&directory.path, None, &header, Some(Vec::new()))
            .expect_err("resume commits require existing manifest");
        assert!(error.to_string().contains("Resume requires run_manifest.json"));
    }

    #[test]
    fn manifest_commit_reader_sorts_and_rejects_duplicate_or_malformed_entries() {
        let manifest = json!({
            "committed_chunks": [
                {"chunk_identifier": 4, "variant_start_index": 4, "variant_stop_index": 6, "row_count": 2, "chunk_file_name": "part-4.parquet"},
                {"chunk_identifier": 0, "variant_start_index": 0, "variant_stop_index": 2, "row_count": 2, "chunk_file_name": "part-0.parquet"},
            ],
        });
        let commits = read_run_manifest_chunk_commits_from_text(&manifest.to_string()).expect("commits read");
        assert_eq!(commits.iter().map(|commit| commit.chunk_identifier).collect::<Vec<_>>(), [0, 4]);

        let duplicate = json!({
            "committed_chunks": [
                {"chunk_identifier": 0, "variant_start_index": 0, "variant_stop_index": 2, "row_count": 2, "chunk_file_name": "first.parquet"},
                {"chunk_identifier": 0, "variant_start_index": 0, "variant_stop_index": 2, "row_count": 2, "chunk_file_name": "second.parquet"},
            ],
        });
        let error = read_run_manifest_chunk_commits_from_text(&duplicate.to_string())
            .expect_err("duplicate commit identifier is rejected");
        assert!(error.to_string().contains("duplicate chunk identifiers"));
        assert!(read_run_manifest_chunk_commits_from_text(r#"{"committed_chunks": {}}"#).is_err());
        assert!(read_run_manifest_chunk_commits_from_text("{}").is_err());
    }
}
