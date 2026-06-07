use std::collections::BTreeMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use serde_json::{Value, json};

const RUN_MANIFEST_FILE_NAME: &str = "run_manifest.json";

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RunManifestChunkCommit {
    pub(crate) chunk_identifier: i64,
    pub(crate) output_format: String,
    pub(crate) compression: String,
    pub(crate) variant_start_index: i64,
    pub(crate) variant_stop_index: i64,
    pub(crate) row_count: usize,
    pub(crate) chunk_file_name: String,
}

pub(crate) fn read_run_manifest_chunk_commits(run_directory: &Path) -> Result<Vec<RunManifestChunkCommit>, String> {
    let manifest_path = run_directory.join(RUN_MANIFEST_FILE_NAME);
    let manifest_text = std::fs::read_to_string(&manifest_path).map_err(|error| error.to_string())?;
    read_run_manifest_chunk_commits_from_text(&manifest_text)
}

pub(crate) fn read_run_manifest_chunk_commits_from_text(
    manifest_json: &str,
) -> Result<Vec<RunManifestChunkCommit>, String> {
    let manifest = serde_json::from_str::<Value>(manifest_json).map_err(|error| error.to_string())?;
    let committed_chunks = manifest
        .get("committed_chunks")
        .and_then(Value::as_array)
        .ok_or_else(|| "Run manifest committed_chunks field must be a list.".to_string())?;
    let mut committed_chunks_by_identifier = BTreeMap::new();
    for committed_chunk in committed_chunks {
        insert_or_validate_chunk_commit(
            &mut committed_chunks_by_identifier,
            read_run_manifest_chunk_commit(committed_chunk)?,
        )?;
    }
    Ok(committed_chunks_by_identifier.into_values().collect())
}

pub(crate) fn record_run_manifest_chunk_commits(
    run_directory: &Path,
    chunk_commits: Vec<RunManifestChunkCommit>,
) -> Result<(), String> {
    if chunk_commits.is_empty() {
        return Ok(());
    }
    update_run_manifest(run_directory, |manifest| {
        let manifest_object =
            manifest.as_object_mut().ok_or_else(|| "Run manifest must contain a JSON object.".to_string())?;
        let committed_chunks = manifest_object
            .entry("committed_chunks")
            .or_insert_with(|| Value::Array(Vec::new()))
            .as_array_mut()
            .ok_or_else(|| "Run manifest committed_chunks field must be a list.".to_string())?;
        let mut committed_chunks_by_identifier = BTreeMap::new();
        for committed_chunk in committed_chunks.iter() {
            let existing_commit = read_run_manifest_chunk_commit(committed_chunk)?;
            insert_or_validate_chunk_commit(&mut committed_chunks_by_identifier, existing_commit)?;
        }
        for chunk_commit in chunk_commits {
            insert_or_validate_chunk_commit(&mut committed_chunks_by_identifier, chunk_commit)?;
        }
        *committed_chunks = committed_chunks_by_identifier.values().map(chunk_commit_to_value).collect();
        Ok(())
    })
}

fn insert_or_validate_chunk_commit(
    committed_chunks_by_identifier: &mut BTreeMap<i64, RunManifestChunkCommit>,
    chunk_commit: RunManifestChunkCommit,
) -> Result<(), String> {
    match committed_chunks_by_identifier.get(&chunk_commit.chunk_identifier) {
        Some(existing_commit) if existing_commit != &chunk_commit => {
            Err(format!("Run manifest has conflicting commit metadata for chunk {}.", chunk_commit.chunk_identifier))
        }
        Some(_) => Ok(()),
        None => {
            committed_chunks_by_identifier.insert(chunk_commit.chunk_identifier, chunk_commit);
            Ok(())
        }
    }
}

fn chunk_commit_to_value(chunk_commit: &RunManifestChunkCommit) -> Value {
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

fn read_run_manifest_chunk_commit(committed_chunk: &Value) -> Result<RunManifestChunkCommit, String> {
    let chunk_file_name = committed_chunk
        .get("chunk_file_name")
        .and_then(Value::as_str)
        .ok_or_else(|| "Run manifest committed chunk entry is missing chunk_file_name.".to_string())?;
    Ok(RunManifestChunkCommit {
        chunk_identifier: read_manifest_integer(committed_chunk, "chunk_identifier")?,
        output_format: read_optional_manifest_string(committed_chunk, "output_format")
            .unwrap_or_else(|| infer_output_format_from_file_name(chunk_file_name).to_string()),
        compression: read_optional_manifest_string(committed_chunk, "compression")
            .unwrap_or_else(|| "none".to_string()),
        variant_start_index: read_manifest_integer(committed_chunk, "variant_start_index")?,
        variant_stop_index: read_manifest_integer(committed_chunk, "variant_stop_index")?,
        row_count: read_manifest_usize(committed_chunk, "row_count")?,
        chunk_file_name: chunk_file_name.to_string(),
    })
}

fn read_optional_manifest_string(committed_chunk: &Value, field_name: &str) -> Option<String> {
    committed_chunk.get(field_name).and_then(Value::as_str).map(str::to_string)
}

fn infer_output_format_from_file_name(chunk_file_name: &str) -> &'static str {
    if chunk_file_name.ends_with(".parquet") {
        return "parquet";
    }
    "arrow"
}

fn read_manifest_integer(committed_chunk: &Value, field_name: &str) -> Result<i64, String> {
    committed_chunk
        .get(field_name)
        .and_then(Value::as_i64)
        .ok_or_else(|| format!("Run manifest committed chunk entry is missing {field_name}."))
}

fn read_manifest_usize(committed_chunk: &Value, field_name: &str) -> Result<usize, String> {
    let value = read_manifest_integer(committed_chunk, field_name)?;
    usize::try_from(value).map_err(|_| format!("Run manifest committed chunk entry {field_name} must be non-negative."))
}

pub(crate) fn mark_run_manifest_finalized(
    final_parquet_path: &Path,
    row_count: usize,
    chunk_file_count: usize,
) -> Result<(), String> {
    let Some(run_directory) = final_parquet_path.parent() else {
        return Ok(());
    };
    update_run_manifest(run_directory, |manifest| {
        let manifest_object =
            manifest.as_object_mut().ok_or_else(|| "Run manifest must contain a JSON object.".to_string())?;
        manifest_object.insert("finalized".to_string(), Value::Bool(true));
        manifest_object.insert("final_parquet".to_string(), Value::String(final_parquet_path.display().to_string()));
        manifest_object.insert("final_row_count".to_string(), json!(row_count));
        manifest_object.insert("final_chunk_file_count".to_string(), json!(chunk_file_count));
        manifest_object.remove("interrupted");
        manifest_object.remove("interrupted_signal");
        Ok(())
    })
}

pub(crate) fn mark_run_manifest_interrupted(run_directory: &Path, signal_name: &str) -> Result<(), String> {
    update_run_manifest(run_directory, |manifest| {
        let manifest_object =
            manifest.as_object_mut().ok_or_else(|| "Run manifest must contain a JSON object.".to_string())?;
        manifest_object.insert("finalized".to_string(), Value::Bool(false));
        manifest_object.insert("interrupted".to_string(), Value::Bool(true));
        manifest_object.insert("interrupted_signal".to_string(), Value::String(signal_name.to_string()));
        manifest_object.remove("final_parquet");
        manifest_object.remove("final_row_count");
        manifest_object.remove("final_chunk_file_count");
        Ok(())
    })
}

fn update_run_manifest(
    run_directory: &Path,
    update_manifest: impl FnOnce(&mut Value) -> Result<(), String>,
) -> Result<(), String> {
    let manifest_path = run_directory.join(RUN_MANIFEST_FILE_NAME);
    if !manifest_path.exists() {
        return Ok(());
    }
    let manifest_lock = get_run_manifest_update_lock();
    let _manifest_guard = manifest_lock.lock().map_err(|_| "Run manifest update lock was poisoned.".to_string())?;
    let manifest_text = std::fs::read_to_string(&manifest_path).map_err(|error| error.to_string())?;
    let mut manifest = serde_json::from_str::<Value>(&manifest_text).map_err(|error| error.to_string())?;
    update_manifest(&mut manifest)?;
    let temporary_manifest_path = manifest_path.with_extension("json.tmp");
    let mut temporary_manifest_file = File::create(&temporary_manifest_path).map_err(|error| error.to_string())?;
    let manifest_bytes = serde_json::to_vec_pretty(&manifest).map_err(|error| error.to_string())?;
    temporary_manifest_file.write_all(&manifest_bytes).map_err(|error| error.to_string())?;
    temporary_manifest_file.write_all(b"\n").map_err(|error| error.to_string())?;
    temporary_manifest_file.sync_all().map_err(|error| error.to_string())?;
    std::fs::rename(&temporary_manifest_path, &manifest_path).map_err(|error| error.to_string())
}

fn get_run_manifest_update_lock() -> &'static Mutex<()> {
    static RUN_MANIFEST_UPDATE_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    RUN_MANIFEST_UPDATE_LOCK.get_or_init(|| Mutex::new(()))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn create_test_directory() -> PathBuf {
        let unique_suffix =
            SystemTime::now().duration_since(UNIX_EPOCH).expect("system time should be after Unix epoch").as_nanos();
        let directory_path = std::env::temp_dir().join(format!("g-output-manifest-test-{unique_suffix}"));
        std::fs::create_dir_all(&directory_path).expect("test directory should be created");
        directory_path
    }

    fn build_chunk_commit(chunk_identifier: i64) -> RunManifestChunkCommit {
        RunManifestChunkCommit {
            chunk_identifier,
            output_format: "arrow".to_string(),
            compression: "none".to_string(),
            variant_start_index: chunk_identifier,
            variant_stop_index: chunk_identifier + 2,
            row_count: 2,
            chunk_file_name: format!("chunk_{chunk_identifier:09}.arrow"),
        }
    }

    #[test]
    fn records_committed_chunks_once_in_identifier_order() {
        let run_directory = create_test_directory();
        let manifest_path = run_directory.join(RUN_MANIFEST_FILE_NAME);
        std::fs::write(&manifest_path, "{\n  \"committed_chunks\": []\n}\n").expect("manifest should be written");

        record_run_manifest_chunk_commits(
            &run_directory,
            vec![build_chunk_commit(2), build_chunk_commit(0), build_chunk_commit(2)],
        )
        .expect("manifest commits should be recorded");

        let manifest_text = std::fs::read_to_string(&manifest_path).expect("manifest should be readable");
        let manifest = serde_json::from_str::<Value>(&manifest_text).expect("manifest should be JSON");
        let committed_chunks =
            manifest.get("committed_chunks").and_then(Value::as_array).expect("committed chunks should be an array");
        let committed_chunk_identifiers = committed_chunks
            .iter()
            .map(|committed_chunk| {
                committed_chunk
                    .get("chunk_identifier")
                    .and_then(Value::as_i64)
                    .expect("chunk identifier should be present")
            })
            .collect::<Vec<_>>();

        assert_eq!(committed_chunk_identifiers, vec![0, 2]);

        std::fs::remove_dir_all(run_directory).expect("test directory should be removed");
    }

    #[test]
    fn rejects_conflicting_duplicate_chunk_commit() {
        let run_directory = create_test_directory();
        let manifest_path = run_directory.join(RUN_MANIFEST_FILE_NAME);
        std::fs::write(&manifest_path, "{\n  \"committed_chunks\": []\n}\n").expect("manifest should be written");
        let mut conflicting_commit = build_chunk_commit(2);
        conflicting_commit.row_count = 3;

        let error = record_run_manifest_chunk_commits(&run_directory, vec![build_chunk_commit(2), conflicting_commit])
            .expect_err("conflicting duplicate chunk should be rejected");

        assert!(error.contains("conflicting commit metadata for chunk 2"));

        std::fs::remove_dir_all(run_directory).expect("test directory should be removed");
    }

    #[test]
    fn reads_manifest_chunk_commits_from_text() {
        let manifest = r#"{
          "committed_chunks": [
            {
              "chunk_identifier": 4,
              "variant_start_index": 4,
              "variant_stop_index": 6,
              "row_count": 2,
              "chunk_file_name": "part_000000000.parquet"
            }
          ]
        }"#;

        let chunk_commits = read_run_manifest_chunk_commits_from_text(manifest).expect("manifest commits should parse");

        assert_eq!(chunk_commits.len(), 1);
        assert_eq!(chunk_commits[0].chunk_identifier, 4);
        assert_eq!(chunk_commits[0].output_format, "parquet");
        assert_eq!(chunk_commits[0].compression, "none");
    }

    #[test]
    fn marks_manifest_interrupted_without_final_outputs() {
        let run_directory = create_test_directory();
        let manifest_path = run_directory.join(RUN_MANIFEST_FILE_NAME);
        std::fs::write(
            &manifest_path,
            "{\n  \"committed_chunks\": [],\n  \"finalized\": true,\n  \"final_parquet\": \"old.parquet\",\n  \"final_row_count\": 1,\n  \"final_chunk_file_count\": 1\n}\n",
        )
        .expect("manifest should be written");

        mark_run_manifest_interrupted(&run_directory, "SIGTERM").expect("manifest should be marked interrupted");

        let manifest_text = std::fs::read_to_string(&manifest_path).expect("manifest should be readable");
        let manifest = serde_json::from_str::<Value>(&manifest_text).expect("manifest should be JSON");

        assert_eq!(manifest.get("finalized").and_then(Value::as_bool), Some(false));
        assert_eq!(manifest.get("interrupted").and_then(Value::as_bool), Some(true));
        assert_eq!(manifest.get("interrupted_signal").and_then(Value::as_str), Some("SIGTERM"));
        assert!(manifest.get("final_parquet").is_none());
        assert!(manifest.get("final_row_count").is_none());
        assert!(manifest.get("final_chunk_file_count").is_none());

        std::fs::remove_dir_all(run_directory).expect("test directory should be removed");
    }

    #[test]
    fn mark_finalized_ignores_path_without_parent() {
        mark_run_manifest_finalized(Path::new("/"), 0, 0).expect("parentless path should be ignored");
    }
}
