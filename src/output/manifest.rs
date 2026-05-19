use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use serde_json::{Value, json};

const RUN_MANIFEST_FILE_NAME: &str = "run_manifest.json";

pub(crate) struct RunManifestChunkCommit {
    pub(crate) chunk_identifier: i64,
    pub(crate) variant_start_index: i64,
    pub(crate) variant_stop_index: i64,
    pub(crate) row_count: usize,
    pub(crate) chunk_file_name: String,
}

pub(crate) fn record_run_manifest_chunk_commits(
    run_directory: &Path,
    chunk_commits: Vec<RunManifestChunkCommit>,
) -> Result<(), String> {
    update_run_manifest(run_directory, |manifest| {
        let manifest_object =
            manifest.as_object_mut().ok_or_else(|| "Run manifest must contain a JSON object.".to_string())?;
        let committed_chunks = manifest_object
            .entry("committed_chunks")
            .or_insert_with(|| Value::Array(Vec::new()))
            .as_array_mut()
            .ok_or_else(|| "Run manifest committed_chunks field must be a list.".to_string())?;
        for chunk_commit in chunk_commits {
            let chunk_identifier = chunk_commit.chunk_identifier;
            let already_committed = committed_chunks.iter().any(|committed_chunk| {
                committed_chunk.get("chunk_identifier").and_then(Value::as_i64) == Some(chunk_identifier)
            });
            if !already_committed {
                committed_chunks.push(json!({
                    "chunk_identifier": chunk_commit.chunk_identifier,
                    "variant_start_index": chunk_commit.variant_start_index,
                    "variant_stop_index": chunk_commit.variant_stop_index,
                    "row_count": chunk_commit.row_count,
                    "chunk_file_name": chunk_commit.chunk_file_name,
                }));
            }
        }
        committed_chunks.sort_by_key(|committed_chunk| {
            committed_chunk.get("chunk_identifier").and_then(Value::as_i64).unwrap_or_default()
        });
        Ok(())
    })
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
