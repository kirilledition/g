use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{Value, json};

use super::{TerminalRunState, finalize_run_manifest_with_writer};
use crate::error::OutputError;
use crate::manifest::chunks::{RunManifestChunkCommit, chunk_commit_to_value};
use crate::manifest::run::write_run_manifest_value_atomic;

struct ManifestFixture {
    directory: PathBuf,
}

impl ManifestFixture {
    fn new(manifest: &Value) -> Self {
        static NEXT_IDENTIFIER: AtomicU64 = AtomicU64::new(0);
        let identifier = NEXT_IDENTIFIER.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!("g-terminal-manifest-{}-{identifier}", std::process::id()));
        std::fs::create_dir_all(&directory).expect("fixture directory creates");
        let fixture = Self { directory };
        write_run_manifest_value_atomic(&fixture.path(), manifest).expect("fixture manifest writes");
        fixture
    }

    fn path(&self) -> PathBuf {
        self.directory.join("run_manifest.json")
    }

    fn manifest(&self) -> Value {
        serde_json::from_slice(&std::fs::read(self.path()).expect("manifest reads")).expect("manifest parses")
    }
}

impl Drop for ManifestFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

fn chunk_commit(file_name: &str) -> RunManifestChunkCommit {
    RunManifestChunkCommit {
        chunk_identifier: 0,
        variant_start_index: 0,
        variant_stop_index: 2,
        row_count: 2,
        chunk_file_name: file_name.to_string(),
    }
}

#[test]
fn successful_finalization_persists_commits_and_state_in_one_write() {
    for terminal_state in [TerminalRunState::Completed, TerminalRunState::Interrupted { signal_name: "SIGTERM" }] {
        let fixture = ManifestFixture::new(
            &json!({"committed_chunks": [], "status": "interrupted", "interrupted_signal": "SIGINT"}),
        );
        let mut write_count = 0;
        finalize_run_manifest_with_writer(
            &fixture.directory,
            vec![chunk_commit("part.parquet")],
            terminal_state,
            |path, manifest| {
                write_count += 1;
                write_run_manifest_value_atomic(path, manifest)
            },
        )
        .expect("combined finalization succeeds");
        assert_eq!(write_count, 1);
        let manifest = fixture.manifest();
        match terminal_state {
            TerminalRunState::Completed => {
                assert_eq!(manifest["status"], "completed");
                assert!(manifest.get("interrupted_signal").is_none());
            }
            TerminalRunState::Interrupted { signal_name } => {
                assert_eq!(manifest["status"], "interrupted");
                assert_eq!(manifest["interrupted_signal"], signal_name);
            }
        }
        assert_eq!(manifest["committed_chunks"], json!([chunk_commit_to_value(&chunk_commit("part.parquet"))]));
    }
}

#[test]
fn failed_terminal_write_preserves_commits_and_exact_previous_lifecycle() {
    let mut existing_commit = chunk_commit("existing.parquet");
    existing_commit.chunk_identifier = 2;
    existing_commit.variant_start_index = 2;
    existing_commit.variant_stop_index = 4;
    let existing_commit_value = chunk_commit_to_value(&existing_commit);
    for previous_manifest in [
        json!({"committed_chunks": []}),
        json!({"committed_chunks": [], "status": null, "interrupted_signal": null}),
        json!({"committed_chunks": [existing_commit_value], "status": "interrupted", "interrupted_signal": "SIGINT"}),
    ] {
        for terminal_state in [TerminalRunState::Completed, TerminalRunState::Interrupted { signal_name: "SIGTERM" }] {
            let fixture = ManifestFixture::new(&previous_manifest);
            let mut write_count = 0;
            let error = finalize_run_manifest_with_writer(
                &fixture.directory,
                vec![chunk_commit("part.parquet")],
                terminal_state,
                |path, manifest| {
                    write_count += 1;
                    if write_count == 1 {
                        return Err(OutputError::Runtime("terminal publication failed".to_string()));
                    }
                    write_run_manifest_value_atomic(path, manifest)
                },
            )
            .expect_err("fallback must not report terminal success");
            assert_eq!(error.to_string(), "terminal publication failed");
            assert_eq!(write_count, 2);
            let observed = fixture.manifest();
            assert_eq!(observed.get("status"), previous_manifest.get("status"));
            assert_eq!(observed.get("interrupted_signal"), previous_manifest.get("interrupted_signal"));
            let mut expected_commits = vec![chunk_commit_to_value(&chunk_commit("part.parquet"))];
            expected_commits
                .extend(previous_manifest["committed_chunks"].as_array().expect("fixture commits").iter().cloned());
            assert_eq!(observed["committed_chunks"], json!(expected_commits));
        }
    }
}

#[test]
fn validation_failure_does_not_attempt_terminal_or_fallback_persistence() {
    for previous_manifest in [
        json!({"committed_chunks": [chunk_commit_to_value(&chunk_commit("original.parquet"))], "status": "running"}),
        json!({"committed_chunks": "invalid", "status": "running"}),
    ] {
        let fixture = ManifestFixture::new(&previous_manifest);
        let original_bytes = std::fs::read(fixture.path()).expect("original manifest reads");
        let mut write_count = 0;
        let result = finalize_run_manifest_with_writer(
            &fixture.directory,
            vec![chunk_commit("conflict.parquet")],
            TerminalRunState::Completed,
            |_path, _manifest| {
                write_count += 1;
                Ok(())
            },
        );
        assert!(matches!(result, Err(OutputError::InvalidInput(_))));
        assert_eq!(write_count, 0);
        assert_eq!(std::fs::read(fixture.path()).expect("manifest remains readable"), original_bytes);
    }
}

#[test]
fn failed_fallback_reports_both_errors_and_preserves_existing_manifest() {
    let fixture = ManifestFixture::new(&json!({"committed_chunks": [], "status": "running"}));
    let original_bytes = std::fs::read(fixture.path()).expect("original manifest reads");
    let mut write_count = 0;
    let error = finalize_run_manifest_with_writer(
        &fixture.directory,
        vec![chunk_commit("part.parquet")],
        TerminalRunState::Completed,
        |_path: &Path, _manifest| {
            write_count += 1;
            Err(OutputError::Runtime(format!("write attempt {write_count} failed")))
        },
    )
    .expect_err("both persistence attempts fail");
    assert_eq!(write_count, 2);
    assert!(error.to_string().contains("write attempt 1 failed"));
    assert!(error.to_string().contains("write attempt 2 failed"));
    assert_eq!(std::fs::read(fixture.path()).expect("manifest remains readable"), original_bytes);
}

#[test]
fn failed_terminal_write_without_new_commits_does_not_retry() {
    let fixture = ManifestFixture::new(&json!({"committed_chunks": [], "status": "running"}));
    let mut write_count = 0;
    let error = finalize_run_manifest_with_writer(
        &fixture.directory,
        Vec::new(),
        TerminalRunState::Completed,
        |_path, _manifest| {
            write_count += 1;
            Err(OutputError::Runtime("terminal publication failed".to_string()))
        },
    )
    .expect_err("terminal persistence fails");
    assert_eq!(error.to_string(), "terminal publication failed");
    assert_eq!(write_count, 1);
    assert_eq!(fixture.manifest()["status"], "running");
}

#[test]
fn terminal_write_without_new_commits_preserves_existing_commit_metadata() {
    let previous_manifest = json!({"committed_chunks": [{"preserved": "status-only update"}], "status": "running"});
    let fixture = ManifestFixture::new(&previous_manifest);
    let mut write_count = 0;
    finalize_run_manifest_with_writer(&fixture.directory, Vec::new(), TerminalRunState::Completed, |path, manifest| {
        write_count += 1;
        write_run_manifest_value_atomic(path, manifest)
    })
    .expect("status-only update preserves the existing commit field");
    assert_eq!(write_count, 1);
    let observed = fixture.manifest();
    assert_eq!(observed["status"], "completed");
    assert_eq!(observed["committed_chunks"], previous_manifest["committed_chunks"]);
}
