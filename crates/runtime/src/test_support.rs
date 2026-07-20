use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::runtime_policy::NativeRunSessionPolicy;

static TEMPORARY_DIRECTORY_COUNTER: AtomicU64 = AtomicU64::new(0);

pub(crate) static RUNTIME_GLOBAL_TEST_MUTEX: Mutex<()> = Mutex::new(());

#[derive(Debug)]
pub(crate) struct TemporaryDirectory {
    path: PathBuf,
}

impl TemporaryDirectory {
    pub(crate) fn new(label: &str) -> Self {
        let sequence = TEMPORARY_DIRECTORY_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("g-runtime-{label}-{}-{sequence}", std::process::id()));
        std::fs::create_dir_all(&path).expect("runtime test directory should be created");
        Self { path }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

pub(crate) fn disabled_session_policy() -> NativeRunSessionPolicy {
    NativeRunSessionPolicy {
        log_filter: "info".to_owned(),
        log_stderr: false,
        log_file: None,
        telemetry_stream_file: None,
        stage_timing_file: None,
        profile_summary_file: None,
        queue_size: 16,
        lossy: false,
        include_source_location: false,
        include_span_events: false,
    }
}

pub(crate) fn execute_isolated_test_body(test_name: &str, child_environment_variable: &str) -> bool {
    if let Some(handshake_path) = std::env::var_os(child_environment_variable) {
        std::fs::write(handshake_path, test_name).expect("isolated Rust test child should record its handshake");
        return true;
    }
    let sequence = TEMPORARY_DIRECTORY_COUNTER.fetch_add(1, Ordering::Relaxed);
    let handshake_path =
        std::env::temp_dir().join(format!("g-runtime-isolated-test-{}-{sequence}.handshake", std::process::id()));
    let test_executable = std::env::current_exe().expect("current Rust test executable should be available");
    let status = std::process::Command::new(test_executable)
        .arg("--exact")
        .arg(test_name)
        .arg("--nocapture")
        .env(child_environment_variable, &handshake_path)
        .status()
        .expect("isolated Rust test subprocess should start");
    let handshake = std::fs::read_to_string(&handshake_path);
    let _ = std::fs::remove_file(&handshake_path);
    assert!(status.success(), "isolated Rust test subprocess should succeed: {status}");
    assert_eq!(handshake.expect("isolated Rust test child should write its handshake"), test_name);
    false
}

impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
