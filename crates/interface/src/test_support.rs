use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_DIRECTORY_IDENTIFIER: AtomicU64 = AtomicU64::new(0);

pub(crate) struct TemporaryDirectory {
    path: PathBuf,
}

impl TemporaryDirectory {
    pub(crate) fn new(test_name: &str) -> Self {
        let directory_identifier = NEXT_DIRECTORY_IDENTIFIER.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("g-interface-{test_name}-{}-{directory_identifier}", std::process::id()));
        std::fs::create_dir(&path).expect("temporary interface-test directory should be created");
        Self { path }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn write(&self, file_name: &str, contents: &str) -> PathBuf {
        let path = self.path.join(file_name);
        std::fs::write(&path, contents).expect("temporary interface-test file should be written");
        path
    }
}

impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
