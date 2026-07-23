use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::error::OutputError;
use crate::persistence::identifier::AttemptIdentifier;

pub(crate) fn path_operation_error(operation: &str, path: &Path, error: &impl std::fmt::Display) -> OutputError {
    OutputError::Runtime(format!("Failed to {operation} '{}': {error}", path.display()))
}

pub(crate) trait OutputIo {
    type File: io::Write + Send;

    fn create_new_file(&self, path: &Path) -> io::Result<Self::File>;
    fn sync_file(&self, file: &Self::File, path: &Path) -> io::Result<()>;
    fn rename_file(&self, source_path: &Path, destination_path: &Path) -> io::Result<()>;
    fn sync_directory(&self, path: &Path) -> io::Result<()>;
    fn file_size(&self, path: &Path) -> io::Result<u64>;
    fn remove_file(&self, path: &Path) -> io::Result<()>;
}

pub(crate) struct StdOutputIo;

impl OutputIo for StdOutputIo {
    type File = File;

    fn create_new_file(&self, path: &Path) -> io::Result<Self::File> {
        OpenOptions::new().write(true).create_new(true).open(path)
    }

    fn sync_file(&self, file: &Self::File, _path: &Path) -> io::Result<()> {
        file.sync_all()
    }

    fn rename_file(&self, source_path: &Path, destination_path: &Path) -> io::Result<()> {
        std::fs::rename(source_path, destination_path)
    }

    fn sync_directory(&self, path: &Path) -> io::Result<()> {
        File::open(path)?.sync_all()
    }

    fn file_size(&self, path: &Path) -> io::Result<u64> {
        std::fs::metadata(path).map(|metadata| metadata.len())
    }

    fn remove_file(&self, path: &Path) -> io::Result<()> {
        std::fs::remove_file(path)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NoReplacePublication {
    Created,
    AlreadyExists,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FileIntegrity {
    pub(crate) size_bytes: u64,
    pub(crate) sha256: String,
}

pub(crate) fn create_directories_durable(directory_path: &Path) -> Result<(), OutputError> {
    let mut missing_directories = Vec::new();
    let mut existing_ancestor = directory_path;
    while !existing_ancestor.exists() {
        missing_directories.push(existing_ancestor.to_path_buf());
        existing_ancestor = existing_ancestor.parent().ok_or_else(|| {
            OutputError::Runtime(format!("Output directory '{}' has no existing ancestor.", directory_path.display()))
        })?;
    }
    if !existing_ancestor.is_dir() {
        return Err(OutputError::InvalidInput(format!(
            "Output path '{}' has a non-directory ancestor '{}'.",
            directory_path.display(),
            existing_ancestor.display()
        )));
    }
    for missing_directory in missing_directories.into_iter().rev() {
        match std::fs::create_dir(&missing_directory) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists && missing_directory.is_dir() => {}
            Err(error) => {
                return Err(path_operation_error("create output directory", &missing_directory, &error));
            }
        }
        let parent_directory = missing_directory.parent().ok_or_else(|| {
            OutputError::Runtime(format!("Output directory '{}' has no parent.", missing_directory.display()))
        })?;
        sync_directory(parent_directory)?;
    }
    Ok(())
}

pub(crate) fn publish_json_no_replace<ValueType: serde::Serialize>(
    destination_path: &Path,
    value: &ValueType,
) -> Result<NoReplacePublication, OutputError> {
    let mut bytes = serde_json::to_vec_pretty(value).map_err(OutputError::runtime)?;
    bytes.push(b'\n');
    publish_bytes_no_replace(destination_path, &bytes)
}

pub(crate) fn publish_bytes_no_replace(
    destination_path: &Path,
    bytes: &[u8],
) -> Result<NoReplacePublication, OutputError> {
    let parent_directory = destination_path.parent().ok_or_else(|| {
        OutputError::InvalidInput(format!("Output publication path '{}' has no parent.", destination_path.display()))
    })?;
    create_directories_durable(parent_directory)?;
    let temporary_path = unique_temporary_path(destination_path);
    let mut temporary_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary_path)
        .map_err(|error| path_operation_error("create immutable-record temporary file", &temporary_path, &error))?;
    let mut cleanup = TemporaryPublication::new(temporary_path.clone());
    temporary_file
        .write_all(bytes)
        .map_err(|error| path_operation_error("write immutable-record temporary file", &temporary_path, &error))?;
    temporary_file.sync_all().map_err(|error| {
        path_operation_error("synchronize immutable-record temporary file", &temporary_path, &error)
    })?;
    drop(temporary_file);
    match std::fs::hard_link(&temporary_path, destination_path) {
        Ok(()) => {
            std::fs::remove_file(&temporary_path).map_err(|error| {
                path_operation_error("remove immutable-record temporary link", &temporary_path, &error)
            })?;
            cleanup.mark_removed();
            sync_directory(parent_directory)?;
            Ok(NoReplacePublication::Created)
        }
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => Ok(NoReplacePublication::AlreadyExists),
        Err(error) => Err(OutputError::Runtime(format!(
            "Failed to publish immutable output record '{}' with hard-link no-replace: {error}",
            destination_path.display()
        ))),
    }
}

pub(crate) fn write_json_atomic<ValueType: serde::Serialize>(
    destination_path: &Path,
    value: &ValueType,
) -> Result<(), OutputError> {
    let mut bytes = serde_json::to_vec_pretty(value).map_err(OutputError::runtime)?;
    bytes.push(b'\n');
    write_bytes_atomic(destination_path, &bytes)
}

pub(crate) fn write_bytes_atomic(destination_path: &Path, bytes: &[u8]) -> Result<(), OutputError> {
    let parent_directory = destination_path.parent().ok_or_else(|| {
        OutputError::InvalidInput(format!("Output publication path '{}' has no parent.", destination_path.display()))
    })?;
    create_directories_durable(parent_directory)?;
    let temporary_path = unique_temporary_path(destination_path);
    let mut temporary_file =
        OpenOptions::new().write(true).create_new(true).open(&temporary_path).map_err(|error| {
            path_operation_error("create atomic-publication temporary file", &temporary_path, &error)
        })?;
    let mut cleanup = TemporaryPublication::new(temporary_path.clone());
    temporary_file
        .write_all(bytes)
        .map_err(|error| path_operation_error("write atomic-publication temporary file", &temporary_path, &error))?;
    temporary_file.sync_all().map_err(|error| {
        path_operation_error("synchronize atomic-publication temporary file", &temporary_path, &error)
    })?;
    drop(temporary_file);
    std::fs::rename(&temporary_path, destination_path).map_err(|error| {
        OutputError::Runtime(format!(
            "Failed to atomically publish '{}' as '{}': {error}",
            temporary_path.display(),
            destination_path.display()
        ))
    })?;
    cleanup.mark_removed();
    sync_directory(parent_directory)
}

pub(crate) fn file_sha256(path: &Path) -> Result<String, OutputError> {
    let mut input_file =
        File::open(path).map_err(|error| path_operation_error("open file for SHA-256", path, &error))?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let bytes_read = input_file
            .read(&mut buffer)
            .map_err(|error| path_operation_error("read file for SHA-256", path, &error))?;
        if bytes_read == 0 {
            break;
        }
        digest.update(&buffer[..bytes_read]);
    }
    Ok(hex::encode(digest.finalize()))
}

pub(crate) fn file_size_and_sha256(path: &Path) -> Result<FileIntegrity, OutputError> {
    let size_bytes =
        std::fs::metadata(path).map_err(|error| path_operation_error("read file metadata", path, &error))?.len();
    Ok(FileIntegrity { size_bytes, sha256: file_sha256(path)? })
}

pub(crate) fn sync_directory(path: &Path) -> Result<(), OutputError> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| path_operation_error("synchronize output directory", path, &error))
}

fn unique_temporary_path(destination_path: &Path) -> PathBuf {
    let file_name = destination_path.file_name().and_then(|name| name.to_str()).unwrap_or("output-record");
    destination_path.with_file_name(format!(".{file_name}.{}.tmp", AttemptIdentifier::generate().as_str()))
}

struct TemporaryPublication {
    path: PathBuf,
    cleanup_required: bool,
}

impl TemporaryPublication {
    fn new(path: PathBuf) -> Self {
        Self { path, cleanup_required: true }
    }

    fn mark_removed(&mut self) {
        self.cleanup_required = false;
    }
}

impl Drop for TemporaryPublication {
    fn drop(&mut self) {
        if self.cleanup_required {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

#[cfg(test)]
mod durable_tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use serde_json::json;

    use super::{NoReplacePublication, file_size_and_sha256, publish_json_no_replace, write_json_atomic};

    #[test]
    fn no_replace_publication_has_one_winner_and_complete_json() {
        let directory = tempfile_directory("no-replace");
        let destination = directory.join("record.json");
        let winner_count = Arc::new(AtomicUsize::new(0));
        std::thread::scope(|scope| {
            for contender in 0..8 {
                let winner_count = Arc::clone(&winner_count);
                let destination = destination.clone();
                scope.spawn(move || {
                    let outcome = publish_json_no_replace(&destination, &json!({"contender": contender}))
                        .expect("hard-link publication succeeds");
                    if outcome == NoReplacePublication::Created {
                        winner_count.fetch_add(1, Ordering::Relaxed);
                    }
                });
            }
        });
        assert_eq!(winner_count.load(Ordering::Relaxed), 1);
        let value: serde_json::Value = serde_json::from_slice(&std::fs::read(&destination).expect("record reads"))
            .expect("record is complete JSON");
        assert!(value["contender"].as_u64().is_some());
    }

    #[test]
    fn atomic_replace_and_hash_cover_published_bytes() {
        let directory = tempfile_directory("atomic-replace");
        let destination = directory.join("record.json");
        write_json_atomic(&destination, &json!({"generation": 1})).expect("first value writes");
        write_json_atomic(&destination, &json!({"generation": 2})).expect("second value replaces");
        let bytes = std::fs::read(&destination).expect("record reads");
        assert_eq!(serde_json::from_slice::<serde_json::Value>(&bytes).expect("record parses")["generation"], 2);
        let integrity = file_size_and_sha256(&destination).expect("file hashes");
        assert_eq!(integrity.size_bytes, u64::try_from(bytes.len()).expect("test bytes fit uint64"));
        assert_eq!(integrity.sha256.len(), 64);
    }

    fn tempfile_directory(label: &str) -> PathBuf {
        let directory = std::env::temp_dir().join(format!(
            "g-output-io-{label}-{}-{}",
            std::process::id(),
            AttemptIdentifier::generate().as_str()
        ));
        std::fs::create_dir(&directory).expect("test directory creates");
        directory
    }

    use std::path::PathBuf;

    use crate::persistence::identifier::AttemptIdentifier;
}
