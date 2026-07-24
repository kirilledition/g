use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::error::OutputError;
use crate::persistence::identifier::AttemptIdentifier;

#[cfg(test)]
std::thread_local! {
    static OWNER_PUBLICATION_SYNC_FAILURES: std::cell::Cell<usize> = const {
        std::cell::Cell::new(0)
    };
    static OWNER_PUBLICATION_FILE_SYNC_FAILURES: std::cell::Cell<usize> = const {
        std::cell::Cell::new(0)
    };
    static IMMUTABLE_PUBLICATION_DIRECTORY_SYNC_FAILURE: std::cell::RefCell<
        Option<ImmutablePublicationDirectorySyncFailure>,
    > = const {
        std::cell::RefCell::new(None)
    };
    static IMMUTABLE_PUBLICATION_FILE_SYNC_FAILURE: std::cell::RefCell<
        Option<ImmutablePublicationFileSyncFailure>,
    > = const {
        std::cell::RefCell::new(None)
    };
}

#[cfg(test)]
pub(crate) fn fail_owner_publication_syncs_for_test(failure_count: usize) {
    OWNER_PUBLICATION_SYNC_FAILURES.with(|remaining_failures| remaining_failures.set(failure_count));
}

#[cfg(test)]
pub(crate) fn fail_owner_publication_file_syncs_for_test(failure_count: usize) {
    OWNER_PUBLICATION_FILE_SYNC_FAILURES.with(|remaining_failures| remaining_failures.set(failure_count));
}

#[cfg(test)]
struct ImmutablePublicationDirectorySyncFailure {
    destination_path: PathBuf,
    remaining_failures: usize,
}

#[cfg(test)]
pub(crate) struct ImmutablePublicationDirectorySyncFailureGuard;

#[cfg(test)]
impl Drop for ImmutablePublicationDirectorySyncFailureGuard {
    fn drop(&mut self) {
        IMMUTABLE_PUBLICATION_DIRECTORY_SYNC_FAILURE.with(|installed_failure| {
            *installed_failure.borrow_mut() = None;
        });
    }
}

#[cfg(test)]
pub(crate) fn install_immutable_publication_directory_sync_failure_for_test(
    destination_path: PathBuf,
    failure_count: usize,
) -> ImmutablePublicationDirectorySyncFailureGuard {
    assert!(failure_count > 0, "an immutable-publication sync failure count must be positive");
    IMMUTABLE_PUBLICATION_DIRECTORY_SYNC_FAILURE.with(|installed_failure| {
        let mut installed_failure = installed_failure.borrow_mut();
        assert!(installed_failure.is_none(), "an immutable-publication sync failure is already installed");
        *installed_failure =
            Some(ImmutablePublicationDirectorySyncFailure { destination_path, remaining_failures: failure_count });
    });
    ImmutablePublicationDirectorySyncFailureGuard
}

#[cfg(test)]
struct ImmutablePublicationFileSyncFailure {
    destination_path: PathBuf,
}

#[cfg(test)]
pub(crate) struct ImmutablePublicationFileSyncFailureGuard;

#[cfg(test)]
impl Drop for ImmutablePublicationFileSyncFailureGuard {
    fn drop(&mut self) {
        IMMUTABLE_PUBLICATION_FILE_SYNC_FAILURE.with(|installed_failure| {
            *installed_failure.borrow_mut() = None;
        });
    }
}

#[cfg(test)]
pub(crate) fn install_immutable_publication_file_sync_failure_for_test(
    destination_path: PathBuf,
) -> ImmutablePublicationFileSyncFailureGuard {
    IMMUTABLE_PUBLICATION_FILE_SYNC_FAILURE.with(|installed_failure| {
        let mut installed_failure = installed_failure.borrow_mut();
        assert!(installed_failure.is_none(), "an immutable-publication file sync failure is already installed");
        *installed_failure = Some(ImmutablePublicationFileSyncFailure { destination_path });
    });
    ImmutablePublicationFileSyncFailureGuard
}

pub(crate) fn path_operation_error(operation: &str, path: &Path, error: &impl std::fmt::Display) -> OutputError {
    OutputError::Runtime(format!("Failed to {operation} '{}': {error}", path.display()))
}

pub(crate) trait OutputIo {
    type File: io::Write + Send;

    fn create_new_file(&self, path: &Path) -> io::Result<Self::File>;
    fn sync_file(&self, file: &Self::File, path: &Path) -> io::Result<()>;
    fn publish_file_no_replace(&self, source_path: &Path, destination_path: &Path) -> io::Result<NoReplacePublication>;
    fn sync_directory(&self, path: &Path) -> io::Result<()>;
    fn file_integrity(&self, path: &Path) -> io::Result<FileIntegrity>;
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

    fn publish_file_no_replace(&self, source_path: &Path, destination_path: &Path) -> io::Result<NoReplacePublication> {
        match std::fs::hard_link(source_path, destination_path) {
            Ok(()) => {
                std::fs::remove_file(source_path)?;
                Ok(NoReplacePublication::Created)
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => Ok(NoReplacePublication::AlreadyExists),
            Err(error) => Err(error),
        }
    }

    fn sync_directory(&self, path: &Path) -> io::Result<()> {
        File::open(path)?.sync_all()
    }

    fn file_integrity(&self, path: &Path) -> io::Result<FileIntegrity> {
        Ok(FileIntegrity { size_bytes: std::fs::metadata(path)?.len(), sha256: file_sha256_io(path)? })
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
    loop {
        match std::fs::metadata(existing_ancestor) {
            Ok(metadata) if metadata.is_dir() => break,
            Ok(_) => {
                return Err(OutputError::InvalidInput(format!(
                    "Output path '{}' has a non-directory ancestor '{}'.",
                    directory_path.display(),
                    existing_ancestor.display()
                )));
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                missing_directories.push(existing_ancestor.to_path_buf());
                existing_ancestor = existing_ancestor.parent().ok_or_else(|| {
                    OutputError::Runtime(format!(
                        "Output directory '{}' has no existing ancestor.",
                        directory_path.display()
                    ))
                })?;
            }
            Err(error) if error.kind() == io::ErrorKind::NotADirectory => {
                return Err(OutputError::InvalidInput(format!(
                    "Output path '{}' contains a non-directory ancestor: {error}",
                    directory_path.display()
                )));
            }
            Err(error) => {
                return Err(path_operation_error("inspect output directory ancestor", existing_ancestor, &error));
            }
        }
    }
    for missing_directory in missing_directories.into_iter().rev() {
        match std::fs::create_dir(&missing_directory) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => match std::fs::metadata(&missing_directory) {
                Ok(metadata) if metadata.is_dir() => {}
                Ok(_) => {
                    return Err(OutputError::InvalidInput(format!(
                        "Output path '{}' became a non-directory during creation.",
                        missing_directory.display()
                    )));
                }
                Err(metadata_error) => {
                    return Err(path_operation_error(
                        "inspect concurrently created output directory",
                        &missing_directory,
                        &metadata_error,
                    ));
                }
            },
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

pub(crate) fn sync_nearest_existing_directory(directory_path: &Path) -> Result<(), OutputError> {
    let mut candidate = directory_path;
    loop {
        match std::fs::symlink_metadata(candidate) {
            Ok(metadata) if metadata.is_dir() => return sync_directory(candidate),
            Ok(_) => {
                return Err(OutputError::InvalidInput(format!(
                    "Output durability path '{}' is not a directory.",
                    candidate.display()
                )));
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                candidate = candidate.parent().ok_or_else(|| {
                    OutputError::Runtime(format!(
                        "Output durability path '{}' has no existing ancestor.",
                        directory_path.display()
                    ))
                })?;
            }
            Err(error) => {
                return Err(path_operation_error("inspect output durability directory", candidate, &error));
            }
        }
    }
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
    sync_immutable_publication_file(&temporary_file, &temporary_path, destination_path)?;
    drop(temporary_file);
    #[cfg(test)]
    crash_owner_claim_publication_at_test_point(destination_path, "before_owner_claim_link");
    #[cfg(test)]
    pause_owner_transition_publication_at_test_barrier(destination_path);
    match std::fs::hard_link(&temporary_path, destination_path) {
        Ok(()) => {
            #[cfg(test)]
            crash_owner_claim_publication_at_test_point(destination_path, "after_owner_claim_link");
            std::fs::remove_file(&temporary_path).map_err(|error| {
                path_operation_error("remove immutable-record temporary link", &temporary_path, &error)
            })?;
            cleanup.mark_removed();
            sync_immutable_publication_directory(destination_path, parent_directory)?;
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
    file_sha256_io(path).map_err(|error| path_operation_error("hash file with SHA-256", path, &error))
}

fn file_sha256_io(path: &Path) -> io::Result<String> {
    let mut input_file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let bytes_read = input_file.read(&mut buffer)?;
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

pub(crate) fn clone_file_no_replace_verified(
    source_path: &Path,
    destination_path: &Path,
    expected_integrity: &FileIntegrity,
) -> Result<NoReplacePublication, OutputError> {
    verify_file_integrity(source_path, expected_integrity, "source")?;
    let destination_directory = destination_path.parent().ok_or_else(|| {
        OutputError::InvalidInput(format!("Output clone path '{}' has no parent.", destination_path.display()))
    })?;
    create_directories_durable(destination_directory)?;
    let publication = match std::fs::hard_link(source_path, destination_path) {
        Ok(()) => {
            sync_directory(destination_directory)?;
            NoReplacePublication::Created
        }
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => NoReplacePublication::AlreadyExists,
        Err(link_error) => {
            copy_file_no_replace_verified(source_path, destination_path, expected_integrity).map_err(|copy_error| {
                OutputError::Runtime(format!(
                    "Failed to hard-link verified output part '{}' as '{}' ({link_error}); copy fallback also failed: {copy_error}",
                    source_path.display(),
                    destination_path.display()
                ))
            })?
        }
    };
    verify_file_integrity(destination_path, expected_integrity, "destination")?;
    Ok(publication)
}

fn copy_file_no_replace_verified(
    source_path: &Path,
    destination_path: &Path,
    expected_integrity: &FileIntegrity,
) -> Result<NoReplacePublication, OutputError> {
    let destination_directory = destination_path.parent().ok_or_else(|| {
        OutputError::InvalidInput(format!("Output clone path '{}' has no parent.", destination_path.display()))
    })?;
    let temporary_path = unique_temporary_path(destination_path);
    let mut source_file = File::open(source_path)
        .map_err(|error| path_operation_error("open verified output source", source_path, &error))?;
    let mut temporary_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary_path)
        .map_err(|error| path_operation_error("create output copy temporary file", &temporary_path, &error))?;
    let mut cleanup = TemporaryPublication::new(temporary_path.clone());
    io::copy(&mut source_file, &mut temporary_file)
        .map_err(|error| path_operation_error("copy verified output source", &temporary_path, &error))?;
    temporary_file
        .sync_all()
        .map_err(|error| path_operation_error("synchronize output copy temporary file", &temporary_path, &error))?;
    drop(temporary_file);
    verify_file_integrity(&temporary_path, expected_integrity, "copied temporary")?;
    let publication = match std::fs::hard_link(&temporary_path, destination_path) {
        Ok(()) => {
            sync_directory(destination_directory)?;
            NoReplacePublication::Created
        }
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => NoReplacePublication::AlreadyExists,
        Err(error) => {
            return Err(OutputError::Runtime(format!(
                "Failed to publish copied output part '{}' as '{}' with hard-link no-replace: {error}",
                temporary_path.display(),
                destination_path.display()
            )));
        }
    };
    std::fs::remove_file(&temporary_path)
        .map_err(|error| path_operation_error("remove output copy temporary link", &temporary_path, &error))?;
    cleanup.mark_removed();
    verify_file_integrity(destination_path, expected_integrity, "destination")?;
    Ok(publication)
}

fn verify_file_integrity(path: &Path, expected_integrity: &FileIntegrity, role: &str) -> Result<(), OutputError> {
    let observed_integrity = file_size_and_sha256(path)?;
    if observed_integrity != *expected_integrity {
        return Err(OutputError::InvalidInput(format!(
            "Verified output {role} '{}' does not match the expected raw byte size and SHA-256.",
            path.display()
        )));
    }
    Ok(())
}

pub(crate) fn sync_directory(path: &Path) -> Result<(), OutputError> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| path_operation_error("synchronize output directory", path, &error))
}

fn sync_immutable_publication_file(
    temporary_file: &File,
    temporary_path: &Path,
    destination_path: &Path,
) -> Result<(), OutputError> {
    #[cfg(not(test))]
    let _ = destination_path;
    #[cfg(test)]
    if is_owner_authority_path(destination_path) {
        let injected_failure = OWNER_PUBLICATION_FILE_SYNC_FAILURES.with(|remaining_failures| {
            let failure_count = remaining_failures.get();
            if failure_count == 0 {
                false
            } else {
                remaining_failures.set(failure_count - 1);
                true
            }
        });
        if injected_failure {
            return Err(OutputError::Runtime(
                "Injected owner-authority temporary-file synchronization failure.".to_string(),
            ));
        }
    }
    #[cfg(test)]
    if fail_immutable_publication_file_sync_for_test(destination_path) {
        return Err(OutputError::Runtime(format!(
            "Injected immutable-publication file synchronization failure for '{}'.",
            destination_path.display()
        )));
    }
    temporary_file
        .sync_all()
        .map_err(|error| path_operation_error("synchronize immutable-record temporary file", temporary_path, &error))
}

#[cfg(test)]
fn fail_immutable_publication_file_sync_for_test(destination_path: &Path) -> bool {
    IMMUTABLE_PUBLICATION_FILE_SYNC_FAILURE.with(|installed_failure| {
        let installed_failure = installed_failure.borrow();
        installed_failure.as_ref().is_some_and(|failure| failure.destination_path == destination_path)
    })
}

pub(crate) fn sync_immutable_publication_directory(
    destination_path: &Path,
    parent_directory: &Path,
) -> Result<(), OutputError> {
    #[cfg(not(test))]
    let _ = destination_path;
    #[cfg(test)]
    if is_owner_authority_path(destination_path) {
        let injected_failure = OWNER_PUBLICATION_SYNC_FAILURES.with(|remaining_failures| {
            let failure_count = remaining_failures.get();
            if failure_count == 0 {
                false
            } else {
                remaining_failures.set(failure_count - 1);
                true
            }
        });
        if injected_failure {
            return Err(OutputError::Runtime(
                "Injected owner-authority publication directory synchronization failure.".to_string(),
            ));
        }
    }
    #[cfg(test)]
    if fail_immutable_publication_directory_sync_for_test(destination_path) {
        return Err(OutputError::Runtime(format!(
            "Injected immutable-publication directory synchronization failure for '{}'.",
            destination_path.display()
        )));
    }
    sync_directory(parent_directory)
}

#[cfg(test)]
fn fail_immutable_publication_directory_sync_for_test(destination_path: &Path) -> bool {
    IMMUTABLE_PUBLICATION_DIRECTORY_SYNC_FAILURE.with(|installed_failure| {
        let mut installed_failure = installed_failure.borrow_mut();
        let Some(installed_failure) = installed_failure.as_mut() else {
            return false;
        };
        if installed_failure.destination_path != destination_path || installed_failure.remaining_failures == 0 {
            return false;
        }
        installed_failure.remaining_failures -= 1;
        true
    })
}

#[cfg(test)]
fn is_owner_authority_path(path: &Path) -> bool {
    path.file_name().and_then(|name| name.to_str()) == Some("session.claim.json")
        || path.parent().and_then(Path::file_name).and_then(|name| name.to_str()) == Some("owner-transitions")
}

fn unique_temporary_path(destination_path: &Path) -> PathBuf {
    let file_name = destination_path.file_name().and_then(|name| name.to_str()).unwrap_or("output-record");
    destination_path.with_file_name(format!(".{file_name}.{}.tmp", AttemptIdentifier::generate().as_str()))
}

#[cfg(test)]
fn crash_owner_claim_publication_at_test_point(destination_path: &Path, expected_failpoint: &str) {
    let configured_failpoint = std::env::var("G_OUTPUT_TEST_CRASH_POINT");
    let is_root_claim = destination_path.file_name().and_then(|name| name.to_str()) == Some("session.claim.json");
    let is_owner_transition =
        destination_path.parent().and_then(Path::file_name).and_then(|name| name.to_str()) == Some("owner-transitions");
    let transition_failpoint = expected_failpoint.replace("owner_claim", "owner_transition");
    if (is_root_claim && configured_failpoint.as_deref() == Ok(expected_failpoint))
        || (is_owner_transition && configured_failpoint.as_deref() == Ok(transition_failpoint.as_str()))
    {
        std::process::exit(86);
    }
}

#[cfg(test)]
fn pause_owner_transition_publication_at_test_barrier(destination_path: &Path) {
    if destination_path.parent().and_then(Path::file_name).and_then(|name| name.to_str()) != Some("owner-transitions") {
        return;
    }
    let Ok(ready_path) = std::env::var("G_OUTPUT_OWNER_TRANSITION_BARRIER_READY") else {
        return;
    };
    let go_path =
        std::env::var("G_OUTPUT_OWNER_TRANSITION_BARRIER_GO").expect("owner transition barrier go path is configured");
    std::fs::write(ready_path, b"ready").expect("owner transition barrier reports readiness");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while !Path::new(&go_path).exists() {
        assert!(std::time::Instant::now() < deadline, "owner transition publication barrier timed out");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
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
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use serde_json::json;

    use super::{
        NoReplacePublication, clone_file_no_replace_verified, copy_file_no_replace_verified,
        create_directories_durable, file_size_and_sha256,
        install_immutable_publication_directory_sync_failure_for_test, publish_json_no_replace,
        sync_immutable_publication_directory, write_json_atomic,
    };
    use crate::persistence::identifier::AttemptIdentifier;

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

    #[test]
    fn verified_clone_is_no_replace_and_rejects_conflicting_destination() {
        let directory = tempfile_directory("verified-clone");
        let source = directory.join("source.parquet");
        let destination = directory.join("destination.parquet");
        std::fs::write(&source, b"immutable part bytes").expect("source writes");
        let integrity = file_size_and_sha256(&source).expect("source hashes");
        assert_eq!(
            clone_file_no_replace_verified(&source, &destination, &integrity).expect("hard-link clone publishes"),
            NoReplacePublication::Created
        );
        assert_eq!(
            clone_file_no_replace_verified(&source, &destination, &integrity).expect("identical replay verifies"),
            NoReplacePublication::AlreadyExists
        );
        std::fs::remove_file(&destination).expect("destination removes");
        std::fs::write(&destination, b"conflicting bytes").expect("conflict writes");
        assert!(clone_file_no_replace_verified(&source, &destination, &integrity).is_err());
    }

    #[test]
    fn verified_copy_fallback_syncs_and_publishes_without_replace() {
        let directory = tempfile_directory("verified-copy");
        let source = directory.join("source.parquet");
        let destination = directory.join("destination.parquet");
        std::fs::write(&source, b"immutable fallback bytes").expect("source writes");
        let integrity = file_size_and_sha256(&source).expect("source hashes");
        assert_eq!(
            copy_file_no_replace_verified(&source, &destination, &integrity).expect("copy fallback publishes"),
            NoReplacePublication::Created
        );
        assert_eq!(std::fs::read(&destination).expect("destination reads"), b"immutable fallback bytes");
        assert!(
            !std::fs::read_dir(&directory).expect("directory reads").filter_map(Result::ok).any(|entry| entry
                .file_name()
                .to_string_lossy()
                .strip_suffix(".tmp")
                .is_some())
        );
    }

    #[cfg(unix)]
    #[test]
    fn durable_directory_creation_preserves_invalid_ancestor_errors() {
        let directory = tempfile_directory("directory-errors");
        let file_ancestor = directory.join("file-ancestor");
        std::fs::write(&file_ancestor, b"not a directory").expect("file ancestor writes");
        let non_directory_error =
            create_directories_durable(&file_ancestor.join("child")).expect_err("non-directory ancestor is rejected");
        assert!(matches!(non_directory_error, crate::OutputError::InvalidInput(_)));

        let loop_ancestor = directory.join("loop-ancestor");
        std::os::unix::fs::symlink("loop-ancestor", &loop_ancestor).expect("symlink loop creates");
        let loop_error =
            create_directories_durable(&loop_ancestor.join("child")).expect_err("symlink-loop I/O error is preserved");
        assert!(matches!(loop_error, crate::OutputError::Runtime(_)));
        assert!(loop_error.to_string().contains("inspect output directory ancestor"));
    }

    #[test]
    fn immutable_publication_sync_failure_matches_exact_destination_and_count() {
        let directory = tempfile_directory("path-targeted-sync-failure");
        let destination = directory.join("genesis.json");
        let other_destination = directory.join("successor.json");
        let guard = install_immutable_publication_directory_sync_failure_for_test(destination.clone(), 3);

        sync_immutable_publication_directory(&other_destination, &directory)
            .expect("a different destination does not consume the failure");
        assert!(sync_immutable_publication_directory(&destination, &directory).is_err());
        sync_immutable_publication_directory(&other_destination, &directory)
            .expect("a different destination still does not consume the failure");
        assert!(sync_immutable_publication_directory(&destination, &directory).is_err());
        assert!(sync_immutable_publication_directory(&destination, &directory).is_err());
        sync_immutable_publication_directory(&destination, &directory)
            .expect("the configured destination succeeds after the exact failure count");
        drop(guard);

        let reset_guard = install_immutable_publication_directory_sync_failure_for_test(destination.clone(), 1);
        drop(reset_guard);
        sync_immutable_publication_directory(&destination, &directory)
            .expect("dropping the guard resets an unconsumed failure");
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
}
