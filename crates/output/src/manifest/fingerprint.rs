use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::error::OutputError;

const FILE_FINGERPRINT_CONTENT_HASH_ALGORITHM: &str = "sha256";
pub(crate) const FILE_FINGERPRINT_METADATA_ONLY: &str = "metadata-only";

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct ManifestFileFingerprint {
    pub path: String,
    pub size: u64,
    pub mtime_ns: i64,
    pub content_hash_algorithm: String,
    pub content_sha256: Option<String>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct ManifestFileFingerprintCacheKey {
    path: PathBuf,
    include_content_hash: bool,
    size: u64,
    mtime_ns: i64,
    device_identifier: u64,
    inode_identifier: u64,
    change_time_nanoseconds: i128,
}

impl ManifestFileFingerprintCacheKey {
    fn from_metadata(
        path: PathBuf,
        include_content_hash: bool,
        metadata: &std::fs::Metadata,
    ) -> Result<Self, OutputError> {
        Ok(Self {
            path,
            include_content_hash,
            size: metadata.len(),
            mtime_ns: file_metadata_mtime_ns(metadata)?,
            device_identifier: metadata.dev(),
            inode_identifier: metadata.ino(),
            change_time_nanoseconds: i128::from(metadata.ctime()) * 1_000_000_000 + i128::from(metadata.ctime_nsec()),
        })
    }
}

#[derive(Debug, Default)]
pub struct ManifestFileFingerprintCache {
    pub(super) fingerprints_by_key: BTreeMap<ManifestFileFingerprintCacheKey, Arc<ManifestFileFingerprint>>,
    indexed_prediction_keys_by_path: BTreeMap<PathBuf, ManifestFileFingerprintCacheKey>,
}

impl ManifestFileFingerprintCache {
    pub(crate) fn build_file_fingerprint(
        &mut self,
        file_path: &Path,
        include_content_hash: bool,
    ) -> Result<Arc<ManifestFileFingerprint>, OutputError> {
        let canonical_path = file_path.canonicalize().map_err(OutputError::runtime)?;
        let metadata = canonical_path.metadata().map_err(OutputError::runtime)?;
        let cache_key =
            ManifestFileFingerprintCacheKey::from_metadata(canonical_path.clone(), include_content_hash, &metadata)?;
        if let Some(cached_fingerprint) = self.fingerprints_by_key.get(&cache_key) {
            return Ok(Arc::clone(cached_fingerprint));
        }
        let file_fingerprint =
            Arc::new(build_manifest_file_fingerprint(&canonical_path, &metadata, include_content_hash)?);
        self.fingerprints_by_key.insert(cache_key, Arc::clone(&file_fingerprint));
        Ok(file_fingerprint)
    }

    /// Reuse an exact prediction digest captured while its source was indexed.
    ///
    /// The caller supplies metadata and the digest from the same verified read.
    /// Source identity is checked again before this cache accepts the digest.
    ///
    /// # Errors
    ///
    /// Returns an error when the path no longer names the indexed source, metadata
    /// cannot be read, or an existing fingerprint disagrees for the same identity.
    pub fn register_indexed_prediction_file_fingerprint(
        &mut self,
        file_path: &Path,
        indexed_metadata: &std::fs::Metadata,
        content_sha256: &[u8; 32],
    ) -> Result<(), OutputError> {
        let canonical_path = file_path.canonicalize().map_err(OutputError::runtime)?;
        let cache_key = ManifestFileFingerprintCacheKey::from_metadata(canonical_path.clone(), true, indexed_metadata)?;
        validate_indexed_prediction_source(file_path, &canonical_path, &cache_key)?;
        if let Some(registered_key) = self.indexed_prediction_keys_by_path.get(file_path)
            && registered_key != &cache_key
        {
            return Err(OutputError::Runtime(format!("LOCO file changed after indexing: {}.", file_path.display())));
        }
        let content_sha256 = hex::encode(content_sha256);
        if let Some(cached_fingerprint) = self.fingerprints_by_key.get(&cache_key) {
            if cached_fingerprint.content_sha256.as_deref() != Some(content_sha256.as_str()) {
                return Err(OutputError::Runtime(format!(
                    "Indexed LOCO fingerprint disagrees with the cached source: {}.",
                    file_path.display()
                )));
            }
            self.indexed_prediction_keys_by_path.insert(file_path.to_path_buf(), cache_key);
            return Ok(());
        }
        let fingerprint = Arc::new(ManifestFileFingerprint {
            path: canonical_path.display().to_string(),
            size: indexed_metadata.len(),
            mtime_ns: file_metadata_mtime_ns(indexed_metadata)?,
            content_hash_algorithm: FILE_FINGERPRINT_CONTENT_HASH_ALGORITHM.to_string(),
            content_sha256: Some(content_sha256),
        });
        self.fingerprints_by_key.insert(cache_key.clone(), fingerprint);
        self.indexed_prediction_keys_by_path.insert(file_path.to_path_buf(), cache_key);
        Ok(())
    }

    /// Build an output-owned LOCO prediction fingerprint.
    ///
    /// # Errors
    ///
    /// Returns an error when the prediction file cannot be read or hashed, or
    /// when a registered indexed source changes before fingerprint retrieval.
    pub fn build_prediction_loco_file_fingerprint(
        &mut self,
        phenotype_name: Arc<str>,
        file_path: &Path,
    ) -> Result<super::header::PredictionLocoFileFingerprint, OutputError> {
        let file_fingerprint = if let Some(registered_key) = self.indexed_prediction_keys_by_path.get(file_path) {
            let canonical_path = file_path.canonicalize().map_err(OutputError::runtime)?;
            validate_indexed_prediction_source(file_path, &canonical_path, registered_key)?;
            Arc::clone(self.fingerprints_by_key.get(registered_key).ok_or_else(|| {
                OutputError::Runtime(format!("Registered LOCO fingerprint is unavailable: {}.", file_path.display()))
            })?)
        } else {
            self.build_file_fingerprint(file_path, true)?
        };
        Ok(super::header::PredictionLocoFileFingerprint { phenotype_name, file_fingerprint })
    }
}

fn validate_indexed_prediction_source(
    file_path: &Path,
    canonical_path: &Path,
    expected_key: &ManifestFileFingerprintCacheKey,
) -> Result<(), OutputError> {
    let source_paths = std::iter::once(canonical_path).chain((file_path != canonical_path).then_some(file_path));
    for source_path in source_paths {
        let observed_metadata = source_path.metadata().map_err(OutputError::runtime)?;
        let observed_key =
            ManifestFileFingerprintCacheKey::from_metadata(canonical_path.to_path_buf(), true, &observed_metadata)?;
        if &observed_key != expected_key {
            return Err(OutputError::Runtime(format!("LOCO file changed after indexing: {}.", file_path.display())));
        }
    }
    Ok(())
}

fn build_manifest_file_fingerprint(
    file_path: &Path,
    metadata: &std::fs::Metadata,
    include_content_hash: bool,
) -> Result<ManifestFileFingerprint, OutputError> {
    let content_hash_algorithm =
        if include_content_hash { FILE_FINGERPRINT_CONTENT_HASH_ALGORITHM } else { FILE_FINGERPRINT_METADATA_ONLY };
    let content_sha256 = if include_content_hash { Some(build_file_content_sha256(file_path)?) } else { None };
    let mtime_ns = file_metadata_mtime_ns(metadata)?;
    Ok(ManifestFileFingerprint {
        path: file_path.display().to_string(),
        size: metadata.len(),
        mtime_ns,
        content_hash_algorithm: content_hash_algorithm.to_string(),
        content_sha256,
    })
}

pub(crate) fn manifest_file_fingerprint_to_value(file_fingerprint: &ManifestFileFingerprint) -> Value {
    json!({
        "path": &file_fingerprint.path,
        "size": file_fingerprint.size,
        "mtime_ns": file_fingerprint.mtime_ns,
        "content_hash_algorithm": &file_fingerprint.content_hash_algorithm,
        "content_sha256": &file_fingerprint.content_sha256,
    })
}

fn build_file_content_sha256(path: &Path) -> Result<String, OutputError> {
    let mut file = File::open(path).map_err(OutputError::runtime)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let bytes_read = file.read(&mut buffer).map_err(OutputError::runtime)?;
        if bytes_read == 0 {
            break;
        }
        digest.update(&buffer[..bytes_read]);
    }
    Ok(hex::encode(digest.finalize()))
}

pub(crate) fn build_manifest_value_sha256(value: &Value) -> Result<String, OutputError> {
    let manifest_bytes = serde_json::to_vec(value).map_err(OutputError::runtime)?;
    let mut digest = Sha256::new();
    digest.update(manifest_bytes);
    Ok(hex::encode(digest.finalize()))
}

fn file_metadata_mtime_ns(metadata: &std::fs::Metadata) -> Result<i64, OutputError> {
    metadata
        .mtime()
        .checked_mul(1_000_000_000)
        .and_then(|mtime_seconds_ns| mtime_seconds_ns.checked_add(metadata.mtime_nsec()))
        .ok_or_else(|| OutputError::Runtime("File modification timestamp overflowed nanoseconds.".to_string()))
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::MetadataExt;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use serde_json::json;
    use sha2::{Digest, Sha256};

    use super::{
        FILE_FINGERPRINT_METADATA_ONLY, ManifestFileFingerprintCache, build_manifest_value_sha256,
        manifest_file_fingerprint_to_value,
    };

    struct TestFile {
        path: PathBuf,
    }

    impl TestFile {
        fn new(contents: &[u8]) -> Self {
            static FILE_COUNTER: AtomicU64 = AtomicU64::new(0);
            let sequence = FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
            let timestamp =
                SystemTime::now().duration_since(UNIX_EPOCH).expect("test time is after Unix epoch").as_nanos();
            let path = std::env::temp_dir()
                .join(format!("g-output-fingerprint-{}-{timestamp}-{sequence}", std::process::id()));
            std::fs::write(&path, contents).expect("test file writes");
            Self { path }
        }
    }

    impl Drop for TestFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    #[test]
    fn content_fingerprint_hashes_bytes_and_reuses_identical_cache_entry() {
        let test_file = TestFile::new(b"abc");
        let mut cache = ManifestFileFingerprintCache::default();
        let first = cache.build_file_fingerprint(&test_file.path, true).expect("fingerprint builds");
        let second = cache.build_file_fingerprint(&test_file.path, true).expect("fingerprint is cached");

        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(first.size, 3);
        assert_eq!(first.content_hash_algorithm, "sha256");
        assert_eq!(
            first.content_sha256.as_deref(),
            Some("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        );
        assert_eq!(manifest_file_fingerprint_to_value(&first)["path"], first.path);
    }

    #[test]
    fn metadata_only_fingerprint_has_distinct_cache_identity_and_no_content_hash() {
        let test_file = TestFile::new(b"content");
        let mut cache = ManifestFileFingerprintCache::default();
        let content = cache.build_file_fingerprint(&test_file.path, true).expect("content fingerprint builds");
        let metadata = cache.build_file_fingerprint(&test_file.path, false).expect("metadata fingerprint builds");

        assert!(!Arc::ptr_eq(&content, &metadata));
        assert_eq!(metadata.content_hash_algorithm, FILE_FINGERPRINT_METADATA_ONLY);
        assert_eq!(metadata.content_sha256, None);
    }

    #[test]
    fn indexed_prediction_digest_reuses_the_exact_manifest_fingerprint() {
        let test_file = TestFile::new(b"FID_IID f_i\r\n\n22 0.125\r\n1 NA");
        let metadata = test_file.path.metadata().expect("indexed file metadata reads");
        let digest: [u8; 32] = Sha256::digest(std::fs::read(&test_file.path).expect("indexed bytes read")).into();
        let mut independent_cache = ManifestFileFingerprintCache::default();
        let expected = independent_cache.build_file_fingerprint(&test_file.path, true).expect("file hashes");
        let mut indexed_cache = ManifestFileFingerprintCache::default();
        indexed_cache
            .register_indexed_prediction_file_fingerprint(&test_file.path, &metadata, &digest)
            .expect("verified indexed fingerprint registers");
        let registered = Arc::clone(indexed_cache.fingerprints_by_key.values().next().expect("entry registers"));
        let observed = indexed_cache.build_file_fingerprint(&test_file.path, true).expect("entry is reused");
        assert!(Arc::ptr_eq(&registered, &observed));
        assert_eq!(manifest_file_fingerprint_to_value(&observed), manifest_file_fingerprint_to_value(&expected));
        indexed_cache
            .register_indexed_prediction_file_fingerprint(&test_file.path, &metadata, &digest)
            .expect("another phenotype can register the same file");
        assert_eq!(indexed_cache.fingerprints_by_key.len(), 1);
    }

    #[test]
    fn indexed_prediction_digest_rejects_changed_bytes_with_restored_mtime() {
        let test_file = TestFile::new(b"old");
        let metadata = test_file.path.metadata().expect("indexed metadata reads");
        let digest: [u8; 32] = Sha256::digest(b"old").into();
        let mut cache = ManifestFileFingerprintCache::default();
        cache
            .register_indexed_prediction_file_fingerprint(&test_file.path, &metadata, &digest)
            .expect("original digest registers");
        // Cross the timestamp boundary on filesystems with one-second ctime precision.
        std::thread::sleep(Duration::from_millis(1_100));
        std::fs::write(&test_file.path, b"new").expect("same-length replacement writes");
        let file = std::fs::OpenOptions::new().write(true).open(&test_file.path).expect("changed file opens");
        file.set_times(std::fs::FileTimes::new().set_modified(metadata.modified().expect("original mtime reads")))
            .expect("original mtime restores");

        let changed_metadata = test_file.path.metadata().expect("changed source metadata reads");
        assert_ne!(metadata.ctime(), changed_metadata.ctime(), "mutation must have a distinct ctime");
        assert_eq!(metadata.len(), changed_metadata.len());
        assert_eq!(metadata.ino(), changed_metadata.ino());
        assert_eq!(
            metadata.modified().expect("indexed mtime reads"),
            changed_metadata.modified().expect("restored mtime reads")
        );
        assert!(cache.register_indexed_prediction_file_fingerprint(&test_file.path, &metadata, &digest).is_err());
        assert!(cache.build_prediction_loco_file_fingerprint(Arc::from("trait"), &test_file.path).is_err());
        let changed_digest: [u8; 32] = Sha256::digest(b"new").into();
        assert!(
            cache
                .register_indexed_prediction_file_fingerprint(&test_file.path, &changed_metadata, &changed_digest)
                .is_err()
        );
        let changed = cache.build_file_fingerprint(&test_file.path, true).expect("changed content hashes anew");
        assert_eq!(changed.content_sha256, Some(hex::encode(Sha256::digest(b"new"))));
    }

    #[test]
    fn indexed_prediction_digest_rejects_replaced_file_with_identical_size_and_mtime() {
        let test_file = TestFile::new(b"old");
        let replacement = TestFile::new(b"new");
        let metadata = test_file.path.metadata().expect("indexed metadata reads");
        let file = std::fs::OpenOptions::new().write(true).open(&replacement.path).expect("replacement opens");
        file.set_times(std::fs::FileTimes::new().set_modified(metadata.modified().expect("indexed mtime reads")))
            .expect("replacement mtime matches indexed source");
        std::fs::rename(&replacement.path, &test_file.path).expect("source is atomically replaced");
        let digest: [u8; 32] = Sha256::digest(b"old").into();
        assert!(
            ManifestFileFingerprintCache::default()
                .register_indexed_prediction_file_fingerprint(&test_file.path, &metadata, &digest)
                .is_err()
        );
    }

    #[test]
    fn indexed_prediction_digest_rejects_conflicting_hash_for_unchanged_source() {
        let test_file = TestFile::new(b"same");
        let metadata = test_file.path.metadata().expect("indexed metadata reads");
        let mut cache = ManifestFileFingerprintCache::default();
        let digest: [u8; 32] = Sha256::digest(b"same").into();
        cache
            .register_indexed_prediction_file_fingerprint(&test_file.path, &metadata, &digest)
            .expect("correct fingerprint registers");
        let conflicting_digest: [u8; 32] = Sha256::digest(b"different").into();
        assert!(
            cache
                .register_indexed_prediction_file_fingerprint(&test_file.path, &metadata, &conflicting_digest)
                .is_err()
        );
    }

    #[test]
    fn registered_prediction_aliases_share_fingerprints_and_reject_later_retargeting() {
        let original = TestFile::new(b"old");
        let replacement = TestFile::new(b"new");
        let alias = TestFile::new(b"");
        std::fs::remove_file(&alias.path).expect("alias placeholder is removed");
        std::os::unix::fs::symlink(&original.path, &alias.path).expect("prediction alias is created");
        let metadata = original.path.metadata().expect("indexed metadata reads");
        let digest: [u8; 32] = Sha256::digest(b"old").into();
        let mut cache = ManifestFileFingerprintCache::default();
        for path in [&original.path, &alias.path] {
            cache
                .register_indexed_prediction_file_fingerprint(path, &metadata, &digest)
                .expect("same indexed source registers through both configured paths");
        }
        let direct = cache
            .build_prediction_loco_file_fingerprint(Arc::from("direct-trait"), &original.path)
            .expect("direct source still verifies");
        let aliased = cache
            .build_prediction_loco_file_fingerprint(Arc::from("aliased-trait"), &alias.path)
            .expect("alias still verifies");
        assert!(Arc::ptr_eq(&direct.file_fingerprint, &aliased.file_fingerprint));
        std::fs::remove_file(&alias.path).expect("original alias unlinks");
        std::os::unix::fs::symlink(&replacement.path, &alias.path).expect("alias is retargeted");
        assert!(cache.build_prediction_loco_file_fingerprint(Arc::from("aliased-trait"), &alias.path).is_err());
        assert!(cache.build_prediction_loco_file_fingerprint(Arc::from("direct-trait"), &original.path).is_ok());
        let replacement_metadata = replacement.path.metadata().expect("replacement metadata reads");
        let replacement_digest: [u8; 32] = Sha256::digest(b"new").into();
        assert!(
            cache
                .register_indexed_prediction_file_fingerprint(&alias.path, &replacement_metadata, &replacement_digest)
                .is_err()
        );
    }

    #[test]
    fn manifest_value_hash_is_deterministic_and_value_sensitive() {
        let first =
            build_manifest_value_sha256(&json!({"schema_version": 0, "name": "alpha"})).expect("manifest hashes");
        let repeated = build_manifest_value_sha256(&json!({"schema_version": 0, "name": "alpha"}))
            .expect("manifest hashes repeatedly");
        let changed = build_manifest_value_sha256(&json!({"schema_version": 0, "name": "beta"}))
            .expect("changed manifest hashes");

        assert_eq!(first, repeated);
        assert_ne!(first, changed);
        assert_eq!(first.len(), 64);
    }
}
