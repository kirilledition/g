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

#[derive(Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct ManifestFileFingerprintCacheKey {
    path: PathBuf,
    include_content_hash: bool,
    size: u64,
    mtime_ns: i64,
}

#[derive(Debug, Default)]
pub struct ManifestFileFingerprintCache {
    pub(super) fingerprints_by_key: BTreeMap<ManifestFileFingerprintCacheKey, Arc<ManifestFileFingerprint>>,
}

impl ManifestFileFingerprintCache {
    pub(crate) fn build_file_fingerprint(
        &mut self,
        file_path: &Path,
        include_content_hash: bool,
    ) -> Result<Arc<ManifestFileFingerprint>, OutputError> {
        let canonical_path = file_path.canonicalize().map_err(OutputError::runtime)?;
        let metadata = canonical_path.metadata().map_err(OutputError::runtime)?;
        let cache_key = ManifestFileFingerprintCacheKey {
            path: canonical_path.clone(),
            include_content_hash,
            size: metadata.len(),
            mtime_ns: file_metadata_mtime_ns(&metadata)?,
        };
        if let Some(cached_fingerprint) = self.fingerprints_by_key.get(&cache_key) {
            return Ok(Arc::clone(cached_fingerprint));
        }
        let file_fingerprint =
            Arc::new(build_manifest_file_fingerprint(&canonical_path, &metadata, include_content_hash)?);
        self.fingerprints_by_key.insert(cache_key, Arc::clone(&file_fingerprint));
        Ok(file_fingerprint)
    }

    /// Build an output-owned LOCO prediction fingerprint.
    ///
    /// # Errors
    ///
    /// Returns an error when the prediction file cannot be read or hashed.
    pub fn build_prediction_loco_file_fingerprint(
        &mut self,
        phenotype_name: Arc<str>,
        file_path: &Path,
    ) -> Result<super::header::PredictionLocoFileFingerprint, OutputError> {
        let file_fingerprint = self.build_file_fingerprint(file_path, true)?;
        Ok(super::header::PredictionLocoFileFingerprint { phenotype_name, file_fingerprint })
    }
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
