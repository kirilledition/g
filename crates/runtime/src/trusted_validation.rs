//! Deterministic trusted BGEN validation cache metadata and writes.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use serde_json::Value;
use sha2::{Digest, Sha256};

const TRUSTED_BGEN_VALIDATION_SCHEMA_VERSION: i64 = 1;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrustedBgenValidationFingerprintInput {
    pub bgen_path: PathBuf,
    pub sample_count: i64,
    pub variant_count: i64,
    pub trusted_no_missing_diploid: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrustedBgenValidationCachePayload {
    pub schema_version: i64,
    pub fingerprint: String,
    pub bgen_path: String,
    pub sample_count: i64,
    pub variant_count: i64,
}

/// Build the cache fingerprint for trusted BGEN validation inputs.
///
/// # Errors
///
/// Returns an error when the BGEN path metadata or canonical path cannot be
/// read, or when the fingerprint payload cannot be serialized.
pub fn build_trusted_bgen_validation_fingerprint(
    input: &TrustedBgenValidationFingerprintInput,
) -> Result<String, std::io::Error> {
    let bgen_metadata = input.bgen_path.metadata()?;
    let resolved_bgen_path = input.bgen_path.canonicalize()?;
    let modified_time_nanoseconds =
        bgen_metadata.mtime().saturating_mul(1_000_000_000).saturating_add(bgen_metadata.mtime_nsec());
    let mut fingerprint_payload = BTreeMap::new();
    fingerprint_payload.insert("bgen_path", Value::String(resolved_bgen_path.display().to_string()));
    fingerprint_payload.insert("mtime_ns", Value::from(modified_time_nanoseconds));
    fingerprint_payload.insert("sample_count", Value::from(input.sample_count));
    fingerprint_payload.insert("schema_version", Value::from(TRUSTED_BGEN_VALIDATION_SCHEMA_VERSION));
    fingerprint_payload.insert("size", Value::from(bgen_metadata.size()));
    fingerprint_payload.insert("trusted_no_missing_diploid", Value::Bool(input.trusted_no_missing_diploid));
    fingerprint_payload.insert("variant_count", Value::from(input.variant_count));
    let fingerprint_bytes = serde_json::to_vec(&fingerprint_payload).map_err(std::io::Error::other)?;
    Ok(finalize_sha256_hex(Sha256::digest(fingerprint_bytes)))
}

/// Build the trusted BGEN validation cache path for a fingerprint.
#[must_use]
pub fn build_trusted_bgen_validation_cache_path(cache_directory: &Path, fingerprint: &str) -> PathBuf {
    cache_directory.join(format!("{fingerprint}.json"))
}

/// Build the trusted BGEN validation cache payload.
///
/// # Errors
///
/// Returns an error when the BGEN path cannot be canonicalized.
pub fn build_trusted_bgen_validation_cache_payload(
    fingerprint: String,
    bgen_path: &Path,
    sample_count: i64,
    variant_count: i64,
) -> Result<TrustedBgenValidationCachePayload, std::io::Error> {
    Ok(TrustedBgenValidationCachePayload {
        schema_version: TRUSTED_BGEN_VALIDATION_SCHEMA_VERSION,
        fingerprint,
        bgen_path: bgen_path.canonicalize()?.display().to_string(),
        sample_count,
        variant_count,
    })
}

/// Write a trusted BGEN validation cache payload atomically.
///
/// # Errors
///
/// Returns an error when the payload cannot be built, the cache directory
/// cannot be created, the temporary payload cannot be written, or the temporary
/// path cannot be renamed into place.
pub fn write_trusted_bgen_validation_cache_payload(
    cache_path: &Path,
    fingerprint: String,
    bgen_path: &Path,
    sample_count: i64,
    variant_count: i64,
) -> Result<(), std::io::Error> {
    let cache_payload =
        build_trusted_bgen_validation_cache_payload(fingerprint, bgen_path, sample_count, variant_count)?;
    write_trusted_bgen_validation_cache_payload_to_path(cache_path, &cache_payload)
}

/// Write an already-built trusted BGEN validation cache payload atomically.
///
/// # Errors
///
/// Returns an error when the cache directory cannot be created, the payload
/// cannot be serialized, the temporary file cannot be written, or the
/// temporary path cannot be renamed into place.
pub fn write_trusted_bgen_validation_cache_payload_to_path(
    cache_path: &Path,
    cache_payload: &TrustedBgenValidationCachePayload,
) -> Result<(), std::io::Error> {
    if let Some(parent_path) = cache_path.parent() {
        fs::create_dir_all(parent_path)?;
    }
    let temporary_cache_path = cache_path.with_extension("json.tmp");
    fs::write(&temporary_cache_path, serialize_trusted_bgen_validation_cache_payload(cache_payload)?)?;
    fs::rename(temporary_cache_path, cache_path)
}

/// Serialize a trusted BGEN validation cache payload to deterministic JSON.
///
/// # Errors
///
/// Returns an error when the payload cannot be serialized as JSON.
pub fn serialize_trusted_bgen_validation_cache_payload(
    cache_payload: &TrustedBgenValidationCachePayload,
) -> Result<String, std::io::Error> {
    let mut serialized_payload = BTreeMap::new();
    serialized_payload.insert("bgen_path", Value::String(cache_payload.bgen_path.clone()));
    serialized_payload.insert("fingerprint", Value::String(cache_payload.fingerprint.clone()));
    serialized_payload.insert("sample_count", Value::from(cache_payload.sample_count));
    serialized_payload.insert("schema_version", Value::from(cache_payload.schema_version));
    serialized_payload.insert("variant_count", Value::from(cache_payload.variant_count));
    let mut payload_text = serde_json::to_string_pretty(&serialized_payload).map_err(std::io::Error::other)?;
    payload_text.push('\n');
    Ok(payload_text)
}

fn finalize_sha256_hex(digest_bytes: impl AsRef<[u8]>) -> String {
    let mut digest_hex = String::with_capacity(digest_bytes.as_ref().len() * 2);
    for byte in digest_bytes.as_ref() {
        write!(&mut digest_hex, "{byte:02x}").expect("writing to String must succeed");
    }
    digest_hex
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trusted_validation_test_directory(test_name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("g-runtime-{test_name}-{}", uuid::Uuid::new_v4()))
    }

    #[test]
    fn serializes_cache_payload_with_stable_sorted_keys() {
        let payload = TrustedBgenValidationCachePayload {
            schema_version: 1,
            fingerprint: "abc123".to_string(),
            bgen_path: "/tmp/study.bgen".to_string(),
            sample_count: 10,
            variant_count: 20,
        };

        let payload_text = serialize_trusted_bgen_validation_cache_payload(&payload).expect("payload serializes");

        assert_eq!(
            payload_text,
            "{\n  \"bgen_path\": \"/tmp/study.bgen\",\n  \"fingerprint\": \"abc123\",\n  \"sample_count\": 10,\n  \"schema_version\": 1,\n  \"variant_count\": 20\n}\n"
        );
    }

    #[test]
    fn writes_cache_payload_through_temporary_file() {
        let test_directory = trusted_validation_test_directory("cache-write");
        let cache_path = test_directory.join("cache").join("abc123.json");
        let bgen_path = test_directory.join("study.bgen");
        fs::create_dir_all(&test_directory).expect("test directory should be created");
        fs::write(&bgen_path, b"bgen").expect("BGEN fixture should be written");

        write_trusted_bgen_validation_cache_payload(&cache_path, "abc123".to_string(), &bgen_path, 10, 20)
            .expect("cache payload should be written");

        let payload_text = fs::read_to_string(&cache_path).expect("cache payload should be readable");
        assert!(payload_text.contains("\"fingerprint\": \"abc123\""));
        assert!(!cache_path.with_extension("json.tmp").exists());

        fs::remove_dir_all(test_directory).expect("test directory should be removable");
    }
}
