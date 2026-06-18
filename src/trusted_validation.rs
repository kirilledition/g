//! Deterministic trusted BGEN validation cache metadata.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use serde_json::Value;
use sha2::{Digest, Sha256};

const TRUSTED_BGEN_VALIDATION_SCHEMA_VERSION: i64 = 1;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TrustedBgenValidationFingerprintInput {
    pub(crate) bgen_path: PathBuf,
    pub(crate) sample_count: i64,
    pub(crate) variant_count: i64,
    pub(crate) trusted_no_missing_diploid: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TrustedBgenValidationCachePayload {
    pub(crate) schema_version: i64,
    pub(crate) fingerprint: String,
    pub(crate) bgen_path: String,
    pub(crate) sample_count: i64,
    pub(crate) variant_count: i64,
}

pub(crate) fn build_trusted_bgen_validation_fingerprint(
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
    let fingerprint_bytes = serde_json::to_vec(&fingerprint_payload).expect("fingerprint payload serialization");
    Ok(finalize_sha256_hex(Sha256::digest(fingerprint_bytes)))
}

pub(crate) fn build_trusted_bgen_validation_cache_path(cache_directory: &Path, fingerprint: &str) -> PathBuf {
    cache_directory.join(format!("{fingerprint}.json"))
}

pub(crate) fn build_trusted_bgen_validation_cache_payload(
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

fn finalize_sha256_hex(digest_bytes: impl AsRef<[u8]>) -> String {
    let mut digest_hex = String::with_capacity(digest_bytes.as_ref().len() * 2);
    for byte in digest_bytes.as_ref() {
        write!(&mut digest_hex, "{byte:02x}").expect("writing to String must succeed");
    }
    digest_hex
}
