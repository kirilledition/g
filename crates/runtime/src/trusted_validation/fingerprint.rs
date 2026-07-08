use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::os::unix::fs::MetadataExt;

use serde_json::Value;
use sha2::{Digest, Sha256};

use super::TRUSTED_BGEN_VALIDATION_SCHEMA_VERSION;
use super::types::TrustedBgenValidationFingerprintInput;

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

fn finalize_sha256_hex(digest_bytes: impl AsRef<[u8]>) -> String {
    let mut digest_hex = String::with_capacity(digest_bytes.as_ref().len() * 2);
    for byte in digest_bytes.as_ref() {
        write!(&mut digest_hex, "{byte:02x}").expect("writing to String must succeed");
    }
    digest_hex
}
