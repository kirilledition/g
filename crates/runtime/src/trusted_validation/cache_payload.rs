use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde_json::Value;

use super::TRUSTED_BGEN_VALIDATION_SCHEMA_VERSION;
use super::types::TrustedBgenValidationCachePayload;

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
