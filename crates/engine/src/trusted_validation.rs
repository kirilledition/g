//! Trusted BGEN validation and its persistent cache.

use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use g_genotype::{BgenError, BgenReaderCore};
use serde::Serialize;
use sha2::{Digest, Sha256};

const CACHE_APPLICATION_DIRECTORY: &str = "g";
const CACHE_DIRECTORY_NAME: &str = "bgen_validation";
const CACHE_SCHEMA_VERSION: i64 = 1;

#[derive(Debug, thiserror::Error)]
pub(crate) enum TrustedBgenValidationError {
    #[error("BGEN sample count exceeds the native validation cache range.")]
    SampleCountRange,
    #[error("BGEN variant count exceeds the native validation cache range.")]
    VariantCountRange,
    #[error("Unable to resolve trusted BGEN validation cache directory: neither XDG_CACHE_HOME nor HOME is set.")]
    MissingCacheDirectory,
    #[error(transparent)]
    Bgen(#[from] BgenError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

#[derive(Serialize)]
struct ValidationFingerprintPayload {
    bgen_path: String,
    mtime_ns: i64,
    sample_count: i64,
    schema_version: i64,
    size: u64,
    trusted_no_missing_diploid: bool,
    variant_count: i64,
}

#[derive(Serialize)]
struct ValidationCachePayload<'payload> {
    bgen_path: String,
    fingerprint: &'payload str,
    sample_count: i64,
    schema_version: i64,
    variant_count: i64,
}

pub(crate) fn default_cache_directory() -> Result<PathBuf, TrustedBgenValidationError> {
    if let Some(cache_root) = non_empty_environment_path("XDG_CACHE_HOME") {
        return Ok(cache_root.join(CACHE_APPLICATION_DIRECTORY).join(CACHE_DIRECTORY_NAME));
    }
    let home_directory = non_empty_environment_path("HOME").ok_or(TrustedBgenValidationError::MissingCacheDirectory)?;
    Ok(home_directory.join(".cache").join(CACHE_APPLICATION_DIRECTORY).join(CACHE_DIRECTORY_NAME))
}

/// Validate a trusted no-missing diploid BGEN reader through a persistent cache.
///
/// # Errors
///
/// Returns an error when BGEN dimensions cannot be represented in cache
/// metadata, BGEN validation fails, or cache fingerprint/payload I/O fails.
pub(crate) fn validate_trusted_no_missing_diploid_with_cache_directory(
    reader: &BgenReaderCore,
    bgen_path: &Path,
    validation_mode: g_plan::TrustedBgenValidationMode,
    cache_directory: &Path,
) -> Result<(), TrustedBgenValidationError> {
    let sample_count =
        i64::try_from(reader.sample_count()).map_err(|_| TrustedBgenValidationError::SampleCountRange)?;
    let variant_count =
        i64::try_from(reader.variant_count()).map_err(|_| TrustedBgenValidationError::VariantCountRange)?;
    let (fingerprint, canonical_bgen_path) = build_validation_fingerprint(bgen_path, sample_count, variant_count)?;
    let cache_path = cache_directory.join(format!("{fingerprint}.json"));
    if validation_mode == g_plan::TrustedBgenValidationMode::CacheOnMiss && cache_path.exists() {
        reader.mark_trusted_no_missing_diploid_validated()?;
        return Ok(());
    }
    reader.validate_trusted_no_missing_diploid()?;
    write_cache_payload(&cache_path, &fingerprint, canonical_bgen_path, sample_count, variant_count)
}

fn build_validation_fingerprint(
    bgen_path: &Path,
    sample_count: i64,
    variant_count: i64,
) -> Result<(String, String), std::io::Error> {
    let metadata = bgen_path.metadata()?;
    let canonical_bgen_path = bgen_path.canonicalize()?.display().to_string();
    let mtime_ns = metadata
        .mtime()
        .checked_mul(1_000_000_000)
        .and_then(|seconds| seconds.checked_add(metadata.mtime_nsec()))
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "BGEN modification timestamp does not fit signed nanoseconds.",
            )
        })?;
    let payload = ValidationFingerprintPayload {
        bgen_path: canonical_bgen_path.clone(),
        mtime_ns,
        sample_count,
        schema_version: CACHE_SCHEMA_VERSION,
        size: metadata.size(),
        trusted_no_missing_diploid: true,
        variant_count,
    };
    let payload_bytes = serde_json::to_vec(&payload).map_err(std::io::Error::other)?;
    Ok((hex::encode(Sha256::digest(payload_bytes)), canonical_bgen_path))
}

fn write_cache_payload(
    cache_path: &Path,
    fingerprint: &str,
    canonical_bgen_path: String,
    sample_count: i64,
    variant_count: i64,
) -> Result<(), TrustedBgenValidationError> {
    if let Some(parent_path) = cache_path.parent() {
        fs::create_dir_all(parent_path)?;
    }
    let payload = ValidationCachePayload {
        bgen_path: canonical_bgen_path,
        fingerprint,
        sample_count,
        schema_version: CACHE_SCHEMA_VERSION,
        variant_count,
    };
    let mut payload_text = serde_json::to_string_pretty(&payload).map_err(std::io::Error::other)?;
    payload_text.push('\n');
    let temporary_cache_path =
        cache_path.with_file_name(format!(".{fingerprint}.{}.{}.tmp", std::process::id(), uuid::Uuid::new_v4()));
    let write_result = (|| -> std::io::Result<()> {
        let mut temporary_cache_file = OpenOptions::new().write(true).create_new(true).open(&temporary_cache_path)?;
        temporary_cache_file.write_all(payload_text.as_bytes())?;
        temporary_cache_file.sync_all()
    })();
    if let Err(error) = write_result {
        let _ = fs::remove_file(&temporary_cache_path);
        return Err(error.into());
    }
    if let Err(error) = fs::rename(&temporary_cache_path, cache_path) {
        let _ = fs::remove_file(&temporary_cache_path);
        if !cache_path.exists() {
            return Err(error.into());
        }
    }
    Ok(())
}

fn non_empty_environment_path(variable_name: &str) -> Option<PathBuf> {
    std::env::var_os(variable_name).filter(|value| !value.is_empty()).map(PathBuf::from)
}
