//! Deterministic trusted BGEN validation cache metadata and writes.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use serde_json::Value;
use sha2::{Digest, Sha256};

const TRUSTED_BGEN_VALIDATION_SCHEMA_VERSION: i64 = 1;
const ASSUME_VALIDATED_UNSAFE_MESSAGE: &str = concat!(
    "Trusted no-missing diploid validation mode 'assume_validated' is unsafe for calculation runs. ",
    "Use 'cache_on_miss' or 'force_validate' so BGEN compatibility is checked before decoding."
);
const TRUSTED_BGEN_VALIDATION_CACHE_APPLICATION_DIRECTORY: &str = "g";
const TRUSTED_BGEN_VALIDATION_CACHE_DIRECTORY_NAME: &str = "bgen_validation";

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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrustedBgenValidationCacheLookupPlan {
    pub should_mark_validated: bool,
    pub should_validate: bool,
    pub should_write_cache: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TrustedBgenValidationCacheLookupError {
    UnsafeAssumeValidatedMode,
    UnsupportedValidationMode(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TrustedBgenValidationCacheDirectoryError {
    MissingHomeDirectory,
}

impl std::fmt::Display for TrustedBgenValidationCacheLookupError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsafeAssumeValidatedMode => formatter.write_str(ASSUME_VALIDATED_UNSAFE_MESSAGE),
            Self::UnsupportedValidationMode(validation_mode) => {
                write!(formatter, "Unsupported trusted BGEN validation mode: {validation_mode}")
            }
        }
    }
}

impl Error for TrustedBgenValidationCacheLookupError {}

impl std::fmt::Display for TrustedBgenValidationCacheDirectoryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingHomeDirectory => formatter.write_str(
                "Unable to resolve trusted BGEN validation cache directory: neither XDG_CACHE_HOME nor HOME is set.",
            ),
        }
    }
}

impl Error for TrustedBgenValidationCacheDirectoryError {}

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

/// Resolve the default trusted BGEN validation cache directory from the process environment.
///
/// # Errors
///
/// Returns an error when neither `XDG_CACHE_HOME` nor `HOME` is available.
pub fn default_trusted_bgen_validation_cache_directory() -> Result<PathBuf, TrustedBgenValidationCacheDirectoryError> {
    let xdg_cache_home = non_empty_environment_path("XDG_CACHE_HOME");
    let home_directory = non_empty_environment_path("HOME");
    build_default_trusted_bgen_validation_cache_directory(xdg_cache_home.as_deref(), home_directory.as_deref())
}

/// Build the default trusted BGEN validation cache directory from optional root paths.
///
/// # Errors
///
/// Returns an error when no XDG cache root or home directory is available.
pub fn build_default_trusted_bgen_validation_cache_directory(
    xdg_cache_home: Option<&Path>,
    home_directory: Option<&Path>,
) -> Result<PathBuf, TrustedBgenValidationCacheDirectoryError> {
    if let Some(cache_directory_root) = xdg_cache_home {
        return Ok(cache_directory_root
            .join(TRUSTED_BGEN_VALIDATION_CACHE_APPLICATION_DIRECTORY)
            .join(TRUSTED_BGEN_VALIDATION_CACHE_DIRECTORY_NAME));
    }
    let home_directory = home_directory.ok_or(TrustedBgenValidationCacheDirectoryError::MissingHomeDirectory)?;
    Ok(home_directory
        .join(".cache")
        .join(TRUSTED_BGEN_VALIDATION_CACHE_APPLICATION_DIRECTORY)
        .join(TRUSTED_BGEN_VALIDATION_CACHE_DIRECTORY_NAME))
}

/// Require a cache-backed trusted BGEN validation mode for calculation runs.
///
/// # Errors
///
/// Returns an error when the validation mode would skip validation or is not
/// known.
pub fn require_cache_backed_trusted_bgen_validation_mode(
    validation_mode: &str,
) -> Result<(), TrustedBgenValidationCacheLookupError> {
    match validation_mode {
        "cache_on_miss" | "force_validate" => Ok(()),
        "assume_validated" => Err(TrustedBgenValidationCacheLookupError::UnsafeAssumeValidatedMode),
        unsupported_validation_mode => Err(TrustedBgenValidationCacheLookupError::UnsupportedValidationMode(
            unsupported_validation_mode.to_string(),
        )),
    }
}

/// Plan cache lookup behavior for trusted BGEN validation.
///
/// # Errors
///
/// Returns an error when the validation mode is not supported for calculation
/// runs.
pub fn plan_trusted_bgen_validation_cache_lookup(
    validation_mode: &str,
    cache_path: &Path,
) -> Result<TrustedBgenValidationCacheLookupPlan, TrustedBgenValidationCacheLookupError> {
    require_cache_backed_trusted_bgen_validation_mode(validation_mode)?;
    match validation_mode {
        "cache_on_miss" if cache_path.exists() => Ok(TrustedBgenValidationCacheLookupPlan {
            should_mark_validated: true,
            should_validate: false,
            should_write_cache: false,
        }),
        "cache_on_miss" | "force_validate" => Ok(TrustedBgenValidationCacheLookupPlan {
            should_mark_validated: false,
            should_validate: true,
            should_write_cache: true,
        }),
        _ => unreachable!("validation mode compatibility was checked before planning"),
    }
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

fn non_empty_environment_path(variable_name: &str) -> Option<PathBuf> {
    std::env::var_os(variable_name).filter(|environment_value| !environment_value.is_empty()).map(PathBuf::from)
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

    #[test]
    fn builds_default_cache_directory_from_xdg_cache_home() {
        let cache_directory = build_default_trusted_bgen_validation_cache_directory(
            Some(Path::new("/tmp/xdg-cache")),
            Some(Path::new("/home/alice")),
        )
        .expect("cache directory should be built");

        assert_eq!(cache_directory, PathBuf::from("/tmp/xdg-cache/g/bgen_validation"));
    }

    #[test]
    fn builds_default_cache_directory_from_home_directory() {
        let cache_directory =
            build_default_trusted_bgen_validation_cache_directory(None, Some(Path::new("/home/alice")))
                .expect("cache directory should be built");

        assert_eq!(cache_directory, PathBuf::from("/home/alice/.cache/g/bgen_validation"));
    }

    #[test]
    fn rejects_missing_default_cache_directory_roots() {
        let error = build_default_trusted_bgen_validation_cache_directory(None, None)
            .expect_err("missing cache roots should be rejected");

        assert_eq!(
            error.to_string(),
            "Unable to resolve trusted BGEN validation cache directory: neither XDG_CACHE_HOME nor HOME is set."
        );
    }

    #[test]
    fn plans_cache_hit_without_python_file_probe() {
        let test_directory = trusted_validation_test_directory("cache-hit");
        let cache_path = test_directory.join("cache").join("abc123.json");
        fs::create_dir_all(cache_path.parent().expect("cache path should have a parent"))
            .expect("cache directory should be created");
        fs::write(&cache_path, b"{}").expect("cache payload should be written");

        let plan = plan_trusted_bgen_validation_cache_lookup("cache_on_miss", &cache_path)
            .expect("cache hit should be planned");

        assert!(plan.should_mark_validated);
        assert!(!plan.should_validate);
        assert!(!plan.should_write_cache);

        fs::remove_dir_all(test_directory).expect("test directory should be removable");
    }

    #[test]
    fn plans_cache_miss_validation_and_write() {
        let test_directory = trusted_validation_test_directory("cache-miss");
        let cache_path = test_directory.join("cache").join("abc123.json");

        let plan = plan_trusted_bgen_validation_cache_lookup("cache_on_miss", &cache_path)
            .expect("cache miss should be planned");

        assert!(!plan.should_mark_validated);
        assert!(plan.should_validate);
        assert!(plan.should_write_cache);
    }

    #[test]
    fn plans_force_validation_even_when_cache_exists() {
        let test_directory = trusted_validation_test_directory("force-validate");
        let cache_path = test_directory.join("cache").join("abc123.json");
        fs::create_dir_all(cache_path.parent().expect("cache path should have a parent"))
            .expect("cache directory should be created");
        fs::write(&cache_path, b"{}").expect("cache payload should be written");

        let plan = plan_trusted_bgen_validation_cache_lookup("force_validate", &cache_path)
            .expect("force validation should be planned");

        assert!(!plan.should_mark_validated);
        assert!(plan.should_validate);
        assert!(plan.should_write_cache);

        fs::remove_dir_all(test_directory).expect("test directory should be removable");
    }

    #[test]
    fn rejects_unsafe_assumed_validation_mode() {
        let error = require_cache_backed_trusted_bgen_validation_mode("assume_validated")
            .expect_err("unsafe validation mode should be rejected");

        assert_eq!(
            error.to_string(),
            "Trusted no-missing diploid validation mode 'assume_validated' is unsafe for calculation runs. Use 'cache_on_miss' or 'force_validate' so BGEN compatibility is checked before decoding."
        );
    }

    #[test]
    fn rejects_unknown_validation_mode() {
        let error = plan_trusted_bgen_validation_cache_lookup("unknown", Path::new("/tmp/cache.json"))
            .expect_err("unknown validation mode should be rejected");

        assert_eq!(error.to_string(), "Unsupported trusted BGEN validation mode: unknown");
    }
}
