//! Deterministic trusted BGEN validation cache metadata and writes.

mod cache_payload;
mod error;
mod fingerprint;
mod lookup;
mod paths;
mod types;

const TRUSTED_BGEN_VALIDATION_SCHEMA_VERSION: i64 = 1;

pub use cache_payload::{
    build_trusted_bgen_validation_cache_payload, serialize_trusted_bgen_validation_cache_payload,
    write_trusted_bgen_validation_cache_payload, write_trusted_bgen_validation_cache_payload_to_path,
};
pub use error::{TrustedBgenValidationCacheDirectoryError, TrustedBgenValidationCacheLookupError};
pub use fingerprint::build_trusted_bgen_validation_fingerprint;
pub use lookup::{plan_trusted_bgen_validation_cache_lookup, require_cache_backed_trusted_bgen_validation_mode};
pub use paths::{
    build_default_trusted_bgen_validation_cache_directory, build_trusted_bgen_validation_cache_path,
    default_trusted_bgen_validation_cache_directory,
};
pub use types::{
    TrustedBgenValidationCacheLookupPlan, TrustedBgenValidationCachePayload, TrustedBgenValidationFingerprintInput,
};

#[cfg(test)]
use std::fs;
#[cfg(test)]
use std::path::{Path, PathBuf};

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
