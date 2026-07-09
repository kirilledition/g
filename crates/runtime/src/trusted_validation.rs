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
