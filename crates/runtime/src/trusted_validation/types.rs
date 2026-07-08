use std::path::PathBuf;

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
