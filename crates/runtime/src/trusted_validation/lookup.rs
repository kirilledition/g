use std::path::Path;

use super::error::TrustedBgenValidationCacheLookupError;
use super::types::TrustedBgenValidationCacheLookupPlan;

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
