//! Trusted BGEN validation orchestration.

use std::path::Path;

use g_genotype::{BgenError, BgenReaderCore};
use g_runtime::{
    TrustedBgenValidationCacheLookupError, TrustedBgenValidationFingerprintInput,
    build_trusted_bgen_validation_cache_path, build_trusted_bgen_validation_fingerprint,
    plan_trusted_bgen_validation_cache_lookup, require_cache_backed_trusted_bgen_validation_mode,
    write_trusted_bgen_validation_cache_payload,
};

#[derive(Debug, thiserror::Error)]
pub enum TrustedBgenValidationError {
    #[error("BGEN sample count exceeds the native validation cache range.")]
    SampleCountRange,
    #[error("BGEN variant count exceeds the native validation cache range.")]
    VariantCountRange,
    #[error(transparent)]
    CacheLookup(#[from] TrustedBgenValidationCacheLookupError),
    #[error(transparent)]
    Bgen(#[from] BgenError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Validate a trusted no-missing diploid BGEN reader through a persistent cache.
///
/// # Errors
///
/// Returns an error when the validation mode is unsupported, BGEN dimensions
/// cannot be represented in cache metadata, BGEN validation fails, or cache
/// fingerprint/payload I/O fails.
pub(crate) fn validate_trusted_no_missing_diploid_with_cache_directory(
    reader: &BgenReaderCore,
    bgen_path: &Path,
    validation_mode: &str,
    cache_directory: &Path,
) -> Result<(), TrustedBgenValidationError> {
    require_cache_backed_trusted_bgen_validation_mode(validation_mode)?;
    let sample_count =
        i64::try_from(reader.sample_count()).map_err(|_| TrustedBgenValidationError::SampleCountRange)?;
    let variant_count =
        i64::try_from(reader.variant_count()).map_err(|_| TrustedBgenValidationError::VariantCountRange)?;
    let fingerprint = build_trusted_bgen_validation_fingerprint(&TrustedBgenValidationFingerprintInput {
        bgen_path: bgen_path.into(),
        sample_count,
        variant_count,
        trusted_no_missing_diploid: true,
    })?;
    let cache_path = build_trusted_bgen_validation_cache_path(cache_directory, &fingerprint);
    let cache_lookup_plan = plan_trusted_bgen_validation_cache_lookup(validation_mode, &cache_path)?;
    if cache_lookup_plan.should_mark_validated {
        reader.mark_trusted_no_missing_diploid_validated()?;
    }
    if !cache_lookup_plan.should_validate {
        return Ok(());
    }
    reader.validate_trusted_no_missing_diploid()?;
    if cache_lookup_plan.should_write_cache {
        write_trusted_bgen_validation_cache_payload(&cache_path, fingerprint, bgen_path, sample_count, variant_count)?;
    }
    Ok(())
}
