//! Trusted BGEN validation orchestration.

use std::path::Path;

use g_genotype::{BgenError, BgenGenotypeSource};
use g_runtime::debug as native_trusted_validation;

#[derive(Debug, thiserror::Error)]
pub enum TrustedBgenValidationError {
    #[error("BGEN sample count exceeds the native validation cache range.")]
    SampleCountRange,
    #[error("BGEN variant count exceeds the native validation cache range.")]
    VariantCountRange,
    #[error(transparent)]
    CacheLookup(#[from] native_trusted_validation::TrustedBgenValidationCacheLookupError),
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
    reader: &BgenGenotypeSource,
    bgen_path: &Path,
    validation_mode: &str,
    cache_directory: &Path,
) -> Result<(), TrustedBgenValidationError> {
    native_trusted_validation::require_cache_backed_trusted_bgen_validation_mode(validation_mode)?;
    let sample_count =
        i64::try_from(reader.sample_count()).map_err(|_| TrustedBgenValidationError::SampleCountRange)?;
    let variant_count =
        i64::try_from(reader.variant_count()).map_err(|_| TrustedBgenValidationError::VariantCountRange)?;
    let fingerprint = native_trusted_validation::build_trusted_bgen_validation_fingerprint(
        &native_trusted_validation::TrustedBgenValidationFingerprintInput {
            bgen_path: bgen_path.into(),
            sample_count,
            variant_count,
            trusted_no_missing_diploid: true,
        },
    )?;
    let cache_path = native_trusted_validation::build_trusted_bgen_validation_cache_path(cache_directory, &fingerprint);
    let cache_lookup_plan =
        native_trusted_validation::plan_trusted_bgen_validation_cache_lookup(validation_mode, &cache_path)?;
    if cache_lookup_plan.should_mark_validated {
        reader.mark_trusted_no_missing_diploid_validated()?;
    }
    if !cache_lookup_plan.should_validate {
        return Ok(());
    }
    reader.validate_trusted_no_missing_diploid()?;
    if cache_lookup_plan.should_write_cache {
        native_trusted_validation::write_trusted_bgen_validation_cache_payload(
            &cache_path,
            fingerprint,
            bgen_path,
            sample_count,
            variant_count,
        )?;
    }
    Ok(())
}
