use std::path::{Path, PathBuf};

use super::error::TrustedBgenValidationCacheDirectoryError;

const TRUSTED_BGEN_VALIDATION_CACHE_APPLICATION_DIRECTORY: &str = "g";
const TRUSTED_BGEN_VALIDATION_CACHE_DIRECTORY_NAME: &str = "bgen_validation";

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

fn non_empty_environment_path(variable_name: &str) -> Option<PathBuf> {
    std::env::var_os(variable_name).filter(|environment_value| !environment_value.is_empty()).map(PathBuf::from)
}
