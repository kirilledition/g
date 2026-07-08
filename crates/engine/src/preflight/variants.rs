use super::error::PreflightError;

/// Resolve the number of variants that a preflight scan must cover.
///
/// # Errors
///
/// Returns an error when the BGEN input or requested scan contains no variants.
pub fn resolve_scanned_variant_count(
    variant_count: usize,
    variant_limit: Option<usize>,
) -> Result<usize, PreflightError> {
    if variant_count == 0 {
        return Err(PreflightError::EmptyBgenInput);
    }
    let scanned_variant_count = variant_limit.map_or(variant_count, |limit| limit.min(variant_count));
    if scanned_variant_count == 0 {
        return Err(PreflightError::EmptyBgenScan);
    }
    Ok(scanned_variant_count)
}

/// Resolve a Python-facing preflight variant count using native scan policy.
///
/// # Errors
///
/// Returns an error when the BGEN input or requested scan contains no variants.
pub fn resolve_preflight_variant_count(variant_count: i64, variant_limit: Option<i64>) -> Result<i64, PreflightError> {
    if variant_count <= 0 {
        return Err(PreflightError::EmptyBgenInput);
    }
    if matches!(variant_limit, Some(limit) if limit <= 0) {
        return Err(PreflightError::EmptyBgenScan);
    }
    Ok(variant_limit.map_or(variant_count, |limit| limit.min(variant_count)))
}
