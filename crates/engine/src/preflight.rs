//! Native preflight report and scan-shape helpers.

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreflightReportPayload {
    pub sample_count: i64,
    pub covariate_count: i64,
    pub chromosome_count: i64,
    pub warning_messages: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum PreflightError {
    #[error("BGEN input contains no variants.")]
    EmptyBgenInput,
    #[error("BGEN scan contains no variants.")]
    EmptyBgenScan,
    #[error("Could not read chromosome boundary metadata: {message}")]
    ChromosomeMetadata { message: String },
    #[error("{label} cannot be negative: {count}")]
    NegativeCount { label: &'static str, count: i64 },
}

#[must_use]
pub fn build_preflight_warnings(
    sample_count: i64,
    covariate_count: i64,
    trusted_no_missing_diploid: bool,
) -> Vec<String> {
    let mut warning_messages = Vec::new();
    let residual_degrees_of_freedom = sample_count - covariate_count;
    if residual_degrees_of_freedom < 10 {
        warning_messages.push("REGENIE step 2 is running with fewer than 10 residual degrees of freedom.".to_string());
    }
    if trusted_no_missing_diploid {
        warning_messages
            .push("Trusted no-missing diploid BGEN path is enabled after compatibility validation.".to_string());
    }
    warning_messages
}

/// Build a native preflight report payload from already-validated dimensions.
///
/// # Errors
///
/// Returns an error when a dimension is negative.
pub fn build_preflight_report_payload(
    sample_count: i64,
    covariate_count: i64,
    chromosome_count: i64,
    trusted_no_missing_diploid: bool,
) -> Result<PreflightReportPayload, PreflightError> {
    validate_non_negative_count("sample count", sample_count)?;
    validate_non_negative_count("covariate count", covariate_count)?;
    validate_non_negative_count("chromosome count", chromosome_count)?;
    Ok(PreflightReportPayload {
        sample_count,
        covariate_count,
        chromosome_count,
        warning_messages: build_preflight_warnings(sample_count, covariate_count, trusted_no_missing_diploid),
    })
}

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

fn validate_non_negative_count(label: &'static str, count: i64) -> Result<(), PreflightError> {
    if count < 0 {
        return Err(PreflightError::NegativeCount { label, count });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_warning_messages_for_low_degrees_of_freedom_and_trusted_path() {
        let warning_messages = build_preflight_warnings(3, 2, true);

        assert_eq!(
            warning_messages,
            vec![
                "REGENIE step 2 is running with fewer than 10 residual degrees of freedom.".to_string(),
                "Trusted no-missing diploid BGEN path is enabled after compatibility validation.".to_string(),
            ],
        );
    }

    #[test]
    fn builds_preflight_report_payload() {
        let payload = build_preflight_report_payload(12, 2, 3, false).unwrap();

        assert_eq!(payload.sample_count, 12);
        assert_eq!(payload.covariate_count, 2);
        assert_eq!(payload.chromosome_count, 3);
        assert!(payload.warning_messages.is_empty());
    }

    #[test]
    fn rejects_negative_report_dimensions() {
        let error = build_preflight_report_payload(-1, 0, 0, false).unwrap_err();

        assert_eq!(error, PreflightError::NegativeCount { label: "sample count", count: -1 });
    }

    #[test]
    fn resolves_scanned_variant_count() {
        assert_eq!(resolve_scanned_variant_count(12, None).unwrap(), 12);
        assert_eq!(resolve_scanned_variant_count(12, Some(5)).unwrap(), 5);
        assert_eq!(resolve_scanned_variant_count(12, Some(50)).unwrap(), 12);
    }

    #[test]
    fn rejects_empty_variant_scans() {
        assert_eq!(resolve_scanned_variant_count(0, None).unwrap_err(), PreflightError::EmptyBgenInput);
        assert_eq!(resolve_scanned_variant_count(1, Some(0)).unwrap_err(), PreflightError::EmptyBgenScan);
    }
}
