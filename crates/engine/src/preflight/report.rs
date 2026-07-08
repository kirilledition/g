use super::common::validate_non_negative_count;
use super::error::PreflightError;
use super::payloads::PreflightReportPayload;

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
