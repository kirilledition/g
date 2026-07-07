use super::PREFLIGHT_WARNING_DIAGNOSTIC_EVENT_NAME;
use super::diagnostics::{
    RunDiagnosticEventPayload, boolean_diagnostic_field, integer_diagnostic_field, text_diagnostic_field,
};

#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn build_preflight_warning_diagnostic_payload(
    message: &str,
    chromosome_count: i64,
    covariate_count: i64,
    preflight_scope: &str,
    sample_count: i64,
    trusted_no_missing_diploid: bool,
    warning_index: i64,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "warning",
        event_name: PREFLIGHT_WARNING_DIAGNOSTIC_EVENT_NAME,
        message: message.to_string(),
        fields: vec![
            integer_diagnostic_field("chromosome_count", chromosome_count),
            integer_diagnostic_field("covariate_count", covariate_count),
            text_diagnostic_field("preflight_scope", preflight_scope),
            integer_diagnostic_field("sample_count", sample_count),
            boolean_diagnostic_field("trusted_no_missing_diploid", trusted_no_missing_diploid),
            integer_diagnostic_field("warning_index", warning_index),
        ],
    }
}
