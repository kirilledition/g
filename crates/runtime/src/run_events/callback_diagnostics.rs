use super::diagnostics::{
    RunDiagnosticEventPayload, boolean_diagnostic_field, integer_diagnostic_field, text_diagnostic_field,
};
use super::names::CALLBACK_NULL_LOGISTIC_NONCONVERGENCE_WARNING_DIAGNOSTIC_EVENT_NAME;

#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn build_callback_null_logistic_nonconvergence_warning_diagnostic_payload(
    message: &str,
    chromosome: &str,
    nonconverged_count: i64,
    phenotype_count: i64,
    policy: &str,
    scalar_convergence: bool,
    total_fit_count: i64,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "warning",
        event_name: CALLBACK_NULL_LOGISTIC_NONCONVERGENCE_WARNING_DIAGNOSTIC_EVENT_NAME,
        message: message.to_string(),
        fields: vec![
            text_diagnostic_field("chromosome", chromosome),
            integer_diagnostic_field("nonconverged_count", nonconverged_count),
            integer_diagnostic_field("phenotype_count", phenotype_count),
            text_diagnostic_field("policy", policy),
            boolean_diagnostic_field("scalar_convergence", scalar_convergence),
            integer_diagnostic_field("total_fit_count", total_fit_count),
        ],
    }
}
