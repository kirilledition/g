use super::diagnostics::{RunDiagnosticEventPayload, text_diagnostic_field};
use super::names::{
    NATIVE_CLI_COMPLETED_LINE_DIAGNOSTIC_EVENT_NAME, NATIVE_CLI_FAILED_LINE_DIAGNOSTIC_EVENT_NAME,
    NATIVE_CLI_INTERRUPTED_LINE_DIAGNOSTIC_EVENT_NAME, RUN_LIFECYCLE_ERROR_LEVEL, RUN_LIFECYCLE_INFO_LEVEL,
    RUN_LIFECYCLE_WARN_LEVEL,
};

#[must_use]
pub fn build_native_cli_interrupted_line_diagnostic_payload(line: &str) -> RunDiagnosticEventPayload {
    build_native_cli_line_diagnostic_payload(
        RUN_LIFECYCLE_WARN_LEVEL,
        NATIVE_CLI_INTERRUPTED_LINE_DIAGNOSTIC_EVENT_NAME,
        "Native CLI interruption detail.",
        line,
    )
}

#[must_use]
pub fn build_native_cli_failed_line_diagnostic_payload(line: &str) -> RunDiagnosticEventPayload {
    build_native_cli_line_diagnostic_payload(
        RUN_LIFECYCLE_ERROR_LEVEL,
        NATIVE_CLI_FAILED_LINE_DIAGNOSTIC_EVENT_NAME,
        "Native CLI failure detail.",
        line,
    )
}

#[must_use]
pub fn build_native_cli_completed_line_diagnostic_payload(line: &str) -> RunDiagnosticEventPayload {
    build_native_cli_line_diagnostic_payload(
        RUN_LIFECYCLE_INFO_LEVEL,
        NATIVE_CLI_COMPLETED_LINE_DIAGNOSTIC_EVENT_NAME,
        "Native CLI completion detail.",
        line,
    )
}

fn build_native_cli_line_diagnostic_payload(
    level: &'static str,
    event_name: &'static str,
    message: &str,
    line: &str,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level,
        event_name,
        message: message.to_string(),
        fields: vec![text_diagnostic_field("line", line)],
    }
}
