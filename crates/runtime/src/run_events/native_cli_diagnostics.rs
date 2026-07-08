use super::diagnostics::{
    RunDiagnosticEventPayload, RunDiagnosticFieldPayload, boolean_diagnostic_field, integer_diagnostic_field,
    text_diagnostic_field,
};
use super::names::{
    NATIVE_CLI_COMPLETED_LINE_DIAGNOSTIC_EVENT_NAME, NATIVE_CLI_FAILED_LINE_DIAGNOSTIC_EVENT_NAME,
    NATIVE_CLI_INTERRUPTED_LINE_DIAGNOSTIC_EVENT_NAME, NATIVE_CLI_STDERR_DIAGNOSTIC_EVENT_NAME,
    NATIVE_CLI_STDOUT_DIAGNOSTIC_EVENT_NAME, RUN_LIFECYCLE_ERROR_LEVEL, RUN_LIFECYCLE_INFO_LEVEL,
    RUN_LIFECYCLE_WARN_LEVEL,
};

#[must_use]
pub fn build_native_cli_stdout_diagnostic_payload(
    output_text: &str,
    max_payload_chars: i64,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: RUN_LIFECYCLE_INFO_LEVEL,
        event_name: NATIVE_CLI_STDOUT_DIAGNOSTIC_EVENT_NAME,
        message: "Native CLI emitted stdout output.".to_string(),
        fields: build_bounded_native_cli_stdout_fields(output_text, max_payload_chars),
    }
}

#[must_use]
pub fn build_native_cli_stderr_diagnostic_payload(
    output_text: &str,
    max_payload_chars: i64,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: RUN_LIFECYCLE_WARN_LEVEL,
        event_name: NATIVE_CLI_STDERR_DIAGNOSTIC_EVENT_NAME,
        message: "Native CLI emitted stderr output.".to_string(),
        fields: build_bounded_native_cli_stderr_fields(output_text, max_payload_chars),
    }
}

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

fn build_bounded_native_cli_stdout_fields(output_text: &str, max_payload_chars: i64) -> Vec<RunDiagnosticFieldPayload> {
    let bounded_payload = BoundedCliOutputPayload::from_output_text(output_text, max_payload_chars);
    let mut fields = vec![
        integer_diagnostic_field("stdout_character_count", bounded_payload.character_count),
        integer_diagnostic_field("stdout_byte_count", bounded_payload.byte_count),
        text_diagnostic_field("stdout_preview", &bounded_payload.preview),
        boolean_diagnostic_field("stdout_truncated", bounded_payload.truncated),
    ];
    if let Some(omitted_character_count) = bounded_payload.omitted_character_count {
        fields.push(integer_diagnostic_field("stdout_omitted_character_count", omitted_character_count));
    }
    fields
}

fn build_bounded_native_cli_stderr_fields(output_text: &str, max_payload_chars: i64) -> Vec<RunDiagnosticFieldPayload> {
    let bounded_payload = BoundedCliOutputPayload::from_output_text(output_text, max_payload_chars);
    let mut fields = vec![
        integer_diagnostic_field("stderr_character_count", bounded_payload.character_count),
        integer_diagnostic_field("stderr_byte_count", bounded_payload.byte_count),
        text_diagnostic_field("stderr_preview", &bounded_payload.preview),
        boolean_diagnostic_field("stderr_truncated", bounded_payload.truncated),
    ];
    if let Some(omitted_character_count) = bounded_payload.omitted_character_count {
        fields.push(integer_diagnostic_field("stderr_omitted_character_count", omitted_character_count));
    }
    fields
}

struct BoundedCliOutputPayload {
    character_count: i64,
    byte_count: i64,
    preview: String,
    truncated: bool,
    omitted_character_count: Option<i64>,
}

impl BoundedCliOutputPayload {
    fn from_output_text(output_text: &str, max_payload_chars: i64) -> Self {
        let character_count = i64::try_from(output_text.chars().count()).unwrap_or(i64::MAX);
        let preview_character_count = usize::try_from(max_payload_chars.max(0)).unwrap_or(usize::MAX);
        let preview = output_text.chars().take(preview_character_count).collect::<String>();
        let preview_count = i64::try_from(preview.chars().count()).unwrap_or(i64::MAX);
        let truncated = character_count > preview_count;
        let omitted_character_count = truncated.then_some(character_count.saturating_sub(preview_count));
        Self {
            character_count,
            byte_count: i64::try_from(output_text.len()).unwrap_or(i64::MAX),
            preview,
            truncated,
            omitted_character_count,
        }
    }
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
