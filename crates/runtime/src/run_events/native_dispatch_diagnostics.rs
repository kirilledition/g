use super::diagnostics::{RunDiagnosticEventPayload, integer_diagnostic_field, text_diagnostic_field};
use super::names::NATIVE_DISPATCH_DELIVERY_FINISHED_DIAGNOSTIC_EVENT_NAME;

#[must_use]
pub fn build_native_dispatch_delivery_finished_diagnostic_payload(
    pipeline_label: &str,
    processed_chunk_count: i64,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "debug",
        event_name: NATIVE_DISPATCH_DELIVERY_FINISHED_DIAGNOSTIC_EVENT_NAME,
        message: format!("{pipeline_label} delivery finished: processed_chunk_count={processed_chunk_count}."),
        fields: vec![
            text_diagnostic_field("pipeline_label", pipeline_label),
            integer_diagnostic_field("processed_chunk_count", processed_chunk_count),
        ],
    }
}
