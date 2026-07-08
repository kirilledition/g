use super::diagnostics::{RunDiagnosticEventPayload, integer_diagnostic_field, text_diagnostic_field};
use super::names::IO_OUTPUT_RESUME_COMMITTED_CHUNKS_DIAGNOSTIC_EVENT_NAME;

#[must_use]
pub fn build_io_output_resume_committed_chunks_diagnostic_payload(
    chunks_directory: &str,
    committed_chunk_count: i64,
    run_directory: &str,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "info",
        event_name: IO_OUTPUT_RESUME_COMMITTED_CHUNKS_DIAGNOSTIC_EVENT_NAME,
        message: format!("Resuming run with {committed_chunk_count} previously committed chunks."),
        fields: vec![
            text_diagnostic_field("chunks_directory", chunks_directory),
            integer_diagnostic_field("committed_chunk_count", committed_chunk_count),
            text_diagnostic_field("run_directory", run_directory),
        ],
    }
}
