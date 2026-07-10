use super::diagnostics::{
    RunDiagnosticEventPayload, boolean_diagnostic_field, integer_diagnostic_field, optional_integer_diagnostic_field,
    text_diagnostic_field,
};
use super::names::{
    NATIVE_DISPATCH_BGEN_ENGINE_CONSTRUCTING_DIAGNOSTIC_EVENT_NAME,
    NATIVE_DISPATCH_BGEN_ENGINE_CONSTRUCTING_DIAGNOSTIC_MESSAGE,
    NATIVE_DISPATCH_DELIVERY_FINISHED_DIAGNOSTIC_EVENT_NAME, NATIVE_DISPATCH_PIPELINE_FINISHED_DIAGNOSTIC_EVENT_NAME,
    NATIVE_DISPATCH_TRUSTED_BGEN_VALIDATION_STARTED_DIAGNOSTIC_EVENT_NAME,
    NATIVE_DISPATCH_TRUSTED_BGEN_VALIDATION_STARTED_DIAGNOSTIC_MESSAGE,
    NATIVE_DISPATCH_WRITER_SESSIONS_FINISH_STARTED_DIAGNOSTIC_EVENT_NAME,
    NATIVE_DISPATCH_WRITER_SESSIONS_FINISH_STARTED_DIAGNOSTIC_MESSAGE,
    NATIVE_DISPATCH_WRITER_SESSIONS_INTERRUPTED_FLUSH_STARTED_DIAGNOSTIC_EVENT_NAME,
};

#[must_use]
pub fn build_native_dispatch_bgen_engine_constructing_diagnostic_payload(
    chunk_size: i64,
    source_path: &str,
    trusted_no_missing_diploid: bool,
    variant_limit: Option<i64>,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "debug",
        event_name: NATIVE_DISPATCH_BGEN_ENGINE_CONSTRUCTING_DIAGNOSTIC_EVENT_NAME,
        message: NATIVE_DISPATCH_BGEN_ENGINE_CONSTRUCTING_DIAGNOSTIC_MESSAGE.to_string(),
        fields: vec![
            integer_diagnostic_field("chunk_size", chunk_size),
            text_diagnostic_field("source_path", source_path),
            boolean_diagnostic_field("trusted_no_missing_diploid", trusted_no_missing_diploid),
            optional_integer_diagnostic_field("variant_limit", variant_limit),
        ],
    }
}

#[must_use]
pub fn build_native_dispatch_trusted_bgen_validation_started_diagnostic_payload(
    source_path: &str,
    trusted_bgen_validation_mode: &str,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "debug",
        event_name: NATIVE_DISPATCH_TRUSTED_BGEN_VALIDATION_STARTED_DIAGNOSTIC_EVENT_NAME,
        message: NATIVE_DISPATCH_TRUSTED_BGEN_VALIDATION_STARTED_DIAGNOSTIC_MESSAGE.to_string(),
        fields: vec![
            text_diagnostic_field("source_path", source_path),
            text_diagnostic_field("trusted_bgen_validation_mode", trusted_bgen_validation_mode),
        ],
    }
}

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

#[must_use]
pub fn build_native_dispatch_pipeline_finished_diagnostic_payload(
    final_parquet_path_count: i64,
    pipeline_label: &str,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "info",
        event_name: NATIVE_DISPATCH_PIPELINE_FINISHED_DIAGNOSTIC_EVENT_NAME,
        message: format!("{pipeline_label} pipeline finished."),
        fields: vec![
            integer_diagnostic_field("final_parquet_path_count", final_parquet_path_count),
            text_diagnostic_field("pipeline_label", pipeline_label),
        ],
    }
}

#[must_use]
pub fn build_native_dispatch_writer_sessions_finish_started_diagnostic_payload(
    requested_thread_count: i64,
    writer_session_count: i64,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "debug",
        event_name: NATIVE_DISPATCH_WRITER_SESSIONS_FINISH_STARTED_DIAGNOSTIC_EVENT_NAME,
        message: NATIVE_DISPATCH_WRITER_SESSIONS_FINISH_STARTED_DIAGNOSTIC_MESSAGE.to_string(),
        fields: vec![
            integer_diagnostic_field("requested_thread_count", requested_thread_count),
            integer_diagnostic_field("writer_session_count", writer_session_count),
        ],
    }
}

#[must_use]
pub fn build_native_dispatch_writer_sessions_interrupted_flush_started_diagnostic_payload(
    requested_thread_count: i64,
    signal_exit_code: i64,
    signal_name: &str,
    signal_number: i64,
    writer_session_count: i64,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "info",
        event_name: NATIVE_DISPATCH_WRITER_SESSIONS_INTERRUPTED_FLUSH_STARTED_DIAGNOSTIC_EVENT_NAME,
        message: format!("Flushing interrupted output writer(s) after {signal_name}."),
        fields: vec![
            integer_diagnostic_field("requested_thread_count", requested_thread_count),
            integer_diagnostic_field("signal_exit_code", signal_exit_code),
            text_diagnostic_field("signal_name", signal_name),
            integer_diagnostic_field("signal_number", signal_number),
            integer_diagnostic_field("writer_session_count", writer_session_count),
        ],
    }
}
