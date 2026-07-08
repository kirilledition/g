use super::diagnostics::{RunDiagnosticEventPayload, integer_diagnostic_field, optional_integer_diagnostic_field};
use super::names::{
    NATIVE_RUNTIME_KNOBS_CONFIGURED_DIAGNOSTIC_EVENT_NAME, NATIVE_RUNTIME_KNOBS_CONFIGURED_DIAGNOSTIC_MESSAGE,
};

#[must_use]
pub fn build_native_runtime_knobs_configured_diagnostic_payload(
    bgen_decode_tile_variant_count: i64,
    threads: Option<i64>,
) -> RunDiagnosticEventPayload {
    RunDiagnosticEventPayload {
        level: "debug",
        event_name: NATIVE_RUNTIME_KNOBS_CONFIGURED_DIAGNOSTIC_EVENT_NAME,
        message: NATIVE_RUNTIME_KNOBS_CONFIGURED_DIAGNOSTIC_MESSAGE.to_string(),
        fields: vec![
            integer_diagnostic_field("bgen_decode_tile_variant_count", bgen_decode_tile_variant_count),
            optional_integer_diagnostic_field("threads", threads),
        ],
    }
}
