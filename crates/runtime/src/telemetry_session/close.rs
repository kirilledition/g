use super::TelemetryWriterCounterSnapshot;

const TELEMETRY_SESSION_CLOSED_EVENT_NAME: &str = "telemetry_session_closed";
const TELEMETRY_SESSION_CLOSED_EVENT_LEVEL: &str = "debug";

#[derive(Clone, Debug, PartialEq)]
pub struct TelemetryCloseMetadataPayload {
    pub writer_counters: TelemetryWriterCounterSnapshot,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TelemetryCloseEventPayload {
    pub event_name: String,
    pub level: String,
    pub writer_counters: TelemetryWriterCounterSnapshot,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TelemetryClosePlan {
    pub should_close: bool,
    pub use_native_close_with_event: bool,
    pub should_emit_legacy_close_event: bool,
    pub legacy_close_event_name: String,
    pub legacy_close_event_level: String,
}

#[must_use]
pub fn build_telemetry_close_metadata(
    writer_counters: TelemetryWriterCounterSnapshot,
) -> TelemetryCloseMetadataPayload {
    TelemetryCloseMetadataPayload { writer_counters }
}

#[must_use]
pub fn build_telemetry_close_event_payload(
    writer_counters: TelemetryWriterCounterSnapshot,
) -> TelemetryCloseEventPayload {
    TelemetryCloseEventPayload {
        event_name: TELEMETRY_SESSION_CLOSED_EVENT_NAME.to_string(),
        level: TELEMETRY_SESSION_CLOSED_EVENT_LEVEL.to_string(),
        writer_counters,
    }
}

#[must_use]
pub fn plan_telemetry_close(has_telemetry_session: bool, is_native_telemetry_session: bool) -> TelemetryClosePlan {
    TelemetryClosePlan {
        should_close: has_telemetry_session && is_native_telemetry_session,
        use_native_close_with_event: has_telemetry_session && is_native_telemetry_session,
        should_emit_legacy_close_event: false,
        legacy_close_event_name: TELEMETRY_SESSION_CLOSED_EVENT_NAME.to_string(),
        legacy_close_event_level: TELEMETRY_SESSION_CLOSED_EVENT_LEVEL.to_string(),
    }
}
