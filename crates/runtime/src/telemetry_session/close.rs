use super::TelemetryWriterCounterSnapshot;
use serde::Serialize;

const TELEMETRY_SESSION_CLOSED_EVENT_NAME: &str = "telemetry_session_closed";
const TELEMETRY_SESSION_CLOSED_EVENT_LEVEL: &str = "debug";

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TelemetryCloseEventPayload {
    #[serde(skip)]
    pub event_name: String,
    #[serde(skip)]
    pub level: String,
    pub writer_counters: TelemetryWriterCounterSnapshot,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TelemetryClosePlan {
    pub should_close: bool,
    pub use_native_close_with_event: bool,
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
    }
}
