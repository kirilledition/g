//! Runtime-owned telemetry session state and payload helpers.

mod cap;
mod close;
mod envelope;
mod serialization;
mod state;

pub use cap::{TelemetryCapAction, TelemetryEventCapState, TelemetryWriterCounterSnapshot};
pub use close::{
    TelemetryCloseEventPayload, TelemetryCloseMetadataPayload, TelemetryClosePlan, build_telemetry_close_event_payload,
    build_telemetry_close_metadata, plan_telemetry_close,
};
pub use envelope::{
    TelemetryEventEnvelope, build_current_telemetry_event_envelope, build_telemetry_event_envelope, generate_run_id,
};
pub use serialization::serialize_telemetry_payload_json_line;
pub use state::{
    TelemetryEventEmissionPlan, TelemetryProgressEmissionPlan, TelemetryProgressThrottleState,
    TelemetryRunSessionState, TelemetryRunSessionWriterPlan, plan_telemetry_event_emission,
    plan_telemetry_progress_emission,
};
