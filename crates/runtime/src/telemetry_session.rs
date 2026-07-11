//! Runtime-owned telemetry session state and payload helpers.

mod counter;
mod envelope;
mod run;
mod serialization;

pub(crate) use counter::{TelemetryEventCounterState, TelemetryWriterCounterSnapshot};
pub(crate) use envelope::generate_run_id;
pub(crate) use envelope::{TelemetryEventEnvelope, build_current_telemetry_event_envelope};
pub use run::{TelemetryRunError, TelemetryRunSession};
pub(crate) use serialization::serialize_telemetry_event_json_line;
