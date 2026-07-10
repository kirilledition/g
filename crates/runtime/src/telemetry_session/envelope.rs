use std::time::{SystemTime, UNIX_EPOCH};

use crate::telemetry_policy;
use serde::Serialize;
use uuid::Uuid;

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct TelemetryEventEnvelope {
    pub schema_version: i64,
    pub run_id: String,
    #[serde(rename = "ts")]
    pub timestamp: String,
    pub level: String,
    pub source: &'static str,
    pub target: &'static str,
    pub event: String,
    #[serde(rename = "pid")]
    pub process_identifier: u32,
    pub thread_name: String,
}

#[must_use]
pub fn build_telemetry_event_envelope(
    run_id: &str,
    event: &str,
    level: &str,
    timestamp: &str,
    process_identifier: u32,
    thread_name: &str,
) -> TelemetryEventEnvelope {
    TelemetryEventEnvelope {
        schema_version: 1,
        run_id: run_id.to_string(),
        timestamp: timestamp.to_string(),
        level: level.to_uppercase(),
        source: "python",
        target: "g.engine.telemetry",
        event: event.to_string(),
        process_identifier,
        thread_name: thread_name.to_string(),
    }
}

#[must_use]
pub fn build_current_telemetry_event_envelope(
    run_id: &str,
    event: &str,
    level: &str,
    thread_name: &str,
) -> TelemetryEventEnvelope {
    build_telemetry_event_envelope(
        run_id,
        event,
        level,
        &current_telemetry_timestamp(),
        std::process::id(),
        thread_name,
    )
}

#[must_use]
pub fn generate_run_id() -> String {
    format!("{:032x}", Uuid::new_v4().as_u128())
}

fn current_telemetry_timestamp() -> String {
    let elapsed_seconds = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0.0, |duration| duration.as_secs_f64());
    telemetry_policy::format_timestamp(elapsed_seconds)
}
