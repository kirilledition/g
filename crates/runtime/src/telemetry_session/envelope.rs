use std::borrow::Cow;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use chrono::{DateTime, SecondsFormat};
use serde::Serialize;
use uuid::Uuid;

#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct TelemetryEventEnvelope<'event> {
    pub schema_version: i64,
    pub run_id: &'event str,
    #[serde(rename = "ts")]
    pub timestamp: String,
    pub level: Cow<'event, str>,
    pub source: &'static str,
    pub target: &'static str,
    pub event: &'event str,
    #[serde(rename = "pid")]
    pub process_identifier: u32,
    pub thread_name: &'event str,
}

#[must_use]
pub fn build_current_telemetry_event_envelope<'event>(
    run_id: &'event str,
    event: &'event str,
    level: &'event str,
    thread_name: &'event str,
) -> TelemetryEventEnvelope<'event> {
    TelemetryEventEnvelope {
        schema_version: 1,
        run_id,
        timestamp: current_telemetry_timestamp(),
        level: uppercase_level(level),
        source: "python",
        target: "g.engine.telemetry",
        event,
        process_identifier: std::process::id(),
        thread_name,
    }
}

#[must_use]
pub fn generate_run_id() -> Arc<str> {
    let run_id = Uuid::new_v4();
    let mut buffer = Uuid::encode_buffer();
    Arc::from(&*run_id.simple().encode_lower(&mut buffer))
}

fn uppercase_level(level: &str) -> Cow<'_, str> {
    match level {
        "error" | "ERROR" => Cow::Borrowed("ERROR"),
        "warn" | "WARN" => Cow::Borrowed("WARN"),
        "warning" | "WARNING" => Cow::Borrowed("WARNING"),
        "info" | "INFO" => Cow::Borrowed("INFO"),
        "debug" | "DEBUG" => Cow::Borrowed("DEBUG"),
        "trace" | "TRACE" => Cow::Borrowed("TRACE"),
        _ if level.chars().all(|character| !character.is_lowercase()) => Cow::Borrowed(level),
        _ => Cow::Owned(level.to_uppercase()),
    }
}

fn current_telemetry_timestamp() -> String {
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).ok().and_then(|duration| {
        i64::try_from(duration.as_secs())
            .ok()
            .and_then(|whole_seconds| DateTime::from_timestamp(whole_seconds, duration.subsec_nanos()))
    });
    timestamp.map_or_else(
        || "1970-01-01T00:00:00.000000Z".to_string(),
        |timestamp| timestamp.to_rfc3339_opts(SecondsFormat::Micros, true),
    )
}
