use std::borrow::Cow;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use chrono::{DateTime, SecondsFormat};
use serde::Serialize;
use uuid::Uuid;

const TELEMETRY_SCHEMA_VERSION: i64 = 0;

#[derive(Debug, Eq, PartialEq, Serialize)]
pub(crate) struct TelemetryEventEnvelope<'event> {
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
pub(crate) fn build_current_telemetry_event_envelope<'event>(
    run_id: &'event str,
    event: &'event str,
    level: &'event str,
    thread_name: &'event str,
) -> TelemetryEventEnvelope<'event> {
    TelemetryEventEnvelope {
        schema_version: TELEMETRY_SCHEMA_VERSION,
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
pub(crate) fn generate_run_id() -> Arc<str> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_uses_prerelease_contract_and_stable_runtime_fields() {
        let envelope = build_current_telemetry_event_envelope("run-id", "chunk_finished", "info", "worker-2");
        assert_eq!(envelope.schema_version, TELEMETRY_SCHEMA_VERSION);
        assert_eq!(envelope.schema_version, 0);
        assert_eq!(envelope.run_id, "run-id");
        assert_eq!(envelope.level, "INFO");
        assert_eq!(envelope.source, "python");
        assert_eq!(envelope.target, "g.engine.telemetry");
        assert_eq!(envelope.event, "chunk_finished");
        assert_eq!(envelope.process_identifier, std::process::id());
        assert_eq!(envelope.thread_name, "worker-2");
        assert!(envelope.timestamp.ends_with('Z'));
        assert!(chrono::DateTime::parse_from_rfc3339(&envelope.timestamp).is_ok());
    }

    #[test]
    fn telemetry_levels_are_normalized_without_allocating_known_values() {
        for (input, expected) in [
            ("error", "ERROR"),
            ("ERROR", "ERROR"),
            ("warn", "WARN"),
            ("warning", "WARNING"),
            ("info", "INFO"),
            ("debug", "DEBUG"),
            ("trace", "TRACE"),
            ("NOTICE", "NOTICE"),
        ] {
            assert_eq!(uppercase_level(input), expected);
            assert!(matches!(uppercase_level(input), Cow::Borrowed(_)));
        }
        assert_eq!(uppercase_level("Notice"), "NOTICE");
        assert!(matches!(uppercase_level("Notice"), Cow::Owned(_)));
    }

    #[test]
    fn generated_run_identifiers_are_unique_lowercase_uuid_hex() {
        let first = generate_run_id();
        let second = generate_run_id();
        assert_ne!(first, second);
        assert_eq!(first.len(), 32);
        assert!(first.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)));
    }
}
