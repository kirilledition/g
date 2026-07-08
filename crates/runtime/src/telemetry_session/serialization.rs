use serde_json::Value as JsonValue;

/// Serialize one telemetry payload as stable JSON text.
///
/// # Errors
///
/// Returns a serialization error when the payload cannot be rendered as JSON.
pub fn serialize_telemetry_payload_json_text(payload: &JsonValue) -> Result<String, serde_json::Error> {
    serde_json::to_string(payload)
}

/// Serialize one telemetry payload as a JSONL record.
///
/// # Errors
///
/// Returns a serialization error when the payload cannot be rendered as JSON.
pub fn serialize_telemetry_payload_json_line(payload: &JsonValue) -> Result<String, serde_json::Error> {
    serialize_telemetry_payload_json_text(payload).map(|json_text| format!("{json_text}\n"))
}
