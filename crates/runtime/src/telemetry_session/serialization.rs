use serde::Serialize;

use super::TelemetryEventEnvelope;

#[derive(Serialize)]
struct TelemetryEventRecord<'record, Fields>
where
    Fields: Serialize + ?Sized,
{
    #[serde(flatten)]
    envelope: &'record TelemetryEventEnvelope,
    #[serde(flatten)]
    fields: &'record Fields,
}

/// Serialize one telemetry payload as a JSONL record.
///
/// # Errors
///
/// Returns a serialization error when the payload cannot be rendered as JSON.
pub fn serialize_telemetry_payload_json_line<Payload>(payload: &Payload) -> Result<String, serde_json::Error>
where
    Payload: Serialize + ?Sized,
{
    serde_json::to_string(payload).map(|json_text| format!("{json_text}\n"))
}

/// Serialize typed telemetry fields into one envelope JSONL record.
///
/// # Errors
///
/// Returns a serialization error when the envelope or fields cannot be
/// rendered as a flattened JSON object.
pub fn serialize_telemetry_event_json_line<Fields>(
    envelope: &TelemetryEventEnvelope,
    fields: &Fields,
) -> Result<String, serde_json::Error>
where
    Fields: Serialize + ?Sized,
{
    serialize_telemetry_payload_json_line(&TelemetryEventRecord { envelope, fields })
}
