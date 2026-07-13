use serde::Serialize;

use super::TelemetryEventEnvelope;

#[derive(Serialize)]
struct TelemetryEventRecord<'record, 'envelope, Fields>
where
    Fields: Serialize + ?Sized,
{
    #[serde(flatten)]
    envelope: &'record TelemetryEventEnvelope<'envelope>,
    #[serde(flatten)]
    fields: &'record Fields,
}

/// Serialize typed telemetry fields into one envelope JSONL record.
///
/// # Errors
///
/// Returns a serialization error when the envelope or fields cannot be
/// rendered as a flattened JSON object.
pub(crate) fn serialize_telemetry_event_json_line<Fields>(
    envelope: &TelemetryEventEnvelope<'_>,
    fields: &Fields,
) -> Result<String, serde_json::Error>
where
    Fields: Serialize + ?Sized,
{
    let mut json_text = serde_json::to_string(&TelemetryEventRecord { envelope, fields })?;
    json_text.push('\n');
    Ok(json_text)
}
