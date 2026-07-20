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

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use serde::ser::Error as _;
    use serde::{Serialize, Serializer};

    use super::*;

    #[derive(Serialize)]
    struct Fields<'fields> {
        chromosome: &'fields str,
        variant_count: u64,
    }

    struct SerializationFailure;

    impl Serialize for SerializationFailure {
        fn serialize<SerializerType>(
            &self,
            _serializer: SerializerType,
        ) -> Result<SerializerType::Ok, SerializerType::Error>
        where
            SerializerType: Serializer,
        {
            Err(SerializerType::Error::custom("intentional serialization failure"))
        }
    }

    fn envelope() -> TelemetryEventEnvelope<'static> {
        TelemetryEventEnvelope {
            schema_version: 0,
            run_id: "run-123",
            timestamp: "2026-07-20T00:00:00.000000Z".to_owned(),
            level: Cow::Borrowed("INFO"),
            source: "python",
            target: "g.engine.telemetry",
            event: "chunk_finished",
            process_identifier: 42,
            thread_name: "worker",
        }
    }

    #[test]
    fn serialization_flattens_fields_into_one_newline_terminated_record() {
        let line =
            serialize_telemetry_event_json_line(&envelope(), &Fields { chromosome: "22", variant_count: 16_384 })
                .expect("telemetry record should serialize");
        assert!(line.ends_with('\n'));
        assert_eq!(line.bytes().filter(|byte| *byte == b'\n').count(), 1);

        let record: serde_json::Value = serde_json::from_str(&line).expect("telemetry record should parse");
        assert_eq!(record["schema_version"], 0);
        assert_eq!(record["run_id"], "run-123");
        assert_eq!(record["event"], "chunk_finished");
        assert_eq!(record["chromosome"], "22");
        assert_eq!(record["variant_count"], 16_384);
    }

    #[test]
    fn serialization_propagates_field_errors() {
        let error = serialize_telemetry_event_json_line(&envelope(), &SerializationFailure)
            .expect_err("failing fields should return serialization error");
        assert!(error.to_string().contains("intentional serialization failure"));
    }
}
