use serde_json::{Map as JsonMap, Number as JsonNumber, Value as JsonValue};

/// Emit a structured diagnostic through the process tracing subscriber.
///
/// # Errors
///
/// Returns an error when the diagnostic fields cannot be serialized.
pub fn emit_diagnostic_event<Fields>(
    level: &str,
    event_name: &str,
    message: &str,
    fields: &Fields,
) -> Result<(), serde_json::Error>
where
    Fields: serde::Serialize,
{
    let fields_json = serde_json::to_string(fields)?;
    match level {
        "error" => {
            tracing::error!(target: "g.native.diagnostic", g_event = event_name, g_fields = %fields_json, "{message}");
        }
        "warn" | "warning" => {
            tracing::warn!(target: "g.native.diagnostic", g_event = event_name, g_fields = %fields_json, "{message}");
        }
        "info" => {
            tracing::info!(target: "g.native.diagnostic", g_event = event_name, g_fields = %fields_json, "{message}");
        }
        "debug" => {
            tracing::debug!(target: "g.native.diagnostic", g_event = event_name, g_fields = %fields_json, "{message}");
        }
        "trace" => {
            tracing::trace!(target: "g.native.diagnostic", g_event = event_name, g_fields = %fields_json, "{message}");
        }
        _ => {
            tracing::warn!(target: "g.native.diagnostic", g_event = event_name, g_fields = %fields_json, requested_level = level, "{message}");
        }
    }
    Ok(())
}

/// Emit a typed run diagnostic payload.
///
/// # Errors
///
/// Returns an error when the diagnostic fields cannot be serialized.
pub fn emit_run_diagnostic_event(event: &RunDiagnosticEventPayload) -> Result<(), serde_json::Error> {
    let mut fields = JsonMap::new();
    for field in &event.fields {
        fields.insert(field.name.to_string(), run_diagnostic_field_value_to_json_value(&field.value));
    }
    emit_diagnostic_event(event.level, event.event_name, &event.message, &JsonValue::Object(fields))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RunDiagnosticFieldValue {
    Integer(i64),
    OptionalInteger(Option<i64>),
    Text(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunDiagnosticFieldPayload {
    pub name: &'static str,
    pub value: RunDiagnosticFieldValue,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunDiagnosticEventPayload {
    pub level: &'static str,
    pub event_name: &'static str,
    pub message: String,
    pub fields: Vec<RunDiagnosticFieldPayload>,
}

pub(super) fn integer_diagnostic_field(name: &'static str, value: i64) -> RunDiagnosticFieldPayload {
    RunDiagnosticFieldPayload { name, value: RunDiagnosticFieldValue::Integer(value) }
}

pub(super) fn optional_integer_diagnostic_field(name: &'static str, value: Option<i64>) -> RunDiagnosticFieldPayload {
    RunDiagnosticFieldPayload { name, value: RunDiagnosticFieldValue::OptionalInteger(value) }
}

pub(super) fn text_diagnostic_field(name: &'static str, value: &str) -> RunDiagnosticFieldPayload {
    RunDiagnosticFieldPayload { name, value: RunDiagnosticFieldValue::Text(value.to_string()) }
}

fn run_diagnostic_field_value_to_json_value(value: &RunDiagnosticFieldValue) -> JsonValue {
    match value {
        RunDiagnosticFieldValue::Integer(value) => JsonValue::Number(JsonNumber::from(*value)),
        RunDiagnosticFieldValue::OptionalInteger(value) => {
            value.map(JsonNumber::from).map_or(JsonValue::Null, JsonValue::Number)
        }
        RunDiagnosticFieldValue::Text(value) => JsonValue::String(value.clone()),
    }
}
