use serde_json::{Map as JsonMap, Number as JsonNumber, Value as JsonValue};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RunDiagnosticFieldValue {
    Boolean(bool),
    Integer(i64),
    OptionalInteger(Option<i64>),
    OptionalText(Option<String>),
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

/// Serialize run diagnostic fields for native diagnostic emission.
///
/// This keeps the diagnostic field JSON shape in `g-runtime`; PyO3 callers only
/// pass the serialized fields through to the logging boundary.
///
/// # Errors
///
/// Returns a serialization error if the diagnostic field payload cannot be
/// encoded as JSON.
pub fn serialize_run_diagnostic_fields_json(fields: &[RunDiagnosticFieldPayload]) -> Result<String, serde_json::Error> {
    let mut payload = JsonMap::new();
    for field in fields {
        payload.insert(field.name.to_string(), run_diagnostic_field_value_to_json_value(&field.value));
    }
    serde_json::to_string(&JsonValue::Object(payload))
}

pub(super) fn boolean_diagnostic_field(name: &'static str, value: bool) -> RunDiagnosticFieldPayload {
    RunDiagnosticFieldPayload { name, value: RunDiagnosticFieldValue::Boolean(value) }
}

pub(super) fn integer_diagnostic_field(name: &'static str, value: i64) -> RunDiagnosticFieldPayload {
    RunDiagnosticFieldPayload { name, value: RunDiagnosticFieldValue::Integer(value) }
}

pub(super) fn optional_integer_diagnostic_field(name: &'static str, value: Option<i64>) -> RunDiagnosticFieldPayload {
    RunDiagnosticFieldPayload { name, value: RunDiagnosticFieldValue::OptionalInteger(value) }
}

pub(super) fn optional_text_diagnostic_field(name: &'static str, value: Option<String>) -> RunDiagnosticFieldPayload {
    RunDiagnosticFieldPayload { name, value: RunDiagnosticFieldValue::OptionalText(value) }
}

pub(super) fn text_diagnostic_field(name: &'static str, value: &str) -> RunDiagnosticFieldPayload {
    RunDiagnosticFieldPayload { name, value: RunDiagnosticFieldValue::Text(value.to_string()) }
}

fn run_diagnostic_field_value_to_json_value(value: &RunDiagnosticFieldValue) -> JsonValue {
    match value {
        RunDiagnosticFieldValue::Boolean(value) => JsonValue::Bool(*value),
        RunDiagnosticFieldValue::Integer(value) => JsonValue::Number(JsonNumber::from(*value)),
        RunDiagnosticFieldValue::OptionalInteger(value) => {
            value.map(JsonNumber::from).map_or(JsonValue::Null, JsonValue::Number)
        }
        RunDiagnosticFieldValue::OptionalText(value) => {
            value.as_ref().map_or(JsonValue::Null, |value| JsonValue::String(value.clone()))
        }
        RunDiagnosticFieldValue::Text(value) => JsonValue::String(value.clone()),
    }
}
