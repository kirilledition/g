use std::collections::BTreeSet;

use serde_json::Value;

use crate::error::OutputError;

pub(super) fn validate_manifest_compatibility_values(
    manifest: &Value,
    current_header: &Value,
) -> Result<(), OutputError> {
    let manifest_object = manifest
        .as_object()
        .ok_or_else(|| OutputError::InvalidInput("Run manifest must contain a JSON object.".to_string()))?;
    let current_header_object = current_header.as_object().ok_or_else(|| {
        OutputError::InvalidInput("Current run manifest header must contain a JSON object.".to_string())
    })?;
    for (field_name, current_value) in current_header_object {
        let Some(manifest_value) = manifest_object.get(field_name) else {
            return Err(OutputError::InvalidInput(format!("Run manifest field '{field_name}' is missing.")));
        };
        if let Some(mismatch_path) = find_first_manifest_mismatch_path(manifest_value, current_value, field_name) {
            return Err(OutputError::InvalidInput(format!(
                "Run manifest field '{mismatch_path}' is incompatible with the requested run."
            )));
        }
    }
    Ok(())
}

fn find_first_manifest_mismatch_path(
    manifest_value: &Value,
    current_value: &Value,
    field_path: &str,
) -> Option<String> {
    match (manifest_value, current_value) {
        (Value::Object(manifest_object), Value::Object(current_object)) => {
            let field_names = manifest_object.keys().chain(current_object.keys()).collect::<BTreeSet<_>>();
            for field_name in field_names {
                let nested_path = format!("{field_path}.{field_name}");
                match (manifest_object.get(field_name), current_object.get(field_name)) {
                    (Some(nested_manifest_value), Some(nested_current_value)) => {
                        if let Some(mismatch_path) =
                            find_first_manifest_mismatch_path(nested_manifest_value, nested_current_value, &nested_path)
                        {
                            return Some(mismatch_path);
                        }
                    }
                    _ => return Some(nested_path),
                }
            }
            None
        }
        (Value::Array(manifest_array), Value::Array(current_array)) => {
            for (index, (manifest_item, current_item)) in manifest_array.iter().zip(current_array).enumerate() {
                let nested_path = format!("{field_path}[{index}]");
                if let Some(mismatch_path) =
                    find_first_manifest_mismatch_path(manifest_item, current_item, &nested_path)
                {
                    return Some(mismatch_path);
                }
            }
            if manifest_array.len() != current_array.len() {
                return Some(field_path.to_string());
            }
            None
        }
        _ if manifest_scalar_values_match(manifest_value, current_value) => None,
        _ => Some(field_path.to_string()),
    }
}

fn manifest_scalar_values_match(manifest_value: &Value, current_value: &Value) -> bool {
    match (manifest_value, current_value) {
        (Value::Number(manifest_number), Value::Number(current_number)) => {
            if let (Some(manifest_integer), Some(current_integer)) = (manifest_number.as_i64(), current_number.as_i64())
            {
                return manifest_integer == current_integer;
            }
            if let (Some(manifest_integer), Some(current_integer)) = (manifest_number.as_u64(), current_number.as_u64())
            {
                return manifest_integer == current_integer;
            }
            manifest_number.as_f64() == current_number.as_f64()
        }
        _ => manifest_value == current_value,
    }
}
