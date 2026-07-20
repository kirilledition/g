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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::validate_manifest_compatibility_values;

    #[test]
    fn compatibility_accepts_equal_nested_values_and_manifest_extensions() {
        let manifest = json!({
            "schema_version": 0,
            "execution_plan": {"names": ["alpha", "beta"], "count": 2},
            "future_extension": true,
        });
        let current = json!({
            "schema_version": 0,
            "execution_plan": {"names": ["alpha", "beta"], "count": 2},
        });

        validate_manifest_compatibility_values(&manifest, &current).expect("compatible manifest is accepted");
    }

    #[test]
    fn compatibility_reports_deterministic_nested_object_and_array_paths() {
        let current = json!({
            "execution_plan": {
                "association_backend": {"genotype_format": "packed8"},
                "covariates": ["age", "sex"],
            },
        });
        let nested_mismatch = json!({
            "execution_plan": {
                "association_backend": {"genotype_format": "dosage"},
                "covariates": ["age", "batch"],
            },
        });
        let error = validate_manifest_compatibility_values(&nested_mismatch, &current)
            .expect_err("nested mismatch is rejected");
        assert!(error.to_string().contains("execution_plan.association_backend.genotype_format"));

        let array_mismatch = json!({
            "execution_plan": {
                "association_backend": {"genotype_format": "packed8"},
                "covariates": ["age", "batch"],
            },
        });
        let error =
            validate_manifest_compatibility_values(&array_mismatch, &current).expect_err("array mismatch is rejected");
        assert!(error.to_string().contains("execution_plan.covariates[1]"));
    }

    #[test]
    fn compatibility_rejects_missing_extra_and_different_length_nested_fields() {
        let current = json!({"execution_plan": {"names": ["alpha", "beta"], "required": true}});
        let cases = [
            (json!({}), "execution_plan"),
            (json!({"execution_plan": {"names": ["alpha", "beta"]}}), "execution_plan.required"),
            (
                json!({"execution_plan": {"names": ["alpha", "beta"], "required": true, "stale": 1}}),
                "execution_plan.stale",
            ),
            (json!({"execution_plan": {"names": ["alpha"], "required": true}}), "execution_plan.names"),
        ];

        for (manifest, expected_path) in cases {
            let error = validate_manifest_compatibility_values(&manifest, &current)
                .expect_err("incompatible manifest is rejected");
            assert!(error.to_string().contains(expected_path), "unexpected error: {error}");
        }
    }

    #[test]
    fn compatibility_rejects_non_object_roots_and_scalar_type_changes() {
        let non_object_manifest = validate_manifest_compatibility_values(&json!([]), &json!({}))
            .expect_err("manifest root must be an object");
        assert!(non_object_manifest.to_string().contains("Run manifest must contain a JSON object"));
        let non_object_header =
            validate_manifest_compatibility_values(&json!({}), &json!([])).expect_err("header root must be an object");
        assert!(non_object_header.to_string().contains("header must contain a JSON object"));

        let error = validate_manifest_compatibility_values(&json!({"count": "2"}), &json!({"count": 2}))
            .expect_err("scalar type change is rejected");
        assert!(error.to_string().contains("field 'count'"));
    }
}
