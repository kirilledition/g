//! Native output-manifest preparation helpers.

use std::path::Path;

use g_input::resolve_prediction_loco_paths;
use g_output::{
    CurrentRunManifestHeaderInput, ManifestFileFingerprint, ManifestFileFingerprintCache, OutputError,
    build_current_run_manifest_header_json_with_cache,
};
use serde_json::{Map, Value, json};

struct PredictionLocoFileFingerprint {
    phenotype: String,
    fingerprint: ManifestFileFingerprint,
}

/// Build a current run manifest header from a flexible JSON input value.
///
/// # Errors
///
/// Returns an error when the input value is not an object, required fields are
/// missing or malformed, prediction LOCO fingerprints cannot be resolved, or
/// the normalized header cannot be serialized.
pub fn build_current_run_manifest_header_json_from_value_with_cache(
    mut current_header_input_value: Value,
    fingerprint_cache: &mut ManifestFileFingerprintCache,
) -> Result<String, OutputError> {
    normalize_current_header_input_json_fields(&mut current_header_input_value, fingerprint_cache)?;
    let current_header_input = parse_current_header_input_value(current_header_input_value)?;
    build_current_run_manifest_header_json_with_cache(current_header_input, fingerprint_cache)
}

fn parse_current_header_input_value(
    current_header_input_value: Value,
) -> Result<CurrentRunManifestHeaderInput, OutputError> {
    serde_json::from_value(current_header_input_value)
        .map_err(|error| OutputError::Runtime(format!("Invalid current_header_input: {error}")))
}

fn normalize_current_header_input_json_fields(
    current_header_input_value: &mut Value,
    fingerprint_cache: &mut ManifestFileFingerprintCache,
) -> Result<(), OutputError> {
    let input_object = current_header_input_value
        .as_object_mut()
        .ok_or_else(|| OutputError::Runtime("Current header input must contain a JSON object.".to_string()))?;
    if !input_object.contains_key("prediction_loco_files_json") {
        let prediction_loco_files = if let Some(value) = input_object.remove("prediction_loco_files") {
            value
        } else {
            let prediction_list_path = input_object
                .get("prediction_list_path")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    OutputError::Runtime("Current header input must include prediction_list_path.".to_string())
                })?
                .to_string();
            let phenotype_names_value = input_object.remove("prediction_input_phenotype_names").ok_or_else(|| {
                OutputError::Runtime("Current header input must include prediction_input_phenotype_names.".to_string())
            })?;
            let phenotype_names = json_string_array_from_value(&phenotype_names_value)?;
            prediction_loco_file_fingerprints_to_json_value(build_prediction_loco_file_fingerprints_with_cache(
                &prediction_list_path,
                &phenotype_names,
                fingerprint_cache,
            )?)?
        };
        input_object.insert(
            "prediction_loco_files_json".to_string(),
            Value::String(
                serde_json::to_string(&prediction_loco_files)
                    .map_err(|error| OutputError::Runtime(format!("Invalid prediction_loco_files value: {error}")))?,
            ),
        );
    }
    if !input_object.contains_key("binary_kernel_config_json") {
        let binary_kernel_config_json = match input_object.remove("binary_kernel_config") {
            None | Some(Value::Null) => Value::Null,
            Some(binary_kernel_config) => Value::String(
                serde_json::to_string(&binary_kernel_config)
                    .map_err(|error| OutputError::Runtime(format!("Invalid binary_kernel_config value: {error}")))?,
            ),
        };
        input_object.insert("binary_kernel_config_json".to_string(), binary_kernel_config_json);
    }
    normalize_current_header_binary_correction_plan(input_object)?;
    Ok(())
}

fn normalize_current_header_binary_correction_plan(input_object: &mut Map<String, Value>) -> Result<(), OutputError> {
    let has_method = input_object.contains_key("binary_correction_plan_method");
    let has_p_threshold = input_object.contains_key("binary_correction_plan_p_threshold");
    let has_firth_se = input_object.contains_key("binary_correction_plan_firth_se");
    let legacy_field_count =
        [has_method, has_p_threshold, has_firth_se].into_iter().filter(|has_field| *has_field).count();
    if legacy_field_count == 3 {
        return Ok(());
    }
    if legacy_field_count != 0 {
        return Err(OutputError::Runtime(
            "Current header input must provide all binary_correction_plan legacy fields or none.".to_string(),
        ));
    }
    let correction_plan = match input_object.remove("binary_correction_plan") {
        Some(Value::Null) | None => legacy_linear_current_header_binary_correction_plan(input_object)?,
        Some(correction_plan_value) => serde_json::from_value::<g_plan::CorrectionPlan>(correction_plan_value)
            .map_err(|error| OutputError::Runtime(format!("Invalid binary_correction_plan: {error}")))?,
    };
    insert_current_header_binary_correction_plan_fields(input_object, &correction_plan);
    Ok(())
}

fn legacy_linear_current_header_binary_correction_plan(
    input_object: &Map<String, Value>,
) -> Result<g_plan::CorrectionPlan, OutputError> {
    let association_mode = input_object
        .get("association_mode")
        .and_then(Value::as_str)
        .ok_or_else(|| OutputError::Runtime("Current header input must include association_mode.".to_string()))?;
    if association_mode != g_plan::AssociationMode::Regenie2Linear.as_str() {
        return Err(OutputError::Runtime(
            "Binary association manifest input must include binary_correction_plan.".to_string(),
        ));
    }
    Ok(g_plan::CorrectionPlan { method: g_plan::BinaryFallbackMethod::ScoreOnly, p_threshold: 0.05, firth_se: false })
}

fn insert_current_header_binary_correction_plan_fields(
    input_object: &mut Map<String, Value>,
    correction_plan: &g_plan::CorrectionPlan,
) {
    input_object.insert(
        "binary_correction_plan_method".to_string(),
        Value::String(correction_plan.method.as_str().to_string()),
    );
    input_object.insert("binary_correction_plan_p_threshold".to_string(), Value::from(correction_plan.p_threshold));
    input_object.insert("binary_correction_plan_firth_se".to_string(), Value::Bool(correction_plan.firth_se));
}

fn json_string_array_from_value(value: &Value) -> Result<Vec<String>, OutputError> {
    let values = value.as_array().ok_or_else(|| OutputError::Runtime("Expected a JSON string array.".to_string()))?;
    values
        .iter()
        .map(|item| {
            item.as_str()
                .map(str::to_string)
                .ok_or_else(|| OutputError::Runtime("Expected a JSON string array.".to_string()))
        })
        .collect()
}

fn prediction_loco_file_fingerprints_to_json_value(
    fingerprints: Vec<PredictionLocoFileFingerprint>,
) -> Result<Value, OutputError> {
    let values = fingerprints
        .into_iter()
        .map(|fingerprint| {
            let content_sha256 = fingerprint.fingerprint.content_sha256.ok_or_else(|| {
                OutputError::Runtime("LOCO prediction file fingerprint must include a content hash.".to_string())
            })?;
            Ok(json!({
                "phenotype": fingerprint.phenotype,
                "path": fingerprint.fingerprint.path,
                "size": fingerprint.fingerprint.size,
                "mtime_ns": fingerprint.fingerprint.mtime_ns,
                "content_hash_algorithm": fingerprint.fingerprint.content_hash_algorithm,
                "content_sha256": content_sha256,
            }))
        })
        .collect::<Result<Vec<_>, OutputError>>()?;
    Ok(Value::Array(values))
}

fn build_prediction_loco_file_fingerprints_with_cache(
    prediction_list_path: &str,
    phenotype_names: &[String],
    fingerprint_cache: &mut ManifestFileFingerprintCache,
) -> Result<Vec<PredictionLocoFileFingerprint>, OutputError> {
    let resolved_loco_paths = resolve_prediction_loco_paths(Path::new(prediction_list_path), phenotype_names)
        .map_err(|error| OutputError::Runtime(error.to_string()))?;
    let mut loco_file_fingerprints = Vec::with_capacity(resolved_loco_paths.len());
    for resolved_loco_path in resolved_loco_paths {
        let file_fingerprint = fingerprint_cache.build_file_fingerprint(&resolved_loco_path.loco_file_path, true)?;
        loco_file_fingerprints.push(PredictionLocoFileFingerprint {
            phenotype: resolved_loco_path.phenotype_name,
            fingerprint: file_fingerprint,
        });
    }
    Ok(loco_file_fingerprints)
}
