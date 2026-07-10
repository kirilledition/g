//! Native output-manifest preparation helpers.

use std::path::Path;

use g_input::resolve_prediction_loco_paths;
use g_output::OutputError;
use g_output::{ManifestFileFingerprint, ManifestFileFingerprintCache};
use serde_json::{Value, json};

struct PredictionLocoFileFingerprint {
    phenotype: String,
    fingerprint: ManifestFileFingerprint,
}

pub(crate) fn build_prediction_loco_files_json_with_cache(
    prediction_list_path: &str,
    phenotype_names: &[String],
    fingerprint_cache: &mut ManifestFileFingerprintCache,
) -> Result<String, OutputError> {
    let prediction_loco_files = prediction_loco_file_fingerprints_to_json_value(
        build_prediction_loco_file_fingerprints_with_cache(prediction_list_path, phenotype_names, fingerprint_cache)?,
    )?;
    serde_json::to_string(&prediction_loco_files)
        .map_err(|error| OutputError::Runtime(format!("Invalid prediction_loco_files value: {error}")))
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
