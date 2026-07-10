//! Native output-manifest preparation helpers.

use std::path::Path;

use g_input::resolve_prediction_loco_paths;
use g_output::{ManifestFileFingerprintCache, OutputError, PredictionLocoFileFingerprint};

pub(crate) fn build_prediction_loco_file_fingerprints_with_cache(
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
            phenotype_name: resolved_loco_path.phenotype_name,
            file_fingerprint,
        });
    }
    Ok(loco_file_fingerprints)
}
