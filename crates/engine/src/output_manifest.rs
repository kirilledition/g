//! Native output-manifest preparation helpers.

use g_output::{ManifestFileFingerprintCache, OutputError, PredictionLocoFileFingerprint};

pub(crate) fn build_prediction_loco_file_fingerprints_with_cache(
    resolved_loco_paths: &[g_input::PredictionLocoPath],
    fingerprint_cache: &mut ManifestFileFingerprintCache,
) -> Result<Vec<PredictionLocoFileFingerprint>, OutputError> {
    let mut loco_file_fingerprints = Vec::with_capacity(resolved_loco_paths.len());
    for resolved_loco_path in resolved_loco_paths {
        loco_file_fingerprints.push(fingerprint_cache.build_prediction_loco_file_fingerprint(
            std::sync::Arc::clone(&resolved_loco_path.phenotype_name),
            &resolved_loco_path.loco_file_path,
        )?);
    }
    Ok(loco_file_fingerprints)
}
