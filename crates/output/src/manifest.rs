mod fingerprint;
mod header;

pub(crate) use fingerprint::ManifestFileFingerprint;
pub use fingerprint::ManifestFileFingerprintCache;
pub(crate) use fingerprint::{build_manifest_value_sha256, manifest_file_fingerprint_to_value};
pub(crate) use header::build_current_run_manifest_header_value_with_cache;
pub use header::{CurrentRunManifestHeaderInput, PredictionLocoFileFingerprint};
pub(crate) const RUN_MANIFEST_SCHEMA_VERSION: i64 = 0;
pub(crate) const OUTPUT_SCHEMA_VERSION: i64 = 0;
const RESUME_POLICY: &str = "lineage_receipts_exact_coverage";

pub(crate) fn read_run_manifest_gpu_genotype_format_from_text(
    manifest_json: &str,
) -> Result<g_plan::GpuGenotypeFormat, crate::error::OutputError> {
    let manifest = serde_json::from_str::<serde_json::Value>(manifest_json)
        .map_err(|error| crate::error::OutputError::InvalidInput(error.to_string()))?;
    if !manifest.is_object() {
        return Err(crate::error::OutputError::InvalidInput("Run manifest must contain a JSON object.".to_string()));
    }
    let genotype_format = manifest
        .pointer("/execution_plan/association_backend/genotype_format")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            crate::error::OutputError::InvalidInput(
                "Run manifest execution_plan.association_backend.genotype_format is missing.".to_string(),
            )
        })?;
    match genotype_format {
        "dosage" => Ok(g_plan::GpuGenotypeFormat::Dosage),
        "packed8" => Ok(g_plan::GpuGenotypeFormat::Packed8),
        unsupported_format => Err(crate::error::OutputError::InvalidInput(format!(
            "Run manifest has unsupported execution-plan GPU genotype format '{unsupported_format}'."
        ))),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::read_run_manifest_gpu_genotype_format_from_text;

    #[test]
    fn manifest_gpu_format_reader_accepts_public_formats_and_rejects_unknown_or_missing() {
        for (format, expected) in
            [("dosage", g_plan::GpuGenotypeFormat::Dosage), ("packed8", g_plan::GpuGenotypeFormat::Packed8)]
        {
            let manifest = json!({
                "execution_plan": {
                    "association_backend": {
                        "genotype_format": format,
                    },
                },
            });
            assert_eq!(
                read_run_manifest_gpu_genotype_format_from_text(&manifest.to_string())
                    .expect("supported genotype format reads"),
                expected
            );
        }
        for manifest in [
            json!({"execution_plan": {"association_backend": {"genotype_format": "unknown"}}}),
            json!({"execution_plan": {"association_backend": {}}}),
            json!([]),
        ] {
            assert!(read_run_manifest_gpu_genotype_format_from_text(&manifest.to_string()).is_err());
        }
    }
}
