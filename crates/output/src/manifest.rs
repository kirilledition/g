mod fingerprint;
mod header;
mod schema_zero;

pub(crate) use fingerprint::ManifestFileFingerprint;
pub use fingerprint::ManifestFileFingerprintCache;
pub(crate) use fingerprint::build_manifest_value_sha256;
pub(crate) use header::build_current_run_manifest_header_value_with_cache;
pub use header::{CurrentRunManifestHeaderInput, PredictionLocoFileFingerprint};
pub(crate) use schema_zero::ExecutionPlanSchemaZero;
#[cfg(test)]
pub(crate) use schema_zero::canonical_execution_plan_schema_zero_test_value;
pub(crate) const RUN_MANIFEST_SCHEMA_VERSION: i64 = 0;
pub(crate) const OUTPUT_SCHEMA_VERSION: i64 = 0;
const RESUME_POLICY: &str = "lineage_receipts_exact_coverage";

#[cfg(test)]
pub(crate) fn read_run_manifest_gpu_genotype_format_from_text(
    manifest_json: &str,
    manifest_path: &std::path::Path,
) -> Result<g_plan::GpuGenotypeFormat, crate::error::OutputError> {
    let manifest = crate::persistence::attempt::parse_attempt_manifest_json(manifest_json.as_bytes(), manifest_path)?;
    crate::persistence::attempt::validate_attempt_manifest_schema_zero_shape(manifest)
        .map(|validated| validated.gpu_genotype_format())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use serde_json::json;

    use super::{build_manifest_value_sha256, read_run_manifest_gpu_genotype_format_from_text};

    fn rehash_execution_plan(manifest: &mut serde_json::Value) {
        manifest["execution_plan_hash"] =
            serde_json::Value::String(build_manifest_value_sha256(&manifest["execution_plan"]).expect("plan hashes"));
    }

    fn manifest_with_gpu_format(format: &str) -> serde_json::Value {
        let mut execution_plan = super::canonical_execution_plan_schema_zero_test_value();
        execution_plan["association_backend"]["kind"] =
            serde_json::Value::String(if format == "packed8" { "jax_packed8" } else { "jax_dosage" }.to_string());
        execution_plan["association_backend"]["genotype_format"] = serde_json::Value::String(format.to_string());
        let execution_plan_hash = build_manifest_value_sha256(&execution_plan).expect("test execution plan hashes");
        json!({
            "schema_version": 0,
            "output_schema_version": 0,
            "execution_plan": execution_plan,
            "execution_plan_hash": execution_plan_hash,
            "attempt_manifest_schema_version": 0,
            "run_set_id": "run-set-test",
            "attempt_id": "attempt-test",
            "phenotype_name": "phenotype",
            "output_directory_name": "trait_0001_phenotype",
            "chunk_plan_hash": std::iter::repeat_n('b', 64).collect::<String>(),
            "status": "running",
            "committed_parts": [],
            "committed_chunks": [],
            "command": {
                "interface": "g regenie",
                "phenotype": "phenotype",
                "effective_config": "/output/attempts/attempt-test/trait_0001_phenotype/effective_config.toml",
            },
            "runtime": {
                "device": "gpu",
                "cpu_threads": null,
                "writer_threads": 1,
                "writer_queue_depth": crate::WRITER_QUEUE_DEPTH,
                "chunks_per_parquet_file": crate::CHUNKS_PER_PARQUET_FILE,
                "parquet_compression": "zstd",
            },
        })
    }

    #[test]
    fn manifest_gpu_format_reader_accepts_exact_schema_and_rejects_unknown_or_missing() {
        for (format, expected) in
            [("dosage", g_plan::GpuGenotypeFormat::Dosage), ("packed8", g_plan::GpuGenotypeFormat::Packed8)]
        {
            let manifest = manifest_with_gpu_format(format);
            assert_eq!(
                read_run_manifest_gpu_genotype_format_from_text(&manifest.to_string(), Path::new("run_manifest.json"))
                    .expect("supported genotype format reads"),
                expected
            );
        }
        let unknown_format = manifest_with_gpu_format("unknown");
        assert!(
            read_run_manifest_gpu_genotype_format_from_text(
                &unknown_format.to_string(),
                Path::new("run_manifest.json")
            )
            .is_err()
        );

        let mut missing_format = manifest_with_gpu_format("dosage");
        missing_format["execution_plan"]["association_backend"]
            .as_object_mut()
            .expect("association backend is an object")
            .remove("genotype_format");
        rehash_execution_plan(&mut missing_format);
        let missing_format_error = read_run_manifest_gpu_genotype_format_from_text(
            &missing_format.to_string(),
            Path::new("run_manifest.json"),
        )
        .expect_err("missing genotype format is rejected after rehashing");
        assert!(
            missing_format_error.to_string().contains("missing field `genotype_format`"),
            "specific nested schema error is preserved: {missing_format_error}"
        );

        let mut unknown_field = manifest_with_gpu_format("dosage");
        unknown_field["unknown"] = serde_json::Value::Null;
        assert!(
            read_run_manifest_gpu_genotype_format_from_text(&unknown_field.to_string(), Path::new("run_manifest.json"))
                .is_err()
        );

        let duplicate_status = manifest_with_gpu_format("dosage").to_string().replacen(
            "\"status\":\"running\"",
            "\"status\":\"running\",\"status\":\"running\"",
            1,
        );
        assert!(
            read_run_manifest_gpu_genotype_format_from_text(&duplicate_status, Path::new("run_manifest.json")).is_err()
        );
    }

    #[test]
    fn manifest_gpu_format_reader_rejects_wrong_typed_schema_zero_fields() {
        let invalid_top_level_fields = [
            ("schema_version", serde_json::Value::Bool(false)),
            ("output_schema_version", json!(0.0)),
            ("execution_plan", serde_json::Value::Array(Vec::new())),
            ("execution_plan_hash", serde_json::Value::Bool(false)),
            ("run_set_id", serde_json::Value::Bool(false)),
            ("attempt_id", serde_json::Value::Bool(false)),
            ("phenotype_name", serde_json::Value::Bool(false)),
            ("output_directory_name", serde_json::Value::Bool(false)),
            ("chunk_plan_hash", serde_json::Value::Bool(false)),
            ("status", serde_json::Value::Bool(false)),
            ("committed_parts", serde_json::Value::Object(serde_json::Map::new())),
            ("committed_chunks", serde_json::Value::Object(serde_json::Map::new())),
            ("command", serde_json::Value::Array(Vec::new())),
            ("runtime", serde_json::Value::Array(Vec::new())),
        ];
        for (field_name, invalid_value) in invalid_top_level_fields {
            let mut manifest = manifest_with_gpu_format("dosage");
            manifest[field_name] = invalid_value;
            assert!(
                read_run_manifest_gpu_genotype_format_from_text(&manifest.to_string(), Path::new("run_manifest.json"))
                    .is_err(),
                "wrong-typed schema-zero field {field_name} is rejected"
            );
        }

        for (label, committed_parts, committed_chunks) in [
            ("receipt-unknown-field", json!([{"unknown": 0}]), serde_json::Value::Array(Vec::new())),
            (
                "chunk-wrong-field-type",
                serde_json::Value::Array(Vec::new()),
                json!([{
                    "chunk_identifier": "0",
                    "variant_start_index": 0,
                    "variant_stop_index": 1,
                    "row_count": 1,
                    "chunk_file_name": "part_000000000.parquet",
                }]),
            ),
            (
                "chunk-unknown-field",
                serde_json::Value::Array(Vec::new()),
                json!([{
                    "chunk_identifier": 0,
                    "variant_start_index": 0,
                    "variant_stop_index": 1,
                    "row_count": 1,
                    "chunk_file_name": "part_000000000.parquet",
                    "unknown": 0,
                }]),
            ),
        ] {
            let mut manifest = manifest_with_gpu_format("dosage");
            manifest["committed_parts"] = committed_parts;
            manifest["committed_chunks"] = committed_chunks;
            assert!(
                read_run_manifest_gpu_genotype_format_from_text(&manifest.to_string(), Path::new("run_manifest.json"))
                    .is_err(),
                "malformed nested schema-zero value {label} is rejected"
            );
        }

        let mut missing_nullable_runtime_field = manifest_with_gpu_format("dosage");
        missing_nullable_runtime_field["runtime"].as_object_mut().expect("runtime is an object").remove("cpu_threads");
        assert!(
            read_run_manifest_gpu_genotype_format_from_text(
                &missing_nullable_runtime_field.to_string(),
                Path::new("run_manifest.json")
            )
            .is_err()
        );
    }
}
