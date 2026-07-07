#![allow(clippy::missing_errors_doc)]

mod chunks;
mod contract;
mod fingerprint;
mod header;
mod run;
mod validation;

pub use chunks::RunManifestChunkCommit;
pub use contract::{
    build_prepared_run_manifest_header_json, build_prepared_run_manifest_header_json_from_current_header_json,
    build_prepared_run_plan_from_current_header_json, build_prepared_run_plan_json_from_current_header_json,
};
pub use fingerprint::{
    ManifestFileFingerprint, ManifestFileFingerprintCache, build_file_content_sha256, build_manifest_file_fingerprint,
    build_manifest_json_sha256,
};
pub(crate) use fingerprint::{build_manifest_value_sha256, manifest_file_fingerprint_to_value};
pub use header::{
    CurrentRunManifestHeaderInput, build_current_run_manifest_header_json,
    build_current_run_manifest_header_json_with_cache,
};
pub use run::{
    InitializedOutputRun, OutputResumeMode, OutputRunPaths, PreparedOutputRun, extend_run_manifest_metadata,
    initialize_output_run, load_run_manifest_json, prepare_output_run,
    read_run_manifest_committed_chunk_identifiers_from_text, resolve_output_run_paths,
    validate_run_manifest_compatibility, write_run_manifest_json,
};
pub(crate) use run::{
    mark_run_manifest_finalized, mark_run_manifest_finalized_output, mark_run_manifest_interrupted,
    read_run_manifest_chunk_commits, read_run_manifest_chunk_commits_from_text, record_run_manifest_chunk_commits,
};

#[cfg(test)]
use fingerprint::FILE_FINGERPRINT_METADATA_ONLY;
#[cfg(test)]
use serde_json::{Value, json};

const RUN_MANIFEST_FILE_NAME: &str = "run_manifest.json";
const RUN_MANIFEST_SCHEMA_VERSION: i64 = 9;
const OUTPUT_SCHEMA_VERSION: i64 = 2;
const JAX_MATMUL_PRECISION_WHEN_UNSET: &str = "float32";
const RESUME_POLICY: &str = "manifest_committed_chunks";

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn create_test_directory() -> PathBuf {
        let unique_suffix =
            SystemTime::now().duration_since(UNIX_EPOCH).expect("system time should be after Unix epoch").as_nanos();
        let directory_path = std::env::temp_dir().join(format!("g-output-manifest-test-{unique_suffix}"));
        std::fs::create_dir_all(&directory_path).expect("test directory should be created");
        directory_path
    }

    fn build_chunk_commit(chunk_identifier: i64) -> RunManifestChunkCommit {
        RunManifestChunkCommit {
            chunk_identifier,
            output_format: "arrow".to_string(),
            compression: "none".to_string(),
            variant_start_index: chunk_identifier,
            variant_stop_index: chunk_identifier + 2,
            row_count: 2,
            chunk_file_name: format!("chunk_{chunk_identifier:09}.arrow"),
        }
    }

    fn convert_file_fingerprint(fingerprint: ManifestFileFingerprint) -> g_plan::ManifestFileFingerprint {
        g_plan::ManifestFileFingerprint {
            path: fingerprint.path,
            size: fingerprint.size,
            mtime_ns: fingerprint.mtime_ns,
            content_hash_algorithm: fingerprint.content_hash_algorithm,
            content_sha256: fingerprint.content_sha256,
        }
    }

    struct TestManifestFiles {
        root_directory: PathBuf,
        bgen_path: PathBuf,
        sample_path: PathBuf,
        phenotype_path: PathBuf,
        covariate_path: PathBuf,
        prediction_list_path: PathBuf,
        prediction_loco_files: Value,
    }

    fn create_manifest_fixture_files() -> TestManifestFiles {
        let root_directory = create_test_directory();
        let bgen_path = root_directory.join("input.bgen");
        let sample_path = root_directory.join("input.sample");
        let phenotype_path = root_directory.join("phenotypes.tsv");
        let covariate_path = root_directory.join("covariates.tsv");
        let prediction_list_path = root_directory.join("pred.list");
        let loco_path = root_directory.join("height.loco");
        std::fs::write(&bgen_path, "bgen").expect("BGEN fixture should be written");
        std::fs::write(&sample_path, "sample").expect("sample fixture should be written");
        std::fs::write(&phenotype_path, "iid height\n1 1.5\n").expect("phenotype fixture should be written");
        std::fs::write(&covariate_path, "iid age\n1 42\n").expect("covariate fixture should be written");
        std::fs::write(&prediction_list_path, "height height.loco\n")
            .expect("prediction list fixture should be written");
        std::fs::write(&loco_path, "loco").expect("LOCO fixture should be written");

        let loco_fingerprint =
            build_manifest_file_fingerprint(&loco_path, true).expect("LOCO fingerprint should build");
        let prediction_loco_files = json!([{
            "phenotype": "height",
            "path": loco_fingerprint.path,
            "size": loco_fingerprint.size,
            "mtime_ns": loco_fingerprint.mtime_ns,
            "content_hash_algorithm": loco_fingerprint.content_hash_algorithm,
            "content_sha256": loco_fingerprint.content_sha256.expect("LOCO hash should be present"),
        }]);
        TestManifestFiles {
            root_directory,
            bgen_path,
            sample_path,
            phenotype_path,
            covariate_path,
            prediction_list_path,
            prediction_loco_files,
        }
    }

    #[test]
    fn manifest_file_fingerprint_cache_reuses_native_fingerprints_by_observed_file_state() {
        let root_directory = create_test_directory();
        let input_path = root_directory.join("input.txt");
        std::fs::write(&input_path, "fingerprint").expect("input fixture should be written");

        let mut fingerprint_cache = ManifestFileFingerprintCache::new();
        let first_fingerprint =
            fingerprint_cache.build_file_fingerprint(&input_path, true).expect("content fingerprint should build");
        let second_fingerprint =
            fingerprint_cache.build_file_fingerprint(&input_path, true).expect("content fingerprint should be cached");

        assert_eq!(first_fingerprint, second_fingerprint);
        assert_eq!(fingerprint_cache.fingerprints_by_key.len(), 1);

        let metadata_only_fingerprint =
            fingerprint_cache.build_file_fingerprint(&input_path, false).expect("metadata fingerprint should build");
        assert_eq!(metadata_only_fingerprint.content_hash_algorithm, FILE_FINGERPRINT_METADATA_ONLY);
        assert_eq!(metadata_only_fingerprint.content_sha256, None);
        assert_eq!(fingerprint_cache.fingerprints_by_key.len(), 2);

        std::fs::remove_dir_all(root_directory).expect("test directory should be removed");
    }

    fn build_test_current_header_json(test_files: &TestManifestFiles) -> String {
        build_current_run_manifest_header_json(CurrentRunManifestHeaderInput {
            association_mode: "regenie2_binary".to_string(),
            association_backend_kind: "jax_packed8".to_string(),
            bgen_path: test_files.bgen_path.clone(),
            sample_path: Some(test_files.sample_path.clone()),
            phenotype_path: test_files.phenotype_path.clone(),
            phenotype_name: "height".to_string(),
            covariate_path: Some(test_files.covariate_path.clone()),
            covariate_names: vec!["age".to_string()],
            prediction_list_path: test_files.prediction_list_path.clone(),
            prediction_loco_files_json: test_files.prediction_loco_files.to_string(),
            sample_count: 12,
            variant_count: 34,
            chunk_size: 8,
            variant_limit: Some(21),
            binary_correction_plan_method: "score_only".to_string(),
            binary_correction_plan_p_threshold: 0.05,
            binary_correction_plan_firth_se: false,
            trusted_no_missing_diploid: true,
            sample_key_mode: "fid_iid".to_string(),
            binary_kernel_config_json: Some(r#"{"minimum_probability":0.0001}"#.to_string()),
            bgen_decode_tile_variant_count: 64,
            trusted_bgen_validation_mode: "cache_on_miss".to_string(),
            jax_device: "gpu".to_string(),
            jax_enable_x64: true,
            jax_matmul_precision: Some("highest".to_string()),
            requested_gpu_genotype_format: "packed8".to_string(),
            gpu_genotype_format: "packed8".to_string(),
            score_dtype: "float32".to_string(),
            firth_dtype: "float64".to_string(),
            multi_phenotype_sample_mode: "single-phenotype".to_string(),
            phenotype_compute_group_mode: None,
            phenotype_compute_group_indices: None,
            phenotype_compute_group_names: None,
            phenotype_compute_group_sample_mode: None,
            sample_set_fingerprint: None,
            covariate_design_fingerprint: None,
            prediction_alignment_fingerprint: None,
            output_format: "parquet".to_string(),
            finalize_parquet: true,
            writer_thread_count: 2,
            writer_queue_depth: 4,
            chunks_per_arrow_file: 8,
            arrow_compression: "zstd".to_string(),
            parquet_compression: "zstd".to_string(),
            output_statistic_dtype: "float32".to_string(),
        })
        .expect("current manifest header should build")
    }

    fn build_test_prepared_run_plan(test_files: &TestManifestFiles) -> g_plan::PreparedRunPlan {
        let bgen_fingerprint = convert_file_fingerprint(
            build_manifest_file_fingerprint(&test_files.bgen_path, false).expect("BGEN fingerprint"),
        );
        let sample_fingerprint = convert_file_fingerprint(
            build_manifest_file_fingerprint(&test_files.sample_path, true).expect("sample fingerprint"),
        );
        let phenotype_fingerprint = convert_file_fingerprint(
            build_manifest_file_fingerprint(&test_files.phenotype_path, true).expect("phenotype fingerprint"),
        );
        let covariate_fingerprint = convert_file_fingerprint(
            build_manifest_file_fingerprint(&test_files.covariate_path, true).expect("covariate fingerprint"),
        );
        let prediction_list_fingerprint = convert_file_fingerprint(
            build_manifest_file_fingerprint(&test_files.prediction_list_path, true)
                .expect("prediction list fingerprint"),
        );
        g_plan::PreparedRunPlan {
            association_mode: g_plan::AssociationMode::Regenie2Binary,
            association_backend: g_plan::AssociationBackendPlan {
                kind: g_plan::AssociationBackendKind::JaxPacked8,
                association_mode: g_plan::AssociationMode::Regenie2Binary,
                device: g_plan::Device::Gpu,
                resolved_genotype_format: g_plan::GpuGenotypeFormat::Packed8,
            },
            input_identity: g_plan::PreparedInputIdentity {
                bgen: bgen_fingerprint,
                sample: Some(sample_fingerprint),
                phenotype_file: phenotype_fingerprint,
                covariate_file: Some(covariate_fingerprint),
                prediction_list: prediction_list_fingerprint.clone(),
                prediction_inputs: g_plan::PredictionInputsIdentity {
                    prediction_list: prediction_list_fingerprint,
                    loco_files: serde_json::from_value(test_files.prediction_loco_files.clone())
                        .expect("LOCO identity should deserialize"),
                },
            },
            phenotype_name: "height".to_string(),
            covariate_names: vec!["age".to_string()],
            sample_count: 12,
            variant_count: 34,
            chunk_size: 8,
            variant_limit: Some(21),
            correction: g_plan::CorrectionPlan {
                method: g_plan::BinaryFallbackMethod::ScoreOnly,
                p_threshold: 0.05,
                firth_se: false,
            },
            binary_kernel_config: Some(json!({"minimum_probability": 0.0001})),
            compute: build_test_prepared_compute_plan(),
            phenotype_compute_group: None,
            output_writer: build_test_prepared_output_writer_plan(),
        }
    }

    fn build_test_prepared_compute_plan() -> g_plan::PreparedComputePlan {
        g_plan::PreparedComputePlan {
            trusted_no_missing_diploid: true,
            trusted_bgen_validation_mode: g_plan::TrustedBgenValidationMode::CacheOnMiss,
            sample_key_mode: g_plan::SampleKeyMode::FidIid,
            bgen_decode_tile_variant_count: 64,
            jax_policy: g_plan::JaxPolicyPlan {
                device: g_plan::Device::Gpu,
                enable_x64: true,
                matmul_precision: Some(g_plan::JaxMatmulPrecision::Highest),
            },
            requested_gpu_genotype_format: g_plan::GpuGenotypeFormat::Packed8,
            resolved_gpu_genotype_format: g_plan::GpuGenotypeFormat::Packed8,
            score_dtype: g_plan::FloatingPointDtype::Float32,
            firth_dtype: g_plan::FloatingPointDtype::Float64,
            sample_mode: g_plan::PreparedSampleMode::SinglePhenotype,
        }
    }

    fn build_test_prepared_output_writer_plan() -> g_plan::PreparedOutputWriterPlan {
        g_plan::PreparedOutputWriterPlan {
            output_format: g_plan::OutputFormat::Parquet,
            finalize_parquet: true,
            writer_thread_count: 2,
            writer_queue_depth: 4,
            chunks_per_arrow_file: 8,
            arrow_compression: g_plan::ArrowCompression::Zstd,
            parquet_compression: g_plan::ParquetCompression::Zstd,
            output_statistic_dtype: g_plan::FloatingPointDtype::Float32,
        }
    }

    #[test]
    fn prepared_run_plan_manifest_matches_current_header_manifest() {
        let test_files = create_manifest_fixture_files();
        let current_header_json = build_test_current_header_json(&test_files);
        let prepared_run_plan = build_test_prepared_run_plan(&test_files);
        let prepared_header_json =
            build_prepared_run_manifest_header_json(&prepared_run_plan).expect("prepared manifest header should build");

        let current_header = serde_json::from_str::<Value>(&current_header_json).expect("current header should parse");
        let prepared_header =
            serde_json::from_str::<Value>(&prepared_header_json).expect("prepared header should parse");
        assert_eq!(prepared_header, current_header);

        std::fs::remove_dir_all(test_files.root_directory).expect("test directory should be removed");
    }

    #[test]
    fn prepared_run_plan_from_current_header_json_matches_prepared_contract() {
        let test_files = create_manifest_fixture_files();
        let current_header_json = build_test_current_header_json(&test_files);
        let prepared_run_plan = build_prepared_run_plan_from_current_header_json(&current_header_json)
            .expect("prepared run plan should build from current header");

        assert_eq!(prepared_run_plan, build_test_prepared_run_plan(&test_files));

        std::fs::remove_dir_all(test_files.root_directory).expect("test directory should be removed");
    }

    #[test]
    fn prepared_run_plan_from_current_header_json_preserves_requested_gpu_format() {
        let test_files = create_manifest_fixture_files();
        let current_header_json = build_test_current_header_json(&test_files);
        let mut current_header = serde_json::from_str::<Value>(&current_header_json)
            .expect("current header should deserialize for test mutation");
        current_header["requested_gpu_genotype_format"] = json!("auto");
        let current_header_json =
            serde_json::to_string(&current_header).expect("mutated current header should serialize");
        let prepared_run_plan = build_prepared_run_plan_from_current_header_json(&current_header_json)
            .expect("prepared run plan should build from current header");

        assert_eq!(prepared_run_plan.compute.requested_gpu_genotype_format, g_plan::GpuGenotypeFormat::Auto);
        assert_eq!(prepared_run_plan.compute.resolved_gpu_genotype_format, g_plan::GpuGenotypeFormat::Packed8,);
        assert_eq!(prepared_run_plan.association_backend.kind, g_plan::AssociationBackendKind::JaxPacked8);

        std::fs::remove_dir_all(test_files.root_directory).expect("test directory should be removed");
    }

    #[test]
    fn prepares_output_run_paths_and_rejects_unsafe_directory_state() {
        let root_directory = create_test_directory();
        let output_root = root_directory.join("result");

        let prepared_output_run = prepare_output_run(&output_root, "regenie2_linear", OutputFileFormat::Parquet, false)
            .expect("output run should prepare");

        assert_eq!(
            prepared_output_run.output_run_paths.run_directory,
            root_directory.join("result.regenie2_linear.run")
        );
        assert_eq!(
            prepared_output_run.output_run_paths.chunks_directory,
            root_directory.join("result.regenie2_linear.run").join("parts")
        );
        assert!(prepared_output_run.output_run_paths.chunks_directory.exists());
        assert_eq!(prepared_output_run.existing_manifest_json, None);

        let stale_output_root = root_directory.join("stale");
        let stale_run_paths = resolve_output_run_paths(&stale_output_root, "regenie2_linear", OutputFileFormat::Arrow);
        std::fs::create_dir_all(&stale_run_paths.run_directory).expect("stale run directory should be created");
        std::fs::write(stale_run_paths.run_directory.join("stale.txt"), "stale").expect("stale file should be written");
        let stale_error = prepare_output_run(&stale_output_root, "regenie2_linear", OutputFileFormat::Arrow, false)
            .expect_err("non-empty output directory should be rejected");
        assert!(stale_error.to_string().contains("already exists and is not empty"));

        let missing_resume_error =
            prepare_output_run(&root_directory.join("missing"), "regenie2_linear", OutputFileFormat::Arrow, true)
                .expect_err("resume without manifest should be rejected");
        assert_eq!(missing_resume_error.to_string(), "Resume requires run_manifest.json.");

        std::fs::remove_dir_all(root_directory).expect("test directory should be removed");
    }

    #[test]
    fn initializes_manifest_lifecycle_and_preserves_preinitialized_metadata() {
        let run_directory = create_test_directory();
        let chunks_directory = run_directory.join("chunks");
        std::fs::create_dir_all(&chunks_directory).expect("chunk directory should be created");
        write_run_manifest_json(&run_directory, r#"{"command":{"interface":"g regenie"},"runtime":{"device":"cpu"}}"#)
            .expect("preinitialized manifest should be written");

        let initialized_output_run = initialize_output_run(
            &run_directory,
            &chunks_directory,
            None,
            r#"{"schema_version":7,"execution_plan":{"chunk_size":2},"execution_plan_hash":"hash"}"#,
            false,
            OutputResumeMode::Fast,
        )
        .expect("output run should initialize");

        assert_eq!(initialized_output_run.committed_chunk_identifiers, Vec::<i64>::new());
        let manifest_json =
            load_run_manifest_json(&run_directory).expect("manifest should load").expect("manifest should exist");
        let manifest = serde_json::from_str::<Value>(&manifest_json).expect("manifest should parse");
        assert_eq!(manifest.pointer("/command/interface").and_then(Value::as_str), Some("g regenie"));
        assert_eq!(manifest.pointer("/runtime/device").and_then(Value::as_str), Some("cpu"));
        assert_eq!(manifest.get("schema_version").and_then(Value::as_i64), Some(7));
        assert_eq!(manifest.get("finalized").and_then(Value::as_bool), Some(false));
        assert_eq!(manifest.get("committed_chunks").and_then(Value::as_array).map(Vec::len), Some(0));

        std::fs::remove_dir_all(run_directory).expect("test directory should be removed");
    }

    #[test]
    fn extends_manifest_metadata_and_preserves_existing_fields() {
        let run_directory = create_test_directory();
        write_run_manifest_json(
            &run_directory,
            r#"{"schema_version":9,"bgen":{"path":"/inputs/input.bgen"},"committed_chunks":[]}"#,
        )
        .expect("manifest should be written");

        extend_run_manifest_metadata(
            &run_directory,
            json!({
                "interface": "g regenie",
                "phenotype": "height",
                "effective_config": "effective_config.toml",
                "output_format": "parquet",
            }),
            json!({
                "device": "gpu",
                "staging_depth": 2,
                "native_callback_batch_size": 3,
                "threads": 4,
                "writer_threads": 5,
                "writer_queue_depth": 6,
                "chunks_per_arrow_file": 7,
                "arrow_compression": "zstd",
                "parquet_compression": "snappy",
                "output_statistic_dtype": "float32",
                "bgen_decode_tile_variant_count": 8,
                "trusted_no_missing_diploid": true,
                "trusted_bgen_validation_mode": "strict",
            }),
        )
        .expect("manifest metadata should be extended");

        let manifest_json =
            load_run_manifest_json(&run_directory).expect("manifest should load").expect("manifest should exist");
        let manifest = serde_json::from_str::<Value>(&manifest_json).expect("manifest should parse");
        assert_eq!(manifest.pointer("/command/interface").and_then(Value::as_str), Some("g regenie"));
        assert_eq!(manifest.pointer("/command/phenotype").and_then(Value::as_str), Some("height"));
        assert_eq!(manifest.pointer("/runtime/device").and_then(Value::as_str), Some("gpu"));
        assert_eq!(manifest.pointer("/runtime/threads").and_then(Value::as_i64), Some(4));
        assert_eq!(manifest.pointer("/bgen/path").and_then(Value::as_str), Some("/inputs/input.bgen"));
        assert_eq!(manifest.get("committed_chunks").and_then(Value::as_array).map(Vec::len), Some(0));

        std::fs::remove_dir_all(run_directory).expect("test directory should be removed");
    }

    #[test]
    fn extend_manifest_metadata_creates_missing_manifest() {
        let run_directory = create_test_directory();

        extend_run_manifest_metadata(
            &run_directory,
            json!({
                "interface": "g regenie",
                "phenotype": "height",
                "effective_config": "effective_config.toml",
                "output_format": "regenie",
            }),
            json!({
                "device": "cpu",
                "staging_depth": 1,
                "native_callback_batch_size": 2,
            }),
        )
        .expect("manifest metadata should be written");

        let manifest_json =
            load_run_manifest_json(&run_directory).expect("manifest should load").expect("manifest should exist");
        let manifest = serde_json::from_str::<Value>(&manifest_json).expect("manifest should parse");
        assert_eq!(manifest.pointer("/command/phenotype").and_then(Value::as_str), Some("height"));
        assert_eq!(manifest.pointer("/runtime/device").and_then(Value::as_str), Some("cpu"));

        std::fs::remove_dir_all(run_directory).expect("test directory should be removed");
    }

    #[test]
    fn initialize_rejects_incompatible_manifest_without_rewrite() {
        let run_directory = create_test_directory();
        let chunks_directory = run_directory.join("chunks");
        std::fs::create_dir_all(&chunks_directory).expect("chunk directory should be created");
        let manifest_json = r#"{"schema_version":7,"execution_plan":{"chunk_size":4},"execution_plan_hash":"old","committed_chunks":[]}"#;
        write_run_manifest_json(&run_directory, manifest_json).expect("manifest should be written");
        let manifest_path = run_directory.join(RUN_MANIFEST_FILE_NAME);
        let original_manifest_bytes = std::fs::read(&manifest_path).expect("manifest should be readable");

        let error = initialize_output_run(
            &run_directory,
            &chunks_directory,
            Some(manifest_json),
            r#"{"schema_version":7,"execution_plan":{"chunk_size":2},"execution_plan_hash":"new"}"#,
            true,
            OutputResumeMode::Fast,
        )
        .expect_err("incompatible manifest should be rejected");

        assert!(error.to_string().contains("execution_plan.chunk_size"));
        assert_eq!(std::fs::read(&manifest_path).expect("manifest should be readable"), original_manifest_bytes);

        std::fs::remove_dir_all(run_directory).expect("test directory should be removed");
    }

    #[test]
    fn records_committed_chunks_once_in_identifier_order() {
        let run_directory = create_test_directory();
        let manifest_path = run_directory.join(RUN_MANIFEST_FILE_NAME);
        std::fs::write(&manifest_path, "{\n  \"committed_chunks\": []\n}\n").expect("manifest should be written");

        record_run_manifest_chunk_commits(
            &run_directory,
            vec![build_chunk_commit(2), build_chunk_commit(0), build_chunk_commit(2)],
        )
        .expect("manifest commits should be recorded");

        let manifest_text = std::fs::read_to_string(&manifest_path).expect("manifest should be readable");
        let manifest = serde_json::from_str::<Value>(&manifest_text).expect("manifest should be JSON");
        let committed_chunks =
            manifest.get("committed_chunks").and_then(Value::as_array).expect("committed chunks should be an array");
        let committed_chunk_identifiers = committed_chunks
            .iter()
            .map(|committed_chunk| {
                committed_chunk
                    .get("chunk_identifier")
                    .and_then(Value::as_i64)
                    .expect("chunk identifier should be present")
            })
            .collect::<Vec<_>>();

        assert_eq!(committed_chunk_identifiers, vec![0, 2]);

        std::fs::remove_dir_all(run_directory).expect("test directory should be removed");
    }

    #[test]
    fn rejects_conflicting_duplicate_chunk_commit() {
        let run_directory = create_test_directory();
        let manifest_path = run_directory.join(RUN_MANIFEST_FILE_NAME);
        std::fs::write(&manifest_path, "{\n  \"committed_chunks\": []\n}\n").expect("manifest should be written");
        let mut conflicting_commit = build_chunk_commit(2);
        conflicting_commit.row_count = 3;

        let error = record_run_manifest_chunk_commits(&run_directory, vec![build_chunk_commit(2), conflicting_commit])
            .expect_err("conflicting duplicate chunk should be rejected");

        assert!(error.contains("conflicting commit metadata for chunk 2"));

        std::fs::remove_dir_all(run_directory).expect("test directory should be removed");
    }

    #[test]
    fn reads_manifest_chunk_commits_from_text() {
        let manifest = r#"{
          "committed_chunks": [
            {
              "chunk_identifier": 4,
              "variant_start_index": 4,
              "variant_stop_index": 6,
              "row_count": 2,
              "chunk_file_name": "part_000000000.parquet"
            }
          ]
        }"#;

        let chunk_commits = read_run_manifest_chunk_commits_from_text(manifest).expect("manifest commits should parse");

        assert_eq!(chunk_commits.len(), 1);
        assert_eq!(chunk_commits[0].chunk_identifier, 4);
        assert_eq!(chunk_commits[0].output_format, "parquet");
        assert_eq!(chunk_commits[0].compression, "none");
    }

    #[test]
    fn marks_manifest_interrupted_without_final_outputs() {
        let run_directory = create_test_directory();
        let manifest_path = run_directory.join(RUN_MANIFEST_FILE_NAME);
        std::fs::write(
            &manifest_path,
            "{\n  \"committed_chunks\": [],\n  \"finalized\": true,\n  \"final_parquet\": \"old.parquet\",\n  \"final_row_count\": 1,\n  \"final_chunk_file_count\": 1\n}\n",
        )
        .expect("manifest should be written");

        mark_run_manifest_interrupted(&run_directory, "SIGTERM").expect("manifest should be marked interrupted");

        let manifest_text = std::fs::read_to_string(&manifest_path).expect("manifest should be readable");
        let manifest = serde_json::from_str::<Value>(&manifest_text).expect("manifest should be JSON");

        assert_eq!(manifest.get("finalized").and_then(Value::as_bool), Some(false));
        assert_eq!(manifest.get("interrupted").and_then(Value::as_bool), Some(true));
        assert_eq!(manifest.get("interrupted_signal").and_then(Value::as_str), Some("SIGTERM"));
        assert!(manifest.get("final_parquet").is_none());
        assert!(manifest.get("final_row_count").is_none());
        assert!(manifest.get("final_chunk_file_count").is_none());

        std::fs::remove_dir_all(run_directory).expect("test directory should be removed");
    }
}
