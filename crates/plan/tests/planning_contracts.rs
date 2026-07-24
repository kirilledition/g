use std::collections::BTreeSet;

use g_genotype_contracts::BgenContentSha256;
use g_plan::{
    AssociationMode, BinaryFallbackMethod, BinaryNullKernelPlan, ComputePlan, CorrectionPlan, Device, DosageThreshold,
    FirthKernelPlan, InputPlan, KernelPlan, LinearKernelPlan, MultiPhenotypeSampleMode, NullFirthKernelPlan,
    NullLogisticNonconvergencePolicy, OutputPlan, PhenotypeComputeGroup, PhenotypeComputeGroupMode, PhenotypeRunPlan,
    PositiveF32, PositiveF64, Probability, ProbabilityFloor, RunPlan, StepScale, TelemetryMode,
    build_phenotype_compute_group_id, build_phenotype_output_directory_name,
};
use serde_json::Value;

const F32_TOLERANCE: f32 = 1.0e-6;
const F64_TOLERANCE: f64 = 1.0e-12;

fn positive_f32(value: f32) -> PositiveF32 {
    PositiveF32::try_from(value).expect("test value should be a valid positive f32")
}

fn positive_f64(value: f64) -> PositiveF64 {
    PositiveF64::try_from(value).expect("test value should be a valid positive f64")
}

fn assert_f32_within_tolerance(actual: f32, expected: f32) {
    let absolute_difference = (actual - expected).abs();
    assert!(
        absolute_difference < F32_TOLERANCE,
        "expected {actual} and {expected} to differ by less than {F32_TOLERANCE}, got {absolute_difference}",
    );
}

fn assert_f64_within_tolerance(actual: f64, expected: f64) {
    let absolute_difference = (actual - expected).abs();
    assert!(
        absolute_difference < F64_TOLERANCE,
        "expected {actual} and {expected} to differ by less than {F64_TOLERANCE}, got {absolute_difference}",
    );
}

fn build_test_run_plan() -> RunPlan {
    RunPlan {
        association_mode: AssociationMode::Regenie2Binary,
        chunk_size: 16_384,
        input: InputPlan {
            bgen_path: "study.bgen".to_string(),
            bgen_content_sha256: Some(BgenContentSha256::from_bytes([0xab; 32])),
            sample_path: "study.sample".to_string(),
            phenotype_path: "phenotypes.tsv".to_string(),
            prediction_list_path: "predictions.list".to_string(),
            covariate_path: Some("covariates.tsv".to_string()),
            covariate_names: vec!["age".to_string(), "sex".to_string()],
        },
        compute: ComputePlan {
            device: Device::Gpu,
            cpu_thread_count: Some(8),
            jax_cache_directory: Some("cache/jax".to_string()),
            multi_phenotype_sample_mode: MultiPhenotypeSampleMode::CompleteCase,
            kernels: KernelPlan {
                linear: LinearKernelPlan {
                    minimum_variance: positive_f32(1.0e-8),
                    relative_variance_tolerance: positive_f32(1.0e-5),
                },
                binary_null: BinaryNullKernelPlan {
                    maximum_iterations: 25,
                    coefficient_tolerance: positive_f32(1.0e-4),
                    nonconvergence_policy: NullLogisticNonconvergencePolicy::Warn,
                    minimum_probability: ProbabilityFloor::try_from(1.0e-6)
                        .expect("test value should be a valid probability floor"),
                    minimum_variance: positive_f32(1.0e-8),
                    relative_variance_tolerance: positive_f32(1.0e-5),
                },
                firth: FirthKernelPlan {
                    batch_size: 512,
                    candidate_capacity: 1_024,
                    maximum_iterations: 100,
                    gradient_tolerance: positive_f64(1.0e-6),
                    maximum_step_size: positive_f64(5.0),
                    pseudo_maximum_iterations: 20,
                    pseudo_inner_maximum_iterations: 10,
                    line_search_maximum_attempts: 25,
                    sparse_carrier_dosage_threshold: DosageThreshold::try_from(0.5)
                        .expect("test value should be a valid dosage threshold"),
                },
                null_firth: NullFirthKernelPlan {
                    maximum_iterations: 100,
                    gradient_tolerance: positive_f64(1.0e-6),
                    maximum_step_size: positive_f64(5.0),
                    fallback_iteration_multiplier: 4,
                    fallback_step_divisor: positive_f64(2.0),
                    line_search_maximum_attempts: 25,
                    step_halving_scale: StepScale::try_from(0.5).expect("test value should be a valid step scale"),
                },
            },
        },
        correction: CorrectionPlan {
            method: BinaryFallbackMethod::FirthApproximate,
            p_threshold: Probability::try_from(0.05).expect("test value should be a valid probability"),
            firth_se: true,
        },
        output: OutputPlan {
            output_run_root: "results/analysis.run".to_string(),
            resume: true,
            recover_attempt: Some("attempt-123".to_string()),
            fenced_owner_claim_id: Some("owner-123".to_string()),
            writer_thread_count: 8,
        },
        telemetry: TelemetryMode::Off,
        phenotype_runs: vec![
            PhenotypeRunPlan {
                phenotype_name: "height".to_string(),
                output_directory_name: "trait_0000_height".to_string(),
            },
            PhenotypeRunPlan {
                phenotype_name: "weight".to_string(),
                output_directory_name: "trait_0001_weight".to_string(),
            },
        ],
    }
}

fn assert_object_keys(value: &Value, expected_keys: &[&str]) {
    let object = value.as_object().expect("contract value should be a JSON object");
    let actual_keys = object.keys().map(String::as_str).collect::<BTreeSet<_>>();
    let expected_keys = expected_keys.iter().copied().collect::<BTreeSet<_>>();
    assert_eq!(actual_keys, expected_keys);
}

fn assert_serialized_run_plan_shape(serialized_plan: &Value) {
    assert_object_keys(
        serialized_plan,
        &["association_mode", "chunk_size", "compute", "correction", "input", "output", "phenotype_runs", "telemetry"],
    );
    assert_object_keys(
        &serialized_plan["input"],
        &[
            "bgen_content_sha256",
            "bgen_path",
            "covariate_names",
            "covariate_path",
            "phenotype_path",
            "prediction_list_path",
            "sample_path",
        ],
    );
    assert_eq!(
        serialized_plan["input"]["bgen_content_sha256"],
        "abababababababababababababababababababababababababababababababab",
    );
    assert_object_keys(
        &serialized_plan["compute"],
        &["cpu_thread_count", "device", "jax_cache_directory", "kernels", "multi_phenotype_sample_mode"],
    );
    assert_object_keys(&serialized_plan["compute"]["kernels"], &["binary_null", "firth", "linear", "null_firth"]);
    assert_object_keys(
        &serialized_plan["compute"]["kernels"]["linear"],
        &["minimum_variance", "relative_variance_tolerance"],
    );
    assert_object_keys(
        &serialized_plan["compute"]["kernels"]["binary_null"],
        &[
            "coefficient_tolerance",
            "maximum_iterations",
            "minimum_probability",
            "minimum_variance",
            "nonconvergence_policy",
            "relative_variance_tolerance",
        ],
    );
    assert_object_keys(
        &serialized_plan["compute"]["kernels"]["firth"],
        &[
            "batch_size",
            "candidate_capacity",
            "gradient_tolerance",
            "line_search_maximum_attempts",
            "maximum_iterations",
            "maximum_step_size",
            "pseudo_inner_maximum_iterations",
            "pseudo_maximum_iterations",
            "sparse_carrier_dosage_threshold",
        ],
    );
    assert_object_keys(
        &serialized_plan["compute"]["kernels"]["null_firth"],
        &[
            "fallback_iteration_multiplier",
            "fallback_step_divisor",
            "gradient_tolerance",
            "line_search_maximum_attempts",
            "maximum_iterations",
            "maximum_step_size",
            "step_halving_scale",
        ],
    );
    assert_object_keys(&serialized_plan["correction"], &["firth_se", "method", "p_threshold"]);
    assert_object_keys(
        &serialized_plan["output"],
        &["fenced_owner_claim_id", "output_run_root", "recover_attempt", "resume", "writer_thread_count"],
    );
    assert_object_keys(&serialized_plan["phenotype_runs"][0], &["output_directory_name", "phenotype_name"]);
}

fn assert_decoded_run_plan_values(decoded_plan: &RunPlan) {
    assert_eq!(decoded_plan.association_mode, AssociationMode::Regenie2Binary);
    assert_eq!(decoded_plan.chunk_size, 16_384);
    assert_eq!(decoded_plan.input.bgen_path, "study.bgen");
    assert_eq!(decoded_plan.input.bgen_content_sha256, Some(BgenContentSha256::from_bytes([0xab; 32])));
    assert_eq!(decoded_plan.input.sample_path, "study.sample");
    assert_eq!(decoded_plan.input.phenotype_path, "phenotypes.tsv");
    assert_eq!(decoded_plan.input.prediction_list_path, "predictions.list");
    assert_eq!(decoded_plan.input.covariate_path.as_deref(), Some("covariates.tsv"));
    assert_eq!(decoded_plan.input.covariate_names, ["age", "sex"]);
    assert_eq!(decoded_plan.compute.device, Device::Gpu);
    assert_eq!(decoded_plan.compute.cpu_thread_count, Some(8));
    assert_eq!(decoded_plan.compute.jax_cache_directory.as_deref(), Some("cache/jax"));
    assert_eq!(decoded_plan.compute.multi_phenotype_sample_mode, MultiPhenotypeSampleMode::CompleteCase);

    let kernels = &decoded_plan.compute.kernels;
    assert_f32_within_tolerance(kernels.linear.minimum_variance.get(), 1.0e-8);
    assert_f32_within_tolerance(kernels.linear.relative_variance_tolerance.get(), 1.0e-5);
    assert_eq!(kernels.binary_null.maximum_iterations, 25);
    assert_f32_within_tolerance(kernels.binary_null.coefficient_tolerance.get(), 1.0e-4);
    assert_eq!(kernels.binary_null.nonconvergence_policy, NullLogisticNonconvergencePolicy::Warn);
    assert_f32_within_tolerance(kernels.binary_null.minimum_probability.get(), 1.0e-6);
    assert_f32_within_tolerance(kernels.binary_null.minimum_variance.get(), 1.0e-8);
    assert_f32_within_tolerance(kernels.binary_null.relative_variance_tolerance.get(), 1.0e-5);

    assert_eq!(kernels.firth.batch_size, 512);
    assert_eq!(kernels.firth.candidate_capacity, 1_024);
    assert_eq!(kernels.firth.maximum_iterations, 100);
    assert_f64_within_tolerance(kernels.firth.gradient_tolerance.get(), 1.0e-6);
    assert_f64_within_tolerance(kernels.firth.maximum_step_size.get(), 5.0);
    assert_eq!(kernels.firth.pseudo_maximum_iterations, 20);
    assert_eq!(kernels.firth.pseudo_inner_maximum_iterations, 10);
    assert_eq!(kernels.firth.line_search_maximum_attempts, 25);
    assert_f64_within_tolerance(kernels.firth.sparse_carrier_dosage_threshold.get(), 0.5);

    assert_eq!(kernels.null_firth.maximum_iterations, 100);
    assert_f64_within_tolerance(kernels.null_firth.gradient_tolerance.get(), 1.0e-6);
    assert_f64_within_tolerance(kernels.null_firth.maximum_step_size.get(), 5.0);
    assert_eq!(kernels.null_firth.fallback_iteration_multiplier, 4);
    assert_f64_within_tolerance(kernels.null_firth.fallback_step_divisor.get(), 2.0);
    assert_eq!(kernels.null_firth.line_search_maximum_attempts, 25);
    assert_f64_within_tolerance(kernels.null_firth.step_halving_scale.get(), 0.5);

    assert_eq!(decoded_plan.correction.method, BinaryFallbackMethod::FirthApproximate);
    assert_f32_within_tolerance(decoded_plan.correction.p_threshold.get(), 0.05);
    assert!(decoded_plan.correction.firth_se);
    assert_eq!(decoded_plan.output.output_run_root, "results/analysis.run");
    assert!(decoded_plan.output.resume);
    assert_eq!(decoded_plan.output.recover_attempt.as_deref(), Some("attempt-123"));
    assert_eq!(decoded_plan.output.fenced_owner_claim_id.as_deref(), Some("owner-123"));
    assert_eq!(decoded_plan.output.writer_thread_count, 8);
    assert_eq!(decoded_plan.telemetry, TelemetryMode::Off);
    assert_eq!(decoded_plan.phenotype_runs.len(), 2);
    assert_eq!(decoded_plan.phenotype_runs[0].phenotype_name, "height");
    assert_eq!(decoded_plan.phenotype_runs[0].output_directory_name, "trait_0000_height");
    assert_eq!(decoded_plan.phenotype_runs[1].phenotype_name, "weight");
    assert_eq!(decoded_plan.phenotype_runs[1].output_directory_name, "trait_0001_weight");
}

#[test]
fn run_plan_round_trip_preserves_the_complete_public_contract() {
    let serialized_plan = serde_json::to_value(build_test_run_plan()).expect("run plan serialization should succeed");
    assert_serialized_run_plan_shape(&serialized_plan);

    let decoded_plan =
        serde_json::from_value::<RunPlan>(serialized_plan).expect("serialized run plan should deserialize");
    assert_decoded_run_plan_values(&decoded_plan);
}

#[test]
fn run_plan_deserialization_enforces_nested_numeric_contracts() {
    let mut serialized_plan =
        serde_json::to_value(build_test_run_plan()).expect("run plan serialization should succeed");
    serialized_plan["correction"]["p_threshold"] = serde_json::json!(1.0);

    let error = serde_json::from_value::<RunPlan>(serialized_plan)
        .expect_err("invalid nested probability should reject the run plan");
    assert!(error.to_string().contains("must be in (0, 1)"));
}

#[test]
fn run_plan_deserialization_rejects_noncanonical_bgen_content_digests() {
    for invalid_digest in
        [serde_json::json!(true), serde_json::json!("ABABABABABABABABABABABABABABABABABABABABABABABABABABABABABAB")]
    {
        let mut serialized_plan =
            serde_json::to_value(build_test_run_plan()).expect("run plan serialization should succeed");
        serialized_plan["input"]["bgen_content_sha256"] = invalid_digest;

        serde_json::from_value::<RunPlan>(serialized_plan)
            .expect_err("noncanonical BGEN content digest should reject the run plan");
    }
}

fn build_test_compute_group() -> PhenotypeComputeGroup {
    PhenotypeComputeGroup {
        group_mode: PhenotypeComputeGroupMode::CompleteCase,
        phenotype_indices: vec![0, 2],
        phenotype_names: vec!["height".to_string(), "weight".to_string()],
        sample_mode: MultiPhenotypeSampleMode::CompleteCase,
        sample_set_fingerprint: "samples-a".to_string(),
        covariate_design_fingerprint: "covariates-a".to_string(),
        phenotype_design_fingerprint: "phenotypes-a".to_string(),
        prediction_alignment_fingerprint: "predictions-a".to_string(),
    }
}

#[test]
fn compute_group_serialization_and_identifier_are_stable() {
    let compute_group = build_test_compute_group();
    let serialized_group = serde_json::to_value(&compute_group).expect("compute group serialization should succeed");
    assert_object_keys(
        &serialized_group,
        &[
            "covariate_design_fingerprint",
            "group_mode",
            "phenotype_design_fingerprint",
            "phenotype_indices",
            "phenotype_names",
            "prediction_alignment_fingerprint",
            "sample_mode",
            "sample_set_fingerprint",
        ],
    );
    assert_eq!(serialized_group["group_mode"], "complete-case");
    assert_eq!(serialized_group["sample_mode"], "complete-case");

    let identifier = build_phenotype_compute_group_id(&compute_group);
    assert_eq!(identifier, "bc1aff2ce06c3776fea9643e86c28aa6358c74f9bb92a59841067366bcb81e2a");
    assert_eq!(identifier.len(), 64);
    assert!(identifier.bytes().all(|identifier_byte| identifier_byte.is_ascii_hexdigit()));

    let decoded_group =
        serde_json::from_value::<PhenotypeComputeGroup>(serialized_group).expect("compute group should deserialize");
    assert_eq!(decoded_group, compute_group);
}

#[test]
fn compute_group_identifier_includes_every_group_field() {
    let base_identifier = build_phenotype_compute_group_id(&build_test_compute_group());

    let mut changed_group_mode = build_test_compute_group();
    changed_group_mode.group_mode = PhenotypeComputeGroupMode::PerPhenotypeCompatible;
    let mut changed_indices = build_test_compute_group();
    changed_indices.phenotype_indices = vec![0, 1];
    let mut changed_names = build_test_compute_group();
    changed_names.phenotype_names = vec!["height".to_string(), "bmi".to_string()];
    let mut changed_sample_mode = build_test_compute_group();
    changed_sample_mode.sample_mode = MultiPhenotypeSampleMode::PerPhenotype;
    let mut changed_samples = build_test_compute_group();
    changed_samples.sample_set_fingerprint = "samples-b".to_string();
    let mut changed_covariates = build_test_compute_group();
    changed_covariates.covariate_design_fingerprint = "covariates-b".to_string();
    let mut changed_phenotypes = build_test_compute_group();
    changed_phenotypes.phenotype_design_fingerprint = "phenotypes-b".to_string();
    let mut changed_predictions = build_test_compute_group();
    changed_predictions.prediction_alignment_fingerprint = "predictions-b".to_string();

    for changed_group in [
        changed_group_mode,
        changed_indices,
        changed_names,
        changed_sample_mode,
        changed_samples,
        changed_covariates,
        changed_phenotypes,
        changed_predictions,
    ] {
        assert_ne!(build_phenotype_compute_group_id(&changed_group), base_identifier);
    }
}

#[test]
fn phenotype_output_directory_names_are_safe_and_deterministic() {
    assert_eq!(build_phenotype_output_directory_name(7, "  Height / BMI  "), "trait_0007_Height_BMI");
    assert_eq!(
        build_phenotype_output_directory_name(12, "alpha.beta_gamma-delta"),
        "trait_0012_alpha.beta_gamma-delta",
    );
    assert_eq!(build_phenotype_output_directory_name(1, "../../height"), "trait_0001_height");
    assert_eq!(build_phenotype_output_directory_name(42, "...---___"), "trait_0042_phenotype");

    let long_name = "a".repeat(81);
    let expected_truncated_name = format!("trait_0003_{}", "a".repeat(80));
    assert_eq!(build_phenotype_output_directory_name(3, &long_name), expected_truncated_name);
}
