use std::sync::Arc;

use serde_json::Value;

use super::{
    PRIMARY_PHENOTYPE, TestDirectory, header, initialize_manager, planned_run_directories, read_manifest, run_plan,
    single_chunk_plan, test_inputs,
};
use crate::OutputManager;

fn assert_legacy_policy_rejected_before_mutation(
    association_mode: g_plan::AssociationMode,
    correction_method: g_plan::BinaryFallbackMethod,
    policy_parent: &str,
    policy_name: &str,
) {
    let directory = TestDirectory::new("legacy-compute-policy");
    let phenotype_names = [PRIMARY_PHENOTYPE];
    let inputs = test_inputs(&directory, &phenotype_names);
    let mut initial_plan = run_plan(&directory, &inputs, &phenotype_names, false, 1);
    let initial_plan_value = Arc::get_mut(&mut initial_plan).expect("test plan has one owner");
    initial_plan_value.association_mode = association_mode;
    initial_plan_value.correction.method = correction_method;
    let run_directory = planned_run_directories(&initial_plan).remove(0);
    let manager = initialize_manager(initial_plan, &inputs, &phenotype_names, &single_chunk_plan(0..1));
    manager.abort().expect("initial manager closes");
    let manifest_path = run_directory.join("run_manifest.json");
    let config_path = run_directory.join("effective_config.toml");
    let original_config_bytes = std::fs::read(&config_path).expect("configuration is readable");
    let compatible_manifest = read_manifest(&run_directory);
    for replacement in [None, Some("previous_numerical_policy")] {
        let mut legacy_manifest = compatible_manifest.clone();
        let policies = legacy_manifest
            .pointer_mut(policy_parent)
            .and_then(Value::as_object_mut)
            .expect("policy parent is an object");
        policies.remove(policy_name).expect("new manifests fingerprint the numerical policy");
        if let Some(replacement) = replacement {
            policies.insert(policy_name.to_string(), Value::String(replacement.to_string()));
        }
        legacy_manifest["execution_plan_hash"] = Value::String(
            crate::manifest::build_manifest_value_sha256(&legacy_manifest["execution_plan"])
                .expect("legacy execution plan hashes"),
        );
        let legacy_manifest_bytes = serde_json::to_vec_pretty(&legacy_manifest).expect("legacy manifest serializes");
        std::fs::write(&manifest_path, &legacy_manifest_bytes).expect("legacy manifest is written");
        let mut resume_plan = run_plan(&directory, &inputs, &phenotype_names, true, 1);
        let resume_plan_value = Arc::get_mut(&mut resume_plan).expect("test plan has one owner");
        resume_plan_value.association_mode = association_mode;
        resume_plan_value.correction.method = correction_method;
        let mut resume_manager = OutputManager::open(resume_plan, "# changed numerical policy\n".to_string())
            .expect("legacy manager opens before compatibility validation");
        let error = resume_manager
            .initialize(vec![header(PRIMARY_PHENOTYPE, &inputs, 1)], &single_chunk_plan(0..1), false)
            .expect_err("missing or mismatched numerical policy rejects legacy resume");
        assert!(error.to_string().contains(policy_name), "unexpected error: {error}");
        drop(resume_manager);
        assert_eq!(std::fs::read(&manifest_path).expect("manifest remains readable"), legacy_manifest_bytes);
        assert_eq!(std::fs::read(&config_path).expect("configuration remains readable"), original_config_bytes);
    }
}

#[test]
fn every_association_mode_rejects_legacy_genotype_summary_policy() {
    for association_mode in [g_plan::AssociationMode::Regenie2Linear, g_plan::AssociationMode::Regenie2Binary] {
        for correction_method in
            [g_plan::BinaryFallbackMethod::ScoreOnly, g_plan::BinaryFallbackMethod::FirthApproximate]
        {
            assert_legacy_policy_rejected_before_mutation(
                association_mode,
                correction_method,
                "/execution_plan",
                "genotype_summary_policy",
            );
        }
    }
}

#[test]
fn linear_resume_rejects_legacy_projection_policy() {
    for policy_name in ["linear_projection_policy", "linear_residual_resolution"] {
        assert_legacy_policy_rejected_before_mutation(
            g_plan::AssociationMode::Regenie2Linear,
            g_plan::BinaryFallbackMethod::ScoreOnly,
            "/execution_plan/jax_policy",
            policy_name,
        );
    }
}

#[test]
fn every_binary_correction_rejects_legacy_null_covariate_and_score_policies() {
    for correction_method in [g_plan::BinaryFallbackMethod::ScoreOnly, g_plan::BinaryFallbackMethod::FirthApproximate] {
        for policy_name in ["binary_null_logistic_policy", "binary_covariate_policy", "binary_score_validity_policy"] {
            assert_legacy_policy_rejected_before_mutation(
                g_plan::AssociationMode::Regenie2Binary,
                correction_method,
                "/execution_plan/jax_policy",
                policy_name,
            );
        }
    }
}
