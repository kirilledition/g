//! Deterministic host-side planning policy shared through the Python boundary.

#![allow(clippy::missing_errors_doc)]

use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde_json::Value;
use sha2::{Digest, Sha256};

const ASSOCIATION_BACKEND_JAX_DOSAGE: &str = "jax_dosage";
const ASSOCIATION_BACKEND_JAX_PACKED8: &str = "jax_packed8";
const ASSOCIATION_MODE_REGENIE2_BINARY: &str = "regenie2_binary";
const ASSOCIATION_MODE_REGENIE2_LINEAR: &str = "regenie2_linear";
const BINARY_FALLBACK_METHOD_FIRTH_APPROXIMATE: &str = "firth_approximate";
const BINARY_FALLBACK_METHOD_SCORE_ONLY: &str = "score_only";
const DEVICE_GPU: &str = "gpu";
const GPU_GENOTYPE_FORMAT_PACKED8: &str = "packed8";
const JAX_CUDA_PLATFORM_NAME: &str = "cuda";
const JAX_CPU_PLATFORM_NAME: &str = "cpu";
const JAX_MATMUL_PRECISION_FLOAT32: &str = "float32";
const MULTI_PHENOTYPE_SAMPLE_MODE_COMPLETE_CASE: &str = "complete-case";
const MULTI_PHENOTYPE_SAMPLE_MODE_PER_PHENOTYPE: &str = "per-phenotype";
const PHENOTYPE_COMPUTE_GROUP_MODE_COMPLETE_CASE: &str = "complete-case";
const PHENOTYPE_COMPUTE_GROUP_MODE_PER_PHENOTYPE_COMPATIBLE: &str = "per-phenotype-compatible";
const PHENOTYPE_COMPUTE_GROUP_MODE_SINGLE_PHENOTYPE: &str = "single-phenotype";
const PHENOTYPE_DIRECTORY_MAXIMUM_SLUG_LENGTH: usize = 80;
const REGENIE_TRAIT_TYPE_BINARY: &str = "binary";
const XLA_AUXILIARY_CACHE_DISABLED: &str = "none";
const XLA_AUXILIARY_CACHE_PER_FUSION_AUTOTUNE: &str = "xla_gpu_per_fusion_autotune_cache_dir";

#[derive(Debug, PartialEq, Eq)]
pub enum HostPolicyError {
    NotImplemented(String),
    Value(String),
}

#[derive(Debug, PartialEq, Eq)]
pub struct AssociationBackendPlanPayload {
    pub backend_kind: &'static str,
    pub association_mode: String,
    pub jax_device: String,
    pub genotype_format: String,
    pub uses_variant_major_packed8_delivery: bool,
}

#[derive(Debug, PartialEq)]
pub struct BinaryCorrectionPlanPayload {
    pub method: &'static str,
    pub p_threshold: f64,
    pub firth_se: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub struct PhenotypeComputeGroupPayload {
    pub group_mode: &'static str,
    pub phenotype_indices: Vec<i64>,
    pub phenotype_names: Vec<String>,
    pub sample_mode: &'static str,
    pub sample_set_fingerprint: Option<String>,
    pub covariate_design_fingerprint: Option<String>,
    pub prediction_alignment_fingerprint: Option<String>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct JaxRuntimeSetupPayload {
    pub requested_device: String,
    pub platform_name: &'static str,
    pub cache_directory: String,
    pub matmul_precision: String,
    pub persistent_cache_enabled: bool,
    pub persistent_cache_min_entry_size_bytes: i64,
    pub persistent_cache_min_compile_time_seconds: i64,
    pub xla_auxiliary_cache_mode: &'static str,
    pub xla_auxiliary_cache_reason: &'static str,
    pub transfer_guard_enabled: bool,
    pub gpu_validation_status: &'static str,
    pub gpu_validation_message: Option<&'static str>,
}

#[must_use]
pub fn plan_association_backend(
    association_mode: &str,
    jax_device: &str,
    gpu_genotype_format: &str,
) -> AssociationBackendPlanPayload {
    let uses_variant_major_packed8_delivery = gpu_genotype_format == GPU_GENOTYPE_FORMAT_PACKED8;
    let backend_kind = if uses_variant_major_packed8_delivery {
        ASSOCIATION_BACKEND_JAX_PACKED8
    } else {
        ASSOCIATION_BACKEND_JAX_DOSAGE
    };
    AssociationBackendPlanPayload {
        backend_kind,
        association_mode: association_mode.to_string(),
        jax_device: jax_device.to_string(),
        genotype_format: gpu_genotype_format.to_string(),
        uses_variant_major_packed8_delivery,
    }
}

#[must_use]
pub fn resolve_association_mode(trait_type: &str) -> &'static str {
    if trait_type == REGENIE_TRAIT_TYPE_BINARY {
        ASSOCIATION_MODE_REGENIE2_BINARY
    } else {
        ASSOCIATION_MODE_REGENIE2_LINEAR
    }
}

#[allow(clippy::fn_params_excessive_bools)]
pub fn normalize_binary_correction(
    firth: bool,
    approx: bool,
    spa: bool,
    p_threshold: f64,
    firth_se: bool,
) -> Result<BinaryCorrectionPlanPayload, HostPolicyError> {
    if !(p_threshold > 0.0 && p_threshold < 1.0) {
        return Err(HostPolicyError::Value("pThresh must be in (0, 1).".to_string()));
    }
    if spa {
        return Err(HostPolicyError::NotImplemented(
            "SPA fallback is not implemented yet. Omit --spa for score-test-only output.".to_string(),
        ));
    }
    if approx && !firth {
        return Err(HostPolicyError::Value("--approx requires --firth.".to_string()));
    }
    if firth && approx {
        return Ok(BinaryCorrectionPlanPayload {
            method: BINARY_FALLBACK_METHOD_FIRTH_APPROXIMATE,
            p_threshold,
            firth_se,
        });
    }
    if firth {
        return Err(HostPolicyError::NotImplemented(
            "Exact REGENIE --firth without --approx is not implemented yet. Use --firth --approx.".to_string(),
        ));
    }
    Ok(BinaryCorrectionPlanPayload { method: BINARY_FALLBACK_METHOD_SCORE_ONLY, p_threshold, firth_se: false })
}

pub fn build_phenotype_compute_groups(
    phenotype_names: &[String],
    multi_phenotype_sample_mode: &str,
) -> Result<Vec<PhenotypeComputeGroupPayload>, HostPolicyError> {
    if phenotype_names.is_empty() {
        return Err(HostPolicyError::Value("At least one phenotype is required for execution planning.".to_string()));
    }
    if phenotype_names.len() == 1 {
        return Ok(vec![PhenotypeComputeGroupPayload {
            group_mode: PHENOTYPE_COMPUTE_GROUP_MODE_SINGLE_PHENOTYPE,
            phenotype_indices: vec![0],
            phenotype_names: phenotype_names.to_vec(),
            sample_mode: MULTI_PHENOTYPE_SAMPLE_MODE_PER_PHENOTYPE,
            sample_set_fingerprint: None,
            covariate_design_fingerprint: None,
            prediction_alignment_fingerprint: None,
        }]);
    }
    let phenotype_indices = (0..phenotype_names.len()).map(phenotype_index_to_i64).collect::<Vec<_>>();
    if multi_phenotype_sample_mode == MULTI_PHENOTYPE_SAMPLE_MODE_COMPLETE_CASE {
        return Ok(vec![PhenotypeComputeGroupPayload {
            group_mode: PHENOTYPE_COMPUTE_GROUP_MODE_COMPLETE_CASE,
            phenotype_indices,
            phenotype_names: phenotype_names.to_vec(),
            sample_mode: MULTI_PHENOTYPE_SAMPLE_MODE_COMPLETE_CASE,
            sample_set_fingerprint: None,
            covariate_design_fingerprint: None,
            prediction_alignment_fingerprint: None,
        }]);
    }
    Ok(phenotype_names
        .iter()
        .enumerate()
        .map(|(phenotype_index, phenotype_name)| PhenotypeComputeGroupPayload {
            group_mode: PHENOTYPE_COMPUTE_GROUP_MODE_PER_PHENOTYPE_COMPATIBLE,
            phenotype_indices: vec![phenotype_index_to_i64(phenotype_index)],
            phenotype_names: vec![phenotype_name.clone()],
            sample_mode: MULTI_PHENOTYPE_SAMPLE_MODE_PER_PHENOTYPE,
            sample_set_fingerprint: None,
            covariate_design_fingerprint: None,
            prediction_alignment_fingerprint: None,
        })
        .collect())
}

fn phenotype_index_to_i64(phenotype_index: usize) -> i64 {
    i64::try_from(phenotype_index).expect("phenotype count must fit in i64")
}

/// Build a deterministic identifier for one phenotype compute group.
///
/// # Panics
///
/// Panics only if serializing the internally constructed JSON value fails.
#[must_use]
pub fn build_phenotype_compute_group_id(
    group_mode: &str,
    phenotype_indices: &[i64],
    phenotype_names: &[String],
    sample_mode: &str,
    sample_set_fingerprint: Option<&str>,
    covariate_design_fingerprint: Option<&str>,
    prediction_alignment_fingerprint: Option<&str>,
) -> String {
    let mut group_payload = BTreeMap::new();
    group_payload.insert("covariate_design_fingerprint", optional_string_value(covariate_design_fingerprint));
    group_payload.insert("group_mode", Value::String(group_mode.to_string()));
    group_payload.insert("phenotype_indices", serde_json::json!(phenotype_indices));
    group_payload.insert("phenotype_names", serde_json::json!(phenotype_names));
    group_payload.insert("prediction_alignment_fingerprint", optional_string_value(prediction_alignment_fingerprint));
    group_payload.insert("sample_mode", Value::String(sample_mode.to_string()));
    group_payload.insert("sample_set_fingerprint", optional_string_value(sample_set_fingerprint));
    let group_payload_bytes = serde_json::to_vec(&group_payload).expect("group payload serialization must succeed");
    finalize_sha256_hex(Sha256::digest(group_payload_bytes))
}

#[must_use]
pub fn build_phenotype_output_directory_name(phenotype_index: i64, phenotype_name: &str) -> String {
    let mut sanitized_slug = String::new();
    let mut previous_character_was_replaced = false;
    for phenotype_character in phenotype_name.chars() {
        if phenotype_character.is_ascii_alphanumeric()
            || phenotype_character == '.'
            || phenotype_character == '_'
            || phenotype_character == '-'
        {
            sanitized_slug.push(phenotype_character);
            previous_character_was_replaced = false;
        } else if !previous_character_was_replaced {
            sanitized_slug.push('_');
            previous_character_was_replaced = true;
        }
    }
    let trimmed_slug = sanitized_slug.trim_matches(['.', '_', '-']).to_string();
    let fallback_slug = if trimmed_slug.is_empty() { "phenotype".to_string() } else { trimmed_slug };
    let truncated_slug = fallback_slug.chars().take(PHENOTYPE_DIRECTORY_MAXIMUM_SLUG_LENGTH).collect::<String>();
    format!("trait_{phenotype_index:04}_{truncated_slug}")
}

#[must_use]
pub fn resolve_jax_runtime_setup(
    requested_device: &str,
    cache_directory: &str,
    matmul_precision: Option<&str>,
    persistent_cache: bool,
    persistent_cache_min_entry_size_bytes: i64,
    persistent_cache_min_compile_time_seconds: i64,
    xla_autotune_cache: bool,
    transfer_guard: bool,
) -> JaxRuntimeSetupPayload {
    let (gpu_validation_status, gpu_validation_message) = if requested_device == DEVICE_GPU {
        ("pending", None)
    } else {
        ("skipped", Some("CPU runtime requested; GPU validation skipped."))
    };
    let platform_name = if requested_device == DEVICE_GPU { JAX_CUDA_PLATFORM_NAME } else { JAX_CPU_PLATFORM_NAME };
    let matmul_precision = matmul_precision.unwrap_or(JAX_MATMUL_PRECISION_FLOAT32).to_string();
    let (xla_auxiliary_cache_mode, xla_auxiliary_cache_reason) = if persistent_cache && xla_autotune_cache {
        (XLA_AUXILIARY_CACHE_PER_FUSION_AUTOTUNE, "XLA auxiliary cache was requested")
    } else if persistent_cache {
        (XLA_AUXILIARY_CACHE_DISABLED, "XLA auxiliary cache was not requested")
    } else {
        (XLA_AUXILIARY_CACHE_DISABLED, "persistent compilation cache is disabled")
    };
    JaxRuntimeSetupPayload {
        requested_device: requested_device.to_string(),
        platform_name,
        cache_directory: cache_directory.to_string(),
        matmul_precision,
        persistent_cache_enabled: persistent_cache,
        persistent_cache_min_entry_size_bytes,
        persistent_cache_min_compile_time_seconds,
        xla_auxiliary_cache_mode,
        xla_auxiliary_cache_reason,
        transfer_guard_enabled: transfer_guard,
        gpu_validation_status,
        gpu_validation_message,
    }
}

fn optional_string_value(value: Option<&str>) -> Value {
    match value {
        Some(text) => Value::String(text.to_string()),
        None => Value::Null,
    }
}

fn finalize_sha256_hex(digest_bytes: sha2::digest::Output<Sha256>) -> String {
    let mut digest_text = String::with_capacity(digest_bytes.len() * 2);
    for digest_byte in digest_bytes {
        write!(&mut digest_text, "{digest_byte:02x}").expect("writing to a string must succeed");
    }
    digest_text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plans_association_backend_from_concrete_genotype_format() {
        let dosage_plan = plan_association_backend(ASSOCIATION_MODE_REGENIE2_LINEAR, JAX_CPU_PLATFORM_NAME, "dosage");
        assert_eq!(dosage_plan.backend_kind, ASSOCIATION_BACKEND_JAX_DOSAGE);
        assert!(!dosage_plan.uses_variant_major_packed8_delivery);

        let packed_plan = plan_association_backend(
            ASSOCIATION_MODE_REGENIE2_BINARY,
            JAX_CUDA_PLATFORM_NAME,
            GPU_GENOTYPE_FORMAT_PACKED8,
        );
        assert_eq!(packed_plan.backend_kind, ASSOCIATION_BACKEND_JAX_PACKED8);
        assert_eq!(packed_plan.genotype_format, GPU_GENOTYPE_FORMAT_PACKED8);
        assert!(packed_plan.uses_variant_major_packed8_delivery);
    }

    #[test]
    fn normalizes_supported_binary_correction_modes() {
        let score_only_plan = normalize_binary_correction(false, false, false, 0.05, true).unwrap();
        assert_eq!(score_only_plan.method, BINARY_FALLBACK_METHOD_SCORE_ONLY);
        assert!(!score_only_plan.firth_se);

        let approximate_firth_plan = normalize_binary_correction(true, true, false, 0.01, true).unwrap();
        assert_eq!(approximate_firth_plan.method, BINARY_FALLBACK_METHOD_FIRTH_APPROXIMATE);
        assert!(approximate_firth_plan.firth_se);

        assert_eq!(
            normalize_binary_correction(false, true, false, 0.01, false),
            Err(HostPolicyError::Value("--approx requires --firth.".to_string())),
        );
    }

    #[test]
    fn builds_config_time_phenotype_compute_groups() {
        let single_groups =
            build_phenotype_compute_groups(&["height".to_string()], MULTI_PHENOTYPE_SAMPLE_MODE_PER_PHENOTYPE).unwrap();
        assert_eq!(single_groups.len(), 1);
        assert_eq!(single_groups[0].group_mode, PHENOTYPE_COMPUTE_GROUP_MODE_SINGLE_PHENOTYPE);
        assert_eq!(single_groups[0].phenotype_indices, vec![0]);

        let complete_case_groups = build_phenotype_compute_groups(
            &["height".to_string(), "weight".to_string()],
            MULTI_PHENOTYPE_SAMPLE_MODE_COMPLETE_CASE,
        )
        .unwrap();
        assert_eq!(complete_case_groups.len(), 1);
        assert_eq!(complete_case_groups[0].sample_mode, MULTI_PHENOTYPE_SAMPLE_MODE_COMPLETE_CASE);
        assert_eq!(complete_case_groups[0].phenotype_indices, vec![0, 1]);

        let per_phenotype_groups = build_phenotype_compute_groups(
            &["height".to_string(), "weight".to_string()],
            MULTI_PHENOTYPE_SAMPLE_MODE_PER_PHENOTYPE,
        )
        .unwrap();
        assert_eq!(per_phenotype_groups.len(), 2);
        assert_eq!(per_phenotype_groups[1].phenotype_names, vec!["weight".to_string()]);
    }

    #[test]
    fn compute_group_identifier_changes_with_alignment_fingerprints() {
        let base_identifier = build_phenotype_compute_group_id(
            PHENOTYPE_COMPUTE_GROUP_MODE_COMPLETE_CASE,
            &[0, 1],
            &["height".to_string(), "weight".to_string()],
            MULTI_PHENOTYPE_SAMPLE_MODE_COMPLETE_CASE,
            Some("samples-a"),
            Some("covariates-a"),
            Some("predictions-a"),
        );
        let changed_identifier = build_phenotype_compute_group_id(
            PHENOTYPE_COMPUTE_GROUP_MODE_COMPLETE_CASE,
            &[0, 1],
            &["height".to_string(), "weight".to_string()],
            MULTI_PHENOTYPE_SAMPLE_MODE_COMPLETE_CASE,
            Some("samples-b"),
            Some("covariates-a"),
            Some("predictions-a"),
        );
        assert_eq!(base_identifier.len(), 64);
        assert_ne!(base_identifier, changed_identifier);
    }

    #[test]
    fn resolves_jax_runtime_setup_payload() {
        let setup = resolve_jax_runtime_setup("gpu", "cache", None, true, 1024, 5, true, true);
        assert_eq!(setup.platform_name, JAX_CUDA_PLATFORM_NAME);
        assert_eq!(setup.matmul_precision, JAX_MATMUL_PRECISION_FLOAT32);
        assert_eq!(setup.xla_auxiliary_cache_mode, XLA_AUXILIARY_CACHE_PER_FUSION_AUTOTUNE);
        assert!(setup.transfer_guard_enabled);

        let cpu_setup = resolve_jax_runtime_setup("cpu", "cache", Some("highest"), false, 0, 0, true, false);
        assert_eq!(cpu_setup.platform_name, JAX_CPU_PLATFORM_NAME);
        assert_eq!(cpu_setup.gpu_validation_status, "skipped");
        assert_eq!(cpu_setup.matmul_precision, "highest");
    }
}
