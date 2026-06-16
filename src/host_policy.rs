//! Deterministic host-side planning policy shared through the Python boundary.

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
pub(crate) enum HostPolicyError {
    NotImplemented(String),
    Value(String),
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct AssociationBackendPlanPayload {
    pub(crate) backend_kind: &'static str,
    pub(crate) association_mode: String,
    pub(crate) jax_device: String,
    pub(crate) genotype_format: String,
    pub(crate) uses_variant_major_packed8_delivery: bool,
}

#[derive(Debug, PartialEq)]
pub(crate) struct BinaryCorrectionPlanPayload {
    pub(crate) method: &'static str,
    pub(crate) p_threshold: f64,
    pub(crate) firth_se: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct PhenotypeComputeGroupPayload {
    pub(crate) group_mode: &'static str,
    pub(crate) phenotype_indices: Vec<i64>,
    pub(crate) phenotype_names: Vec<String>,
    pub(crate) sample_mode: &'static str,
    pub(crate) sample_set_fingerprint: Option<String>,
    pub(crate) covariate_design_fingerprint: Option<String>,
    pub(crate) prediction_alignment_fingerprint: Option<String>,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct JaxRuntimeSetupPayload {
    pub(crate) requested_device: String,
    pub(crate) platform_name: &'static str,
    pub(crate) cache_directory: String,
    pub(crate) matmul_precision: String,
    pub(crate) persistent_cache_enabled: bool,
    pub(crate) persistent_cache_min_entry_size_bytes: i64,
    pub(crate) persistent_cache_min_compile_time_seconds: i64,
    pub(crate) xla_auxiliary_cache_mode: &'static str,
    pub(crate) xla_auxiliary_cache_reason: &'static str,
    pub(crate) transfer_guard_enabled: bool,
    pub(crate) gpu_validation_status: &'static str,
    pub(crate) gpu_validation_message: Option<&'static str>,
}

pub(crate) fn plan_association_backend(
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

pub(crate) fn resolve_association_mode(trait_type: &str) -> &'static str {
    if trait_type == REGENIE_TRAIT_TYPE_BINARY {
        ASSOCIATION_MODE_REGENIE2_BINARY
    } else {
        ASSOCIATION_MODE_REGENIE2_LINEAR
    }
}

pub(crate) fn normalize_binary_correction(
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

pub(crate) fn build_phenotype_compute_groups(
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
    let phenotype_indices =
        (0..phenotype_names.len()).map(|phenotype_index| phenotype_index as i64).collect::<Vec<_>>();
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
            phenotype_indices: vec![phenotype_index as i64],
            phenotype_names: vec![phenotype_name.clone()],
            sample_mode: MULTI_PHENOTYPE_SAMPLE_MODE_PER_PHENOTYPE,
            sample_set_fingerprint: None,
            covariate_design_fingerprint: None,
            prediction_alignment_fingerprint: None,
        })
        .collect())
}

pub(crate) fn build_phenotype_compute_group_id(
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

pub(crate) fn build_phenotype_output_directory_name(phenotype_index: i64, phenotype_name: &str) -> String {
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

pub(crate) fn resolve_jax_runtime_setup(
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
