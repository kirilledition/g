//! Deterministic host-side planning policy.

#![allow(clippy::missing_errors_doc)]

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::fmt::Write as _;

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::enums::{BinaryFallbackMethod, MultiPhenotypeSampleMode, PhenotypeComputeGroupMode};
use crate::request::{CorrectionPlan, PhenotypeComputeGroup};

const PHENOTYPE_DIRECTORY_MAXIMUM_SLUG_LENGTH: usize = 80;

#[derive(Debug, PartialEq, Eq)]
pub enum HostPolicyError {
    NotImplemented(String),
    Value(String),
}

impl fmt::Display for HostPolicyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotImplemented(message) | Self::Value(message) => formatter.write_str(message),
        }
    }
}

impl Error for HostPolicyError {}

#[allow(clippy::fn_params_excessive_bools)]
pub fn normalize_binary_correction(
    firth: bool,
    approx: bool,
    spa: bool,
    p_threshold: f64,
    firth_se: bool,
) -> Result<CorrectionPlan, HostPolicyError> {
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
        return Ok(CorrectionPlan { method: BinaryFallbackMethod::FirthApproximate, p_threshold, firth_se });
    }
    if firth {
        return Err(HostPolicyError::NotImplemented(
            "Exact REGENIE --firth without --approx is not implemented yet. Use --firth --approx.".to_string(),
        ));
    }
    Ok(CorrectionPlan { method: BinaryFallbackMethod::ScoreOnly, p_threshold, firth_se: false })
}

pub fn build_phenotype_compute_groups(
    phenotype_names: &[String],
    multi_phenotype_sample_mode: MultiPhenotypeSampleMode,
) -> Result<Vec<PhenotypeComputeGroup>, HostPolicyError> {
    if phenotype_names.is_empty() {
        return Err(HostPolicyError::Value("At least one phenotype is required for execution planning.".to_string()));
    }
    if phenotype_names.len() == 1 {
        return Ok(vec![PhenotypeComputeGroup {
            group_mode: PhenotypeComputeGroupMode::SinglePhenotype,
            phenotype_indices: vec![0],
            phenotype_names: phenotype_names.to_vec(),
            sample_mode: MultiPhenotypeSampleMode::PerPhenotype,
            sample_set_fingerprint: None,
            covariate_design_fingerprint: None,
            prediction_alignment_fingerprint: None,
        }]);
    }
    let phenotype_indices = (0..phenotype_names.len())
        .map(|phenotype_index| {
            u32::try_from(phenotype_index)
                .map_err(|_| HostPolicyError::Value("Phenotype count exceeds native u32 capacity.".to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if multi_phenotype_sample_mode == MultiPhenotypeSampleMode::CompleteCase {
        return Ok(vec![PhenotypeComputeGroup {
            group_mode: PhenotypeComputeGroupMode::CompleteCase,
            phenotype_indices,
            phenotype_names: phenotype_names.to_vec(),
            sample_mode: MultiPhenotypeSampleMode::CompleteCase,
            sample_set_fingerprint: None,
            covariate_design_fingerprint: None,
            prediction_alignment_fingerprint: None,
        }]);
    }
    phenotype_names
        .iter()
        .enumerate()
        .map(|(phenotype_index, phenotype_name)| {
            Ok(PhenotypeComputeGroup {
                group_mode: PhenotypeComputeGroupMode::PerPhenotypeCompatible,
                phenotype_indices: vec![
                    u32::try_from(phenotype_index).map_err(|_| {
                        HostPolicyError::Value("Phenotype count exceeds native u32 capacity.".to_string())
                    })?,
                ],
                phenotype_names: vec![phenotype_name.clone()],
                sample_mode: MultiPhenotypeSampleMode::PerPhenotype,
                sample_set_fingerprint: None,
                covariate_design_fingerprint: None,
                prediction_alignment_fingerprint: None,
            })
        })
        .collect()
}

/// Build a deterministic identifier for one phenotype compute group.
///
/// # Panics
///
/// Panics only if serializing the internally constructed JSON value fails.
#[must_use]
pub fn build_phenotype_compute_group_id(phenotype_compute_group: &PhenotypeComputeGroup) -> String {
    let mut group_payload = BTreeMap::new();
    group_payload.insert(
        "covariate_design_fingerprint",
        optional_string_value(phenotype_compute_group.covariate_design_fingerprint.as_deref()),
    );
    group_payload.insert("group_mode", Value::String(phenotype_compute_group.group_mode.as_str().to_string()));
    group_payload.insert("phenotype_indices", serde_json::json!(&phenotype_compute_group.phenotype_indices));
    group_payload.insert("phenotype_names", serde_json::json!(&phenotype_compute_group.phenotype_names));
    group_payload.insert(
        "prediction_alignment_fingerprint",
        optional_string_value(phenotype_compute_group.prediction_alignment_fingerprint.as_deref()),
    );
    group_payload.insert("sample_mode", Value::String(phenotype_compute_group.sample_mode.as_str().to_string()));
    group_payload.insert(
        "sample_set_fingerprint",
        optional_string_value(phenotype_compute_group.sample_set_fingerprint.as_deref()),
    );
    let group_payload_bytes = serde_json::to_vec(&group_payload).expect("group payload serialization must succeed");
    finalize_sha256_hex(Sha256::digest(group_payload_bytes))
}

#[must_use]
pub fn build_phenotype_output_directory_name(phenotype_index: u32, phenotype_name: &str) -> String {
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
