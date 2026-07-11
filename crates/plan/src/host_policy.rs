//! Deterministic host-side planning policy.

#![allow(clippy::missing_errors_doc)]

use std::collections::BTreeMap;

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::enums::{MultiPhenotypeSampleMode, PhenotypeComputeGroupMode};
use crate::request::PhenotypeComputeGroup;

const PHENOTYPE_DIRECTORY_MAXIMUM_SLUG_LENGTH: usize = 80;

pub fn build_phenotype_compute_groups(
    phenotype_names: &[String],
    multi_phenotype_sample_mode: MultiPhenotypeSampleMode,
) -> Result<Vec<PhenotypeComputeGroup>, String> {
    if phenotype_names.is_empty() {
        return Err("At least one phenotype is required for execution planning.".to_string());
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
            u32::try_from(phenotype_index).map_err(|_| "Phenotype count exceeds native u32 capacity.".to_string())
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
                    u32::try_from(phenotype_index)
                        .map_err(|_| "Phenotype count exceeds native u32 capacity.".to_string())?,
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
    hex::encode(Sha256::digest(group_payload_bytes))
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
