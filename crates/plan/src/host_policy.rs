//! Deterministic host-side planning policy.

use std::collections::BTreeMap;

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::request::PhenotypeComputeGroup;

const PHENOTYPE_DIRECTORY_MAXIMUM_SLUG_LENGTH: usize = 80;

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
        Value::String(phenotype_compute_group.covariate_design_fingerprint.clone()),
    );
    group_payload.insert("group_mode", Value::String(phenotype_compute_group.group_mode.as_str().to_string()));
    group_payload.insert("phenotype_indices", serde_json::json!(&phenotype_compute_group.phenotype_indices));
    group_payload.insert("phenotype_names", serde_json::json!(&phenotype_compute_group.phenotype_names));
    group_payload.insert(
        "phenotype_design_fingerprint",
        Value::String(phenotype_compute_group.phenotype_design_fingerprint.clone()),
    );
    group_payload.insert(
        "prediction_alignment_fingerprint",
        Value::String(phenotype_compute_group.prediction_alignment_fingerprint.clone()),
    );
    group_payload.insert("sample_mode", Value::String(phenotype_compute_group.sample_mode.as_str().to_string()));
    group_payload
        .insert("sample_set_fingerprint", Value::String(phenotype_compute_group.sample_set_fingerprint.clone()));
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
