use std::collections::HashMap;

use super::types::{SampleAlignmentError, SampleKeyMode};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) enum SampleKey {
    Iid(String),
    FidIid { family_identifier: String, individual_identifier: String },
}

pub(super) fn sample_key_mode_value(sample_key_mode: SampleKeyMode) -> &'static str {
    match sample_key_mode {
        SampleKeyMode::Iid => "iid",
        SampleKeyMode::FidIid => "fid_iid",
    }
}

pub(super) fn validate_sample_identifier_lengths(
    sample_indices: &[i64],
    family_identifiers: &[String],
    individual_identifiers: &[String],
) -> Result<(), SampleAlignmentError> {
    if sample_indices.len() != family_identifiers.len() || sample_indices.len() != individual_identifiers.len() {
        return Err(SampleAlignmentError::new(format!(
            "Sample alignment arrays must have equal length: sample_indices={}, family_identifiers={}, individual_identifiers={}.",
            sample_indices.len(),
            family_identifiers.len(),
            individual_identifiers.len(),
        )));
    }
    Ok(())
}

pub(super) fn validate_sample_identifier_keys(
    sample_key_mode: SampleKeyMode,
    family_identifiers: &[String],
    individual_identifiers: &[String],
) -> Result<(), SampleAlignmentError> {
    match sample_key_mode {
        SampleKeyMode::Iid => {
            reject_duplicate_individual_identifiers(individual_identifiers, "BGEN/sample identifiers")?;
        }
        SampleKeyMode::FidIid => {
            reject_duplicate_sample_keys(family_identifiers, individual_identifiers, "BGEN/sample identifiers")?;
        }
    }
    Ok(())
}

pub(super) fn build_sample_key(
    sample_key_mode: SampleKeyMode,
    family_identifier: &str,
    individual_identifier: &str,
) -> SampleKey {
    match sample_key_mode {
        SampleKeyMode::Iid => SampleKey::Iid(individual_identifier.to_string()),
        SampleKeyMode::FidIid => SampleKey::FidIid {
            family_identifier: family_identifier.to_string(),
            individual_identifier: individual_identifier.to_string(),
        },
    }
}

pub(super) fn build_sample_row_indices_by_key(
    sample_key_mode: SampleKeyMode,
    family_identifiers: &[String],
    individual_identifiers: &[String],
) -> HashMap<SampleKey, usize> {
    let mut sample_row_indices_by_key = HashMap::with_capacity(individual_identifiers.len());
    for (sample_array_index, (family_identifier, individual_identifier)) in
        family_identifiers.iter().zip(individual_identifiers.iter()).enumerate()
    {
        if individual_identifier.is_empty() {
            continue;
        }
        let sample_key = build_sample_key(sample_key_mode, family_identifier, individual_identifier);
        sample_row_indices_by_key.insert(sample_key, sample_array_index);
    }
    sample_row_indices_by_key
}

pub(super) fn duplicate_table_sample_key_error(
    source_name: &str,
    sample_key_mode: SampleKeyMode,
    family_identifier: &str,
    individual_identifier: &str,
) -> String {
    if sample_key_mode == SampleKeyMode::FidIid {
        return format!(
            "Duplicate sample key '{family_identifier}_{individual_identifier}' found in {source_name}; sample_key_mode='fid_iid' requires unique (FID, IID) values."
        );
    }
    format!(
        "Duplicate IID '{individual_identifier}' found in {source_name}; sample_key_mode='iid' requires unique non-null IID values."
    )
}

fn reject_duplicate_individual_identifiers(
    individual_identifiers: &[String],
    source_name: &str,
) -> Result<(), SampleAlignmentError> {
    let mut observed_identifiers: HashMap<&str, usize> = HashMap::new();
    for individual_identifier in individual_identifiers {
        if individual_identifier.is_empty() {
            continue;
        }
        let occurrence_count = observed_identifiers.entry(individual_identifier.as_str()).or_insert(0);
        *occurrence_count += 1;
        if *occurrence_count > 1 {
            return Err(SampleAlignmentError::new(format!(
                "Duplicate IID '{individual_identifier}' found in {source_name}; sample_key_mode='iid' requires unique non-null IID values. Use sample_key_mode='fid_iid' for datasets with non-globally-unique IID."
            )));
        }
    }
    Ok(())
}

fn reject_duplicate_sample_keys(
    family_identifiers: &[String],
    individual_identifiers: &[String],
    source_name: &str,
) -> Result<(), SampleAlignmentError> {
    let mut observed_identifiers: HashMap<(&str, &str), usize> = HashMap::new();
    for (family_identifier, individual_identifier) in family_identifiers.iter().zip(individual_identifiers.iter()) {
        if individual_identifier.is_empty() {
            continue;
        }
        let sample_key = (family_identifier.as_str(), individual_identifier.as_str());
        let occurrence_count = observed_identifiers.entry(sample_key).or_insert(0);
        *occurrence_count += 1;
        if *occurrence_count > 1 {
            return Err(SampleAlignmentError::new(format!(
                "Duplicate sample key '{family_identifier}_{individual_identifier}' found in {source_name}; sample_key_mode='fid_iid' requires unique (FID, IID) values."
            )));
        }
    }
    Ok(())
}
