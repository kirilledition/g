use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use super::PredictionError;
use super::loco::LocoSampleIndex;

#[derive(Debug)]
pub(super) enum LocoSampleAlignment {
    Identity,
    Indices(Vec<usize>),
}

pub(super) fn validate_loco_sample_keys(loco_sample_index: &LocoSampleIndex) -> Result<(), PredictionError> {
    let mut observed_sample_keys = HashSet::with_capacity(loco_sample_index.family_identifiers.len());
    for (family_identifier, individual_identifier) in
        loco_sample_index.family_identifiers.iter().zip(loco_sample_index.individual_identifiers.iter())
    {
        let sample_key = (family_identifier.as_str(), individual_identifier.as_str());
        if !observed_sample_keys.insert(sample_key) {
            return Err(PredictionError::DuplicateLocoSampleKey {
                sample_key: format!("{family_identifier}_{individual_identifier}"),
            });
        }
    }
    Ok(())
}

pub(super) fn build_sample_alignment(
    loco_sample_index: &LocoSampleIndex,
    target_family_identifiers: &[String],
    target_individual_identifiers: &[String],
    target_sample_indices: &[usize],
) -> Result<LocoSampleAlignment, PredictionError> {
    if target_family_identifiers.len() != target_individual_identifiers.len() {
        return Err(PredictionError::TargetSampleLengthMismatch);
    }

    if target_sample_indices.len() == loco_sample_index.family_identifiers.len()
        && target_sample_indices
            .iter()
            .zip(loco_sample_index.family_identifiers.iter().zip(&loco_sample_index.individual_identifiers))
            .all(|(target_sample_index, (source_family, source_individual))| {
                target_family_identifiers[*target_sample_index] == *source_family
                    && target_individual_identifiers[*target_sample_index] == *source_individual
            })
    {
        return Ok(LocoSampleAlignment::Identity);
    }
    let mut loco_lookup = HashMap::with_capacity(loco_sample_index.family_identifiers.len());
    for (sample_index, (family_identifier, individual_identifier)) in
        loco_sample_index.family_identifiers.iter().zip(loco_sample_index.individual_identifiers.iter()).enumerate()
    {
        loco_lookup.insert((family_identifier.as_str(), individual_identifier.as_str()), sample_index);
    }

    let mut alignment_indices = Vec::with_capacity(target_sample_indices.len());
    let mut missing_samples = Vec::new();
    for target_sample_index in target_sample_indices {
        let family_identifier = &target_family_identifiers[*target_sample_index];
        let individual_identifier = &target_individual_identifiers[*target_sample_index];
        let key = (family_identifier.as_str(), individual_identifier.as_str());
        if let Some(sample_index) = loco_lookup.get(&key) {
            alignment_indices.push(*sample_index);
        } else {
            missing_samples.push(format!("{family_identifier}_{individual_identifier}"));
        }
    }

    if !missing_samples.is_empty() {
        return Err(PredictionError::MissingTargetSamples(format_missing_samples(&missing_samples)));
    }
    Ok(LocoSampleAlignment::Indices(alignment_indices))
}

fn format_missing_samples(missing_samples: &[String]) -> String {
    let mut sample_list = missing_samples.iter().take(5).cloned().collect::<Vec<String>>().join(", ");
    if missing_samples.len() > 5 {
        let _ = write!(sample_list, ", ... ({} total)", missing_samples.len());
    }
    sample_list
}
