use std::fmt::Write as _;

use ahash::{HashMap, HashMapExt, HashSet, HashSetExt};

use super::PredictionError;
use super::loco::LocoSampleIndex;

#[derive(Debug)]
pub(super) enum LocoSampleAlignment {
    Identity,
    Indices(Vec<usize>),
}

pub(super) fn validate_loco_sample_keys(loco_sample_index: &LocoSampleIndex) -> Result<(), PredictionError> {
    let sample_identifiers = loco_sample_index.identifiers();
    let mut observed_sample_keys = HashSet::with_capacity(sample_identifiers.len());
    for sample_key in sample_identifiers {
        if !observed_sample_keys.insert(sample_key) {
            return Err(PredictionError::DuplicateLocoSampleKey { sample_key: sample_key.to_string() });
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
    validate_target_sample_keys(target_family_identifiers, target_individual_identifiers)?;

    let source_sample_identifiers = loco_sample_index.identifiers();
    if target_sample_indices.len() == source_sample_identifiers.len()
        && target_sample_indices.iter().zip(source_sample_identifiers).all(
            |(target_sample_index, source_identifier)| {
                source_identifier
                    .strip_prefix(target_family_identifiers[*target_sample_index].as_str())
                    .and_then(|remaining_identifier| remaining_identifier.strip_prefix('_'))
                    == Some(target_individual_identifiers[*target_sample_index].as_str())
            },
        )
    {
        return Ok(LocoSampleAlignment::Identity);
    }
    let source_sample_identifiers = loco_sample_index.identifiers();
    let mut loco_lookup = HashMap::with_capacity(source_sample_identifiers.len());
    for (sample_index, sample_identifier) in source_sample_identifiers.enumerate() {
        loco_lookup.insert(sample_identifier, sample_index);
    }

    let mut alignment_indices = Vec::with_capacity(target_sample_indices.len());
    let mut missing_samples = Vec::new();
    let mut sample_key = String::new();
    for target_sample_index in target_sample_indices {
        let family_identifier = &target_family_identifiers[*target_sample_index];
        let individual_identifier = &target_individual_identifiers[*target_sample_index];
        sample_key.clear();
        let _ = write!(sample_key, "{family_identifier}_{individual_identifier}");
        if let Some(sample_index) = loco_lookup.get(sample_key.as_str()) {
            alignment_indices.push(*sample_index);
        } else {
            missing_samples.push(sample_key.clone());
        }
    }

    if !missing_samples.is_empty() {
        return Err(PredictionError::MissingTargetSamples(format_missing_samples(&missing_samples)));
    }
    Ok(LocoSampleAlignment::Indices(alignment_indices))
}

fn validate_target_sample_keys(
    target_family_identifiers: &[String],
    target_individual_identifiers: &[String],
) -> Result<(), PredictionError> {
    let mut observed_sample_keys = HashMap::new();
    for (sample_index, (family_identifier, individual_identifier)) in
        target_family_identifiers.iter().zip(target_individual_identifiers).enumerate()
    {
        // A serialization collision requires an underscore inside at least one component.
        if !family_identifier.contains('_') && !individual_identifier.contains('_') {
            continue;
        }
        let sample_key = format!("{family_identifier}_{individual_identifier}");
        if let Some(previous_sample_index) = observed_sample_keys.insert(sample_key.clone(), sample_index)
            && (target_family_identifiers[previous_sample_index] != *family_identifier
                || target_individual_identifiers[previous_sample_index] != *individual_identifier)
        {
            return Err(PredictionError::AmbiguousTargetSampleKey { sample_key });
        }
    }
    Ok(())
}

fn format_missing_samples(missing_samples: &[String]) -> String {
    let mut sample_list = missing_samples.iter().take(5).cloned().collect::<Vec<String>>().join(", ");
    if missing_samples.len() > 5 {
        let _ = write!(sample_list, ", ... ({} total)", missing_samples.len());
    }
    sample_list
}
