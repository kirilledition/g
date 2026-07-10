use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::Arc;

use g_plan::SampleKeyMode;

use super::PredictionError;
use super::loco::LocoSampleIndex;

pub(super) fn align_prediction_values(prediction_values: &Arc<[f32]>, alignment_indices: &[usize]) -> Arc<[f32]> {
    if is_identity_alignment(alignment_indices, prediction_values.len()) {
        return Arc::clone(prediction_values);
    }
    alignment_indices.iter().map(|sample_index| prediction_values[*sample_index]).collect::<Vec<f32>>().into()
}

pub(super) fn validate_target_sample_keys(
    target_family_identifiers: &[&str],
    target_individual_identifiers: &[&str],
) -> Result<(), PredictionError> {
    if target_family_identifiers.len() != target_individual_identifiers.len() {
        return Err(PredictionError::TargetSampleLengthMismatch);
    }
    let mut observed_sample_keys = HashMap::with_capacity(target_family_identifiers.len());
    for (family_identifier, individual_identifier) in
        target_family_identifiers.iter().zip(target_individual_identifiers.iter())
    {
        let sample_key = (*family_identifier, *individual_identifier);
        let occurrence_count = observed_sample_keys.entry(sample_key).or_insert(0);
        *occurrence_count += 1;
        if *occurrence_count > 1 {
            return Err(PredictionError::DuplicateTargetSampleKey {
                sample_key: format!("{family_identifier}_{individual_identifier}"),
            });
        }
    }
    Ok(())
}

pub(super) fn validate_loco_sample_keys(loco_sample_index: &LocoSampleIndex) -> Result<(), PredictionError> {
    let mut observed_sample_keys = HashMap::with_capacity(loco_sample_index.family_identifiers.len());
    for (family_identifier, individual_identifier) in
        loco_sample_index.family_identifiers.iter().zip(loco_sample_index.individual_identifiers.iter())
    {
        let sample_key = (family_identifier.as_str(), individual_identifier.as_str());
        let occurrence_count = observed_sample_keys.entry(sample_key).or_insert(0);
        *occurrence_count += 1;
        if *occurrence_count > 1 {
            return Err(PredictionError::DuplicateLocoSampleKey {
                sample_key: format!("{family_identifier}_{individual_identifier}"),
            });
        }
    }
    Ok(())
}

pub(super) fn validate_unique_target_individual_identifiers(
    target_individual_identifiers: &[&str],
) -> Result<(), PredictionError> {
    let mut observed_individual_identifiers = HashMap::with_capacity(target_individual_identifiers.len());
    for individual_identifier in target_individual_identifiers {
        if individual_identifier.is_empty() {
            return Err(PredictionError::EmptyTargetIid);
        }
        let occurrence_count = observed_individual_identifiers.entry(*individual_identifier).or_insert(0);
        *occurrence_count += 1;
        if *occurrence_count > 1 {
            return Err(PredictionError::DuplicateTargetIid {
                individual_identifier: (*individual_identifier).to_string(),
            });
        }
    }
    Ok(())
}

pub(super) fn validate_unique_loco_individual_identifiers(
    loco_sample_index: &LocoSampleIndex,
) -> Result<(), PredictionError> {
    let mut observed_individual_identifiers = HashMap::with_capacity(loco_sample_index.individual_identifiers.len());
    for individual_identifier in &loco_sample_index.individual_identifiers {
        if individual_identifier.is_empty() {
            return Err(PredictionError::EmptyLocoIid);
        }
        let occurrence_count = observed_individual_identifiers.entry(individual_identifier.as_str()).or_insert(0);
        *occurrence_count += 1;
        if *occurrence_count > 1 {
            return Err(PredictionError::DuplicateLocoIid { individual_identifier: individual_identifier.clone() });
        }
    }
    Ok(())
}

pub(super) fn build_sample_alignment_indices(
    loco_sample_index: &LocoSampleIndex,
    target_family_identifiers: &[&str],
    target_individual_identifiers: &[&str],
    sample_key_mode: SampleKeyMode,
) -> Result<Vec<usize>, PredictionError> {
    if target_family_identifiers.len() != target_individual_identifiers.len() {
        return Err(PredictionError::TargetSampleLengthMismatch);
    }

    if sample_key_mode == SampleKeyMode::Iid {
        return build_individual_identifier_alignment_indices(loco_sample_index, target_individual_identifiers);
    }
    build_family_individual_identifier_alignment_indices(
        loco_sample_index,
        target_family_identifiers,
        target_individual_identifiers,
    )
}

fn build_individual_identifier_alignment_indices(
    loco_sample_index: &LocoSampleIndex,
    target_individual_identifiers: &[&str],
) -> Result<Vec<usize>, PredictionError> {
    let mut loco_lookup = HashMap::with_capacity(loco_sample_index.individual_identifiers.len());
    for (sample_index, individual_identifier) in loco_sample_index.individual_identifiers.iter().enumerate() {
        loco_lookup.insert(individual_identifier.as_str(), sample_index);
    }

    let mut alignment_indices = Vec::with_capacity(target_individual_identifiers.len());
    let mut missing_samples = Vec::new();
    for individual_identifier in target_individual_identifiers {
        if let Some(sample_index) = loco_lookup.get(individual_identifier) {
            alignment_indices.push(*sample_index);
        } else {
            missing_samples.push((*individual_identifier).to_string());
        }
    }

    if !missing_samples.is_empty() {
        return Err(PredictionError::MissingTargetSamples(format_missing_samples(&missing_samples)));
    }
    Ok(alignment_indices)
}

fn build_family_individual_identifier_alignment_indices(
    loco_sample_index: &LocoSampleIndex,
    target_family_identifiers: &[&str],
    target_individual_identifiers: &[&str],
) -> Result<Vec<usize>, PredictionError> {
    let mut loco_lookup = HashMap::with_capacity(loco_sample_index.family_identifiers.len());
    for (sample_index, (family_identifier, individual_identifier)) in
        loco_sample_index.family_identifiers.iter().zip(loco_sample_index.individual_identifiers.iter()).enumerate()
    {
        loco_lookup.insert((family_identifier.as_str(), individual_identifier.as_str()), sample_index);
    }

    let mut alignment_indices = Vec::with_capacity(target_family_identifiers.len());
    let mut missing_samples = Vec::new();
    for (family_identifier, individual_identifier) in
        target_family_identifiers.iter().zip(target_individual_identifiers.iter())
    {
        let key = (*family_identifier, *individual_identifier);
        if let Some(sample_index) = loco_lookup.get(&key) {
            alignment_indices.push(*sample_index);
        } else {
            missing_samples.push(format!("{family_identifier}_{individual_identifier}"));
        }
    }

    if !missing_samples.is_empty() {
        return Err(PredictionError::MissingTargetSamples(format_missing_samples(&missing_samples)));
    }
    Ok(alignment_indices)
}

fn is_identity_alignment(alignment_indices: &[usize], source_sample_count: usize) -> bool {
    alignment_indices.len() == source_sample_count
        && alignment_indices
            .iter()
            .enumerate()
            .all(|(expected_sample_index, sample_index)| expected_sample_index == *sample_index)
}

fn format_missing_samples(missing_samples: &[String]) -> String {
    let mut sample_list = missing_samples.iter().take(5).cloned().collect::<Vec<String>>().join(", ");
    if missing_samples.len() > 5 {
        let _ = write!(sample_list, ", ... ({} total)", missing_samples.len());
    }
    sample_list
}
