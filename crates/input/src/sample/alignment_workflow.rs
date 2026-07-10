use std::path::Path;

use crate::error::InputResult;
use crate::regenie::PredictionSourceLoader;

use super::alignment::build_aligned_phenotype_group_drafts;
use super::fingerprints::build_phenotype_compute_group;
use super::keys::{
    build_sample_row_indices_by_key, validate_sample_identifier_keys, validate_sample_identifier_lengths,
};
use super::tables::{load_covariate_table, multi_phenotype_parse_candidate_mask, read_multi_phenotype_table};
use super::types::{AlignedPhenotypeGroup, AlignedPhenotypeGroupDraft, PhenotypeGroupLoadRequest};

pub fn load_aligned_phenotype_groups(request: &PhenotypeGroupLoadRequest) -> InputResult<Vec<AlignedPhenotypeGroup>> {
    if request.phenotype_names.is_empty() {
        return Err("At least one phenotype is required for alignment.".to_string().into());
    }
    validate_sample_identifier_lengths(
        &request.sample_identifiers.sample_indices,
        &request.sample_identifiers.family_identifiers,
        &request.sample_identifiers.individual_identifiers,
    )?;
    validate_sample_identifier_keys(
        request.sample_key_mode,
        &request.sample_identifiers.family_identifiers,
        &request.sample_identifiers.individual_identifiers,
    )?;

    let sample_row_indices_by_key = build_sample_row_indices_by_key(
        request.sample_key_mode,
        &request.sample_identifiers.family_identifiers,
        &request.sample_identifiers.individual_identifiers,
    );
    let phenotype_table = read_multi_phenotype_table(
        Path::new(&request.phenotype_path),
        &request.phenotype_names,
        request.is_binary_trait,
        request.sample_key_mode,
        &sample_row_indices_by_key,
        request.sample_identifiers.sample_indices.len(),
    )?;
    let parse_candidate_mask = multi_phenotype_parse_candidate_mask(&phenotype_table);
    let covariate_table = load_covariate_table(
        request.covariate_path.as_deref(),
        request.covariate_names.as_deref(),
        request.sample_key_mode,
        &sample_row_indices_by_key,
        &parse_candidate_mask,
        request.sample_identifiers.sample_indices.len(),
    )?;
    let group_drafts = build_aligned_phenotype_group_drafts(request, &phenotype_table, &covariate_table)?;
    let mut prediction_source_loader = PredictionSourceLoader::new(Path::new(&request.prediction_list_path))?;

    group_drafts
        .into_iter()
        .map(|draft| build_aligned_phenotype_group(request, draft, &mut prediction_source_loader))
        .collect()
}

fn build_aligned_phenotype_group(
    request: &PhenotypeGroupLoadRequest,
    draft: AlignedPhenotypeGroupDraft,
    prediction_source_loader: &mut PredictionSourceLoader,
) -> InputResult<AlignedPhenotypeGroup> {
    let family_identifiers = draft
        .sample_array_indices
        .iter()
        .map(|sample_array_index| request.sample_identifiers.family_identifiers[*sample_array_index].as_str())
        .collect::<Vec<_>>();
    let individual_identifiers = draft
        .sample_array_indices
        .iter()
        .map(|sample_array_index| request.sample_identifiers.individual_identifiers[*sample_array_index].as_str())
        .collect::<Vec<_>>();
    let phenotype_names = draft
        .phenotype_indices
        .iter()
        .map(|phenotype_index| request.phenotype_names[*phenotype_index].clone())
        .collect::<Vec<_>>();
    let prediction_source = prediction_source_loader.load(
        &phenotype_names,
        &family_identifiers,
        &individual_identifiers,
        request.sample_key_mode,
    )?;
    let phenotype_group = build_phenotype_compute_group(request, &draft, &family_identifiers, &individual_identifiers)?;
    let sample_count = draft.sample_indices.len();
    let phenotype_value_count = phenotype_group
        .phenotype_names
        .len()
        .checked_mul(sample_count)
        .ok_or_else(|| "Aligned phenotype matrix dimensions overflowed usize.".to_string())?;
    if draft.phenotype_values.len() != phenotype_value_count {
        return Err(format!(
            "Aligned phenotype matrix contains {} values, expected {phenotype_value_count}.",
            draft.phenotype_values.len()
        )
        .into());
    }
    let covariate_value_count = sample_count
        .checked_mul(draft.covariate_names.len())
        .ok_or_else(|| "Aligned covariate matrix dimensions overflowed usize.".to_string())?;
    if draft.covariate_values.len() != covariate_value_count {
        return Err(format!(
            "Aligned covariate matrix contains {} values, expected {covariate_value_count}.",
            draft.covariate_values.len()
        )
        .into());
    }
    Ok(AlignedPhenotypeGroup {
        phenotype_group,
        sample_indices: draft.sample_indices,
        phenotype_values: draft.phenotype_values,
        covariate_names: draft.covariate_names,
        covariate_values: draft.covariate_values,
        prediction_source,
    })
}
