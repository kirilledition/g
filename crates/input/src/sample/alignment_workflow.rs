use std::path::Path;

use crate::error::InputResult;
use crate::regenie::PredictionSourceLoader;

use super::alignment::build_aligned_phenotype_group_drafts;
use super::fingerprints::build_phenotype_compute_group;
use super::keys::build_sample_row_indices_by_key;
use super::tables::{load_covariate_table, multi_phenotype_parse_candidate_mask, read_multi_phenotype_table};
use super::types::{AlignedPhenotypeGroup, AlignedPhenotypeGroupDraft, PhenotypeGroupLoadRequest};

pub fn load_aligned_phenotype_groups(
    request: &PhenotypeGroupLoadRequest<'_>,
) -> InputResult<Vec<AlignedPhenotypeGroup>> {
    if request.phenotype_names.is_empty() {
        return Err("At least one phenotype is required for alignment.".to_string().into());
    }
    let group_drafts = {
        let sample_row_indices_by_key = build_sample_row_indices_by_key(
            request.sample_identifiers.family_identifiers.as_slice(),
            request.sample_identifiers.individual_identifiers.as_slice(),
        )?;
        let phenotype_table = read_multi_phenotype_table(
            Path::new(request.phenotype_path),
            request.phenotype_names,
            request.is_binary_trait,
            &sample_row_indices_by_key,
            request.sample_identifiers.family_identifiers.len(),
        )?;
        let parse_candidate_mask = multi_phenotype_parse_candidate_mask(&phenotype_table);
        let covariate_table = load_covariate_table(
            request.covariate_path,
            request.covariate_names,
            &sample_row_indices_by_key,
            &parse_candidate_mask,
            request.sample_identifiers.family_identifiers.len(),
        )?;
        build_aligned_phenotype_group_drafts(request, &phenotype_table, &covariate_table)?
    };
    let mut prediction_source_loader = PredictionSourceLoader::new(request.prediction_loco_paths);

    group_drafts
        .into_iter()
        .map(|draft| build_aligned_phenotype_group(request, draft, &mut prediction_source_loader))
        .collect()
}

fn build_aligned_phenotype_group(
    request: &PhenotypeGroupLoadRequest<'_>,
    draft: AlignedPhenotypeGroupDraft,
    prediction_source_loader: &mut PredictionSourceLoader,
) -> InputResult<AlignedPhenotypeGroup> {
    let prediction_source = prediction_source_loader.load(
        &draft.phenotype_indices,
        request.sample_identifiers.family_identifiers.as_slice(),
        request.sample_identifiers.individual_identifiers.as_slice(),
        &draft.sample_array_indices,
    )?;
    let prediction_alignment_source_digest = prediction_source.alignment_source_digest();
    let phenotype_group = build_phenotype_compute_group(request, &draft, &prediction_alignment_source_digest)?;
    let sample_count = draft.sample_array_indices.len();
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
        sample_indices: draft.sample_array_indices,
        phenotype_values: draft.phenotype_values,
        covariate_names: draft.covariate_names,
        covariate_values: draft.covariate_values,
        prediction_source,
    })
}
