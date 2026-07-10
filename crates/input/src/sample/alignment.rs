use std::collections::HashMap;

use super::tables::{CovariateTable, MultiPhenotypeTable, is_complete_multi_phenotype_sample};
use super::types::{AlignedPhenotypeGroupDraft, PhenotypeGroupLoadRequest};

type SampleAlignmentResult<T> = Result<T, String>;

pub(super) fn build_aligned_phenotype_group_drafts(
    request: &PhenotypeGroupLoadRequest,
    phenotype_table: &MultiPhenotypeTable,
    covariate_table: &CovariateTable,
) -> SampleAlignmentResult<Vec<AlignedPhenotypeGroupDraft>> {
    if request.phenotype_names.len() == 1 || request.sample_mode == g_plan::MultiPhenotypeSampleMode::CompleteCase {
        let phenotype_indices = (0..phenotype_table.phenotype_count).collect::<Vec<_>>();
        let sample_array_indices = complete_case_sample_array_indices(
            &request.sample_identifiers.sample_indices,
            phenotype_table,
            covariate_table,
        );
        if sample_array_indices.is_empty() {
            return Err("No aligned samples remain after joining phenotype and covariate tables.".to_string());
        }
        return Ok(vec![build_group_draft(
            request,
            phenotype_table,
            covariate_table,
            phenotype_indices,
            sample_array_indices,
        )]);
    }

    build_compatible_sample_group_drafts(request, phenotype_table, covariate_table)
}

fn build_compatible_sample_group_drafts(
    request: &PhenotypeGroupLoadRequest,
    phenotype_table: &MultiPhenotypeTable,
    covariate_table: &CovariateTable,
) -> SampleAlignmentResult<Vec<AlignedPhenotypeGroupDraft>> {
    let mut group_index_by_sample_array_indices: HashMap<Vec<usize>, usize> = HashMap::new();
    let mut sample_array_indices_by_group: Vec<Vec<usize>> = Vec::new();
    let mut phenotype_indices_by_group: Vec<Vec<usize>> = Vec::new();
    let sorted_sample_array_indices =
        sorted_sample_array_indices_by_sample_index(&request.sample_identifiers.sample_indices);
    let mut complete_sample_array_indices = Vec::with_capacity(request.sample_identifiers.sample_indices.len());

    for phenotype_index in 0..phenotype_table.phenotype_count {
        collect_complete_trait_sample_array_indices(
            &sorted_sample_array_indices,
            phenotype_table,
            covariate_table,
            phenotype_index,
            &mut complete_sample_array_indices,
        );
        if complete_sample_array_indices.is_empty() {
            return Err(format!(
                "No aligned samples remain after joining phenotype '{}' and covariate tables.",
                request.phenotype_names[phenotype_index]
            ));
        }
        let group_index = match group_index_by_sample_array_indices.get(&complete_sample_array_indices) {
            Some(existing_group_index) => *existing_group_index,
            None => {
                let new_group_index = sample_array_indices_by_group.len();
                let stored_sample_array_indices = std::mem::take(&mut complete_sample_array_indices);
                group_index_by_sample_array_indices.insert(stored_sample_array_indices.clone(), new_group_index);
                sample_array_indices_by_group.push(stored_sample_array_indices);
                phenotype_indices_by_group.push(Vec::new());
                complete_sample_array_indices = Vec::with_capacity(request.sample_identifiers.sample_indices.len());
                new_group_index
            }
        };
        phenotype_indices_by_group[group_index].push(phenotype_index);
    }

    Ok(phenotype_indices_by_group
        .into_iter()
        .zip(sample_array_indices_by_group)
        .map(|(phenotype_indices, sample_array_indices)| {
            build_group_draft(request, phenotype_table, covariate_table, phenotype_indices, sample_array_indices)
        })
        .collect())
}

fn build_group_draft(
    request: &PhenotypeGroupLoadRequest,
    phenotype_table: &MultiPhenotypeTable,
    covariate_table: &CovariateTable,
    phenotype_indices: Vec<usize>,
    sample_array_indices: Vec<usize>,
) -> AlignedPhenotypeGroupDraft {
    let sample_count = sample_array_indices.len();
    let covariate_names = returned_covariate_names(&covariate_table.selected_covariate_names);
    let mut sample_indices = Vec::with_capacity(sample_count);
    let mut phenotype_values = Vec::with_capacity(phenotype_indices.len() * sample_count);
    let mut covariate_values = Vec::with_capacity(sample_count * covariate_names.len());

    for sample_array_index in &sample_array_indices {
        sample_indices.push(request.sample_identifiers.sample_indices[*sample_array_index]);
        push_covariate_row(&mut covariate_values, covariate_table, *sample_array_index);
    }
    for phenotype_index in &phenotype_indices {
        for sample_array_index in &sample_array_indices {
            let value_index = phenotype_index * phenotype_table.sample_count + sample_array_index;
            phenotype_values.push(phenotype_table.phenotype_values[value_index]);
        }
    }

    AlignedPhenotypeGroupDraft {
        phenotype_indices,
        sample_array_indices,
        sample_indices,
        phenotype_values,
        covariate_names,
        covariate_values,
    }
}

fn complete_case_sample_array_indices(
    sample_indices: &[usize],
    phenotype_table: &MultiPhenotypeTable,
    covariate_table: &CovariateTable,
) -> Vec<usize> {
    sorted_sample_array_indices_by_sample_index(sample_indices)
        .into_iter()
        .filter(|sample_array_index| {
            is_complete_multi_phenotype_sample(phenotype_table, *sample_array_index)
                && covariate_table.covariate_mask[*sample_array_index]
        })
        .collect()
}

fn sorted_sample_array_indices_by_sample_index(sample_indices: &[usize]) -> Vec<usize> {
    let mut sorted_sample_array_indices: Vec<usize> = (0..sample_indices.len()).collect();
    sorted_sample_array_indices.sort_by_key(|sample_array_index| sample_indices[*sample_array_index]);
    sorted_sample_array_indices
}

fn collect_complete_trait_sample_array_indices(
    sorted_sample_array_indices: &[usize],
    phenotype_table: &MultiPhenotypeTable,
    covariate_table: &CovariateTable,
    phenotype_index: usize,
    complete_sample_array_indices: &mut Vec<usize>,
) {
    complete_sample_array_indices.clear();
    complete_sample_array_indices.extend(sorted_sample_array_indices.iter().copied().filter(|sample_array_index| {
        let phenotype_mask_index = phenotype_index * phenotype_table.sample_count + sample_array_index;
        phenotype_table.phenotype_masks[phenotype_mask_index] && covariate_table.covariate_mask[*sample_array_index]
    }));
}

fn push_covariate_row(covariate_values: &mut Vec<f32>, covariate_table: &CovariateTable, sample_array_index: usize) {
    covariate_values.push(1.0);
    if covariate_table.selected_covariate_count == 0 {
        return;
    }
    let row_start = sample_array_index * covariate_table.selected_covariate_count;
    covariate_values
        .extend(&covariate_table.covariate_values[row_start..row_start + covariate_table.selected_covariate_count]);
}

fn returned_covariate_names(selected_covariate_names: &[String]) -> Vec<String> {
    let mut covariate_names = Vec::with_capacity(selected_covariate_names.len() + 1);
    covariate_names.push("intercept".to_string());
    covariate_names.extend(selected_covariate_names.iter().cloned());
    covariate_names
}
