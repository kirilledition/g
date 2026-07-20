use ahash::{HashMap, HashMapExt};

use super::SampleAlignmentResult;
use super::tables::{CovariateTable, MultiPhenotypeTable, is_complete_multi_phenotype_sample};
use super::types::{AlignedPhenotypeGroupDraft, PhenotypeGroupLoadRequest};

pub(super) fn build_aligned_phenotype_group_drafts(
    request: &PhenotypeGroupLoadRequest<'_>,
    phenotype_table: &MultiPhenotypeTable,
    covariate_table: &CovariateTable,
) -> SampleAlignmentResult<Vec<AlignedPhenotypeGroupDraft>> {
    if request.phenotype_names.len() == 1 || request.sample_mode == g_plan::MultiPhenotypeSampleMode::CompleteCase {
        let phenotype_indices = (0..phenotype_table.phenotype_count).collect::<Vec<_>>();
        let sample_array_indices = complete_case_sample_array_indices(phenotype_table, covariate_table);
        if sample_array_indices.is_empty() {
            return Err("No aligned samples remain after joining phenotype and covariate tables.".to_string());
        }
        return Ok(vec![build_group_draft(phenotype_table, covariate_table, phenotype_indices, sample_array_indices)]);
    }

    build_compatible_sample_group_drafts(request, phenotype_table, covariate_table)
}

fn build_compatible_sample_group_drafts(
    request: &PhenotypeGroupLoadRequest<'_>,
    phenotype_table: &MultiPhenotypeTable,
    covariate_table: &CovariateTable,
) -> SampleAlignmentResult<Vec<AlignedPhenotypeGroupDraft>> {
    let mut groups_by_sample_array_indices: HashMap<Vec<usize>, (usize, Vec<usize>)> = HashMap::new();
    let sample_count = request.sample_identifiers.family_identifiers.len();
    let sample_array_indices = 0..sample_count;
    let mut complete_sample_array_indices = Vec::with_capacity(sample_count);

    for phenotype_index in 0..phenotype_table.phenotype_count {
        collect_complete_trait_sample_array_indices(
            sample_array_indices.clone(),
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
        if let Some((_group_order, phenotype_indices)) =
            groups_by_sample_array_indices.get_mut(complete_sample_array_indices.as_slice())
        {
            phenotype_indices.push(phenotype_index);
        } else {
            let group_order = groups_by_sample_array_indices.len();
            let stored_sample_array_indices = std::mem::take(&mut complete_sample_array_indices);
            groups_by_sample_array_indices.insert(stored_sample_array_indices, (group_order, vec![phenotype_index]));
            complete_sample_array_indices = Vec::with_capacity(sample_count);
        }
    }

    let mut ordered_groups = groups_by_sample_array_indices
        .into_iter()
        .map(|(sample_array_indices, (group_order, phenotype_indices))| {
            (group_order, phenotype_indices, sample_array_indices)
        })
        .collect::<Vec<_>>();
    ordered_groups.sort_by_key(|(group_order, _, _)| *group_order);
    Ok(ordered_groups
        .into_iter()
        .map(|(_, phenotype_indices, sample_array_indices)| {
            build_group_draft(phenotype_table, covariate_table, phenotype_indices, sample_array_indices)
        })
        .collect())
}

fn build_group_draft(
    phenotype_table: &MultiPhenotypeTable,
    covariate_table: &CovariateTable,
    phenotype_indices: Vec<usize>,
    sample_array_indices: Vec<usize>,
) -> AlignedPhenotypeGroupDraft {
    let sample_count = sample_array_indices.len();
    let covariate_names = returned_covariate_names(&covariate_table.selected_covariate_names);
    let mut phenotype_values = Vec::with_capacity(phenotype_indices.len() * sample_count);
    let mut covariate_values = Vec::with_capacity(sample_count * covariate_names.len());

    for sample_array_index in &sample_array_indices {
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
        phenotype_values,
        covariate_names,
        covariate_values,
    }
}

fn complete_case_sample_array_indices(
    phenotype_table: &MultiPhenotypeTable,
    covariate_table: &CovariateTable,
) -> Vec<usize> {
    (0..phenotype_table.sample_count)
        .filter(|sample_array_index| {
            is_complete_multi_phenotype_sample(phenotype_table, *sample_array_index)
                && covariate_table.covariate_mask[*sample_array_index]
        })
        .collect()
}

fn collect_complete_trait_sample_array_indices(
    sample_array_indices: impl Iterator<Item = usize>,
    phenotype_table: &MultiPhenotypeTable,
    covariate_table: &CovariateTable,
    phenotype_index: usize,
    complete_sample_array_indices: &mut Vec<usize>,
) {
    complete_sample_array_indices.clear();
    complete_sample_array_indices.extend(sample_array_indices.filter(|sample_array_index| {
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
