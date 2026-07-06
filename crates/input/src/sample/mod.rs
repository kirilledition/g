//! Native sample alignment and Oxford sample-file parsing.

#![allow(clippy::missing_errors_doc)]
#![allow(clippy::float_cmp)]
#![allow(clippy::single_match_else)]
#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;
use std::path::Path;

mod fingerprints;
mod grouping;
mod identity;
mod keys;
mod tables;
mod types;

pub use fingerprints::{
    resolve_complete_case_compute_group, resolve_per_phenotype_compute_group, resolve_single_phenotype_compute_group,
};
pub use grouping::{build_group_sample_position_array, build_union_sample_indices};
pub use identity::load_sample_identifier_data_from_sample_file;
use keys::{build_sample_row_indices_by_key, validate_sample_identifier_keys, validate_sample_identifier_lengths};
use tables::{
    CovariateTable, MultiPhenotypeTable, SinglePhenotypeTable, is_complete_multi_phenotype_sample,
    load_covariate_table, multi_phenotype_parse_candidate_mask, read_multi_phenotype_table,
    read_single_phenotype_table,
};
pub use types::{
    AlignedPhenotypeGroup, AlignedSampleData, AlignmentInputs, GroupedAlignedSampleData, MultiAlignedSampleData,
    MultiAlignmentInputs, ResolvedPhenotypeComputeGroup, SampleAlignmentError, SampleIdentifierData, SampleKeyMode,
};

#[cfg(test)]
mod tests;

type SampleAlignmentResult<T> = Result<T, SampleAlignmentError>;

pub fn align_sample_data(inputs: AlignmentInputs) -> Result<AlignedSampleData, SampleAlignmentError> {
    validate_sample_identifier_lengths(
        &inputs.sample_indices,
        &inputs.family_identifiers,
        &inputs.individual_identifiers,
    )?;
    validate_sample_identifier_keys(
        inputs.sample_key_mode,
        &inputs.family_identifiers,
        &inputs.individual_identifiers,
    )?;

    let sample_row_indices_by_key = build_sample_row_indices_by_key(
        inputs.sample_key_mode,
        &inputs.family_identifiers,
        &inputs.individual_identifiers,
    );
    let phenotype_table = read_single_phenotype_table(
        Path::new(&inputs.phenotype_path),
        &inputs.phenotype_name,
        inputs.is_binary_trait,
        inputs.sample_key_mode,
        &sample_row_indices_by_key,
        inputs.sample_indices.len(),
    )?;

    let covariate_table = load_covariate_table(
        inputs.covariate_path.as_deref(),
        inputs.covariate_names.as_deref(),
        inputs.sample_key_mode,
        &sample_row_indices_by_key,
        &phenotype_table.phenotype_mask,
        inputs.sample_indices.len(),
    )?;

    build_single_aligned_sample_data(inputs, &phenotype_table, &covariate_table)
}

/// Align several phenotypes to one shared complete-case sample set.
///
/// This intentionally intersects all per-trait valid sample sets and therefore
/// is not equivalent to running each phenotype through `align_sample_data`.
pub fn align_multi_sample_data(inputs: MultiAlignmentInputs) -> Result<MultiAlignedSampleData, SampleAlignmentError> {
    if inputs.phenotype_names.is_empty() {
        return Err(SampleAlignmentError::new("At least one phenotype is required for multi-phenotype alignment."));
    }
    validate_sample_identifier_lengths(
        &inputs.sample_indices,
        &inputs.family_identifiers,
        &inputs.individual_identifiers,
    )?;
    validate_sample_identifier_keys(
        inputs.sample_key_mode,
        &inputs.family_identifiers,
        &inputs.individual_identifiers,
    )?;

    let sample_row_indices_by_key = build_sample_row_indices_by_key(
        inputs.sample_key_mode,
        &inputs.family_identifiers,
        &inputs.individual_identifiers,
    );
    let phenotype_table = read_multi_phenotype_table(
        Path::new(&inputs.phenotype_path),
        &inputs.phenotype_names,
        inputs.is_binary_trait,
        inputs.sample_key_mode,
        &sample_row_indices_by_key,
        inputs.sample_indices.len(),
    )?;
    let parse_candidate_mask = multi_phenotype_parse_candidate_mask(&phenotype_table);
    let covariate_table = load_covariate_table(
        inputs.covariate_path.as_deref(),
        inputs.covariate_names.as_deref(),
        inputs.sample_key_mode,
        &sample_row_indices_by_key,
        &parse_candidate_mask,
        inputs.sample_indices.len(),
    )?;

    build_multi_aligned_sample_data(inputs, &phenotype_table, &covariate_table)
}

/// Align several phenotypes independently, then group traits that share one
/// sample/covariate layout.
pub fn align_grouped_sample_data(
    inputs: &MultiAlignmentInputs,
) -> Result<GroupedAlignedSampleData, SampleAlignmentError> {
    if inputs.phenotype_names.is_empty() {
        return Err(SampleAlignmentError::new("At least one phenotype is required for grouped phenotype alignment."));
    }
    validate_sample_identifier_lengths(
        &inputs.sample_indices,
        &inputs.family_identifiers,
        &inputs.individual_identifiers,
    )?;
    validate_sample_identifier_keys(
        inputs.sample_key_mode,
        &inputs.family_identifiers,
        &inputs.individual_identifiers,
    )?;

    let sample_row_indices_by_key = build_sample_row_indices_by_key(
        inputs.sample_key_mode,
        &inputs.family_identifiers,
        &inputs.individual_identifiers,
    );
    let phenotype_table = read_multi_phenotype_table(
        Path::new(&inputs.phenotype_path),
        &inputs.phenotype_names,
        inputs.is_binary_trait,
        inputs.sample_key_mode,
        &sample_row_indices_by_key,
        inputs.sample_indices.len(),
    )?;
    let parse_candidate_mask = multi_phenotype_parse_candidate_mask(&phenotype_table);
    let covariate_table = load_covariate_table(
        inputs.covariate_path.as_deref(),
        inputs.covariate_names.as_deref(),
        inputs.sample_key_mode,
        &sample_row_indices_by_key,
        &parse_candidate_mask,
        inputs.sample_indices.len(),
    )?;

    build_grouped_aligned_sample_data(inputs, &phenotype_table, &covariate_table)
}

fn build_single_aligned_sample_data(
    inputs: AlignmentInputs,
    phenotype_table: &SinglePhenotypeTable,
    covariate_table: &CovariateTable,
) -> SampleAlignmentResult<AlignedSampleData> {
    let complete_sample_array_indices = complete_single_sample_array_indices(&inputs, phenotype_table, covariate_table);
    if complete_sample_array_indices.is_empty() {
        return Err(SampleAlignmentError::new(
            "No aligned samples remain after joining phenotype and covariate tables.",
        ));
    }

    let aligned_sample_count = complete_sample_array_indices.len();
    let covariate_names = returned_covariate_names(&covariate_table.selected_covariate_names);
    let covariate_column_count = covariate_names.len();
    let mut sample_indices = Vec::with_capacity(aligned_sample_count);
    let mut family_identifiers = Vec::with_capacity(aligned_sample_count);
    let mut individual_identifiers = Vec::with_capacity(aligned_sample_count);
    let mut phenotype_vector = Vec::with_capacity(aligned_sample_count);
    let mut covariate_matrix_values = Vec::with_capacity(aligned_sample_count * covariate_column_count);

    for sample_array_index in complete_sample_array_indices {
        sample_indices.push(inputs.sample_indices[sample_array_index]);
        family_identifiers.push(inputs.family_identifiers[sample_array_index].clone());
        individual_identifiers.push(inputs.individual_identifiers[sample_array_index].clone());
        phenotype_vector.push(phenotype_table.phenotype_values[sample_array_index]);
        push_covariate_matrix_row(&mut covariate_matrix_values, covariate_table, sample_array_index);
    }

    Ok(AlignedSampleData {
        sample_indices,
        family_identifiers,
        individual_identifiers,
        phenotype_name: inputs.phenotype_name,
        phenotype_vector,
        covariate_names,
        covariate_matrix_values,
        covariate_row_count: aligned_sample_count,
        covariate_column_count,
        is_binary_trait: inputs.is_binary_trait,
    })
}

fn build_multi_aligned_sample_data(
    inputs: MultiAlignmentInputs,
    phenotype_table: &MultiPhenotypeTable,
    covariate_table: &CovariateTable,
) -> SampleAlignmentResult<MultiAlignedSampleData> {
    let complete_sample_array_indices = complete_multi_sample_array_indices(&inputs, phenotype_table, covariate_table);
    if complete_sample_array_indices.is_empty() {
        return Err(SampleAlignmentError::new(
            "No aligned samples remain after complete-case multi-phenotype intersection.",
        ));
    }

    let aligned_sample_count = complete_sample_array_indices.len();
    let covariate_names = returned_covariate_names(&covariate_table.selected_covariate_names);
    let covariate_column_count = covariate_names.len();
    let mut sample_indices = Vec::with_capacity(aligned_sample_count);
    let mut family_identifiers = Vec::with_capacity(aligned_sample_count);
    let mut individual_identifiers = Vec::with_capacity(aligned_sample_count);
    let mut phenotype_matrix_values = Vec::with_capacity(phenotype_table.phenotype_count * aligned_sample_count);
    let mut covariate_matrix_values = Vec::with_capacity(aligned_sample_count * covariate_column_count);

    for sample_array_index in &complete_sample_array_indices {
        sample_indices.push(inputs.sample_indices[*sample_array_index]);
        family_identifiers.push(inputs.family_identifiers[*sample_array_index].clone());
        individual_identifiers.push(inputs.individual_identifiers[*sample_array_index].clone());
        push_covariate_matrix_row(&mut covariate_matrix_values, covariate_table, *sample_array_index);
    }
    for phenotype_index in 0..phenotype_table.phenotype_count {
        for sample_array_index in &complete_sample_array_indices {
            let value_index = phenotype_index * phenotype_table.sample_count + sample_array_index;
            phenotype_matrix_values.push(phenotype_table.phenotype_values[value_index]);
        }
    }

    Ok(MultiAlignedSampleData {
        sample_indices,
        family_identifiers,
        individual_identifiers,
        phenotype_names: inputs.phenotype_names,
        phenotype_matrix_values,
        phenotype_row_count: phenotype_table.phenotype_count,
        phenotype_column_count: aligned_sample_count,
        covariate_names,
        covariate_matrix_values,
        covariate_row_count: aligned_sample_count,
        covariate_column_count,
        is_binary_trait: inputs.is_binary_trait,
    })
}

fn build_grouped_aligned_sample_data(
    inputs: &MultiAlignmentInputs,
    phenotype_table: &MultiPhenotypeTable,
    covariate_table: &CovariateTable,
) -> SampleAlignmentResult<GroupedAlignedSampleData> {
    let mut group_indices_by_sample_indices: HashMap<Vec<usize>, usize> = HashMap::new();
    let mut group_sample_array_indices: Vec<Vec<usize>> = Vec::new();
    let mut phenotype_indices_by_group: Vec<Vec<usize>> = Vec::new();
    let sorted_sample_array_indices = sorted_sample_array_indices_by_sample_index(&inputs.sample_indices);
    let mut complete_sample_array_indices = Vec::with_capacity(inputs.sample_indices.len());

    for phenotype_index in 0..phenotype_table.phenotype_count {
        collect_complete_grouped_trait_sample_array_indices(
            &sorted_sample_array_indices,
            phenotype_table,
            covariate_table,
            phenotype_index,
            &mut complete_sample_array_indices,
        );
        if complete_sample_array_indices.is_empty() {
            return Err(SampleAlignmentError::new(format!(
                "No aligned samples remain after joining phenotype '{}' and covariate tables.",
                inputs.phenotype_names[phenotype_index]
            )));
        }
        let group_index = match group_indices_by_sample_indices.get(&complete_sample_array_indices) {
            Some(existing_group_index) => *existing_group_index,
            None => {
                let new_group_index = group_sample_array_indices.len();
                let stored_sample_array_indices = std::mem::take(&mut complete_sample_array_indices);
                group_indices_by_sample_indices.insert(stored_sample_array_indices.clone(), new_group_index);
                group_sample_array_indices.push(stored_sample_array_indices);
                phenotype_indices_by_group.push(Vec::new());
                complete_sample_array_indices = Vec::with_capacity(inputs.sample_indices.len());
                new_group_index
            }
        };
        phenotype_indices_by_group[group_index].push(phenotype_index);
    }

    let groups = phenotype_indices_by_group
        .into_iter()
        .zip(group_sample_array_indices)
        .map(|(phenotype_indices, complete_sample_array_indices)| {
            build_aligned_phenotype_group(
                inputs,
                phenotype_table,
                covariate_table,
                phenotype_indices,
                &complete_sample_array_indices,
            )
        })
        .collect::<Vec<_>>();
    Ok(GroupedAlignedSampleData { groups })
}

fn build_aligned_phenotype_group(
    inputs: &MultiAlignmentInputs,
    phenotype_table: &MultiPhenotypeTable,
    covariate_table: &CovariateTable,
    phenotype_indices: Vec<usize>,
    complete_sample_array_indices: &[usize],
) -> AlignedPhenotypeGroup {
    let aligned_sample_count = complete_sample_array_indices.len();
    let covariate_names = returned_covariate_names(&covariate_table.selected_covariate_names);
    let covariate_column_count = covariate_names.len();
    let mut sample_indices = Vec::with_capacity(aligned_sample_count);
    let mut family_identifiers = Vec::with_capacity(aligned_sample_count);
    let mut individual_identifiers = Vec::with_capacity(aligned_sample_count);
    let mut phenotype_matrix_values = Vec::with_capacity(phenotype_indices.len() * aligned_sample_count);
    let mut covariate_matrix_values = Vec::with_capacity(aligned_sample_count * covariate_column_count);

    for sample_array_index in complete_sample_array_indices {
        sample_indices.push(inputs.sample_indices[*sample_array_index]);
        family_identifiers.push(inputs.family_identifiers[*sample_array_index].clone());
        individual_identifiers.push(inputs.individual_identifiers[*sample_array_index].clone());
        push_covariate_matrix_row(&mut covariate_matrix_values, covariate_table, *sample_array_index);
    }
    for phenotype_index in &phenotype_indices {
        for sample_array_index in complete_sample_array_indices {
            let value_index = phenotype_index * phenotype_table.sample_count + sample_array_index;
            phenotype_matrix_values.push(phenotype_table.phenotype_values[value_index]);
        }
    }

    let phenotype_names =
        phenotype_indices.iter().map(|phenotype_index| inputs.phenotype_names[*phenotype_index].clone()).collect();
    let phenotype_row_count = phenotype_indices.len();
    AlignedPhenotypeGroup {
        phenotype_indices,
        aligned_sample_data: MultiAlignedSampleData {
            sample_indices,
            family_identifiers,
            individual_identifiers,
            phenotype_names,
            phenotype_matrix_values,
            phenotype_row_count,
            phenotype_column_count: aligned_sample_count,
            covariate_names,
            covariate_matrix_values,
            covariate_row_count: aligned_sample_count,
            covariate_column_count,
            is_binary_trait: inputs.is_binary_trait,
        },
    }
}

fn complete_single_sample_array_indices(
    inputs: &AlignmentInputs,
    phenotype_table: &SinglePhenotypeTable,
    covariate_table: &CovariateTable,
) -> Vec<usize> {
    sorted_sample_array_indices_by_sample_index(&inputs.sample_indices)
        .into_iter()
        .filter(|sample_array_index| {
            phenotype_table.phenotype_mask[*sample_array_index] && covariate_table.covariate_mask[*sample_array_index]
        })
        .collect()
}

fn complete_multi_sample_array_indices(
    inputs: &MultiAlignmentInputs,
    phenotype_table: &MultiPhenotypeTable,
    covariate_table: &CovariateTable,
) -> Vec<usize> {
    sorted_sample_array_indices_by_sample_index(&inputs.sample_indices)
        .into_iter()
        .filter(|sample_array_index| {
            is_complete_multi_phenotype_sample(phenotype_table, *sample_array_index)
                && covariate_table.covariate_mask[*sample_array_index]
        })
        .collect()
}

fn sorted_sample_array_indices_by_sample_index(sample_indices: &[i64]) -> Vec<usize> {
    let mut sorted_sample_array_indices: Vec<usize> = (0..sample_indices.len()).collect();
    sorted_sample_array_indices.sort_by_key(|sample_array_index| sample_indices[*sample_array_index]);
    sorted_sample_array_indices
}

fn collect_complete_grouped_trait_sample_array_indices(
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

fn push_covariate_matrix_row(
    covariate_matrix_values: &mut Vec<f32>,
    covariate_table: &CovariateTable,
    sample_array_index: usize,
) {
    covariate_matrix_values.push(1.0);
    if covariate_table.selected_covariate_count == 0 {
        return;
    }
    let row_start = sample_array_index * covariate_table.selected_covariate_count;
    covariate_matrix_values
        .extend(&covariate_table.covariate_values[row_start..row_start + covariate_table.selected_covariate_count]);
}

fn returned_covariate_names(selected_covariate_names: &[String]) -> Vec<String> {
    let mut covariate_names = Vec::with_capacity(selected_covariate_names.len() + 1);
    covariate_names.push("intercept".to_string());
    covariate_names.extend(selected_covariate_names.iter().cloned());
    covariate_names
}
