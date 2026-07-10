use std::path::Path;

use crate::error::InputResult;

use super::alignment::{
    build_grouped_aligned_sample_data, build_multi_aligned_sample_data, build_single_aligned_sample_data,
};
use super::keys::{
    build_sample_row_indices_by_key, validate_sample_identifier_keys, validate_sample_identifier_lengths,
};
use super::tables::{
    load_covariate_table, multi_phenotype_parse_candidate_mask, read_multi_phenotype_table, read_single_phenotype_table,
};
use super::types::{
    AlignedPhenotypeGroup, AlignedSampleData, AlignmentInputs, MultiAlignedSampleData, MultiAlignmentInputs,
};

pub fn align_sample_data(inputs: AlignmentInputs) -> InputResult<AlignedSampleData> {
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

    build_single_aligned_sample_data(inputs, &phenotype_table, &covariate_table).map_err(Into::into)
}

/// Align several phenotypes to one shared complete-case sample set.
///
/// This intentionally intersects all per-trait valid sample sets and therefore
/// is not equivalent to running each phenotype through `align_sample_data`.
pub fn align_multi_sample_data(inputs: MultiAlignmentInputs) -> InputResult<MultiAlignedSampleData> {
    if inputs.phenotype_names.is_empty() {
        return Err(super::SampleAlignmentError::new(
            "At least one phenotype is required for multi-phenotype alignment.",
        )
        .into());
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

    build_multi_aligned_sample_data(inputs, &phenotype_table, &covariate_table).map_err(Into::into)
}

/// Align several phenotypes independently, then group traits that share one
/// sample/covariate layout.
pub fn align_grouped_sample_data(inputs: &MultiAlignmentInputs) -> InputResult<Vec<AlignedPhenotypeGroup>> {
    if inputs.phenotype_names.is_empty() {
        return Err(super::SampleAlignmentError::new(
            "At least one phenotype is required for grouped phenotype alignment.",
        )
        .into());
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

    build_grouped_aligned_sample_data(inputs, &phenotype_table, &covariate_table).map_err(Into::into)
}
