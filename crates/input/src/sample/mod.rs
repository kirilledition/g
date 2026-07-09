//! Native sample alignment and Oxford sample-file parsing.

#![allow(clippy::missing_errors_doc)]
#![allow(clippy::float_cmp)]
#![allow(clippy::single_match_else)]
#![allow(clippy::too_many_arguments)]

use std::path::Path;

mod alignment;
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
    load_covariate_table, multi_phenotype_parse_candidate_mask, read_multi_phenotype_table, read_single_phenotype_table,
};
pub use types::{
    AlignedPhenotypeGroup, AlignedSampleData, AlignmentInputs, GroupedAlignedSampleData, MultiAlignedSampleData,
    MultiAlignmentInputs, ResolvedPhenotypeComputeGroup, SampleAlignmentError, SampleIdentifierData, SampleKeyMode,
};

use alignment::{build_grouped_aligned_sample_data, build_multi_aligned_sample_data, build_single_aligned_sample_data};

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
