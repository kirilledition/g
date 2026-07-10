//! Native sample alignment and Oxford sample-file parsing.

#![allow(clippy::missing_errors_doc)]
#![allow(clippy::float_cmp)]
#![allow(clippy::single_match_else)]
#![allow(clippy::too_many_arguments)]

mod alignment;
mod alignment_workflow;
mod fingerprints;
mod grouping;
mod identity;
mod keys;
mod tables;
mod types;

pub use alignment_workflow::{align_grouped_sample_data, align_multi_sample_data, align_sample_data};
pub use fingerprints::{
    resolve_complete_case_compute_group, resolve_per_phenotype_compute_group, resolve_single_phenotype_compute_group,
};
pub use grouping::{build_group_sample_position_array, build_union_sample_indices};
pub use identity::load_sample_identifier_data_from_sample_file;
pub use types::{
    AlignedPhenotypeGroup, AlignedSampleData, AlignmentInputs, MultiAlignedSampleData, MultiAlignmentInputs,
    ResolvedPhenotypeComputeGroup, SampleAlignmentError, SampleIdentifierData, SampleKeyMode,
};
