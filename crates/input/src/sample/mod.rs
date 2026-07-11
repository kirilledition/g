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

type SampleAlignmentResult<T> = Result<T, String>;

pub use alignment_workflow::load_aligned_phenotype_groups;
pub use grouping::{build_group_sample_position_arrays, build_union_sample_indices};
pub use identity::load_sample_identifier_data_from_sample_file;
pub use types::{AlignedPhenotypeGroup, PhenotypeGroupLoadRequest, SampleIdentifierData};
