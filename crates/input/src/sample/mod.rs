//! Native sample alignment and Oxford sample-file parsing.

mod alignment;
mod alignment_workflow;
mod fingerprints;
mod identity;
mod keys;
mod tables;
mod types;

type SampleAlignmentResult<T> = Result<T, String>;

pub use alignment_workflow::load_aligned_phenotype_groups;
pub use identity::load_sample_identifier_data_from_sample_file;
pub use types::{AlignedPhenotypeGroup, PhenotypeGroupLoadRequest, SampleIdentifierData};
