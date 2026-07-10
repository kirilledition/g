//! Public input crate facade.

pub use crate::error::InputError;
pub use crate::regenie::{
    ChromosomePredictionMatrix, PredictionError, PredictionLocoPath, resolve_prediction_loco_paths,
};
pub use crate::sample::{
    AlignedPhenotypeGroup, PhenotypeGroupLoadRequest, SampleIdentifierData, build_group_sample_position_array,
    build_union_sample_indices, load_aligned_phenotype_groups, load_sample_identifier_data_from_sample_file,
};
