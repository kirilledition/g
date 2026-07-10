//! Public input crate facade.

pub use crate::error::{InputError, InputResult};
pub use crate::regenie::{
    ChromosomePredictionMatrix, MultiPredictionSource, PredictionError, PredictionLocoPath, PredictionSource,
    resolve_prediction_loco_paths,
};
pub use crate::sample::{
    AlignedPhenotypeGroup, AlignedSampleData, AlignmentInputs, MultiAlignedSampleData, MultiAlignmentInputs,
    ResolvedPhenotypeComputeGroup, SampleAlignmentError, SampleIdentifierData, SampleKeyMode,
    align_grouped_sample_data, align_multi_sample_data, align_sample_data, build_group_sample_position_array,
    build_union_sample_indices, load_sample_identifier_data_from_sample_file, resolve_complete_case_compute_group,
    resolve_per_phenotype_compute_group, resolve_single_phenotype_compute_group,
};
