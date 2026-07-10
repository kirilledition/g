use crate::regenie::{ChromosomePredictionMatrix, PredictionError, PredictionSource};

#[derive(Debug)]
pub struct AlignedPhenotypeGroup {
    pub phenotype_group: g_plan::PhenotypeComputeGroup,
    pub sample_indices: Vec<usize>,
    pub phenotype_values: Vec<f32>,
    pub covariate_names: Vec<String>,
    pub covariate_values: Vec<f32>,
    pub(super) prediction_source: PredictionSource,
}

impl AlignedPhenotypeGroup {
    /// Returns the shared trait-major LOCO prediction matrix for a chromosome.
    ///
    /// # Errors
    ///
    /// Returns an error when the prediction source has no aligned matrix for
    /// the requested chromosome.
    pub fn chromosome_prediction_matrix(
        &self,
        chromosome: &str,
    ) -> Result<std::sync::Arc<ChromosomePredictionMatrix>, PredictionError> {
        self.prediction_source.chromosome_prediction_matrix(chromosome)
    }
}

#[derive(Clone, Debug)]
pub struct PhenotypeGroupLoadRequest {
    pub sample_identifiers: SampleIdentifierData,
    pub phenotype_path: String,
    pub prediction_list_path: String,
    pub phenotype_names: Vec<String>,
    pub covariate_path: Option<String>,
    pub covariate_names: Option<Vec<String>>,
    pub is_binary_trait: bool,
    pub sample_key_mode: g_plan::SampleKeyMode,
    pub sample_mode: g_plan::MultiPhenotypeSampleMode,
}

pub(super) struct AlignedPhenotypeGroupDraft {
    pub phenotype_indices: Vec<usize>,
    pub sample_array_indices: Vec<usize>,
    pub sample_indices: Vec<usize>,
    pub phenotype_values: Vec<f32>,
    pub covariate_names: Vec<String>,
    pub covariate_values: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct SampleIdentifierData {
    pub sample_indices: Vec<usize>,
    pub family_identifiers: Vec<String>,
    pub individual_identifiers: Vec<String>,
}
