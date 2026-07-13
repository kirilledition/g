use std::sync::Arc;

use crate::regenie::{ChromosomePredictionMatrix, PredictionError, PredictionLocoPath, PredictionSource};

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
    /// Replace prediction use counts with post-resume chromosome blocks.
    ///
    /// Only chromosomes left after resume reconciliation are validated for
    /// later lazy materialization. Repeated noncontiguous blocks must appear
    /// repeatedly.
    ///
    /// # Errors
    ///
    /// Returns an error when a planned chromosome is absent from a trait's
    /// indexed LOCO file.
    pub fn plan_prediction_uses(&mut self, chromosome_blocks: &[Arc<str>]) -> Result<(), PredictionError> {
        self.prediction_source.plan_uses(chromosome_blocks)
    }

    /// Lazily reads and takes the aligned matrix for one chromosome execution.
    ///
    /// Repeated noncontiguous chromosome blocks receive a safe clone until the
    /// final planned use, which transfers the source allocation.
    ///
    /// # Errors
    ///
    /// Returns an error when the prediction source has no remaining aligned
    /// matrix for the requested chromosome.
    pub fn take_chromosome_prediction_matrix(
        &mut self,
        chromosome: &str,
    ) -> Result<ChromosomePredictionMatrix, PredictionError> {
        self.prediction_source.take_chromosome_prediction_matrix(chromosome)
    }
}

#[derive(Debug)]
pub struct PhenotypeGroupLoadRequest<'input> {
    pub sample_identifiers: &'input SampleIdentifierData,
    pub phenotype_path: &'input str,
    pub prediction_loco_paths: &'input [PredictionLocoPath],
    pub phenotype_names: &'input [String],
    pub covariate_path: Option<&'input str>,
    pub covariate_names: Option<&'input [String]>,
    pub is_binary_trait: bool,
    pub sample_mode: g_plan::MultiPhenotypeSampleMode,
}

pub(super) struct AlignedPhenotypeGroupDraft {
    pub phenotype_indices: Vec<usize>,
    pub sample_array_indices: Vec<usize>,
    pub phenotype_values: Vec<f32>,
    pub covariate_names: Vec<String>,
    pub covariate_values: Vec<f32>,
}

#[derive(Debug)]
pub struct SampleIdentifierData {
    pub(crate) family_identifiers: Vec<String>,
    pub(crate) individual_identifiers: Vec<String>,
}
