//! Python-free association backend contract.

/// Owned trait-major values with shape `traits x samples`.
#[derive(Debug, PartialEq)]
pub struct TraitMajorMatrix {
    pub values: Vec<f32>,
    pub trait_count: usize,
    pub sample_count: usize,
}

/// Owned sample-major covariate values with shape `samples x covariates`.
#[derive(Debug, PartialEq)]
pub struct SampleMajorCovariateMatrix {
    pub values: Vec<f32>,
    pub sample_count: usize,
    pub covariate_count: usize,
}

/// Backend inputs shared by every chromosome in one phenotype group.
#[derive(Debug, PartialEq)]
pub struct GroupPreparationInput {
    pub phenotypes: TraitMajorMatrix,
    pub covariates: SampleMajorCovariateMatrix,
}

/// Prepared chromosome state and null-logistic convergence policy input.
#[derive(Debug, PartialEq)]
pub struct PreparedChromosome<State> {
    pub state: State,
    pub null_logistic_converged: Option<Vec<bool>>,
}

/// One genotype batch submitted to the device backend.
#[derive(Debug, PartialEq)]
pub struct GenotypeBatchInput {
    pub variant_count: usize,
    pub sample_count: usize,
    pub genotypes: g_genotype::OwnedGenotypeBuffer,
    pub statistics: g_genotype::ChunkComputeStatistics,
}

/// Chunk-oriented association compute implemented by the device runtime.
pub trait AssociationBackend: Send + Sync {
    type GroupState: Send;
    type ChromosomeState: Send;
    type DeviceResult: Send;
    type Error: std::error::Error + Send + Sync + 'static;

    /// Prepare reusable device state for one phenotype group.
    ///
    /// # Errors
    ///
    /// Returns an error when the phenotype or covariate data cannot be prepared.
    fn prepare_group(&self, input: GroupPreparationInput) -> Result<Self::GroupState, Self::Error>;

    /// Release group state after its final chromosome has been prepared.
    fn release_group(&self, group: Self::GroupState) {
        drop(group);
    }

    /// Prepare reusable state and null-logistic policy input for one chromosome.
    ///
    /// # Errors
    ///
    /// Returns an error when LOCO predictions are invalid or the chromosome
    /// state cannot be prepared.
    fn prepare_chromosome(
        &self,
        group: &Self::GroupState,
        predictions: g_input::ChromosomePredictionMatrix,
    ) -> Result<PreparedChromosome<Self::ChromosomeState>, Self::Error>;

    /// Release one chromosome state on the backend execution thread.
    ///
    /// Backends with thread- or runtime-affine reference management can
    /// override this hook. The scheduler calls it only after every submitted
    /// batch using the state has been materialized and received.
    fn release_chromosome(&self, chromosome: Self::ChromosomeState) {
        drop(chromosome);
    }

    /// Submit one genotype batch and return an opaque device result.
    ///
    /// # Errors
    ///
    /// Returns an error when the genotype batch is invalid or device execution
    /// cannot be submitted.
    fn compute_batch(
        &self,
        chromosome: &Self::ChromosomeState,
        input: GenotypeBatchInput,
    ) -> Result<Self::DeviceResult, Self::Error>;

    /// Select active traits, transfer one result to host, and retain only its
    /// logical variant rows.
    ///
    /// # Errors
    ///
    /// Returns an error when selection, conversion, or device-to-host transfer
    /// fails.
    fn materialize_batch(
        &self,
        result: Self::DeviceResult,
        active_trait_indices: Option<&[usize]>,
        logical_variant_count: usize,
    ) -> Result<g_output::Regenie2StatisticBatch, Self::Error>;
}
