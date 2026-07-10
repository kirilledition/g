//! Python-free association backend contract.

/// Borrowed trait-major phenotype values with shape `traits x samples`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TraitMajorPhenotypeMatrixView<'view> {
    pub values: &'view [f32],
    pub trait_count: usize,
    pub sample_count: usize,
}

/// Borrowed sample-major covariate values with shape `samples x covariates`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleMajorCovariateMatrixView<'view> {
    pub values: &'view [f32],
    pub sample_count: usize,
    pub covariate_count: usize,
}

/// Backend inputs shared by every chromosome in one phenotype group.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroupPreparationInput<'view> {
    pub phenotypes: TraitMajorPhenotypeMatrixView<'view>,
    pub covariates: SampleMajorCovariateMatrixView<'view>,
}

/// Borrowed trait-major LOCO predictions with shape `traits x samples`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TraitMajorPredictionMatrixView<'view> {
    pub values: &'view [f32],
    pub trait_count: usize,
    pub sample_count: usize,
}

/// Prepared chromosome state and null-logistic convergence policy input.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedChromosome<State> {
    pub state: State,
    pub null_logistic_converged: Option<Vec<bool>>,
}

/// Borrowed variant-major dosage values with shape `variants x samples`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VariantMajorDosageMatrixView<'view> {
    pub values: &'view [f32],
    pub variant_count: usize,
    pub sample_count: usize,
}

/// Borrowed variant-major packed probability pairs.
///
/// Values have shape `variants x samples x 2`; the final dimension contains
/// the two stored BGEN probabilities for one sample and variant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VariantMajorPacked8MatrixView<'view> {
    pub values: &'view [u8],
    pub variant_count: usize,
    pub sample_count: usize,
}

/// Supported host genotype representations at the compute boundary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GenotypeMatrixView<'view> {
    Dosage(VariantMajorDosageMatrixView<'view>),
    Packed8(VariantMajorPacked8MatrixView<'view>),
}

/// Borrowed per-variant statistics required by association kernels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenotypeBatchStatisticsView<'view> {
    pub dosage_sum: &'view [f32],
    pub observation_count: &'view [i32],
    pub imputed_dosage_square_sum: Option<&'view [f32]>,
    pub sparse_candidate_mask: Option<&'view [bool]>,
}

/// One borrowed genotype batch submitted to the device backend.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenotypeBatchInput<'view> {
    pub variant_start_index: usize,
    pub genotypes: GenotypeMatrixView<'view>,
    pub statistics: GenotypeBatchStatisticsView<'view>,
}

/// Materialization policy applied on-device before transfer to the host.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaterializationInput<'view> {
    pub active_trait_indices: &'view [usize],
}

/// Trait-major association statistic matrices with shape `traits x variants`.
#[derive(Clone, Debug, PartialEq)]
pub struct HostAssociationStatisticMatrix<Statistic> {
    pub trait_count: usize,
    pub variant_count: usize,
    pub beta: Vec<Statistic>,
    pub standard_error: Vec<Statistic>,
    pub chi_squared: Vec<Statistic>,
    pub log10_p_value: Vec<Statistic>,
}

/// Float32 association statistics materialized from the device.
pub type HostAssociationStatistics = HostAssociationStatisticMatrix<f32>;

/// Trait-major binary correction codes with shape `traits x variants`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostCorrectionCodeMatrix {
    pub trait_count: usize,
    pub variant_count: usize,
    pub values: Vec<i32>,
}

/// One fully materialized host association batch.
#[derive(Clone, Debug, PartialEq)]
pub struct HostAssociationBatch {
    pub statistics: HostAssociationStatistics,
    pub correction_codes: Option<HostCorrectionCodeMatrix>,
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
    fn prepare_group(&self, input: GroupPreparationInput<'_>) -> Result<Self::GroupState, Self::Error>;

    /// Prepare reusable state and null-logistic policy input for one chromosome.
    ///
    /// # Errors
    ///
    /// Returns an error when LOCO predictions are invalid or the chromosome
    /// state cannot be prepared.
    fn prepare_chromosome(
        &self,
        group: &Self::GroupState,
        predictions: TraitMajorPredictionMatrixView<'_>,
    ) -> Result<PreparedChromosome<Self::ChromosomeState>, Self::Error>;

    /// Submit one genotype batch and return an opaque device result.
    ///
    /// # Errors
    ///
    /// Returns an error when the genotype batch is invalid or device execution
    /// cannot be submitted.
    fn compute_batch(
        &self,
        chromosome: &Self::ChromosomeState,
        input: GenotypeBatchInput<'_>,
    ) -> Result<Self::DeviceResult, Self::Error>;

    /// Select active traits, narrow statistics, and transfer one result to host.
    ///
    /// # Errors
    ///
    /// Returns an error when selection, conversion, or device-to-host transfer
    /// fails.
    fn materialize_batch(
        &self,
        result: Self::DeviceResult,
        input: MaterializationInput<'_>,
    ) -> Result<HostAssociationBatch, Self::Error>;
}
