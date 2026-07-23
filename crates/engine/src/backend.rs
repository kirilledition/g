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

/// Genotype transfer support advertised by an association backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GenotypeDeliveryCapability {
    /// Accept only dosage or packed8 values decoded on the host.
    HostOnly,
    /// Accept host-decoded values and zlib members stripped to raw DEFLATE.
    RawDeflatePacked8,
}

/// Per-group genotype transfer state prepared before chromosome execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GenotypeTransferPreparation {
    /// No compressed-transfer state is required.
    Host,
    /// Upload and retain sample selection for raw-DEFLATE packed8 batches.
    CompressedPacked8(g_genotype::CompressedPacked8Transfer),
}

/// Backend inputs shared by every chromosome in one phenotype group.
#[derive(Debug, PartialEq)]
pub struct GroupPreparationInput {
    pub phenotypes: TraitMajorMatrix,
    pub covariates: SampleMajorCovariateMatrix,
    pub genotype_transfer: GenotypeTransferPreparation,
}

/// Prepared chromosome state and null-logistic convergence policy input.
#[derive(Debug, PartialEq)]
pub struct PreparedChromosome<State> {
    pub state: State,
    pub null_logistic_converged: Option<Vec<bool>>,
}

/// Association values and optional device-produced packed8 summaries.
#[derive(Debug, PartialEq)]
pub struct MaterializedAssociationBatch {
    pub association: g_output::Regenie2StatisticBatch,
    pub genotype_statistics: MaterializedGenotypeStatistics,
}

/// Exactly one source of output-facing genotype statistics.
#[derive(Debug, PartialEq)]
pub enum MaterializedGenotypeStatistics {
    /// Statistics computed while decoding genotypes on the host.
    Ready(g_genotype_contracts::ChunkOutputStatistics),
    /// Exact packed8 integer summaries computed on the device.
    Packed8Raw(g_genotype::Packed8RawStatistics),
}

/// Chunk-oriented association compute implemented by the device runtime.
pub trait AssociationBackend: Send + Sync {
    type GroupState: Send;
    type ChromosomeState: Send + 'static;
    type TransferredInput: Send + 'static;
    type DeviceResult: Send + 'static;
    type Error: std::error::Error + Send + Sync + 'static;

    /// Return the runtime-selected implementations that affect reproducibility.
    ///
    /// Backends without optional implementations retain the empty default.
    fn association_implementation_state(&self) -> crate::AssociationImplementationState {
        crate::AssociationImplementationState::default()
    }

    /// Return the genotype delivery modes supported by this backend instance.
    fn genotype_delivery_capability(&self) -> GenotypeDeliveryCapability;

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

    /// Asynchronously transfer one validated genotype batch to the device.
    /// The delivery thread may call this concurrently with `compute_batch` and
    /// `materialize_batch` calls on the pipeline workers.
    ///
    /// # Errors
    ///
    /// Returns an error when the genotype batch cannot be transferred.
    fn transfer_batch(
        &self,
        group: &Self::GroupState,
        input: g_genotype::GenotypeBatch,
    ) -> Result<Self::TransferredInput, Self::Error>;

    /// Submit one transferred genotype batch and return an opaque device result.
    ///
    /// # Errors
    ///
    /// Returns an error when the genotype batch is invalid or device execution
    /// cannot be submitted.
    fn compute_batch(
        &self,
        chromosome: &Self::ChromosomeState,
        input: Self::TransferredInput,
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
    ) -> Result<MaterializedAssociationBatch, Self::Error>;
}
