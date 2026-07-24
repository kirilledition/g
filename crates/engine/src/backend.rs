//! Python-free association backend contract.

use crate::cuda_capability::ptx_target_matches_compute_capability;

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
    /// Accept host-decoded values and raw-DEFLATE packed8 through this artifact.
    RawDeflatePacked8(RawDeflatePacked8ArtifactIdentity),
}

/// Stable identity of the raw-DEFLATE packed8 typed-XLA artifact and ABI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RawDeflatePacked8ArtifactIdentity {
    ffi_target: &'static str,
    ffi_api_version: u32,
    handler_sha256: &'static str,
    ptx_sha256: &'static str,
    ptx_isa: &'static str,
    ptx_target: &'static str,
    capability_requirements: RawDeflatePacked8CapabilityRequirements,
}

/// Minimum CUDA runtime capabilities required by a raw-DEFLATE packed8 artifact.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RawDeflatePacked8CapabilityRequirements {
    cuda_driver_version: i32,
    compute_capability_major: i32,
    compute_capability_minor: i32,
}

impl RawDeflatePacked8CapabilityRequirements {
    /// Build validated minimum CUDA runtime requirements.
    #[must_use]
    pub const fn new(
        minimum_cuda_driver_version: i32,
        minimum_compute_capability_major: i32,
        minimum_compute_capability_minor: i32,
    ) -> Option<Self> {
        if minimum_cuda_driver_version <= 0
            || minimum_compute_capability_major <= 0
            || minimum_compute_capability_minor < 0
        {
            return None;
        }
        Some(Self {
            cuda_driver_version: minimum_cuda_driver_version,
            compute_capability_major: minimum_compute_capability_major,
            compute_capability_minor: minimum_compute_capability_minor,
        })
    }

    /// Return the reviewed minimum CUDA driver API version.
    #[must_use]
    pub const fn minimum_cuda_driver_version(self) -> i32 {
        self.cuda_driver_version
    }

    /// Return the minimum compute-capability major version.
    #[must_use]
    pub const fn minimum_compute_capability_major(self) -> i32 {
        self.compute_capability_major
    }

    /// Return the minimum compute-capability minor version.
    #[must_use]
    pub const fn minimum_compute_capability_minor(self) -> i32 {
        self.compute_capability_minor
    }
}

impl RawDeflatePacked8ArtifactIdentity {
    /// Build a validated raw-DEFLATE packed8 artifact identity.
    #[must_use]
    pub const fn new(
        ffi_target: &'static str,
        ffi_api_version: u32,
        handler_sha256: &'static str,
        ptx_sha256: &'static str,
        ptx_isa: &'static str,
        ptx_target: &'static str,
        capability_requirements: RawDeflatePacked8CapabilityRequirements,
    ) -> Option<Self> {
        if ffi_target.is_empty()
            || ffi_api_version == 0
            || !is_canonical_sha256(handler_sha256)
            || !is_canonical_sha256(ptx_sha256)
            || ptx_isa.is_empty()
            || ptx_target.is_empty()
            || !ptx_target_matches_compute_capability(
                ptx_target,
                capability_requirements.minimum_compute_capability_major(),
                capability_requirements.minimum_compute_capability_minor(),
            )
        {
            return None;
        }
        Some(Self {
            ffi_target,
            ffi_api_version,
            handler_sha256,
            ptx_sha256,
            ptx_isa,
            ptx_target,
            capability_requirements,
        })
    }

    /// Return the exact typed-XLA FFI target.
    #[must_use]
    pub const fn ffi_target(self) -> &'static str {
        self.ffi_target
    }

    /// Return the typed-XLA FFI API version.
    #[must_use]
    pub const fn ffi_api_version(self) -> u32 {
        self.ffi_api_version
    }

    /// Return the framed native handler and ABI input SHA-256.
    #[must_use]
    pub const fn handler_sha256(self) -> &'static str {
        self.handler_sha256
    }

    /// Return the canonical lowercase SHA-256 of the embedded PTX.
    #[must_use]
    pub const fn ptx_sha256(self) -> &'static str {
        self.ptx_sha256
    }

    /// Return the PTX ISA version used by the embedded artifact.
    #[must_use]
    pub const fn ptx_isa(self) -> &'static str {
        self.ptx_isa
    }

    /// Return the PTX target used by the embedded artifact.
    #[must_use]
    pub const fn ptx_target(self) -> &'static str {
        self.ptx_target
    }

    /// Return the reviewed minimum CUDA driver API version.
    #[must_use]
    pub const fn minimum_cuda_driver_version(self) -> i32 {
        self.capability_requirements.minimum_cuda_driver_version()
    }

    /// Return the embedded PTX target's minimum compute-capability major version.
    #[must_use]
    pub const fn minimum_compute_capability_major(self) -> i32 {
        self.capability_requirements.minimum_compute_capability_major()
    }

    /// Return the embedded PTX target's minimum compute-capability minor version.
    #[must_use]
    pub const fn minimum_compute_capability_minor(self) -> i32 {
        self.capability_requirements.minimum_compute_capability_minor()
    }
}

const fn is_canonical_sha256(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() != 64 {
        return false;
    }
    let mut index = 0;
    while index < bytes.len() {
        if !matches!(bytes[index], b'0'..=b'9' | b'a'..=b'f') {
            return false;
        }
        index += 1;
    }
    true
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

/// Chunk-oriented association compute implemented by the current JAX runtime.
///
/// Every production implementation must report exact JAX/`JAXlib` state. A
/// future non-JAX backend requires an explicit implementation-state and output
/// schema extension rather than fabricated JAX versions.
pub trait AssociationBackend: Send + Sync {
    type GroupState: Send + Sync + 'static;
    type ChromosomeState: Send + 'static;
    type TransferredInput: Send + 'static;
    type DeviceResult: Send + 'static;
    type Error: std::error::Error + Send + Sync + 'static;

    /// Return the runtime-selected implementations that affect reproducibility.
    fn association_implementation_state(&self) -> crate::AssociationImplementationState;

    /// Return the genotype delivery modes supported by this backend instance.
    fn genotype_delivery_capability(&self) -> GenotypeDeliveryCapability;

    /// Prepare reusable device state for one phenotype group.
    ///
    /// # Errors
    ///
    /// Returns an error when the phenotype or covariate data cannot be prepared.
    fn prepare_group(&self, input: GroupPreparationInput) -> Result<Self::GroupState, Self::Error>;

    /// Release group state after final chromosome completion and worker teardown.
    fn release_group(&self, group: Self::GroupState) {
        drop(group);
    }

    /// Prepare reusable state and null-logistic policy input for one chromosome.
    ///
    /// The association scheduler invokes this hook on the same backend
    /// execution worker that invokes `compute_batch` and
    /// `release_chromosome`.
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
    /// override this hook. The scheduler calls it only after materialization
    /// is quiescent and every produced device result has been materialized or
    /// dropped. Graceful chromosome transitions additionally require every
    /// submitted batch using the state to have been received.
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
