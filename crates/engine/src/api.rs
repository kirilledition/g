//! Public engine crate facade.

pub use crate::association_implementation::{
    AssociationImplementationState, FirthComponentsFallbackReason, FirthComponentsImplementation,
    FirthComponentsImplementationState, JaxRuntimeVersions,
};
pub use crate::backend::{
    AssociationBackend, GenotypeDeliveryCapability, GenotypeTransferPreparation, GroupPreparationInput,
    MaterializedAssociationBatch, MaterializedGenotypeStatistics, PreparedChromosome, SampleMajorCovariateMatrix,
    TraitMajorMatrix,
};
pub use crate::run::RunHooks;
pub use crate::run_coordinator::{EngineRunError, PhenotypeRunArtifact, execute_coordinated_run};
