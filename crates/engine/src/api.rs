//! Public engine crate facade.

pub use crate::backend::{
    AssociationBackend, GenotypeDeliveryCapability, GenotypeTransferPreparation, GroupPreparationInput,
    MaterializedAssociationBatch, MaterializedGenotypeStatistics, PreparedChromosome, SampleMajorCovariateMatrix,
    TraitMajorMatrix,
};
pub use crate::run::{
    EnginePostSessionCleanup, EnginePostSessionCleanupError, EnginePostSessionCleanupPurpose, RunHooks,
};
pub use crate::run_coordinator::{
    ClaimedCoordinatedRun, EngineClaimError, EngineExecutionOutcome, EngineRunError, PhenotypeRunArtifact,
    claim_coordinated_run, execute_coordinated_run,
};
