//! Public engine crate facade.

pub use crate::backend::{
    AssociationBackend, GenotypeBatchInput, GenotypeBatchStatistics, GroupPreparationInput, HostAssociationBatch,
    OwnedGenotypeBuffer, PreparedChromosome, SampleMajorCovariateMatrix, TraitMajorMatrix,
};
pub use crate::run::RunHooks;
pub use crate::run_coordinator::{EngineRunError, PhenotypeRunArtifact, execute_coordinated_run};
