//! Public engine crate facade.

pub use crate::backend::{
    AssociationBackend, GenotypeBatchInput, GenotypeBatchStatisticsView, GenotypeMatrixView, GroupPreparationInput,
    HostAssociationBatch, HostAssociationStatisticMatrix, HostAssociationStatistics, HostCorrectionCodeMatrix,
    MaterializationInput, PreparedChromosome, SampleMajorCovariateMatrixView, TraitMajorPhenotypeMatrixView,
    TraitMajorPredictionMatrixView, VariantMajorDosageMatrixView, VariantMajorPacked8MatrixView,
};
pub use crate::backend_settings::{
    JaxApproximateFirthSettings, JaxBackendSettings, JaxBinaryNullLogisticSettings, JaxBinaryNumericalSettings,
    JaxBinarySettings, JaxCorrectionSettings, JaxFirthCandidateSettings, JaxLinearSettings, JaxNullFirthSettings,
};
pub use crate::run::RunHooks;
pub use crate::run_coordinator::{EngineRunError, execute_coordinated_run};
