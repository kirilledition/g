//! Public engine crate facade.

pub use crate::association_scheduler::SchedulerError;
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
pub use crate::delivery_execution::DeliveryError;
pub use crate::preparation::PipelineOutputPreparationError;
pub use crate::run::{RunExecutionError, RunHooks, RunPreparationError};
pub use crate::run_coordinator::{CoordinatedRunError, execute_coordinated_run};
pub use crate::trusted_validation::TrustedBgenValidationError;
pub use g_genotype::{BgenError, GenotypeError};
pub use g_input::{InputError, PredictionError};
pub use g_output::OutputError;
