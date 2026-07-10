//! Public engine crate facade.

pub use crate::association_scheduler::SchedulerError;
pub use crate::backend::{
    AssociationBackend, GenotypeBatchInput, GenotypeBatchStatisticsView, GenotypeMatrixView, GroupPreparationInput,
    HostAssociationBatch, HostAssociationStatisticMatrix, HostAssociationStatistics, HostCorrectionCodeMatrix,
    MaterializationInput, PreparedChromosome, SampleMajorCovariateMatrixView, TraitMajorPhenotypeMatrixView,
    TraitMajorPredictionMatrixView, VariantMajorDosageMatrixView, VariantMajorPacked8MatrixView,
};
pub use crate::delivery_execution::{AssociationDeliveryReport, DeliveryError, DeliveryWarning};
pub use crate::null_logistic_policy::NullLogisticPolicyError;
pub use crate::preflight::PreflightError;
pub use crate::preparation::PipelineOutputPreparationError;
pub use crate::progress::{RunProgressError, RunProgressReporter};
pub use crate::run::{
    PreparedRun, RunEngine, RunExecution, RunExecutionError, RunHooks, RunPreparationError, validate_jax_integer_domain,
};
pub use crate::run_coordinator::{CoordinatedRunError, execute_coordinated_run};
pub use crate::trusted_validation::TrustedBgenValidationError;
pub use g_genotype::{BgenError, GenotypeError};
pub use g_input::{InputError, PredictionError};
pub use g_output::{CompletedOutputRun, OutputError};
