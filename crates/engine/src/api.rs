//! Public engine crate facade.

pub use crate::association_scheduler::{
    AssociationBatchPipeline, CompletedAssociationBatch, OwnedGenotypeBuffer, ScheduledAssociationBatch, SchedulerError,
};
pub use crate::backend::{
    AssociationBackend, BackendError, BinaryBatchDiagnostics, ChromosomePreparationInput, GenotypeBatchInput,
    GenotypeBatchStatisticsView, GenotypeMatrixView, GroupPreparationInput, HostAssociationBatch,
    HostAssociationStatisticMatrix, HostAssociationStatistics, HostExtraCodeMatrix, MaterializationInput,
    NullModelDiagnostics, PreparedChromosome, SampleMajorCovariateMatrixView, TraitMajorPhenotypeMatrixView,
    TraitMajorPredictionMatrixView, VariantMajorDosageMatrixView, VariantMajorPacked8MatrixView,
};
pub use crate::delivery_schedule::{
    GpuGenotypeFormatResolutionPlan, plan_auto_gpu_genotype_format_after_trusted_validation,
    plan_gpu_genotype_format_auto_to_dosage, plan_single_trait_binary_gpu_genotype_format_resolution,
    resolve_effective_trusted_no_missing_diploid, resolve_manifest_gpu_genotype_format,
};
pub use crate::null_logistic_policy::{
    NullLogisticNonconvergenceAction, NullLogisticNonconvergencePlan, NullLogisticPolicyError,
    plan_null_logistic_nonconvergence,
};
pub use crate::output_schedule::{
    MultiTraitChunkWritePlan, intersect_committed_chunk_identifier_sets, plan_multi_trait_chunk_write,
};
pub use crate::pipeline::Regenie2RunEngineCore;
pub use crate::preflight::{
    MultiTraitPreflightShapePayload, PreflightError, PreflightReportPayload, SingleTraitPreflightShapePayload,
    build_preflight_report_payload, validate_multi_prediction_values, validate_multi_trait_preflight_values,
    validate_single_prediction_values, validate_single_trait_preflight_values,
};
pub use crate::preparation::{
    PipelineOutputInitialization, PipelineOutputPreparationBatch, PipelineOutputPreparationError,
    PipelineResumeCompatibilityError, RuntimeOutputGroup, RuntimeOutputGroupInput, RuntimeOutputPhenotypeComputeGroup,
    RuntimeOutputPlan, RuntimeOutputPreparationGroup, RuntimeOutputPreparedRun,
    build_output_resume_committed_chunk_diagnostic_payloads, build_runtime_output_preparation_group,
};
pub use crate::schedule_error::ScheduleError;
pub use crate::trusted_validation::TrustedBgenValidationError;
