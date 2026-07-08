//! Typed telemetry event kind policy.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RunTelemetryEventKind {
    AssociationBackendSelected,
    BgenEngineOpened,
    BinaryCorrectionSummary,
    EffectiveConfigWritten,
    ExecutionPlanPrepared,
    GpuGenotypeFormatResolved,
    MultiPhenotypeSampleSummary,
    PredictionSourceLoaded,
    PreflightCompleted,
    RunCompleted,
    RunFailed,
    RunInterrupted,
    RunStarted,
    SampleAlignmentCompleted,
    WriterFinished,
}

impl RunTelemetryEventKind {
    #[must_use]
    pub fn event_name(self) -> &'static str {
        match self {
            Self::AssociationBackendSelected => super::super::names::ASSOCIATION_BACKEND_SELECTED_EVENT_NAME,
            Self::BgenEngineOpened => super::super::names::BGEN_ENGINE_OPENED_EVENT_NAME,
            Self::BinaryCorrectionSummary => super::super::names::BINARY_CORRECTION_SUMMARY_EVENT_NAME,
            Self::EffectiveConfigWritten => super::super::names::EFFECTIVE_CONFIG_WRITTEN_EVENT_NAME,
            Self::ExecutionPlanPrepared => super::super::names::EXECUTION_PLAN_PREPARED_EVENT_NAME,
            Self::GpuGenotypeFormatResolved => super::super::names::GPU_GENOTYPE_FORMAT_RESOLVED_EVENT_NAME,
            Self::MultiPhenotypeSampleSummary => super::super::names::MULTI_PHENOTYPE_SAMPLE_SUMMARY_EVENT_NAME,
            Self::PredictionSourceLoaded => super::super::names::PREDICTION_SOURCE_LOADED_EVENT_NAME,
            Self::PreflightCompleted => super::super::names::PREFLIGHT_COMPLETED_EVENT_NAME,
            Self::RunCompleted => super::super::names::RUN_COMPLETED_EVENT_NAME,
            Self::RunFailed | Self::RunInterrupted => super::super::names::RUN_FAILED_EVENT_NAME,
            Self::RunStarted => super::super::names::RUN_STARTED_EVENT_NAME,
            Self::SampleAlignmentCompleted => super::super::names::SAMPLE_ALIGNMENT_COMPLETED_EVENT_NAME,
            Self::WriterFinished => super::super::names::WRITER_FINISHED_EVENT_NAME,
        }
    }

    #[must_use]
    pub fn level(self) -> &'static str {
        match self {
            Self::RunFailed => super::super::names::RUN_LIFECYCLE_ERROR_LEVEL,
            Self::RunInterrupted => super::super::names::RUN_LIFECYCLE_WARN_LEVEL,
            _ => super::super::names::RUN_LIFECYCLE_INFO_LEVEL,
        }
    }
}
