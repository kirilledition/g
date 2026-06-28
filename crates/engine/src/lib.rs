#![warn(clippy::pedantic)]

pub mod backend;
pub mod callback_diagnostics;
pub mod callback_progress;
pub mod callback_summary;
pub mod coordinator;
pub mod effects;
pub mod fake_backend;
pub mod fake_effects;
pub mod phase;
pub mod pipeline;
pub mod preflight;
pub mod preparation;
pub mod schedule;

pub use backend::{
    AssociationBackend, AssociationBatchResult, BackendError, GenotypeBatchView, PredictionView, PreparedGroupInput,
};
pub use callback_diagnostics::{
    CallbackDiagnosticsError, NullLogisticNonconvergenceAction, NullLogisticNonconvergencePlan,
    plan_null_logistic_nonconvergence,
};
pub use callback_progress::{
    CallbackChunkIdentity, CallbackProgressCompletion, CallbackProgressState, CallbackProgressTelemetryEvent,
    CallbackProgressTelemetryPlan, CallbackProgressTelemetryRecord, CallbackProgressUpdate,
};
pub use callback_summary::{BinaryChunkDiagnosticsInput, BinaryCorrectionSummaryState};
pub use coordinator::{EngineCoordinator, EngineError, EngineRunInput, EngineRunReport, InjectedCoordinatorFailure};
pub use effects::{EngineEffectError, EngineEffectOperation, EngineRunEffects, NoopEngineRunEffects};
pub use fake_backend::{FakeBackend, FakeBackendFailure, FakeChromosomeState, FakeGroupState};
pub use fake_effects::{FakeEngineRunEffects, FakeOutputLifecycleState, FakeOutputState, FakeRunEffectState};
pub use phase::RunPhase;
pub use pipeline::Regenie2RunEngineCore;
pub use preflight::{
    PreflightError, PreflightReportPayload, build_preflight_report_payload, build_preflight_warnings,
    resolve_scanned_variant_count, validate_binary_phenotype_case_control_counts, validate_binary_phenotype_coding,
    validate_covariate_matrix_rank, validate_finite_array, validate_multi_prediction_preflight_shape,
    validate_multi_trait_preflight_shape_payload, validate_single_prediction_preflight_shape,
    validate_single_trait_preflight_shape_payload,
};
pub use preparation::{
    PipelineResumeCompatibilityError, initialize_pipeline_output_runs, validate_pipeline_resume_compatibility,
};
pub use schedule::{
    BgenDeliveryCleanupPlan, BgenDeliveryInvocationPlan, BgenDeliveryMethod, CallbackQueueOperationObservationPlan,
    CallbackQueueStageObservationPlan, CallbackWorkerAbortPlan, CallbackWorkerFinishPlan, CallbackWorkerJoinPlan,
    CallbackWorkerLifecycleState, CallbackWorkerShutdownTimeouts, CallbackWorkerStopPlan, CallbackWorkerStopPollPlan,
    DosageBufferPoolState, DosageBufferReusePlan, GpuGenotypeFormatResolutionPlan, MultiTraitOutputWritePlan,
    NativeCallbackQueueLimits, ResultInFlightSlotState, ScheduleError, SingleTraitOutputWritePlan,
    VariantMajorDosageBatchHandoffPlan, WriterFinishExecutionPlan, callback_worker_backpressure_poll_timeout_seconds,
    callback_worker_shutdown_timeouts, format_dosage_callback_worker_error_message,
    format_result_callback_worker_error_message, intersect_committed_chunk_identifier_sets,
    plan_auto_gpu_genotype_format_after_trusted_validation, plan_bgen_delivery_cleanup, plan_bgen_delivery_invocation,
    plan_callback_queue_operation_observation, plan_callback_queue_stage_observation, plan_callback_worker_abort,
    plan_callback_worker_finish, plan_callback_worker_stop_poll, plan_dosage_buffer_reuse,
    plan_dosage_callback_worker_join, plan_dosage_callback_worker_stop, plan_gpu_genotype_format_auto_to_dosage,
    plan_multi_trait_output_write, plan_result_callback_worker_join, plan_result_callback_worker_stop,
    plan_single_trait_binary_gpu_genotype_format_resolution, plan_single_trait_output_write,
    plan_variant_major_dosage_batch_handoff, plan_writer_finish_execution, resolve_bgen_delivery_method,
    resolve_callback_worker_stop_poll_timeout_seconds, resolve_delivery_callback_batch_size,
    resolve_effective_trusted_no_missing_diploid, resolve_grouped_union_callback_batch_size,
    resolve_manifest_gpu_genotype_format, resolve_native_callback_queue_limits, resolve_writer_finish_thread_count,
    should_attempt_callback_worker_stop,
};
