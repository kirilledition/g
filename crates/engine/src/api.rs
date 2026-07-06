//! Public engine crate facade.

pub use crate::backend::{
    AssociationBackend, AssociationBatchResult, BackendError, GenotypeBatchView, PredictionView, PreparedGroupInput,
};
pub use crate::callback_diagnostics::{
    CallbackDiagnosticsError, NullLogisticNonconvergenceAction, NullLogisticNonconvergencePlan,
    plan_null_logistic_nonconvergence,
};
pub use crate::callback_progress::{
    CallbackChunkIdentity, CallbackProgressCompletion, CallbackProgressState, CallbackProgressTelemetryEvent,
    CallbackProgressTelemetryPlan, CallbackProgressTelemetryRecord, CallbackProgressUpdate,
};
pub use crate::callback_queue::BoundedCallbackQueue;
pub use crate::callback_summary::{
    BinaryChunkDiagnosticsInput, BinaryCorrectionDiagnosticsRecordPlan, BinaryCorrectionSummaryEmitPlan,
    BinaryCorrectionSummaryState,
};
pub use crate::coordinator::{
    EngineChromosomeRunInput, EngineChromosomeRunReport, EngineCoordinator, EngineGroupChromosomeInput,
    EngineGroupRunInput, EngineGroupRunReport, EngineRunInput, EngineRunReport, InjectedCoordinatorFailure,
};
pub use crate::effects::{EngineEffectError, EngineEffectOperation, EngineRunEffects, NoopEngineRunEffects};
pub use crate::error::{EngineError, EngineResult};
pub use crate::output_manifest::build_current_run_manifest_header_json_from_value_with_cache;
pub use crate::phase::RunPhase;
pub use crate::pipeline::Regenie2RunEngineCore;
pub use crate::preflight::{
    MultiTraitPreflightShapePayload, PreflightError, PreflightReportPayload, SingleTraitPreflightShapePayload,
    build_preflight_report_payload, build_preflight_warnings, resolve_preflight_variant_count,
    resolve_scanned_variant_count, validate_binary_phenotype_case_control_counts, validate_binary_phenotype_coding,
    validate_covariate_matrix_rank, validate_finite_array, validate_multi_prediction_preflight_shape,
    validate_multi_trait_preflight_shape_payload, validate_single_prediction_preflight_shape,
    validate_single_trait_preflight_shape_payload,
};
pub use crate::preparation::{
    PipelineOutputInitialization, PipelineOutputPreparationBatch, PipelineResumeCompatibilityError,
    initialize_pipeline_output_run_batch, initialize_pipeline_output_runs, validate_pipeline_resume_compatibility,
};
pub use crate::schedule::{
    BgenDeliveryCleanupPlan, BgenDeliveryInvocationPlan, BgenDeliveryMethod, CallbackQueueBackpressureObservation,
    CallbackQueueGetAttemptPlan, CallbackQueueGetObservationPlan, CallbackQueueOccupancyState,
    CallbackQueueOperationObservationPlan, CallbackQueuePutAttemptPlan, CallbackQueuePutObservationPlan,
    CallbackQueueStageBackpressureObservation, CallbackQueueStageObservationPlan, CallbackSchedulerState,
    CallbackWorkerAbortPlan, CallbackWorkerErrorRaisePlan, CallbackWorkerErrorUpdatePlan, CallbackWorkerFinishPlan,
    CallbackWorkerJoinPlan, CallbackWorkerLifecycleState, CallbackWorkerShutdownTimeouts,
    CallbackWorkerStartAttemptPlan, CallbackWorkerStartPlan, CallbackWorkerStopPlan, CallbackWorkerStopPollPlan,
    ChunkBatchPlan, DosageBufferAcquireAttemptPlan, DosageBufferDiscardAttemptPlan, DosageBufferPoolObservationPlan,
    DosageBufferPoolState, DosageBufferRegisterAttemptPlan, DosageBufferReturnAttemptPlan, DosageBufferReusePlan,
    DosageWorkDrainCompletionPlan, DosageWorkHandoffPlan, DosageWorkItemDispatchPlan, DosageWorkItemStageDurationPlan,
    GpuGenotypeFormatResolutionPlan, MultiTraitChunkWritePlan, MultiTraitOutputWritePlan, NativeCallbackQueueLimits,
    ResultInFlightAcquireAttemptPlan, ResultInFlightAcquireObservationPlan, ResultInFlightReleaseAttemptPlan,
    ResultInFlightReleaseObservationPlan, ResultInFlightSlotState, ResultWriteDrainCompletionPlan,
    ResultWriteHandoffPlan, ResultWriteItemDispatchPlan, ResultWriteItemResourceReleasePlan, ScheduleError,
    SingleTraitOutputWritePlan, VariantMajorDosageBatchHandoffPlan, WriterFinishExecutionPlan,
    callback_worker_backpressure_poll_timeout_seconds, callback_worker_shutdown_timeouts,
    format_dosage_callback_worker_error_message, format_result_callback_worker_error_message,
    intersect_committed_chunk_identifier_sets, plan_auto_gpu_genotype_format_after_trusted_validation,
    plan_bgen_delivery_cleanup, plan_bgen_delivery_invocation, plan_callback_queue_backpressure_observation,
    plan_callback_queue_get_observation, plan_callback_queue_operation_observation,
    plan_callback_queue_put_observation, plan_callback_queue_stage_backpressure_observation,
    plan_callback_queue_stage_observation, plan_callback_worker_abort, plan_callback_worker_finish,
    plan_callback_worker_start, plan_callback_worker_stop_poll, plan_chunk_batches,
    plan_dosage_buffer_pool_observation, plan_dosage_buffer_reuse, plan_dosage_callback_worker_join,
    plan_dosage_callback_worker_stop, plan_dosage_work_handoff, plan_dosage_work_item_dispatch,
    plan_dosage_work_item_stage_duration, plan_gpu_genotype_format_auto_to_dosage, plan_multi_trait_chunk_write,
    plan_multi_trait_output_write, plan_result_callback_worker_join, plan_result_callback_worker_stop,
    plan_result_in_flight_slot_acquire_observation, plan_result_in_flight_slot_release_observation,
    plan_result_write_handoff, plan_result_write_item_dispatch,
    plan_single_trait_binary_gpu_genotype_format_resolution, plan_single_trait_output_write,
    plan_variant_major_dosage_batch_handoff, plan_writer_finish_execution, resolve_bgen_delivery_method,
    resolve_callback_worker_stop_poll_timeout_seconds, resolve_delivery_callback_batch_size,
    resolve_effective_trusted_no_missing_diploid, resolve_grouped_union_callback_batch_size,
    resolve_manifest_gpu_genotype_format, resolve_native_callback_queue_limits, resolve_writer_finish_thread_count,
    should_attempt_callback_worker_stop,
};
pub use crate::trusted_validation::TrustedBgenValidationError;
