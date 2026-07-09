//! Pure scheduling and resume policy helpers for engine-owned delivery.

pub use crate::callback_observation_schedule::{
    CallbackQueueBackpressureObservation, CallbackQueueOperationObservationPlan,
    CallbackQueueStageBackpressureObservation, CallbackQueueStageObservationPlan,
    plan_callback_queue_backpressure_observation, plan_callback_queue_operation_observation,
    plan_callback_queue_stage_backpressure_observation, plan_callback_queue_stage_observation,
};
pub(crate) use crate::callback_observation_schedule::{
    DOSAGE_BUFFER_POOL_NAME, DOSAGE_QUEUE_NAME, QUEUE_ALLOCATE_OPERATION, QUEUE_CONSUMER_WAIT_OPERATION,
    QUEUE_DISCARD_OPERATION, QUEUE_PRODUCER_BLOCKING_OPERATION, QUEUE_PUT_OPERATION, QUEUE_RETURN_OPERATION,
    QUEUE_REUSE_OPERATION, RESULT_IN_FLIGHT_SLOTS_NAME, RESULT_QUEUE_NAME, RESULT_SLOT_ACQUIRE_OPERATION,
    RESULT_SLOT_RELEASE_OPERATION,
};
pub use crate::callback_worker_schedule::{
    CallbackWorkerAbortPlan, CallbackWorkerErrorRaisePlan, CallbackWorkerErrorUpdatePlan, CallbackWorkerFinishAction,
    CallbackWorkerFinishPlan, CallbackWorkerJoinPlan, CallbackWorkerLifecycleState, CallbackWorkerShutdownTimeouts,
    CallbackWorkerStartAction, CallbackWorkerStartAttemptPlan, CallbackWorkerStartPlan, CallbackWorkerStopPlan,
    CallbackWorkerStopPollPlan, callback_worker_backpressure_poll_timeout_seconds, callback_worker_shutdown_timeouts,
    format_dosage_callback_worker_error_message, format_result_callback_worker_error_message,
    plan_callback_worker_abort, plan_callback_worker_finish, plan_callback_worker_start,
    plan_callback_worker_stop_poll, plan_dosage_callback_worker_join, plan_dosage_callback_worker_stop,
    plan_result_callback_worker_join, plan_result_callback_worker_stop,
    resolve_callback_worker_stop_poll_timeout_seconds, should_attempt_callback_worker_stop,
};
pub(crate) use crate::callback_worker_schedule::{
    plan_callback_worker_error_raise, plan_callback_worker_start_attempt, update_callback_worker_error,
};
pub use crate::delivery_schedule::{
    BgenDeliveryCleanupAction, BgenDeliveryCleanupOutcome, BgenDeliveryCleanupPlan, BgenDeliveryInvocationPlan,
    BgenDeliveryMethod, GpuGenotypeFormatResolutionPlan, plan_auto_gpu_genotype_format_after_trusted_validation,
    plan_bgen_delivery_cleanup, plan_bgen_delivery_invocation, plan_gpu_genotype_format_auto_to_dosage,
    plan_single_trait_binary_gpu_genotype_format_resolution, resolve_bgen_delivery_method,
    resolve_delivery_callback_batch_size, resolve_effective_trusted_no_missing_diploid,
    resolve_grouped_union_callback_batch_size, resolve_manifest_gpu_genotype_format,
};
pub use crate::output_schedule::{
    MultiTraitChunkWritePlan, MultiTraitOutputWritePlan, SingleTraitOutputWritePlan, WriterFinishExecutionPlan,
    intersect_committed_chunk_identifier_sets, plan_multi_trait_chunk_write, plan_multi_trait_output_write,
    plan_single_trait_output_write, plan_writer_finish_execution, resolve_writer_finish_thread_count,
};
pub use callback_queue::{
    CallbackQueueGetAttemptPlan, CallbackQueueGetObservationPlan, CallbackQueueOccupancyState,
    CallbackQueuePutAttemptPlan, CallbackQueuePutObservationPlan, plan_callback_queue_get_observation,
    plan_callback_queue_put_observation,
};
use callback_queue::{plan_callback_queue_get_attempt, plan_callback_queue_put_attempt};
pub use chunk_batch::{
    ChunkBatchPlan, DosageWorkHandoffPlan, VariantMajorDosageBatchHandoffPlan, plan_chunk_batches,
    plan_dosage_work_handoff, plan_variant_major_dosage_batch_handoff,
};
pub use dosage_buffer_pool::{
    DosageBufferAcquireAttemptPlan, DosageBufferDiscardAttemptPlan, DosageBufferPoolObservationPlan,
    DosageBufferPoolState, DosageBufferRegisterAttemptPlan, DosageBufferReturnAttemptPlan, DosageBufferReusePlan,
    plan_dosage_buffer_pool_observation, plan_dosage_buffer_reuse,
};
use dosage_buffer_pool::{
    plan_dosage_buffer_acquire_attempt, plan_dosage_buffer_discard_attempt, plan_dosage_buffer_register_attempt,
    plan_dosage_buffer_return_attempt,
};
pub use dosage_work::{
    DosageWorkDrainCompletionPlan, DosageWorkItemDispatchPlan, DosageWorkItemKind, DosageWorkItemStageDurationPlan,
    plan_dosage_work_item_dispatch, plan_dosage_work_item_stage_duration,
};
pub use result_slots::{
    ResultInFlightAcquireAttemptPlan, ResultInFlightAcquireObservationPlan, ResultInFlightReleaseAttemptPlan,
    ResultInFlightReleaseObservationPlan, ResultInFlightSlotState, plan_result_in_flight_slot_acquire_observation,
    plan_result_in_flight_slot_release_observation,
};
use result_slots::{plan_result_in_flight_slot_acquire_attempt, plan_result_in_flight_slot_release_attempt};
pub use result_write::{
    ResultWriteDrainCompletionPlan, ResultWriteHandoffPlan, ResultWriteItemDispatchPlan, ResultWriteItemKind,
    ResultWriteItemResourceReleasePlan, plan_result_write_handoff, plan_result_write_item_dispatch,
    plan_result_write_item_dispatch_for_kinds,
};

mod callback_queue;
mod callback_scheduler;
mod chunk_batch;
mod dosage_buffer_pool;
mod dosage_work;
mod error;
mod result_slots;
mod result_write;

pub use callback_scheduler::{CallbackSchedulerState, NativeCallbackQueueLimits, resolve_native_callback_queue_limits};
pub use error::ScheduleError;

fn normalize_callback_queue_wait_timeout_seconds(wait_timeout_seconds: f64) -> f64 {
    if wait_timeout_seconds.is_finite() && wait_timeout_seconds > 0.0 { wait_timeout_seconds } else { 0.0 }
}
