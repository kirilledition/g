//! Pure scheduling and resume policy helpers for engine-owned delivery.

use std::collections::BTreeSet;

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
#[cfg(test)]
pub(crate) use crate::callback_worker_schedule::{
    CALLBACK_WORKER_FINISH_COMPLETE_PROGRESS_ACTION, CALLBACK_WORKER_FINISH_EMIT_BINARY_CORRECTION_SUMMARY_ACTION,
    CALLBACK_WORKER_FINISH_JOIN_DOSAGE_WORKER_ACTION, CALLBACK_WORKER_FINISH_JOIN_RESULT_WORKER_ACTION,
    CALLBACK_WORKER_FINISH_RAISE_WORKER_ERROR_ACTION, CALLBACK_WORKER_FINISH_STOP_DOSAGE_WORKER_ACTION,
    CALLBACK_WORKER_FINISH_STOP_RESULT_WORKER_ACTION, CALLBACK_WORKER_START_DOSAGE_WORKER_ACTION,
    CALLBACK_WORKER_START_RESULT_WORKER_ACTION,
};
pub use crate::callback_worker_schedule::{
    CallbackWorkerAbortPlan, CallbackWorkerErrorRaisePlan, CallbackWorkerErrorUpdatePlan, CallbackWorkerFinishPlan,
    CallbackWorkerJoinPlan, CallbackWorkerLifecycleState, CallbackWorkerShutdownTimeouts,
    CallbackWorkerStartAttemptPlan, CallbackWorkerStartPlan, CallbackWorkerStopPlan, CallbackWorkerStopPollPlan,
    callback_worker_backpressure_poll_timeout_seconds, callback_worker_shutdown_timeouts,
    format_dosage_callback_worker_error_message, format_result_callback_worker_error_message,
    plan_callback_worker_abort, plan_callback_worker_finish, plan_callback_worker_start,
    plan_callback_worker_stop_poll, plan_dosage_callback_worker_join, plan_dosage_callback_worker_stop,
    plan_result_callback_worker_join, plan_result_callback_worker_stop,
    resolve_callback_worker_stop_poll_timeout_seconds, should_attempt_callback_worker_stop,
};
pub(crate) use crate::callback_worker_schedule::{
    plan_callback_worker_error_raise, plan_callback_worker_start_attempt, update_callback_worker_error,
};
#[cfg(test)]
pub(crate) use crate::delivery_schedule::{
    BGEN_DELIVERY_CLEANUP_ACTION_ABORT_CALLBACK, BGEN_DELIVERY_CLEANUP_ACTION_ABORT_WRITER_SESSIONS,
    BGEN_DELIVERY_CLEANUP_ACTION_DRAIN_CALLBACK, BGEN_DELIVERY_CLEANUP_ACTION_FINISH_INTERRUPTED_WRITER_SESSIONS,
    BGEN_DELIVERY_CLEANUP_ACTION_FINISH_WRITER_SESSIONS, BGEN_DELIVERY_CLEANUP_ACTION_WRITE_STAGE_TIMING_SNAPSHOT,
    BGEN_DELIVERY_CLEANUP_FAILURE, BGEN_DELIVERY_CLEANUP_INTERRUPTED,
    BGEN_DELIVERY_CLEANUP_INTERRUPTED_CLEANUP_FAILURE, BGEN_DELIVERY_CLEANUP_SUCCESS, build_bgen_delivery_cleanup_plan,
};
pub use crate::delivery_schedule::{
    BgenDeliveryCleanupPlan, BgenDeliveryInvocationPlan, BgenDeliveryMethod, GpuGenotypeFormatResolutionPlan,
    plan_auto_gpu_genotype_format_after_trusted_validation, plan_bgen_delivery_cleanup, plan_bgen_delivery_invocation,
    plan_gpu_genotype_format_auto_to_dosage, plan_single_trait_binary_gpu_genotype_format_resolution,
    resolve_bgen_delivery_method, resolve_delivery_callback_batch_size, resolve_effective_trusted_no_missing_diploid,
    resolve_grouped_union_callback_batch_size, resolve_manifest_gpu_genotype_format,
};
pub use crate::output_schedule::{
    MultiTraitChunkWritePlan, MultiTraitOutputWritePlan, SingleTraitOutputWritePlan, WriterFinishExecutionPlan,
    intersect_committed_chunk_identifier_sets, plan_multi_trait_chunk_write, plan_multi_trait_output_write,
    plan_single_trait_output_write, plan_writer_finish_execution, resolve_writer_finish_thread_count,
};
#[cfg(test)]
pub(crate) use crate::output_schedule::{REGENIE2_NATIVE_CHUNK_WRITE_F64_METHOD, REGENIE2_NATIVE_CHUNK_WRITE_METHOD};
use g_genotype::ChunkSpec;

const RESULT_WRITE_ITEM_KIND_SINGLE_RESULT: &str = "single_result";
const RESULT_WRITE_ITEM_KIND_MULTI_RESULT: &str = "multi_result";
const RESULT_WRITE_ITEM_KIND_STOP_SIGNAL: &str = "stop_signal";
const DOSAGE_WORK_ITEM_KIND_SAMPLE_MAJOR_DOSAGE: &str = "sample_major_dosage";
const DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_DOSAGE: &str = "variant_major_dosage";
const DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_DOSAGE_BATCH: &str = "variant_major_dosage_batch";
const DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_PACKED8_PROBABILITY_PAIR: &str = "variant_major_packed8_probability_pair";
const DOSAGE_WORK_ITEM_KIND_STOP_SIGNAL: &str = "stop_signal";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeCallbackQueueLimits {
    pub dosage_queue_depth: usize,
    pub result_queue_depth: usize,
    pub result_in_flight_limit: usize,
    pub dosage_buffer_limit: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DosageBufferReusePlan {
    pub requires_slice: bool,
    pub slice_dimensions: Vec<usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VariantMajorDosageBatchHandoffPlan {
    pub chunk_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkBatchPlan {
    pub chunk_batches: Vec<Vec<ChunkSpec>>,
}

impl ChunkBatchPlan {
    #[must_use]
    pub fn chunk_batch_count(&self) -> usize {
        self.chunk_batches.len()
    }

    #[must_use]
    pub fn chunk_count(&self) -> usize {
        self.chunk_batches.iter().map(Vec::len).sum()
    }

    #[must_use]
    pub fn into_chunk_batches(self) -> Vec<Vec<ChunkSpec>> {
        self.chunk_batches
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DosageWorkHandoffPlan {
    pub chunk_count: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CallbackBoundedResourceOccupancy {
    queue_depth: usize,
    queue_capacity: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CallbackQueuePutAttemptPlan {
    pub should_put: bool,
    pub should_wait: bool,
    pub wait_timeout_seconds: f64,
    pub queue_depth: usize,
    pub queue_capacity: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CallbackQueuePutObservationPlan {
    pub queue_name: String,
    pub operation_name: String,
    pub blocked: bool,
    pub should_retry_put: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CallbackQueueGetAttemptPlan {
    pub should_get: bool,
    pub should_wait: bool,
    pub has_release_error: bool,
    pub wait_timeout_seconds: f64,
    pub queue_depth: usize,
    pub queue_capacity: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CallbackQueueGetObservationPlan {
    pub queue_name: String,
    pub operation_name: String,
    pub blocked: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResultInFlightAcquireAttemptPlan {
    pub should_acquire: bool,
    pub should_wait: bool,
    pub wait_timeout_seconds: f64,
    pub occupied_count: usize,
    pub slot_limit: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResultInFlightAcquireObservationPlan {
    pub resource_name: String,
    pub operation_name: String,
    pub blocked: bool,
    pub should_retry_acquisition: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResultInFlightReleaseAttemptPlan {
    pub should_release: bool,
    pub has_release_error: bool,
    pub occupied_count: usize,
    pub slot_limit: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResultInFlightReleaseObservationPlan {
    pub resource_name: String,
    pub operation_name: String,
    pub blocked: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResultWriteItemResourceReleasePlan {
    pub should_release_host_buffer: bool,
    pub should_release_result_in_flight_slot: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResultWriteHandoffPlan {
    pub should_enqueue: bool,
    pub has_result_work_item: bool,
    pub is_stop_signal: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResultWriteDrainCompletionPlan {
    pub should_stop: bool,
    pub should_flush_binary_correction_diagnostics: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResultWriteItemDispatchPlan {
    pub result_work_item_kind: String,
    pub expected_result_work_item_kind: String,
    pub should_process_result_write_item: bool,
    pub should_process_multi_result_write_item: bool,
    pub has_dispatch_error: bool,
    pub error_message: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DosageWorkDrainCompletionPlan {
    pub should_stop: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DosageWorkItemDispatchPlan {
    pub dosage_work_item_kind: String,
    pub processing_path: Option<String>,
    pub error_message: Option<String>,
}

impl DosageWorkItemDispatchPlan {
    #[must_use]
    pub fn should_process_sample_major_dosage(&self) -> bool {
        self.processing_path.as_deref() == Some(DOSAGE_WORK_ITEM_KIND_SAMPLE_MAJOR_DOSAGE)
    }

    #[must_use]
    pub fn should_process_variant_major_dosage(&self) -> bool {
        self.processing_path.as_deref() == Some(DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_DOSAGE)
    }

    #[must_use]
    pub fn should_process_variant_major_dosage_batch(&self) -> bool {
        self.processing_path.as_deref() == Some(DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_DOSAGE_BATCH)
    }

    #[must_use]
    pub fn should_process_variant_major_packed8_probability_pair(&self) -> bool {
        self.processing_path.as_deref() == Some(DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_PACKED8_PROBABILITY_PAIR)
    }

    #[must_use]
    pub fn has_dispatch_error(&self) -> bool {
        self.error_message.is_some()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DosageWorkItemStageDurationPlan {
    pub chunk_count: usize,
    pub duration_per_chunk: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DosageBufferAcquireAttemptPlan {
    pub should_take_free_buffer: bool,
    pub should_allocate: bool,
    pub should_wait: bool,
    pub wait_timeout_seconds: f64,
    pub free_buffer_count: usize,
    pub allocated_count: usize,
    pub buffer_limit: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DosageBufferRegisterAttemptPlan {
    pub should_register: bool,
    pub has_registration_error: bool,
    pub allocated_count: usize,
    pub buffer_limit: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DosageBufferReturnAttemptPlan {
    pub should_return: bool,
    pub allocated_count: usize,
    pub buffer_limit: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DosageBufferDiscardAttemptPlan {
    pub should_discard: bool,
    pub allocated_count: usize,
    pub buffer_limit: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DosageBufferPoolObservationPlan {
    pub operation_name: String,
    pub blocked: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CallbackQueueOccupancyState {
    queue_capacity: usize,
    occupied_count: usize,
}

impl CallbackQueueOccupancyState {
    #[must_use]
    pub const fn new(queue_capacity: usize) -> Self {
        Self { queue_capacity, occupied_count: 0 }
    }

    #[must_use]
    pub const fn queue_capacity(&self) -> usize {
        self.queue_capacity
    }

    #[must_use]
    pub const fn occupied_count(&self) -> usize {
        self.occupied_count
    }

    #[must_use]
    pub const fn has_available_slot(&self) -> bool {
        self.occupied_count < self.queue_capacity
    }

    pub fn acquire_slot(&mut self) -> bool {
        if !self.has_available_slot() {
            return false;
        }
        self.occupied_count += 1;
        true
    }

    pub fn release_slot(&mut self) -> bool {
        if self.occupied_count == 0 {
            return false;
        }
        self.occupied_count -= 1;
        true
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DosageBufferPoolState {
    buffer_limit: usize,
    buffer_identifiers: BTreeSet<usize>,
}

impl DosageBufferPoolState {
    #[must_use]
    pub fn new(buffer_limit: usize) -> Self {
        Self { buffer_limit, buffer_identifiers: BTreeSet::new() }
    }

    #[must_use]
    pub const fn buffer_limit(&self) -> usize {
        self.buffer_limit
    }

    #[must_use]
    pub fn allocated_count(&self) -> usize {
        self.buffer_identifiers.len()
    }

    #[must_use]
    pub fn buffer_identifiers(&self) -> Vec<usize> {
        self.buffer_identifiers.iter().copied().collect()
    }

    #[must_use]
    pub fn has_available_slot(&self) -> bool {
        self.allocated_count() < self.buffer_limit
    }

    #[must_use]
    pub fn owns_buffer(&self, buffer_identifier: usize) -> bool {
        self.buffer_identifiers.contains(&buffer_identifier)
    }

    pub fn register_buffer(&mut self, buffer_identifier: usize) -> bool {
        if !self.has_available_slot() || self.owns_buffer(buffer_identifier) {
            return false;
        }
        self.buffer_identifiers.insert(buffer_identifier)
    }

    pub fn discard_buffer(&mut self, buffer_identifier: usize) -> bool {
        self.buffer_identifiers.remove(&buffer_identifier)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResultInFlightSlotState {
    slot_limit: usize,
    occupied_count: usize,
}

impl ResultInFlightSlotState {
    #[must_use]
    pub const fn new(slot_limit: usize) -> Self {
        Self { slot_limit, occupied_count: 0 }
    }

    #[must_use]
    pub const fn slot_limit(&self) -> usize {
        self.slot_limit
    }

    #[must_use]
    pub const fn occupied_count(&self) -> usize {
        self.occupied_count
    }

    #[must_use]
    pub const fn has_available_slot(&self) -> bool {
        self.occupied_count < self.slot_limit
    }

    pub fn acquire_slot(&mut self) -> bool {
        if !self.has_available_slot() {
            return false;
        }
        self.occupied_count += 1;
        true
    }

    pub fn release_slot(&mut self) -> bool {
        if self.occupied_count == 0 {
            return false;
        }
        self.occupied_count -= 1;
        true
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CallbackSchedulerState {
    queue_limits: NativeCallbackQueueLimits,
    native_callback_batch_size: usize,
    dosage_queue_state: CallbackQueueOccupancyState,
    result_queue_state: CallbackQueueOccupancyState,
    result_in_flight_slot_state: ResultInFlightSlotState,
    dosage_buffer_pool_state: DosageBufferPoolState,
    worker_lifecycle_state: CallbackWorkerLifecycleState,
    dosage_worker_error_message: Option<String>,
    result_worker_error_message: Option<String>,
}

impl CallbackSchedulerState {
    /// Build the native callback scheduler state for one callback runner.
    ///
    /// # Errors
    ///
    /// Returns an error when queue limits or bounded resource limits are invalid.
    pub fn new(
        staging_depth: i64,
        native_callback_batch_size: i64,
        result_in_flight_limit: Option<i64>,
        dosage_buffer_limit: Option<i64>,
    ) -> Result<Self, ScheduleError> {
        let queue_limits = resolve_native_callback_queue_limits(
            staging_depth,
            native_callback_batch_size,
            result_in_flight_limit,
            dosage_buffer_limit,
        )?;
        let native_callback_batch_size = usize::try_from(native_callback_batch_size).map_err(|_| {
            ScheduleError::CallbackBatchSizeOverflow { callback_batch_size: native_callback_batch_size }
        })?;
        Ok(Self {
            queue_limits,
            native_callback_batch_size,
            dosage_queue_state: CallbackQueueOccupancyState::new(queue_limits.dosage_queue_depth),
            result_queue_state: CallbackQueueOccupancyState::new(queue_limits.result_queue_depth),
            result_in_flight_slot_state: ResultInFlightSlotState::new(queue_limits.result_in_flight_limit),
            dosage_buffer_pool_state: DosageBufferPoolState::new(queue_limits.dosage_buffer_limit),
            worker_lifecycle_state: CallbackWorkerLifecycleState::new(),
            dosage_worker_error_message: None,
            result_worker_error_message: None,
        })
    }

    #[must_use]
    pub const fn queue_limits(&self) -> NativeCallbackQueueLimits {
        self.queue_limits
    }

    #[must_use]
    pub const fn native_callback_batch_size(&self) -> usize {
        self.native_callback_batch_size
    }

    #[must_use]
    pub const fn dosage_queue_depth(&self) -> usize {
        self.queue_limits.dosage_queue_depth
    }

    #[must_use]
    pub const fn dosage_queue_capacity(&self) -> usize {
        self.dosage_queue_state.queue_capacity()
    }

    #[must_use]
    pub const fn dosage_queue_occupied_count(&self) -> usize {
        self.dosage_queue_state.occupied_count()
    }

    #[must_use]
    pub const fn has_available_dosage_queue_slot(&self) -> bool {
        self.dosage_queue_state.has_available_slot()
    }

    pub fn acquire_dosage_queue_slot(&mut self) -> bool {
        self.dosage_queue_state.acquire_slot()
    }

    pub fn release_dosage_queue_slot(&mut self) -> bool {
        self.dosage_queue_state.release_slot()
    }

    #[must_use]
    pub fn plan_dosage_queue_put_attempt(&mut self, wait_timeout_seconds: f64) -> CallbackQueuePutAttemptPlan {
        plan_callback_queue_put_attempt(&mut self.dosage_queue_state, wait_timeout_seconds)
    }

    #[must_use]
    pub fn plan_dosage_queue_put_backpressure_attempt(&mut self) -> CallbackQueuePutAttemptPlan {
        self.plan_dosage_queue_put_attempt(callback_worker_backpressure_poll_timeout_seconds())
    }

    #[must_use]
    pub fn plan_dosage_queue_put_observation(&self, queued: bool) -> CallbackQueuePutObservationPlan {
        debug_assert!(self.dosage_queue_state.queue_capacity() > 0);
        plan_callback_queue_put_observation(DOSAGE_QUEUE_NAME, queued)
    }

    #[must_use]
    pub fn plan_dosage_queue_get_attempt(&mut self, has_queued_item: bool) -> CallbackQueueGetAttemptPlan {
        plan_callback_queue_get_attempt(&mut self.dosage_queue_state, has_queued_item)
    }

    #[must_use]
    pub fn plan_dosage_queue_get_observation(&self) -> CallbackQueueGetObservationPlan {
        debug_assert!(self.dosage_queue_state.queue_capacity() > 0);
        plan_callback_queue_get_observation(DOSAGE_QUEUE_NAME)
    }

    #[must_use]
    pub const fn result_queue_depth(&self) -> usize {
        self.queue_limits.result_queue_depth
    }

    #[must_use]
    pub const fn result_queue_capacity(&self) -> usize {
        self.result_queue_state.queue_capacity()
    }

    #[must_use]
    pub const fn result_queue_occupied_count(&self) -> usize {
        self.result_queue_state.occupied_count()
    }

    #[must_use]
    pub const fn has_available_result_queue_slot(&self) -> bool {
        self.result_queue_state.has_available_slot()
    }

    pub fn acquire_result_queue_slot(&mut self) -> bool {
        self.result_queue_state.acquire_slot()
    }

    pub fn release_result_queue_slot(&mut self) -> bool {
        self.result_queue_state.release_slot()
    }

    #[must_use]
    pub fn plan_result_queue_put_attempt(&mut self, wait_timeout_seconds: f64) -> CallbackQueuePutAttemptPlan {
        plan_callback_queue_put_attempt(&mut self.result_queue_state, wait_timeout_seconds)
    }

    #[must_use]
    pub fn plan_result_queue_put_backpressure_attempt(&mut self) -> CallbackQueuePutAttemptPlan {
        self.plan_result_queue_put_attempt(callback_worker_backpressure_poll_timeout_seconds())
    }

    #[must_use]
    pub fn plan_result_queue_put_observation(&self, queued: bool) -> CallbackQueuePutObservationPlan {
        debug_assert!(self.result_queue_state.queue_capacity() > 0);
        plan_callback_queue_put_observation(RESULT_QUEUE_NAME, queued)
    }

    #[must_use]
    pub fn plan_result_queue_get_attempt(&mut self, has_queued_item: bool) -> CallbackQueueGetAttemptPlan {
        plan_callback_queue_get_attempt(&mut self.result_queue_state, has_queued_item)
    }

    #[must_use]
    pub fn plan_result_queue_get_observation(&self) -> CallbackQueueGetObservationPlan {
        debug_assert!(self.result_queue_state.queue_capacity() > 0);
        plan_callback_queue_get_observation(RESULT_QUEUE_NAME)
    }

    #[must_use]
    pub const fn result_in_flight_limit(&self) -> usize {
        self.queue_limits.result_in_flight_limit
    }

    #[must_use]
    pub const fn dosage_buffer_limit(&self) -> usize {
        self.queue_limits.dosage_buffer_limit
    }

    #[must_use]
    pub const fn has_started(&self) -> bool {
        self.worker_lifecycle_state.has_started()
    }

    pub fn mark_started(&mut self) -> bool {
        self.worker_lifecycle_state.mark_started()
    }

    #[must_use]
    pub fn plan_worker_start(&self) -> CallbackWorkerStartPlan {
        plan_callback_worker_start(self.has_started())
    }

    #[must_use]
    pub fn plan_worker_start_attempt(&mut self) -> CallbackWorkerStartAttemptPlan {
        plan_callback_worker_start_attempt(&mut self.worker_lifecycle_state)
    }

    #[must_use]
    pub const fn result_in_flight_slot_limit(&self) -> usize {
        self.result_in_flight_slot_state.slot_limit()
    }

    #[must_use]
    pub const fn result_in_flight_occupied_count(&self) -> usize {
        self.result_in_flight_slot_state.occupied_count()
    }

    #[must_use]
    pub const fn has_available_result_in_flight_slot(&self) -> bool {
        self.result_in_flight_slot_state.has_available_slot()
    }

    pub fn acquire_result_in_flight_slot(&mut self) -> bool {
        self.result_in_flight_slot_state.acquire_slot()
    }

    pub fn release_result_in_flight_slot(&mut self) -> bool {
        self.result_in_flight_slot_state.release_slot()
    }

    #[must_use]
    pub fn plan_result_in_flight_slot_acquire_attempt(
        &mut self,
        wait_timeout_seconds: f64,
    ) -> ResultInFlightAcquireAttemptPlan {
        plan_result_in_flight_slot_acquire_attempt(&mut self.result_in_flight_slot_state, wait_timeout_seconds)
    }

    #[must_use]
    pub fn plan_result_in_flight_slot_acquire_backpressure_attempt(&mut self) -> ResultInFlightAcquireAttemptPlan {
        self.plan_result_in_flight_slot_acquire_attempt(callback_worker_backpressure_poll_timeout_seconds())
    }

    #[must_use]
    pub fn plan_result_in_flight_slot_acquire_observation(
        &self,
        acquire_attempt_plan: &ResultInFlightAcquireAttemptPlan,
    ) -> ResultInFlightAcquireObservationPlan {
        debug_assert_eq!(acquire_attempt_plan.slot_limit, self.result_in_flight_slot_state.slot_limit());
        plan_result_in_flight_slot_acquire_observation(acquire_attempt_plan)
    }

    #[must_use]
    pub fn plan_result_in_flight_slot_release_attempt(&mut self) -> ResultInFlightReleaseAttemptPlan {
        plan_result_in_flight_slot_release_attempt(&mut self.result_in_flight_slot_state)
    }

    #[must_use]
    pub fn plan_result_in_flight_slot_release_observation(&self) -> ResultInFlightReleaseObservationPlan {
        debug_assert!(self.result_in_flight_slot_state.slot_limit() > 0);
        plan_result_in_flight_slot_release_observation()
    }

    #[must_use]
    pub const fn plan_result_write_item_pre_write_resource_release(
        &self,
        has_host_dosage_buffer: bool,
    ) -> ResultWriteItemResourceReleasePlan {
        ResultWriteItemResourceReleasePlan {
            should_release_host_buffer: has_host_dosage_buffer,
            should_release_result_in_flight_slot: false,
        }
    }

    #[must_use]
    #[allow(clippy::fn_params_excessive_bools)]
    pub const fn plan_result_write_item_final_resource_release(
        &self,
        has_host_dosage_buffer: bool,
        has_released_host_dosage_buffer: bool,
        release_in_flight_slot: bool,
    ) -> ResultWriteItemResourceReleasePlan {
        ResultWriteItemResourceReleasePlan {
            should_release_host_buffer: has_host_dosage_buffer && !has_released_host_dosage_buffer,
            should_release_result_in_flight_slot: release_in_flight_slot,
        }
    }

    #[must_use]
    pub const fn plan_result_write_handoff(&self, has_result_work_item: bool) -> ResultWriteHandoffPlan {
        plan_result_write_handoff(has_result_work_item)
    }

    #[must_use]
    pub const fn plan_result_write_drain_completion(
        &self,
        has_result_work_item: bool,
        flush_binary_correction_diagnostics_on_stop: bool,
    ) -> ResultWriteDrainCompletionPlan {
        ResultWriteDrainCompletionPlan {
            should_stop: !has_result_work_item,
            should_flush_binary_correction_diagnostics: !has_result_work_item
                && flush_binary_correction_diagnostics_on_stop,
        }
    }

    /// Plan which result consumer should process a dequeued work item.
    ///
    /// # Errors
    ///
    /// Returns an error when either work-item kind is unsupported.
    pub fn plan_result_write_item_dispatch(
        &self,
        result_work_item_kind: &str,
        expected_result_work_item_kind: &str,
    ) -> Result<ResultWriteItemDispatchPlan, ScheduleError> {
        plan_result_write_item_dispatch(result_work_item_kind, expected_result_work_item_kind)
    }

    #[must_use]
    pub const fn plan_dosage_work_drain_completion(&self, has_dosage_work_item: bool) -> DosageWorkDrainCompletionPlan {
        DosageWorkDrainCompletionPlan { should_stop: !has_dosage_work_item }
    }

    /// Plan which dosage consumer path should process a dequeued work item.
    ///
    /// # Errors
    ///
    /// Returns an error when the work-item kind is unsupported.
    pub fn plan_dosage_work_item_dispatch(
        &self,
        dosage_work_item_kind: &str,
    ) -> Result<DosageWorkItemDispatchPlan, ScheduleError> {
        plan_dosage_work_item_dispatch(dosage_work_item_kind)
    }

    /// Plan chunk-level timing attribution for one dosage work item.
    ///
    /// # Errors
    ///
    /// Returns an error when the work-item kind or chunk count is invalid.
    pub fn plan_dosage_work_item_stage_duration(
        &self,
        dosage_work_item_kind: &str,
        chunk_count: usize,
        elapsed_seconds: f64,
    ) -> Result<DosageWorkItemStageDurationPlan, ScheduleError> {
        plan_dosage_work_item_stage_duration(dosage_work_item_kind, chunk_count, elapsed_seconds)
    }

    #[must_use]
    pub const fn dosage_buffer_pool_limit(&self) -> usize {
        self.dosage_buffer_pool_state.buffer_limit()
    }

    #[must_use]
    pub fn dosage_buffer_allocated_count(&self) -> usize {
        self.dosage_buffer_pool_state.allocated_count()
    }

    #[must_use]
    pub fn dosage_buffer_identifiers(&self) -> Vec<usize> {
        self.dosage_buffer_pool_state.buffer_identifiers()
    }

    #[must_use]
    pub fn has_available_dosage_buffer_slot(&self) -> bool {
        self.dosage_buffer_pool_state.has_available_slot()
    }

    #[must_use]
    pub fn owns_dosage_buffer(&self, buffer_identifier: usize) -> bool {
        self.dosage_buffer_pool_state.owns_buffer(buffer_identifier)
    }

    pub fn register_dosage_buffer(&mut self, buffer_identifier: usize) -> bool {
        self.dosage_buffer_pool_state.register_buffer(buffer_identifier)
    }

    pub fn discard_dosage_buffer(&mut self, buffer_identifier: usize) -> bool {
        self.dosage_buffer_pool_state.discard_buffer(buffer_identifier)
    }

    #[must_use]
    pub fn plan_dosage_buffer_acquire_attempt(
        &self,
        free_buffer_count: usize,
        wait_timeout_seconds: f64,
    ) -> DosageBufferAcquireAttemptPlan {
        plan_dosage_buffer_acquire_attempt(&self.dosage_buffer_pool_state, free_buffer_count, wait_timeout_seconds)
    }

    #[must_use]
    pub fn plan_dosage_buffer_acquire_backpressure_attempt(
        &self,
        free_buffer_count: usize,
    ) -> DosageBufferAcquireAttemptPlan {
        self.plan_dosage_buffer_acquire_attempt(free_buffer_count, callback_worker_backpressure_poll_timeout_seconds())
    }

    #[must_use]
    pub fn plan_dosage_buffer_register_attempt(&mut self, buffer_identifier: usize) -> DosageBufferRegisterAttemptPlan {
        plan_dosage_buffer_register_attempt(&mut self.dosage_buffer_pool_state, buffer_identifier)
    }

    #[must_use]
    pub fn plan_dosage_buffer_return_attempt(&self, buffer_identifier: usize) -> DosageBufferReturnAttemptPlan {
        plan_dosage_buffer_return_attempt(&self.dosage_buffer_pool_state, buffer_identifier)
    }

    #[must_use]
    pub fn plan_dosage_buffer_discard_attempt(&mut self, buffer_identifier: usize) -> DosageBufferDiscardAttemptPlan {
        plan_dosage_buffer_discard_attempt(&mut self.dosage_buffer_pool_state, buffer_identifier)
    }

    #[must_use]
    pub fn plan_dosage_buffer_pool_reuse_observation(&self) -> DosageBufferPoolObservationPlan {
        debug_assert!(self.dosage_buffer_pool_state.buffer_limit() > 0);
        plan_dosage_buffer_pool_observation(QUEUE_REUSE_OPERATION, false)
    }

    #[must_use]
    pub fn plan_dosage_buffer_pool_return_observation(&self) -> DosageBufferPoolObservationPlan {
        debug_assert!(self.dosage_buffer_pool_state.buffer_limit() > 0);
        plan_dosage_buffer_pool_observation(QUEUE_RETURN_OPERATION, false)
    }

    #[must_use]
    pub fn plan_dosage_buffer_pool_allocate_observation(&self) -> DosageBufferPoolObservationPlan {
        debug_assert!(self.dosage_buffer_pool_state.buffer_limit() > 0);
        plan_dosage_buffer_pool_observation(QUEUE_ALLOCATE_OPERATION, false)
    }

    #[must_use]
    pub fn plan_dosage_buffer_pool_discard_observation(&self) -> DosageBufferPoolObservationPlan {
        debug_assert!(self.dosage_buffer_pool_state.buffer_limit() > 0);
        plan_dosage_buffer_pool_observation(QUEUE_DISCARD_OPERATION, false)
    }

    #[must_use]
    pub fn plan_dosage_buffer_pool_consumer_wait_observation(&self) -> DosageBufferPoolObservationPlan {
        debug_assert!(self.dosage_buffer_pool_state.buffer_limit() > 0);
        plan_dosage_buffer_pool_observation(QUEUE_CONSUMER_WAIT_OPERATION, true)
    }

    #[must_use]
    pub fn plan_dosage_buffer_reuse(
        &self,
        buffered_shape: &[usize],
        expected_shape: &[usize],
    ) -> Option<DosageBufferReusePlan> {
        plan_dosage_buffer_reuse(buffered_shape, expected_shape)
    }

    /// Plan a variant-major dosage batch handoff into the callback queue.
    ///
    /// # Errors
    ///
    /// Returns an error when the metadata, genotype matrix, and chunk-stat
    /// batches have different lengths, or when the batch is empty.
    pub fn plan_variant_major_dosage_batch_handoff(
        &self,
        metadata_count: usize,
        genotype_matrix_by_variant_count: usize,
        chunk_stats_count: usize,
    ) -> Result<VariantMajorDosageBatchHandoffPlan, ScheduleError> {
        plan_variant_major_dosage_batch_handoff(metadata_count, genotype_matrix_by_variant_count, chunk_stats_count)
    }

    /// Plan a dosage work handoff into the callback queue.
    ///
    /// # Errors
    ///
    /// Returns an error when the handoff contains no chunks.
    pub fn plan_dosage_work_handoff(&self, chunk_count: usize) -> Result<DosageWorkHandoffPlan, ScheduleError> {
        plan_dosage_work_handoff(chunk_count)
    }

    #[must_use]
    pub fn dosage_worker_error_message(&self) -> Option<&str> {
        self.dosage_worker_error_message.as_deref()
    }

    #[must_use]
    pub fn result_worker_error_message(&self) -> Option<&str> {
        self.result_worker_error_message.as_deref()
    }

    #[must_use]
    pub fn has_dosage_worker_error(&self) -> bool {
        self.dosage_worker_error_message.is_some()
    }

    #[must_use]
    pub fn has_result_worker_error(&self) -> bool {
        self.result_worker_error_message.is_some()
    }

    #[must_use]
    pub fn plan_worker_error_raise(&self) -> CallbackWorkerErrorRaisePlan {
        plan_callback_worker_error_raise(self.dosage_worker_error_message(), self.result_worker_error_message())
    }

    pub fn record_dosage_worker_error(&mut self, error_message: &str) {
        self.dosage_worker_error_message = Some(format_dosage_callback_worker_error_message(error_message));
    }

    pub fn record_result_worker_error(&mut self, error_message: &str) {
        self.result_worker_error_message = Some(format_result_callback_worker_error_message(error_message));
    }

    pub fn update_dosage_worker_error(&mut self, error_message: Option<&str>) -> CallbackWorkerErrorUpdatePlan {
        update_callback_worker_error(
            &mut self.dosage_worker_error_message,
            error_message,
            format_dosage_callback_worker_error_message,
        )
    }

    pub fn update_result_worker_error(&mut self, error_message: Option<&str>) -> CallbackWorkerErrorUpdatePlan {
        update_callback_worker_error(
            &mut self.result_worker_error_message,
            error_message,
            format_result_callback_worker_error_message,
        )
    }

    pub fn clear_dosage_worker_error(&mut self) -> bool {
        let had_error = self.has_dosage_worker_error();
        self.dosage_worker_error_message = None;
        had_error
    }

    pub fn clear_result_worker_error(&mut self) -> bool {
        let had_error = self.has_result_worker_error();
        self.result_worker_error_message = None;
        had_error
    }

    #[must_use]
    pub const fn backpressure_poll_timeout_seconds(&self) -> f64 {
        callback_worker_backpressure_poll_timeout_seconds()
    }

    #[must_use]
    pub fn plan_worker_finish(&self) -> CallbackWorkerFinishPlan {
        plan_callback_worker_finish()
    }

    #[must_use]
    pub fn plan_worker_abort(&self) -> CallbackWorkerAbortPlan {
        plan_callback_worker_abort()
    }

    /// Plan one aggregate callback queue or bounded-resource observation.
    ///
    /// # Errors
    ///
    /// Returns an error when the queue/resource and operation pair is not part
    /// of the callback scheduler observation contract.
    pub fn plan_queue_operation_observation(
        &self,
        queue_name: &str,
        operation_name: &str,
        elapsed_seconds: f64,
        blocked: bool,
    ) -> Result<CallbackQueueOperationObservationPlan, ScheduleError> {
        plan_callback_queue_operation_observation(queue_name, operation_name, elapsed_seconds, blocked)
    }

    /// Plan one aggregate callback queue or bounded-resource backpressure observation.
    ///
    /// # Errors
    ///
    /// Returns an error when the queue/resource and operation pair is not part
    /// of the callback scheduler observation contract.
    pub fn plan_queue_backpressure_observation(
        &self,
        queue_name: &str,
        operation_name: &str,
        queue_depth: usize,
        queue_capacity: usize,
        elapsed_seconds: f64,
        blocked: bool,
    ) -> Result<CallbackQueueBackpressureObservation, ScheduleError> {
        plan_callback_queue_backpressure_observation(
            queue_name,
            operation_name,
            queue_depth,
            queue_capacity,
            elapsed_seconds,
            blocked,
        )
    }

    /// Plan a callback queue or result-slot observation using native occupancy.
    ///
    /// # Errors
    ///
    /// Returns an error when the queue/resource and operation pair is not part
    /// of the native-owned callback scheduler observation contract.
    pub fn plan_current_queue_backpressure_observation(
        &self,
        queue_name: &str,
        operation_name: &str,
        elapsed_seconds: f64,
        blocked: bool,
    ) -> Result<CallbackQueueBackpressureObservation, ScheduleError> {
        let occupancy = self.current_queue_occupancy(queue_name, operation_name)?;
        plan_callback_queue_backpressure_observation(
            queue_name,
            operation_name,
            occupancy.queue_depth,
            occupancy.queue_capacity,
            elapsed_seconds,
            blocked,
        )
    }

    /// Plan a dosage-buffer pool observation using Python-owned free depth.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation is not part of the dosage-buffer
    /// pool observation contract.
    pub fn plan_dosage_buffer_pool_backpressure_observation(
        &self,
        operation_name: &str,
        free_buffer_count: usize,
        elapsed_seconds: f64,
        blocked: bool,
    ) -> Result<CallbackQueueBackpressureObservation, ScheduleError> {
        plan_callback_queue_backpressure_observation(
            DOSAGE_BUFFER_POOL_NAME,
            operation_name,
            free_buffer_count,
            self.dosage_buffer_pool_state.buffer_limit(),
            elapsed_seconds,
            blocked,
        )
    }

    /// Plan one timed callback queue or bounded-resource observation.
    ///
    /// # Errors
    ///
    /// Returns an error when the queue/resource and operation pair does not
    /// have a canonical callback timing stage.
    pub fn plan_queue_stage_observation(
        &self,
        queue_name: &str,
        operation_name: &str,
        elapsed_seconds: f64,
        blocked: bool,
    ) -> Result<CallbackQueueStageObservationPlan, ScheduleError> {
        plan_callback_queue_stage_observation(queue_name, operation_name, elapsed_seconds, blocked)
    }

    /// Plan one timed callback queue or bounded-resource backpressure observation.
    ///
    /// # Errors
    ///
    /// Returns an error when the queue/resource and operation pair does not
    /// have a canonical callback timing stage.
    pub fn plan_queue_stage_backpressure_observation(
        &self,
        queue_name: &str,
        operation_name: &str,
        queue_depth: usize,
        queue_capacity: usize,
        elapsed_seconds: f64,
        blocked: bool,
    ) -> Result<CallbackQueueStageBackpressureObservation, ScheduleError> {
        plan_callback_queue_stage_backpressure_observation(
            queue_name,
            operation_name,
            queue_depth,
            queue_capacity,
            elapsed_seconds,
            blocked,
        )
    }

    /// Plan a timed callback queue or result-slot observation using native occupancy.
    ///
    /// # Errors
    ///
    /// Returns an error when the queue/resource and operation pair does not
    /// have a canonical callback timing stage in the native scheduler contract.
    pub fn plan_current_queue_stage_backpressure_observation(
        &self,
        queue_name: &str,
        operation_name: &str,
        elapsed_seconds: f64,
        blocked: bool,
    ) -> Result<CallbackQueueStageBackpressureObservation, ScheduleError> {
        let occupancy = self.current_queue_occupancy(queue_name, operation_name)?;
        plan_callback_queue_stage_backpressure_observation(
            queue_name,
            operation_name,
            occupancy.queue_depth,
            occupancy.queue_capacity,
            elapsed_seconds,
            blocked,
        )
    }

    /// Plan a timed dosage-buffer pool observation using Python-owned free depth.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation does not have a canonical
    /// dosage-buffer pool timing stage.
    pub fn plan_dosage_buffer_pool_stage_backpressure_observation(
        &self,
        operation_name: &str,
        free_buffer_count: usize,
        elapsed_seconds: f64,
        blocked: bool,
    ) -> Result<CallbackQueueStageBackpressureObservation, ScheduleError> {
        plan_callback_queue_stage_backpressure_observation(
            DOSAGE_BUFFER_POOL_NAME,
            operation_name,
            free_buffer_count,
            self.dosage_buffer_pool_state.buffer_limit(),
            elapsed_seconds,
            blocked,
        )
    }

    #[must_use]
    pub fn plan_dosage_worker_join(&self, timeout_seconds: Option<f64>) -> CallbackWorkerJoinPlan {
        plan_dosage_callback_worker_join(timeout_seconds, self.has_started())
    }

    #[must_use]
    pub fn plan_result_worker_join(&self, timeout_seconds: Option<f64>) -> CallbackWorkerJoinPlan {
        plan_result_callback_worker_join(timeout_seconds, self.has_started())
    }

    #[must_use]
    pub fn plan_dosage_worker_stop(
        &self,
        timeout_seconds: Option<f64>,
        is_worker_alive: bool,
    ) -> CallbackWorkerStopPlan {
        plan_dosage_callback_worker_stop(
            timeout_seconds,
            self.has_started(),
            self.has_dosage_worker_error(),
            is_worker_alive,
        )
    }

    #[must_use]
    pub fn plan_result_worker_stop(
        &self,
        timeout_seconds: Option<f64>,
        is_worker_alive: bool,
    ) -> CallbackWorkerStopPlan {
        plan_result_callback_worker_stop(
            timeout_seconds,
            self.has_started(),
            self.has_result_worker_error(),
            is_worker_alive,
        )
    }

    #[must_use]
    pub fn plan_dosage_worker_stop_poll(
        &self,
        remaining_timeout_seconds: f64,
        is_worker_alive: bool,
    ) -> CallbackWorkerStopPollPlan {
        plan_callback_worker_stop_poll(
            remaining_timeout_seconds,
            self.has_started(),
            self.has_dosage_worker_error(),
            is_worker_alive,
        )
    }

    #[must_use]
    pub fn plan_result_worker_stop_poll(
        &self,
        remaining_timeout_seconds: f64,
        is_worker_alive: bool,
    ) -> CallbackWorkerStopPollPlan {
        plan_callback_worker_stop_poll(
            remaining_timeout_seconds,
            self.has_started(),
            self.has_result_worker_error(),
            is_worker_alive,
        )
    }

    fn current_queue_occupancy(
        &self,
        queue_name: &str,
        operation_name: &str,
    ) -> Result<CallbackBoundedResourceOccupancy, ScheduleError> {
        let occupancy = match queue_name {
            DOSAGE_QUEUE_NAME => CallbackBoundedResourceOccupancy {
                queue_depth: self.dosage_queue_state.occupied_count(),
                queue_capacity: self.dosage_queue_state.queue_capacity(),
            },
            RESULT_QUEUE_NAME => CallbackBoundedResourceOccupancy {
                queue_depth: self.result_queue_state.occupied_count(),
                queue_capacity: self.result_queue_state.queue_capacity(),
            },
            RESULT_IN_FLIGHT_SLOTS_NAME => CallbackBoundedResourceOccupancy {
                queue_depth: self.result_in_flight_slot_state.occupied_count(),
                queue_capacity: self.result_in_flight_slot_state.slot_limit(),
            },
            _ => {
                return Err(ScheduleError::UnsupportedCallbackQueueOperation {
                    queue_name: queue_name.to_string(),
                    operation_name: operation_name.to_string(),
                });
            }
        };
        Ok(occupancy)
    }
}

fn plan_callback_queue_put_attempt(
    queue_state: &mut CallbackQueueOccupancyState,
    wait_timeout_seconds: f64,
) -> CallbackQueuePutAttemptPlan {
    if queue_state.acquire_slot() {
        return CallbackQueuePutAttemptPlan {
            should_put: true,
            should_wait: false,
            wait_timeout_seconds: 0.0,
            queue_depth: queue_state.occupied_count(),
            queue_capacity: queue_state.queue_capacity(),
        };
    }
    let normalized_wait_timeout_seconds = normalize_callback_queue_wait_timeout_seconds(wait_timeout_seconds);
    CallbackQueuePutAttemptPlan {
        should_put: false,
        should_wait: normalized_wait_timeout_seconds > 0.0,
        wait_timeout_seconds: normalized_wait_timeout_seconds,
        queue_depth: queue_state.occupied_count(),
        queue_capacity: queue_state.queue_capacity(),
    }
}

#[must_use]
pub fn plan_callback_queue_put_observation(queue_name: &str, queued: bool) -> CallbackQueuePutObservationPlan {
    if queued {
        return CallbackQueuePutObservationPlan {
            queue_name: queue_name.to_string(),
            operation_name: QUEUE_PUT_OPERATION.to_string(),
            blocked: false,
            should_retry_put: false,
        };
    }
    CallbackQueuePutObservationPlan {
        queue_name: queue_name.to_string(),
        operation_name: QUEUE_PRODUCER_BLOCKING_OPERATION.to_string(),
        blocked: true,
        should_retry_put: true,
    }
}

fn plan_callback_queue_get_attempt(
    queue_state: &mut CallbackQueueOccupancyState,
    has_queued_item: bool,
) -> CallbackQueueGetAttemptPlan {
    if has_queued_item {
        let released_slot = queue_state.release_slot();
        return CallbackQueueGetAttemptPlan {
            should_get: released_slot,
            should_wait: false,
            has_release_error: !released_slot,
            wait_timeout_seconds: 0.0,
            queue_depth: queue_state.occupied_count(),
            queue_capacity: queue_state.queue_capacity(),
        };
    }
    CallbackQueueGetAttemptPlan {
        should_get: false,
        should_wait: true,
        has_release_error: false,
        wait_timeout_seconds: callback_worker_backpressure_poll_timeout_seconds(),
        queue_depth: queue_state.occupied_count(),
        queue_capacity: queue_state.queue_capacity(),
    }
}

#[must_use]
pub fn plan_callback_queue_get_observation(queue_name: &str) -> CallbackQueueGetObservationPlan {
    CallbackQueueGetObservationPlan {
        queue_name: queue_name.to_string(),
        operation_name: QUEUE_CONSUMER_WAIT_OPERATION.to_string(),
        blocked: true,
    }
}

fn normalize_callback_queue_wait_timeout_seconds(wait_timeout_seconds: f64) -> f64 {
    if wait_timeout_seconds.is_finite() && wait_timeout_seconds > 0.0 { wait_timeout_seconds } else { 0.0 }
}

fn plan_result_in_flight_slot_acquire_attempt(
    slot_state: &mut ResultInFlightSlotState,
    wait_timeout_seconds: f64,
) -> ResultInFlightAcquireAttemptPlan {
    if slot_state.acquire_slot() {
        return ResultInFlightAcquireAttemptPlan {
            should_acquire: true,
            should_wait: false,
            wait_timeout_seconds: 0.0,
            occupied_count: slot_state.occupied_count(),
            slot_limit: slot_state.slot_limit(),
        };
    }
    let normalized_wait_timeout_seconds = normalize_callback_queue_wait_timeout_seconds(wait_timeout_seconds);
    ResultInFlightAcquireAttemptPlan {
        should_acquire: false,
        should_wait: normalized_wait_timeout_seconds > 0.0,
        wait_timeout_seconds: normalized_wait_timeout_seconds,
        occupied_count: slot_state.occupied_count(),
        slot_limit: slot_state.slot_limit(),
    }
}

#[must_use]
pub fn plan_result_in_flight_slot_acquire_observation(
    acquire_attempt_plan: &ResultInFlightAcquireAttemptPlan,
) -> ResultInFlightAcquireObservationPlan {
    if acquire_attempt_plan.should_acquire {
        return ResultInFlightAcquireObservationPlan {
            resource_name: RESULT_IN_FLIGHT_SLOTS_NAME.to_string(),
            operation_name: RESULT_SLOT_ACQUIRE_OPERATION.to_string(),
            blocked: false,
            should_retry_acquisition: false,
        };
    }
    ResultInFlightAcquireObservationPlan {
        resource_name: RESULT_IN_FLIGHT_SLOTS_NAME.to_string(),
        operation_name: QUEUE_PRODUCER_BLOCKING_OPERATION.to_string(),
        blocked: true,
        should_retry_acquisition: true,
    }
}

fn plan_result_in_flight_slot_release_attempt(
    slot_state: &mut ResultInFlightSlotState,
) -> ResultInFlightReleaseAttemptPlan {
    let released_slot = slot_state.release_slot();
    ResultInFlightReleaseAttemptPlan {
        should_release: released_slot,
        has_release_error: !released_slot,
        occupied_count: slot_state.occupied_count(),
        slot_limit: slot_state.slot_limit(),
    }
}

#[must_use]
pub fn plan_result_in_flight_slot_release_observation() -> ResultInFlightReleaseObservationPlan {
    ResultInFlightReleaseObservationPlan {
        resource_name: RESULT_IN_FLIGHT_SLOTS_NAME.to_string(),
        operation_name: RESULT_SLOT_RELEASE_OPERATION.to_string(),
        blocked: false,
    }
}

fn plan_dosage_buffer_acquire_attempt(
    buffer_pool_state: &DosageBufferPoolState,
    free_buffer_count: usize,
    wait_timeout_seconds: f64,
) -> DosageBufferAcquireAttemptPlan {
    let should_take_free_buffer = free_buffer_count > 0;
    let should_allocate = !should_take_free_buffer && buffer_pool_state.has_available_slot();
    let normalized_wait_timeout_seconds = if should_take_free_buffer || should_allocate {
        0.0
    } else {
        normalize_callback_queue_wait_timeout_seconds(wait_timeout_seconds)
    };
    DosageBufferAcquireAttemptPlan {
        should_take_free_buffer,
        should_allocate,
        should_wait: !should_take_free_buffer && !should_allocate && normalized_wait_timeout_seconds > 0.0,
        wait_timeout_seconds: normalized_wait_timeout_seconds,
        free_buffer_count,
        allocated_count: buffer_pool_state.allocated_count(),
        buffer_limit: buffer_pool_state.buffer_limit(),
    }
}

fn plan_dosage_buffer_register_attempt(
    buffer_pool_state: &mut DosageBufferPoolState,
    buffer_identifier: usize,
) -> DosageBufferRegisterAttemptPlan {
    let registered_buffer = buffer_pool_state.register_buffer(buffer_identifier);
    DosageBufferRegisterAttemptPlan {
        should_register: registered_buffer,
        has_registration_error: !registered_buffer,
        allocated_count: buffer_pool_state.allocated_count(),
        buffer_limit: buffer_pool_state.buffer_limit(),
    }
}

fn plan_dosage_buffer_return_attempt(
    buffer_pool_state: &DosageBufferPoolState,
    buffer_identifier: usize,
) -> DosageBufferReturnAttemptPlan {
    DosageBufferReturnAttemptPlan {
        should_return: buffer_pool_state.owns_buffer(buffer_identifier),
        allocated_count: buffer_pool_state.allocated_count(),
        buffer_limit: buffer_pool_state.buffer_limit(),
    }
}

fn plan_dosage_buffer_discard_attempt(
    buffer_pool_state: &mut DosageBufferPoolState,
    buffer_identifier: usize,
) -> DosageBufferDiscardAttemptPlan {
    let discarded_buffer = buffer_pool_state.discard_buffer(buffer_identifier);
    DosageBufferDiscardAttemptPlan {
        should_discard: discarded_buffer,
        allocated_count: buffer_pool_state.allocated_count(),
        buffer_limit: buffer_pool_state.buffer_limit(),
    }
}

#[must_use]
pub fn plan_dosage_buffer_pool_observation(operation_name: &str, blocked: bool) -> DosageBufferPoolObservationPlan {
    DosageBufferPoolObservationPlan { operation_name: operation_name.to_string(), blocked }
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ScheduleError {
    #[error("staging_depth must be positive.")]
    NonPositiveStagingDepth,
    #[error("native_callback_batch_size must be positive.")]
    NonPositiveCallbackBatchSize,
    #[error("native_callback_batch_size > 1 is not supported for packed8 BGEN delivery.")]
    Packed8CallbackBatchSize,
    #[error("native_callback_batch_size > 1 is not supported for grouped union BGEN delivery.")]
    GroupedUnionCallbackBatchSize,
    #[error("native_callback_batch_size must not exceed the effective dosage_buffer_limit ({dosage_buffer_limit}).")]
    CallbackBatchSizeExceedsDosageBufferLimit { dosage_buffer_limit: usize },
    #[error("result_in_flight_limit must be positive when provided.")]
    NonPositiveResultInFlightLimit,
    #[error("dosage_buffer_limit must be positive when provided.")]
    NonPositiveDosageBufferLimit,
    #[error("staging_depth exceeds platform capacity: {staging_depth}")]
    StagingDepthOverflow { staging_depth: i64 },
    #[error("native_callback_batch_size exceeds platform capacity: {callback_batch_size}")]
    CallbackBatchSizeOverflow { callback_batch_size: i64 },
    #[error("result_in_flight_limit exceeds platform capacity: {result_in_flight_limit}")]
    ResultInFlightLimitOverflow { result_in_flight_limit: i64 },
    #[error("dosage_buffer_limit exceeds platform capacity: {dosage_buffer_limit}")]
    DosageBufferLimitOverflow { dosage_buffer_limit: i64 },
    #[error("queue limit default for staging_depth exceeds platform capacity: {staging_depth}")]
    QueueLimitDefaultOverflow { staging_depth: usize },
    #[error("Writer finish thread count must be positive.")]
    NonPositiveWriterFinishThreadCount,
    #[error("Writer session count exceeds platform capacity: {session_count}")]
    WriterSessionCountOverflow { session_count: i64 },
    #[error("Writer finish thread count exceeds platform capacity: {thread_count}")]
    WriterFinishThreadCountOverflow { thread_count: i64 },
    #[error("Variant-major dosage batch inputs must have identical lengths.")]
    VariantMajorDosageBatchLengthMismatch,
    #[error("Variant-major dosage batch must contain at least one chunk.")]
    EmptyVariantMajorDosageBatch,
    #[error("Dosage work handoff must contain at least one chunk.")]
    EmptyDosageWorkHandoff,
    #[error(
        "Committed chunk identifier set count ({committed_set_count}) must match writer session count ({writer_session_count})."
    )]
    MultiTraitCommittedChunkSetCountMismatch { writer_session_count: usize, committed_set_count: usize },
    #[error("Unsupported public statistic output dtype: {output_statistic_dtype}")]
    UnsupportedOutputStatisticDtype { output_statistic_dtype: String },
    #[error("Unsupported GPU genotype format: {gpu_genotype_format}")]
    UnsupportedGpuGenotypeFormat { gpu_genotype_format: String },
    #[error("Unsupported JAX device: {jax_device}")]
    UnsupportedJaxDevice { jax_device: String },
    #[error("Unsupported callback queue stage operation: {queue_name}.{operation_name}")]
    UnsupportedCallbackQueueStageOperation { queue_name: String, operation_name: String },
    #[error("Unsupported callback queue operation: {queue_name}.{operation_name}")]
    UnsupportedCallbackQueueOperation { queue_name: String, operation_name: String },
    #[error("Unsupported BGEN delivery cleanup outcome: {outcome}")]
    UnsupportedBgenDeliveryCleanupOutcome { outcome: String },
    #[error("Unsupported result write item kind: {result_work_item_kind}")]
    UnsupportedResultWriteItemKind { result_work_item_kind: String },
    #[error("Unsupported dosage work item kind: {dosage_work_item_kind}")]
    UnsupportedDosageWorkItemKind { dosage_work_item_kind: String },
    #[error("Dosage work item stage duration attribution requires at least one chunk.")]
    EmptyDosageWorkItemStageDuration,
    #[error("Cannot record stage duration for a dosage work stop signal.")]
    DosageWorkItemStageDurationStopSignal,
    #[error("Dosage work item kind {dosage_work_item_kind} must attribute to exactly one chunk, got {chunk_count}.")]
    DosageWorkItemStageDurationChunkCountMismatch { dosage_work_item_kind: String, chunk_count: usize },
    #[error("Dosage work item stage duration chunk count exceeds floating-point attribution capacity: {chunk_count}.")]
    DosageWorkItemStageDurationChunkCountOverflow { chunk_count: usize },
}

#[must_use]
pub fn plan_dosage_buffer_reuse(buffered_shape: &[usize], expected_shape: &[usize]) -> Option<DosageBufferReusePlan> {
    if buffered_shape.len() != expected_shape.len() {
        return None;
    }
    if buffered_shape
        .iter()
        .zip(expected_shape)
        .any(|(buffered_dimension, expected_dimension)| buffered_dimension < expected_dimension)
    {
        return None;
    }
    Some(DosageBufferReusePlan {
        requires_slice: buffered_shape != expected_shape,
        slice_dimensions: expected_shape.to_vec(),
    })
}

/// Plan a variant-major dosage batch handoff into the callback queue.
///
/// # Errors
///
/// Returns an error when the metadata, genotype matrix, and chunk-stat batches
/// have different lengths, or when the batch is empty.
pub fn plan_variant_major_dosage_batch_handoff(
    metadata_count: usize,
    genotype_matrix_by_variant_count: usize,
    chunk_stats_count: usize,
) -> Result<VariantMajorDosageBatchHandoffPlan, ScheduleError> {
    if metadata_count != genotype_matrix_by_variant_count || metadata_count != chunk_stats_count {
        return Err(ScheduleError::VariantMajorDosageBatchLengthMismatch);
    }
    if metadata_count == 0 {
        return Err(ScheduleError::EmptyVariantMajorDosageBatch);
    }
    let handoff_plan = plan_dosage_work_handoff(metadata_count)?;
    Ok(VariantMajorDosageBatchHandoffPlan { chunk_count: handoff_plan.chunk_count })
}

/// Partition planned genotype chunks into callback batches.
///
/// # Errors
///
/// Returns an error when the callback batch size is zero.
pub fn plan_chunk_batches(
    chunk_specs: &[ChunkSpec],
    callback_batch_size: usize,
) -> Result<ChunkBatchPlan, ScheduleError> {
    if callback_batch_size == 0 {
        return Err(ScheduleError::NonPositiveCallbackBatchSize);
    }
    let chunk_batches = chunk_specs.chunks(callback_batch_size).map(<[ChunkSpec]>::to_vec).collect();
    Ok(ChunkBatchPlan { chunk_batches })
}

/// Plan a dosage work handoff into the callback queue.
///
/// # Errors
///
/// Returns an error when the handoff contains no chunks.
pub fn plan_dosage_work_handoff(chunk_count: usize) -> Result<DosageWorkHandoffPlan, ScheduleError> {
    if chunk_count == 0 {
        return Err(ScheduleError::EmptyDosageWorkHandoff);
    }
    Ok(DosageWorkHandoffPlan { chunk_count })
}

#[must_use]
pub const fn plan_result_write_handoff(has_result_work_item: bool) -> ResultWriteHandoffPlan {
    ResultWriteHandoffPlan { should_enqueue: true, has_result_work_item, is_stop_signal: !has_result_work_item }
}

/// Plan which result write processing path should consume a dequeued item.
///
/// # Errors
///
/// Returns an error when either work-item kind is unsupported.
pub fn plan_result_write_item_dispatch(
    result_work_item_kind: &str,
    expected_result_work_item_kind: &str,
) -> Result<ResultWriteItemDispatchPlan, ScheduleError> {
    validate_result_write_item_kind(result_work_item_kind)?;
    validate_result_write_item_kind(expected_result_work_item_kind)?;

    if result_work_item_kind == RESULT_WRITE_ITEM_KIND_STOP_SIGNAL {
        return Ok(ResultWriteItemDispatchPlan {
            result_work_item_kind: result_work_item_kind.to_owned(),
            expected_result_work_item_kind: expected_result_work_item_kind.to_owned(),
            should_process_result_write_item: false,
            should_process_multi_result_write_item: false,
            has_dispatch_error: true,
            error_message: Some("Native result write dispatch plan continued without a work item.".to_owned()),
        });
    }

    if result_work_item_kind != expected_result_work_item_kind {
        return Ok(ResultWriteItemDispatchPlan {
            result_work_item_kind: result_work_item_kind.to_owned(),
            expected_result_work_item_kind: expected_result_work_item_kind.to_owned(),
            should_process_result_write_item: false,
            should_process_multi_result_write_item: false,
            has_dispatch_error: true,
            error_message: Some(format!(
                "Native result write dispatch plan expected {expected_result_work_item_kind} but received {result_work_item_kind}."
            )),
        });
    }

    Ok(ResultWriteItemDispatchPlan {
        result_work_item_kind: result_work_item_kind.to_owned(),
        expected_result_work_item_kind: expected_result_work_item_kind.to_owned(),
        should_process_result_write_item: result_work_item_kind == RESULT_WRITE_ITEM_KIND_SINGLE_RESULT,
        should_process_multi_result_write_item: result_work_item_kind == RESULT_WRITE_ITEM_KIND_MULTI_RESULT,
        has_dispatch_error: false,
        error_message: None,
    })
}

fn validate_result_write_item_kind(result_work_item_kind: &str) -> Result<(), ScheduleError> {
    match result_work_item_kind {
        RESULT_WRITE_ITEM_KIND_SINGLE_RESULT
        | RESULT_WRITE_ITEM_KIND_MULTI_RESULT
        | RESULT_WRITE_ITEM_KIND_STOP_SIGNAL => Ok(()),
        _ => Err(ScheduleError::UnsupportedResultWriteItemKind {
            result_work_item_kind: result_work_item_kind.to_owned(),
        }),
    }
}

/// Plan which dosage processing path should consume a dequeued work item.
///
/// # Errors
///
/// Returns an error when the work-item kind is unsupported.
pub fn plan_dosage_work_item_dispatch(
    dosage_work_item_kind: &str,
) -> Result<DosageWorkItemDispatchPlan, ScheduleError> {
    validate_dosage_work_item_kind(dosage_work_item_kind)?;

    if dosage_work_item_kind == DOSAGE_WORK_ITEM_KIND_STOP_SIGNAL {
        return Ok(DosageWorkItemDispatchPlan {
            dosage_work_item_kind: dosage_work_item_kind.to_owned(),
            processing_path: None,
            error_message: Some("Native dosage work dispatch plan continued without a work item.".to_owned()),
        });
    }

    Ok(DosageWorkItemDispatchPlan {
        dosage_work_item_kind: dosage_work_item_kind.to_owned(),
        processing_path: Some(dosage_work_item_kind.to_owned()),
        error_message: None,
    })
}

fn validate_dosage_work_item_kind(dosage_work_item_kind: &str) -> Result<(), ScheduleError> {
    match dosage_work_item_kind {
        DOSAGE_WORK_ITEM_KIND_SAMPLE_MAJOR_DOSAGE
        | DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_DOSAGE
        | DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_DOSAGE_BATCH
        | DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_PACKED8_PROBABILITY_PAIR
        | DOSAGE_WORK_ITEM_KIND_STOP_SIGNAL => Ok(()),
        _ => Err(ScheduleError::UnsupportedDosageWorkItemKind {
            dosage_work_item_kind: dosage_work_item_kind.to_owned(),
        }),
    }
}

/// Plan chunk-level timing attribution for one dosage work item.
///
/// # Errors
///
/// Returns an error when the work-item kind or chunk count is invalid.
pub fn plan_dosage_work_item_stage_duration(
    dosage_work_item_kind: &str,
    chunk_count: usize,
    elapsed_seconds: f64,
) -> Result<DosageWorkItemStageDurationPlan, ScheduleError> {
    validate_dosage_work_item_kind(dosage_work_item_kind)?;
    if dosage_work_item_kind == DOSAGE_WORK_ITEM_KIND_STOP_SIGNAL {
        return Err(ScheduleError::DosageWorkItemStageDurationStopSignal);
    }
    if chunk_count == 0 {
        return Err(ScheduleError::EmptyDosageWorkItemStageDuration);
    }
    if dosage_work_item_kind != DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_DOSAGE_BATCH && chunk_count != 1 {
        return Err(ScheduleError::DosageWorkItemStageDurationChunkCountMismatch {
            dosage_work_item_kind: dosage_work_item_kind.to_owned(),
            chunk_count,
        });
    }
    let chunk_count_for_duration = u32::try_from(chunk_count)
        .map_err(|_| ScheduleError::DosageWorkItemStageDurationChunkCountOverflow { chunk_count })?;
    Ok(DosageWorkItemStageDurationPlan {
        chunk_count,
        duration_per_chunk: elapsed_seconds / f64::from(chunk_count_for_duration),
    })
}

/// Resolve native callback queue depths and bounded resource limits.
///
/// # Errors
///
/// Returns an error when a configured limit is non-positive, cannot fit in
/// `usize`, the default `staging_depth + 1` limit would overflow, or the
/// callback batch size cannot fit in the effective dosage buffer limit.
pub fn resolve_native_callback_queue_limits(
    staging_depth: i64,
    native_callback_batch_size: i64,
    result_in_flight_limit: Option<i64>,
    dosage_buffer_limit: Option<i64>,
) -> Result<NativeCallbackQueueLimits, ScheduleError> {
    if staging_depth <= 0 {
        return Err(ScheduleError::NonPositiveStagingDepth);
    }
    if native_callback_batch_size <= 0 {
        return Err(ScheduleError::NonPositiveCallbackBatchSize);
    }
    if matches!(result_in_flight_limit, Some(limit) if limit <= 0) {
        return Err(ScheduleError::NonPositiveResultInFlightLimit);
    }
    if matches!(dosage_buffer_limit, Some(limit) if limit <= 0) {
        return Err(ScheduleError::NonPositiveDosageBufferLimit);
    }

    let staging_depth =
        usize::try_from(staging_depth).map_err(|_| ScheduleError::StagingDepthOverflow { staging_depth })?;
    let native_callback_batch_size = usize::try_from(native_callback_batch_size)
        .map_err(|_| ScheduleError::CallbackBatchSizeOverflow { callback_batch_size: native_callback_batch_size })?;
    let default_limit =
        staging_depth.checked_add(1).ok_or(ScheduleError::QueueLimitDefaultOverflow { staging_depth })?;
    let result_in_flight_limit = result_in_flight_limit
        .map(|limit| {
            usize::try_from(limit)
                .map_err(|_| ScheduleError::ResultInFlightLimitOverflow { result_in_flight_limit: limit })
        })
        .transpose()?
        .unwrap_or(default_limit);
    let dosage_buffer_limit = dosage_buffer_limit
        .map(|limit| {
            usize::try_from(limit).map_err(|_| ScheduleError::DosageBufferLimitOverflow { dosage_buffer_limit: limit })
        })
        .transpose()?
        .unwrap_or(default_limit);
    if dosage_buffer_limit < native_callback_batch_size {
        return Err(ScheduleError::CallbackBatchSizeExceedsDosageBufferLimit { dosage_buffer_limit });
    }

    Ok(NativeCallbackQueueLimits {
        dosage_queue_depth: staging_depth,
        result_queue_depth: staging_depth,
        result_in_flight_limit,
        dosage_buffer_limit,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk_spec(variant_start_index: usize, variant_stop_index: usize) -> ChunkSpec {
        ChunkSpec { variant_start_index, variant_stop_index }
    }

    #[test]
    fn returns_empty_set_for_empty_inputs() {
        let shared_chunk_identifiers = intersect_committed_chunk_identifier_sets(&[]);

        assert!(shared_chunk_identifiers.is_empty());
    }

    #[test]
    fn intersects_committed_chunk_identifiers_across_outputs() {
        let committed_chunk_identifier_sets =
            [BTreeSet::from([0_usize, 32, 64]), BTreeSet::from([32, 64, 96]), BTreeSet::from([32, 128])];

        let shared_chunk_identifiers = intersect_committed_chunk_identifier_sets(&committed_chunk_identifier_sets);

        assert_eq!(shared_chunk_identifiers, BTreeSet::from([32]));
    }

    #[test]
    fn preserves_single_output_committed_chunk_identifiers() {
        let committed_chunk_identifier_sets = [BTreeSet::from([64_usize, 0])];

        let shared_chunk_identifiers = intersect_committed_chunk_identifier_sets(&committed_chunk_identifier_sets);

        assert_eq!(shared_chunk_identifiers, BTreeSet::from([0, 64]));
    }

    #[test]
    fn resolves_manifest_gpu_genotype_format_with_legacy_backend_fallback() {
        assert_eq!(resolve_manifest_gpu_genotype_format(true, Some("packed8"), Some("dosage")), Some("packed8"),);
        assert_eq!(resolve_manifest_gpu_genotype_format(true, Some("invalid"), Some("dosage")), None);
        assert_eq!(resolve_manifest_gpu_genotype_format(true, None, Some("dosage")), Some("dosage"));
        assert_eq!(resolve_manifest_gpu_genotype_format(false, Some("packed8"), None), None);
    }

    #[test]
    fn resolves_effective_trusted_no_missing_diploid() {
        assert!(!resolve_effective_trusted_no_missing_diploid(false, false));
        assert!(resolve_effective_trusted_no_missing_diploid(true, false));
        assert!(resolve_effective_trusted_no_missing_diploid(false, true));
        assert!(resolve_effective_trusted_no_missing_diploid(true, true));
    }

    #[test]
    fn plans_auto_to_dosage_gpu_genotype_format_resolution() {
        let resolution_plan =
            plan_gpu_genotype_format_auto_to_dosage("auto", "multi_trait_or_linear_pipeline").unwrap();

        assert_eq!(
            resolution_plan,
            GpuGenotypeFormatResolutionPlan {
                requested_gpu_genotype_format: "auto".to_string(),
                resolved_gpu_genotype_format: Some("dosage".to_string()),
                resolution_reason: Some("multi_trait_or_linear_pipeline".to_string()),
                fallback_error: None,
                requires_trusted_validation: false,
            },
        );
        assert!(resolution_plan.is_resolved());
        assert!(resolution_plan.should_log_auto_resolution());

        let explicit_plan = plan_gpu_genotype_format_auto_to_dosage("packed8", "unused").unwrap();
        assert_eq!(explicit_plan.resolved_gpu_genotype_format.as_deref(), Some("packed8"));
        assert_eq!(explicit_plan.resolution_reason.as_deref(), Some("explicit"));
        assert!(!explicit_plan.should_log_auto_resolution());
    }

    #[test]
    fn plans_single_trait_binary_gpu_genotype_format_resolution_before_validation() {
        let manifest_plan =
            plan_single_trait_binary_gpu_genotype_format_resolution("auto", Some("packed8"), None, true, "gpu")
                .unwrap();
        assert_eq!(manifest_plan.resolved_gpu_genotype_format.as_deref(), Some("packed8"));
        assert_eq!(manifest_plan.resolution_reason.as_deref(), Some("resume_manifest"));
        assert!(!manifest_plan.requires_trusted_validation);
        assert!(manifest_plan.should_log_auto_resolution());

        let cpu_plan =
            plan_single_trait_binary_gpu_genotype_format_resolution("auto", None, None, false, "cpu").unwrap();
        assert_eq!(cpu_plan.resolved_gpu_genotype_format.as_deref(), Some("dosage"));
        assert_eq!(cpu_plan.resolution_reason.as_deref(), Some("non_gpu_device"));
        assert!(!cpu_plan.requires_trusted_validation);

        let validation_plan =
            plan_single_trait_binary_gpu_genotype_format_resolution("auto", None, None, false, "gpu").unwrap();
        assert_eq!(validation_plan.resolved_gpu_genotype_format, None);
        assert_eq!(validation_plan.resolution_reason, None);
        assert!(validation_plan.requires_trusted_validation);
        assert!(!validation_plan.should_log_auto_resolution());

        let explicit_plan =
            plan_single_trait_binary_gpu_genotype_format_resolution("dosage", None, None, false, "gpu").unwrap();
        assert_eq!(explicit_plan.resolved_gpu_genotype_format.as_deref(), Some("dosage"));
        assert_eq!(explicit_plan.resolution_reason.as_deref(), Some("explicit"));
        assert!(!explicit_plan.requires_trusted_validation);
    }

    #[test]
    fn plans_auto_gpu_genotype_format_after_trusted_validation() {
        let passed_plan = plan_auto_gpu_genotype_format_after_trusted_validation(None);
        assert_eq!(passed_plan.resolved_gpu_genotype_format.as_deref(), Some("packed8"));
        assert_eq!(passed_plan.resolution_reason.as_deref(), Some("trusted_validation_passed"));
        assert!(passed_plan.should_log_auto_resolution());

        let failed_plan = plan_auto_gpu_genotype_format_after_trusted_validation(Some("packed8 incompatible"));
        assert_eq!(failed_plan.resolved_gpu_genotype_format.as_deref(), Some("dosage"));
        assert_eq!(failed_plan.resolution_reason.as_deref(), Some("trusted_validation_failed"));
        assert_eq!(failed_plan.fallback_error.as_deref(), Some("packed8 incompatible"));
    }

    #[test]
    fn rejects_invalid_gpu_genotype_format_resolution_inputs() {
        assert_eq!(
            plan_gpu_genotype_format_auto_to_dosage("unknown", "unused").unwrap_err(),
            ScheduleError::UnsupportedGpuGenotypeFormat { gpu_genotype_format: "unknown".to_string() },
        );
        assert_eq!(
            plan_single_trait_binary_gpu_genotype_format_resolution("auto", None, None, false, "tpu").unwrap_err(),
            ScheduleError::UnsupportedJaxDevice { jax_device: "tpu".to_string() },
        );
    }

    #[test]
    fn resolves_delivery_callback_batch_size_default_and_explicit_values() {
        assert_eq!(resolve_delivery_callback_batch_size(None, false).unwrap(), 1);
        assert_eq!(resolve_delivery_callback_batch_size(Some(3), false).unwrap(), 3);
        assert_eq!(resolve_delivery_callback_batch_size(Some(1), true).unwrap(), 1);
    }

    #[test]
    fn rejects_invalid_delivery_callback_batch_sizes() {
        assert_eq!(
            resolve_delivery_callback_batch_size(Some(0), false).unwrap_err(),
            ScheduleError::NonPositiveCallbackBatchSize,
        );
        assert_eq!(
            resolve_delivery_callback_batch_size(Some(2), true).unwrap_err(),
            ScheduleError::Packed8CallbackBatchSize,
        );
    }

    #[test]
    fn plans_bgen_delivery_invocation() {
        assert_eq!(
            plan_bgen_delivery_invocation(Some(3), false, true, true).unwrap(),
            BgenDeliveryInvocationPlan {
                delivery_method: BgenDeliveryMethod::DosageNativeMultiAlignedSamples,
                callback_batch_size: 3,
            },
        );
        assert_eq!(
            plan_bgen_delivery_invocation(None, false, false, true).unwrap(),
            BgenDeliveryInvocationPlan {
                delivery_method: BgenDeliveryMethod::DosageNativeAlignedSamples,
                callback_batch_size: 1,
            },
        );
        assert_eq!(
            plan_bgen_delivery_invocation(Some(1), true, false, false).unwrap(),
            BgenDeliveryInvocationPlan {
                delivery_method: BgenDeliveryMethod::Packed8SampleIndices,
                callback_batch_size: 1,
            },
        );
    }

    #[test]
    fn rejects_invalid_bgen_delivery_invocation_batch_size() {
        assert_eq!(
            plan_bgen_delivery_invocation(Some(2), true, false, false).unwrap_err(),
            ScheduleError::Packed8CallbackBatchSize,
        );
    }

    #[test]
    fn resolves_grouped_union_callback_batch_size() {
        assert_eq!(resolve_grouped_union_callback_batch_size(1).unwrap(), 1);
    }

    #[test]
    fn rejects_invalid_grouped_union_callback_batch_sizes() {
        assert_eq!(
            resolve_grouped_union_callback_batch_size(0).unwrap_err(),
            ScheduleError::NonPositiveCallbackBatchSize,
        );
        assert_eq!(
            resolve_grouped_union_callback_batch_size(2).unwrap_err(),
            ScheduleError::GroupedUnionCallbackBatchSize,
        );
    }

    #[test]
    fn plans_dosage_buffer_reuse_for_exact_and_larger_shapes() {
        assert_eq!(
            plan_dosage_buffer_reuse(&[2, 3], &[2, 3]).unwrap(),
            DosageBufferReusePlan { requires_slice: false, slice_dimensions: vec![2, 3] },
        );
        assert_eq!(
            plan_dosage_buffer_reuse(&[4, 5], &[2, 3]).unwrap(),
            DosageBufferReusePlan { requires_slice: true, slice_dimensions: vec![2, 3] },
        );
    }

    #[test]
    fn rejects_incompatible_dosage_buffer_reuse_shapes() {
        assert_eq!(plan_dosage_buffer_reuse(&[2, 3], &[2, 3, 1]), None);
        assert_eq!(plan_dosage_buffer_reuse(&[2, 3], &[3, 2]), None);
    }

    #[test]
    fn plans_callback_scheduler_dosage_buffer_reuse() {
        let scheduler_state = CallbackSchedulerState::new(3, 2, Some(7), Some(8)).unwrap();

        assert_eq!(
            scheduler_state.plan_dosage_buffer_reuse(&[4, 5], &[2, 3]).unwrap(),
            DosageBufferReusePlan { requires_slice: true, slice_dimensions: vec![2, 3] },
        );
        assert_eq!(scheduler_state.plan_dosage_buffer_reuse(&[2, 3], &[3, 2]), None);
    }

    #[test]
    fn plans_variant_major_dosage_batch_handoff() {
        assert_eq!(
            plan_variant_major_dosage_batch_handoff(2, 2, 2).unwrap(),
            VariantMajorDosageBatchHandoffPlan { chunk_count: 2 },
        );
    }

    #[test]
    fn plans_chunk_batches_from_callback_batch_size() {
        let chunk_specs =
            [chunk_spec(0, 4), chunk_spec(4, 8), chunk_spec(8, 12), chunk_spec(12, 16), chunk_spec(16, 20)];

        let batch_plan = plan_chunk_batches(&chunk_specs, 2).unwrap();

        assert_eq!(batch_plan.chunk_batch_count(), 3);
        assert_eq!(batch_plan.chunk_count(), 5);
        assert_eq!(
            batch_plan.into_chunk_batches(),
            vec![
                vec![chunk_spec(0, 4), chunk_spec(4, 8)],
                vec![chunk_spec(8, 12), chunk_spec(12, 16)],
                vec![chunk_spec(16, 20)],
            ],
        );
    }

    #[test]
    fn plans_no_chunk_batches_for_empty_chunk_specs() {
        let batch_plan = plan_chunk_batches(&[], 2).unwrap();

        assert_eq!(batch_plan.chunk_batch_count(), 0);
        assert_eq!(batch_plan.chunk_count(), 0);
        assert!(batch_plan.into_chunk_batches().is_empty());
    }

    #[test]
    fn rejects_zero_chunk_batch_size() {
        assert_eq!(
            plan_chunk_batches(&[chunk_spec(0, 4)], 0).unwrap_err(),
            ScheduleError::NonPositiveCallbackBatchSize,
        );
    }

    #[test]
    fn plans_callback_scheduler_variant_major_dosage_batch_handoff() {
        let scheduler_state = CallbackSchedulerState::new(3, 2, Some(7), Some(8)).unwrap();

        assert_eq!(
            scheduler_state.plan_variant_major_dosage_batch_handoff(2, 2, 2).unwrap(),
            VariantMajorDosageBatchHandoffPlan { chunk_count: 2 },
        );
        assert_eq!(
            scheduler_state.plan_variant_major_dosage_batch_handoff(2, 1, 2).unwrap_err(),
            ScheduleError::VariantMajorDosageBatchLengthMismatch,
        );
    }

    #[test]
    fn rejects_invalid_variant_major_dosage_batch_handoffs() {
        assert_eq!(
            plan_variant_major_dosage_batch_handoff(2, 1, 2).unwrap_err(),
            ScheduleError::VariantMajorDosageBatchLengthMismatch,
        );
        assert_eq!(
            plan_variant_major_dosage_batch_handoff(0, 0, 0).unwrap_err(),
            ScheduleError::EmptyVariantMajorDosageBatch,
        );
    }

    #[test]
    fn plans_dosage_work_handoff() {
        assert_eq!(plan_dosage_work_handoff(2).unwrap(), DosageWorkHandoffPlan { chunk_count: 2 });
        assert_eq!(plan_dosage_work_handoff(0).unwrap_err(), ScheduleError::EmptyDosageWorkHandoff);

        let scheduler_state = CallbackSchedulerState::new(3, 2, Some(7), Some(8)).unwrap();
        assert_eq!(scheduler_state.plan_dosage_work_handoff(1).unwrap(), DosageWorkHandoffPlan { chunk_count: 1 });
    }

    #[test]
    fn plans_multi_trait_chunk_write_for_uncommitted_traits() {
        assert_eq!(
            plan_multi_trait_chunk_write(
                3,
                32,
                &[BTreeSet::from([0_usize]), BTreeSet::from([32_usize]), BTreeSet::from([64_usize]),],
            )
            .unwrap(),
            MultiTraitChunkWritePlan { active_trait_indices: vec![0, 2], total_trait_count: 3 },
        );
    }

    #[test]
    fn plans_multi_trait_chunk_write_when_all_traits_committed() {
        let write_plan =
            plan_multi_trait_chunk_write(2, 32, &[BTreeSet::from([32_usize]), BTreeSet::from([0_usize, 32])]).unwrap();

        assert_eq!(write_plan.active_trait_indices, Vec::<usize>::new());
        assert_eq!(write_plan.active_trait_count(), 0);
        assert!(write_plan.all_traits_committed());
    }

    #[test]
    fn rejects_mismatched_multi_trait_committed_chunk_set_counts() {
        assert_eq!(
            plan_multi_trait_chunk_write(2, 32, &[BTreeSet::new()]).unwrap_err(),
            ScheduleError::MultiTraitCommittedChunkSetCountMismatch { writer_session_count: 2, committed_set_count: 1 },
        );
    }

    #[test]
    fn tracks_dosage_buffer_pool_slots() {
        let mut buffer_pool_state = DosageBufferPoolState::new(2);

        assert_eq!(buffer_pool_state.buffer_limit(), 2);
        assert_eq!(buffer_pool_state.allocated_count(), 0);
        assert!(buffer_pool_state.has_available_slot());
        assert!(buffer_pool_state.register_buffer(11));
        assert!(buffer_pool_state.owns_buffer(11));
        assert!(!buffer_pool_state.register_buffer(11));
        assert!(buffer_pool_state.register_buffer(7));
        assert_eq!(buffer_pool_state.allocated_count(), 2);
        assert_eq!(buffer_pool_state.buffer_identifiers(), vec![7, 11]);
        assert!(!buffer_pool_state.has_available_slot());
        assert!(!buffer_pool_state.register_buffer(13));
        assert!(buffer_pool_state.discard_buffer(11));
        assert!(!buffer_pool_state.owns_buffer(11));
        assert!(buffer_pool_state.has_available_slot());
        assert!(!buffer_pool_state.discard_buffer(99));
    }

    #[test]
    fn plans_dosage_buffer_pool_attempts() {
        let mut scheduler_state = CallbackSchedulerState::new(1, 1, None, Some(1)).unwrap();

        assert_eq!(
            scheduler_state.plan_dosage_buffer_acquire_attempt(0, 0.25),
            DosageBufferAcquireAttemptPlan {
                should_take_free_buffer: false,
                should_allocate: true,
                should_wait: false,
                wait_timeout_seconds: 0.0,
                free_buffer_count: 0,
                allocated_count: 0,
                buffer_limit: 1,
            },
        );
        assert_eq!(
            scheduler_state.plan_dosage_buffer_register_attempt(11),
            DosageBufferRegisterAttemptPlan {
                should_register: true,
                has_registration_error: false,
                allocated_count: 1,
                buffer_limit: 1,
            },
        );
        assert_eq!(
            scheduler_state.plan_dosage_buffer_register_attempt(13),
            DosageBufferRegisterAttemptPlan {
                should_register: false,
                has_registration_error: true,
                allocated_count: 1,
                buffer_limit: 1,
            },
        );
        assert_eq!(
            scheduler_state.plan_dosage_buffer_acquire_attempt(0, 0.25),
            DosageBufferAcquireAttemptPlan {
                should_take_free_buffer: false,
                should_allocate: false,
                should_wait: true,
                wait_timeout_seconds: 0.25,
                free_buffer_count: 0,
                allocated_count: 1,
                buffer_limit: 1,
            },
        );
        assert_eq!(
            scheduler_state.plan_dosage_buffer_acquire_backpressure_attempt(0),
            DosageBufferAcquireAttemptPlan {
                should_take_free_buffer: false,
                should_allocate: false,
                should_wait: true,
                wait_timeout_seconds: 0.1,
                free_buffer_count: 0,
                allocated_count: 1,
                buffer_limit: 1,
            },
        );
        assert_eq!(
            scheduler_state.plan_dosage_buffer_acquire_attempt(1, 0.25),
            DosageBufferAcquireAttemptPlan {
                should_take_free_buffer: true,
                should_allocate: false,
                should_wait: false,
                wait_timeout_seconds: 0.0,
                free_buffer_count: 1,
                allocated_count: 1,
                buffer_limit: 1,
            },
        );
        assert_eq!(
            scheduler_state.plan_dosage_buffer_return_attempt(11),
            DosageBufferReturnAttemptPlan { should_return: true, allocated_count: 1, buffer_limit: 1 },
        );
        assert_eq!(
            scheduler_state.plan_dosage_buffer_return_attempt(13),
            DosageBufferReturnAttemptPlan { should_return: false, allocated_count: 1, buffer_limit: 1 },
        );
        assert_eq!(
            scheduler_state.plan_dosage_buffer_discard_attempt(11),
            DosageBufferDiscardAttemptPlan { should_discard: true, allocated_count: 0, buffer_limit: 1 },
        );
        assert_eq!(
            scheduler_state.plan_dosage_buffer_discard_attempt(11),
            DosageBufferDiscardAttemptPlan { should_discard: false, allocated_count: 0, buffer_limit: 1 },
        );
        assert_eq!(
            scheduler_state.plan_dosage_buffer_acquire_attempt(0, f64::NAN),
            DosageBufferAcquireAttemptPlan {
                should_take_free_buffer: false,
                should_allocate: true,
                should_wait: false,
                wait_timeout_seconds: 0.0,
                free_buffer_count: 0,
                allocated_count: 0,
                buffer_limit: 1,
            },
        );
    }

    #[test]
    fn plans_dosage_buffer_pool_observations() {
        let scheduler_state = CallbackSchedulerState::new(1, 1, None, Some(1)).unwrap();

        assert_eq!(
            scheduler_state.plan_dosage_buffer_pool_reuse_observation(),
            DosageBufferPoolObservationPlan { operation_name: QUEUE_REUSE_OPERATION.to_string(), blocked: false },
        );
        assert_eq!(
            scheduler_state.plan_dosage_buffer_pool_return_observation(),
            DosageBufferPoolObservationPlan { operation_name: QUEUE_RETURN_OPERATION.to_string(), blocked: false },
        );
        assert_eq!(
            scheduler_state.plan_dosage_buffer_pool_allocate_observation(),
            DosageBufferPoolObservationPlan { operation_name: QUEUE_ALLOCATE_OPERATION.to_string(), blocked: false },
        );
        assert_eq!(
            scheduler_state.plan_dosage_buffer_pool_discard_observation(),
            DosageBufferPoolObservationPlan { operation_name: QUEUE_DISCARD_OPERATION.to_string(), blocked: false },
        );
        assert_eq!(
            scheduler_state.plan_dosage_buffer_pool_consumer_wait_observation(),
            DosageBufferPoolObservationPlan {
                operation_name: QUEUE_CONSUMER_WAIT_OPERATION.to_string(),
                blocked: true,
            },
        );
    }

    #[test]
    fn tracks_result_in_flight_slots() {
        let mut slot_state = ResultInFlightSlotState::new(2);

        assert_eq!(slot_state.slot_limit(), 2);
        assert_eq!(slot_state.occupied_count(), 0);
        assert!(slot_state.has_available_slot());
        assert!(slot_state.acquire_slot());
        assert_eq!(slot_state.occupied_count(), 1);
        assert!(slot_state.acquire_slot());
        assert_eq!(slot_state.occupied_count(), 2);
        assert!(!slot_state.has_available_slot());
        assert!(!slot_state.acquire_slot());
        assert!(slot_state.release_slot());
        assert_eq!(slot_state.occupied_count(), 1);
        assert!(slot_state.release_slot());
        assert_eq!(slot_state.occupied_count(), 0);
        assert!(!slot_state.release_slot());
    }

    #[test]
    fn plans_result_in_flight_slot_attempts() {
        let mut scheduler_state = CallbackSchedulerState::new(1, 1, Some(1), None).unwrap();

        assert_eq!(
            scheduler_state.plan_result_in_flight_slot_acquire_attempt(0.25),
            ResultInFlightAcquireAttemptPlan {
                should_acquire: true,
                should_wait: false,
                wait_timeout_seconds: 0.0,
                occupied_count: 1,
                slot_limit: 1,
            },
        );
        assert_eq!(
            scheduler_state.plan_result_in_flight_slot_acquire_observation(&ResultInFlightAcquireAttemptPlan {
                should_acquire: true,
                should_wait: false,
                wait_timeout_seconds: 0.0,
                occupied_count: 1,
                slot_limit: 1,
            }),
            ResultInFlightAcquireObservationPlan {
                resource_name: RESULT_IN_FLIGHT_SLOTS_NAME.to_string(),
                operation_name: RESULT_SLOT_ACQUIRE_OPERATION.to_string(),
                blocked: false,
                should_retry_acquisition: false,
            },
        );
        assert_eq!(
            scheduler_state.plan_result_in_flight_slot_acquire_attempt(0.25),
            ResultInFlightAcquireAttemptPlan {
                should_acquire: false,
                should_wait: true,
                wait_timeout_seconds: 0.25,
                occupied_count: 1,
                slot_limit: 1,
            },
        );
        assert_eq!(
            scheduler_state.plan_result_in_flight_slot_acquire_observation(&ResultInFlightAcquireAttemptPlan {
                should_acquire: false,
                should_wait: true,
                wait_timeout_seconds: 0.25,
                occupied_count: 1,
                slot_limit: 1,
            }),
            ResultInFlightAcquireObservationPlan {
                resource_name: RESULT_IN_FLIGHT_SLOTS_NAME.to_string(),
                operation_name: QUEUE_PRODUCER_BLOCKING_OPERATION.to_string(),
                blocked: true,
                should_retry_acquisition: true,
            },
        );
        assert_eq!(
            scheduler_state.plan_result_in_flight_slot_acquire_backpressure_attempt(),
            ResultInFlightAcquireAttemptPlan {
                should_acquire: false,
                should_wait: true,
                wait_timeout_seconds: 0.1,
                occupied_count: 1,
                slot_limit: 1,
            },
        );
        assert_eq!(
            scheduler_state.plan_result_in_flight_slot_release_attempt(),
            ResultInFlightReleaseAttemptPlan {
                should_release: true,
                has_release_error: false,
                occupied_count: 0,
                slot_limit: 1,
            },
        );
        assert_eq!(
            scheduler_state.plan_result_in_flight_slot_release_attempt(),
            ResultInFlightReleaseAttemptPlan {
                should_release: false,
                has_release_error: true,
                occupied_count: 0,
                slot_limit: 1,
            },
        );
        assert_eq!(
            scheduler_state.plan_result_in_flight_slot_acquire_attempt(f64::NAN),
            ResultInFlightAcquireAttemptPlan {
                should_acquire: true,
                should_wait: false,
                wait_timeout_seconds: 0.0,
                occupied_count: 1,
                slot_limit: 1,
            },
        );
        assert_eq!(
            scheduler_state.plan_result_in_flight_slot_acquire_attempt(f64::NAN),
            ResultInFlightAcquireAttemptPlan {
                should_acquire: false,
                should_wait: false,
                wait_timeout_seconds: 0.0,
                occupied_count: 1,
                slot_limit: 1,
            },
        );
    }

    #[test]
    fn plans_result_in_flight_slot_release_observation() {
        let scheduler_state = CallbackSchedulerState::new(1, 1, Some(1), Some(1)).unwrap();

        assert_eq!(
            scheduler_state.plan_result_in_flight_slot_release_observation(),
            ResultInFlightReleaseObservationPlan {
                resource_name: RESULT_IN_FLIGHT_SLOTS_NAME.to_string(),
                operation_name: RESULT_SLOT_RELEASE_OPERATION.to_string(),
                blocked: false,
            },
        );
    }

    #[test]
    fn plans_result_write_item_resource_release() {
        let scheduler_state = CallbackSchedulerState::new(1, 1, Some(1), Some(1)).unwrap();

        assert_eq!(
            scheduler_state.plan_result_write_item_pre_write_resource_release(true),
            ResultWriteItemResourceReleasePlan {
                should_release_host_buffer: true,
                should_release_result_in_flight_slot: false,
            },
        );
        assert_eq!(
            scheduler_state.plan_result_write_item_pre_write_resource_release(false),
            ResultWriteItemResourceReleasePlan {
                should_release_host_buffer: false,
                should_release_result_in_flight_slot: false,
            },
        );
        assert_eq!(
            scheduler_state.plan_result_write_item_final_resource_release(true, true, true),
            ResultWriteItemResourceReleasePlan {
                should_release_host_buffer: false,
                should_release_result_in_flight_slot: true,
            },
        );
        assert_eq!(
            scheduler_state.plan_result_write_item_final_resource_release(true, false, false),
            ResultWriteItemResourceReleasePlan {
                should_release_host_buffer: true,
                should_release_result_in_flight_slot: false,
            },
        );
    }

    #[test]
    fn plans_result_write_handoff() {
        let scheduler_state = CallbackSchedulerState::new(1, 1, Some(1), Some(1)).unwrap();

        assert_eq!(
            plan_result_write_handoff(true),
            ResultWriteHandoffPlan { should_enqueue: true, has_result_work_item: true, is_stop_signal: false },
        );
        assert_eq!(
            scheduler_state.plan_result_write_handoff(false),
            ResultWriteHandoffPlan { should_enqueue: true, has_result_work_item: false, is_stop_signal: true },
        );
    }

    #[test]
    fn plans_result_write_drain_completion() {
        let scheduler_state = CallbackSchedulerState::new(1, 1, Some(1), Some(1)).unwrap();

        assert_eq!(
            scheduler_state.plan_result_write_drain_completion(true, true),
            ResultWriteDrainCompletionPlan { should_stop: false, should_flush_binary_correction_diagnostics: false },
        );
        assert_eq!(
            scheduler_state.plan_result_write_drain_completion(false, true),
            ResultWriteDrainCompletionPlan { should_stop: true, should_flush_binary_correction_diagnostics: true },
        );
        assert_eq!(
            scheduler_state.plan_result_write_drain_completion(false, false),
            ResultWriteDrainCompletionPlan { should_stop: true, should_flush_binary_correction_diagnostics: false },
        );
    }

    #[test]
    fn plans_result_write_item_dispatch() {
        let scheduler_state = CallbackSchedulerState::new(1, 1, Some(1), Some(1)).unwrap();

        assert_eq!(
            scheduler_state
                .plan_result_write_item_dispatch(
                    RESULT_WRITE_ITEM_KIND_SINGLE_RESULT,
                    RESULT_WRITE_ITEM_KIND_SINGLE_RESULT,
                )
                .unwrap(),
            ResultWriteItemDispatchPlan {
                result_work_item_kind: RESULT_WRITE_ITEM_KIND_SINGLE_RESULT.to_owned(),
                expected_result_work_item_kind: RESULT_WRITE_ITEM_KIND_SINGLE_RESULT.to_owned(),
                should_process_result_write_item: true,
                should_process_multi_result_write_item: false,
                has_dispatch_error: false,
                error_message: None,
            },
        );
        assert_eq!(
            plan_result_write_item_dispatch(RESULT_WRITE_ITEM_KIND_MULTI_RESULT, RESULT_WRITE_ITEM_KIND_MULTI_RESULT)
                .unwrap(),
            ResultWriteItemDispatchPlan {
                result_work_item_kind: RESULT_WRITE_ITEM_KIND_MULTI_RESULT.to_owned(),
                expected_result_work_item_kind: RESULT_WRITE_ITEM_KIND_MULTI_RESULT.to_owned(),
                should_process_result_write_item: false,
                should_process_multi_result_write_item: true,
                has_dispatch_error: false,
                error_message: None,
            },
        );

        let missing_item_plan =
            plan_result_write_item_dispatch(RESULT_WRITE_ITEM_KIND_STOP_SIGNAL, RESULT_WRITE_ITEM_KIND_SINGLE_RESULT)
                .unwrap();
        assert!(missing_item_plan.has_dispatch_error);
        assert_eq!(
            missing_item_plan.error_message.as_deref(),
            Some("Native result write dispatch plan continued without a work item."),
        );

        let mismatched_item_plan =
            plan_result_write_item_dispatch(RESULT_WRITE_ITEM_KIND_SINGLE_RESULT, RESULT_WRITE_ITEM_KIND_MULTI_RESULT)
                .unwrap();
        assert!(mismatched_item_plan.has_dispatch_error);
        assert_eq!(
            mismatched_item_plan.error_message.as_deref(),
            Some("Native result write dispatch plan expected multi_result but received single_result."),
        );
        assert_eq!(
            plan_result_write_item_dispatch("unknown", RESULT_WRITE_ITEM_KIND_SINGLE_RESULT).unwrap_err(),
            ScheduleError::UnsupportedResultWriteItemKind { result_work_item_kind: "unknown".to_owned() },
        );
    }

    #[test]
    fn plans_dosage_work_drain_completion() {
        let scheduler_state = CallbackSchedulerState::new(1, 1, Some(1), Some(1)).unwrap();

        assert_eq!(
            scheduler_state.plan_dosage_work_drain_completion(true),
            DosageWorkDrainCompletionPlan { should_stop: false },
        );
        assert_eq!(
            scheduler_state.plan_dosage_work_drain_completion(false),
            DosageWorkDrainCompletionPlan { should_stop: true },
        );
    }

    #[test]
    fn plans_dosage_work_item_dispatch() {
        let scheduler_state = CallbackSchedulerState::new(1, 1, Some(1), Some(1)).unwrap();

        assert_eq!(
            scheduler_state.plan_dosage_work_item_dispatch(DOSAGE_WORK_ITEM_KIND_SAMPLE_MAJOR_DOSAGE).unwrap(),
            DosageWorkItemDispatchPlan {
                dosage_work_item_kind: DOSAGE_WORK_ITEM_KIND_SAMPLE_MAJOR_DOSAGE.to_owned(),
                processing_path: Some(DOSAGE_WORK_ITEM_KIND_SAMPLE_MAJOR_DOSAGE.to_owned()),
                error_message: None,
            },
        );
        assert_eq!(
            plan_dosage_work_item_dispatch(DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_DOSAGE).unwrap(),
            DosageWorkItemDispatchPlan {
                dosage_work_item_kind: DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_DOSAGE.to_owned(),
                processing_path: Some(DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_DOSAGE.to_owned()),
                error_message: None,
            },
        );
        assert_eq!(
            plan_dosage_work_item_dispatch(DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_DOSAGE_BATCH).unwrap(),
            DosageWorkItemDispatchPlan {
                dosage_work_item_kind: DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_DOSAGE_BATCH.to_owned(),
                processing_path: Some(DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_DOSAGE_BATCH.to_owned()),
                error_message: None,
            },
        );
        assert_eq!(
            plan_dosage_work_item_dispatch(DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_PACKED8_PROBABILITY_PAIR).unwrap(),
            DosageWorkItemDispatchPlan {
                dosage_work_item_kind: DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_PACKED8_PROBABILITY_PAIR.to_owned(),
                processing_path: Some(DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_PACKED8_PROBABILITY_PAIR.to_owned()),
                error_message: None,
            },
        );

        let missing_item_plan = plan_dosage_work_item_dispatch(DOSAGE_WORK_ITEM_KIND_STOP_SIGNAL).unwrap();
        assert!(missing_item_plan.has_dispatch_error());
        assert_eq!(
            missing_item_plan.error_message.as_deref(),
            Some("Native dosage work dispatch plan continued without a work item."),
        );
        assert_eq!(
            plan_dosage_work_item_dispatch("unknown").unwrap_err(),
            ScheduleError::UnsupportedDosageWorkItemKind { dosage_work_item_kind: "unknown".to_owned() },
        );
    }

    #[test]
    fn plans_dosage_work_item_stage_duration_attribution() {
        let scheduler_state = CallbackSchedulerState::new(1, 1, Some(1), Some(1)).unwrap();

        assert_eq!(
            scheduler_state
                .plan_dosage_work_item_stage_duration(DOSAGE_WORK_ITEM_KIND_SAMPLE_MAJOR_DOSAGE, 1, 3.0)
                .unwrap(),
            DosageWorkItemStageDurationPlan { chunk_count: 1, duration_per_chunk: 3.0 },
        );
        assert_eq!(
            plan_dosage_work_item_stage_duration(DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_DOSAGE_BATCH, 2, 5.0).unwrap(),
            DosageWorkItemStageDurationPlan { chunk_count: 2, duration_per_chunk: 2.5 },
        );
        assert_eq!(
            plan_dosage_work_item_stage_duration(DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_DOSAGE_BATCH, 0, 5.0).unwrap_err(),
            ScheduleError::EmptyDosageWorkItemStageDuration,
        );
        assert_eq!(
            plan_dosage_work_item_stage_duration(DOSAGE_WORK_ITEM_KIND_STOP_SIGNAL, 1, 5.0).unwrap_err(),
            ScheduleError::DosageWorkItemStageDurationStopSignal,
        );
        assert_eq!(
            plan_dosage_work_item_stage_duration(DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_DOSAGE, 2, 5.0).unwrap_err(),
            ScheduleError::DosageWorkItemStageDurationChunkCountMismatch {
                dosage_work_item_kind: DOSAGE_WORK_ITEM_KIND_VARIANT_MAJOR_DOSAGE.to_owned(),
                chunk_count: 2,
            },
        );
    }

    #[test]
    fn plans_callback_scheduler_worker_error_raise() {
        let mut scheduler_state = CallbackSchedulerState::new(1, 1, None, None).unwrap();

        assert_eq!(
            scheduler_state.plan_worker_error_raise(),
            CallbackWorkerErrorRaisePlan {
                should_raise: false,
                raise_dosage_worker_error: false,
                raise_result_worker_error: false,
                error_message: None,
            },
        );
        scheduler_state.record_result_worker_error("writer failed");
        assert_eq!(
            scheduler_state.plan_worker_error_raise(),
            CallbackWorkerErrorRaisePlan {
                should_raise: true,
                raise_dosage_worker_error: false,
                raise_result_worker_error: true,
                error_message: Some("native pipeline result writer worker failed: writer failed".to_string()),
            },
        );
        scheduler_state.record_dosage_worker_error("dosage failed");
        assert_eq!(
            scheduler_state.plan_worker_error_raise(),
            CallbackWorkerErrorRaisePlan {
                should_raise: true,
                raise_dosage_worker_error: true,
                raise_result_worker_error: false,
                error_message: Some("native pipeline callback worker failed: dosage failed".to_string()),
            },
        );
    }

    #[test]
    fn updates_callback_scheduler_worker_errors() {
        let mut scheduler_state = CallbackSchedulerState::new(1, 1, None, None).unwrap();

        assert_eq!(
            scheduler_state.update_dosage_worker_error(Some("dosage failed")),
            CallbackWorkerErrorUpdatePlan {
                had_error: false,
                has_error: true,
                error_message: Some("native pipeline callback worker failed: dosage failed".to_string()),
            },
        );
        assert_eq!(
            scheduler_state.dosage_worker_error_message(),
            Some("native pipeline callback worker failed: dosage failed")
        );
        assert_eq!(
            scheduler_state.update_dosage_worker_error(None),
            CallbackWorkerErrorUpdatePlan { had_error: true, has_error: false, error_message: None },
        );
        assert_eq!(scheduler_state.dosage_worker_error_message(), None);
        assert_eq!(
            scheduler_state.update_result_worker_error(Some("writer failed")),
            CallbackWorkerErrorUpdatePlan {
                had_error: false,
                has_error: true,
                error_message: Some("native pipeline result writer worker failed: writer failed".to_string()),
            },
        );
        assert_eq!(
            scheduler_state.result_worker_error_message(),
            Some("native pipeline result writer worker failed: writer failed"),
        );
    }

    #[test]
    fn tracks_callback_worker_lifecycle_start() {
        let mut lifecycle_state = CallbackWorkerLifecycleState::new();

        assert!(!lifecycle_state.has_started());
        assert!(lifecycle_state.mark_started());
        assert!(lifecycle_state.has_started());
        assert!(!lifecycle_state.mark_started());
    }

    #[test]
    fn plans_callback_worker_start_policy() {
        let start_plan = plan_callback_worker_start(false);

        assert!(start_plan.should_start());
        assert!(start_plan.start_result_worker());
        assert!(start_plan.start_dosage_worker());
        assert_eq!(
            start_plan.start_actions,
            vec![
                CALLBACK_WORKER_START_RESULT_WORKER_ACTION.to_string(),
                CALLBACK_WORKER_START_DOSAGE_WORKER_ACTION.to_string(),
            ],
        );

        let already_started_plan = plan_callback_worker_start(true);
        assert!(!already_started_plan.should_start());
        assert!(!already_started_plan.start_result_worker());
        assert!(!already_started_plan.start_dosage_worker());
        assert!(already_started_plan.start_actions.is_empty());
    }

    #[test]
    fn plans_callback_scheduler_worker_start_attempts() {
        let mut scheduler_state = CallbackSchedulerState::new(1, 1, None, None).unwrap();

        assert_eq!(
            scheduler_state.plan_worker_start_attempt(),
            CallbackWorkerStartAttemptPlan {
                start_actions: vec![
                    CALLBACK_WORKER_START_RESULT_WORKER_ACTION.to_string(),
                    CALLBACK_WORKER_START_DOSAGE_WORKER_ACTION.to_string(),
                ],
                has_marked_started: true,
                has_start_error: false,
                error_message: None,
            },
        );
        assert!(scheduler_state.has_started());
        assert_eq!(
            scheduler_state.plan_worker_start_attempt(),
            CallbackWorkerStartAttemptPlan {
                start_actions: Vec::new(),
                has_marked_started: false,
                has_start_error: false,
                error_message: None,
            },
        );
    }

    #[must_use]
    fn expected_callback_worker_finish_plan() -> CallbackWorkerFinishPlan {
        CallbackWorkerFinishPlan {
            finish_actions: vec![
                CALLBACK_WORKER_FINISH_STOP_DOSAGE_WORKER_ACTION.to_string(),
                CALLBACK_WORKER_FINISH_JOIN_DOSAGE_WORKER_ACTION.to_string(),
                CALLBACK_WORKER_FINISH_STOP_RESULT_WORKER_ACTION.to_string(),
                CALLBACK_WORKER_FINISH_JOIN_RESULT_WORKER_ACTION.to_string(),
                CALLBACK_WORKER_FINISH_RAISE_WORKER_ERROR_ACTION.to_string(),
                CALLBACK_WORKER_FINISH_COMPLETE_PROGRESS_ACTION.to_string(),
                CALLBACK_WORKER_FINISH_EMIT_BINARY_CORRECTION_SUMMARY_ACTION.to_string(),
            ],
            dosage_stop_timeout_seconds: 60.0,
            dosage_join_timeout_seconds: 300.0,
            result_stop_timeout_seconds: 60.0,
            result_join_timeout_seconds: 300.0,
        }
    }

    #[must_use]
    fn expected_callback_worker_abort_plan() -> CallbackWorkerAbortPlan {
        CallbackWorkerAbortPlan {
            abort_actions: vec![
                CALLBACK_WORKER_FINISH_STOP_DOSAGE_WORKER_ACTION.to_string(),
                CALLBACK_WORKER_FINISH_STOP_RESULT_WORKER_ACTION.to_string(),
            ],
            dosage_stop_timeout_seconds: 1.0,
            result_stop_timeout_seconds: 1.0,
        }
    }

    #[test]
    fn tracks_callback_scheduler_state() {
        let mut scheduler_state = CallbackSchedulerState::new(3, 2, Some(7), Some(8)).unwrap();

        assert_eq!(scheduler_state.queue_limits().dosage_queue_depth, 3);
        assert_eq!(scheduler_state.native_callback_batch_size(), 2);
        assert_eq!(scheduler_state.dosage_queue_depth(), 3);
        assert_eq!(scheduler_state.dosage_queue_capacity(), 3);
        assert_eq!(scheduler_state.dosage_queue_occupied_count(), 0);
        assert!(scheduler_state.has_available_dosage_queue_slot());
        assert_eq!(scheduler_state.result_queue_depth(), 3);
        assert_eq!(scheduler_state.result_queue_capacity(), 3);
        assert_eq!(scheduler_state.result_queue_occupied_count(), 0);
        assert!(scheduler_state.has_available_result_queue_slot());
        assert_eq!(scheduler_state.result_in_flight_limit(), 7);
        assert_eq!(scheduler_state.result_in_flight_slot_limit(), 7);
        assert_eq!(scheduler_state.dosage_buffer_limit(), 8);
        assert_eq!(scheduler_state.dosage_buffer_pool_limit(), 8);
        assert!(!scheduler_state.has_started());
        assert_eq!(scheduler_state.plan_worker_start(), plan_callback_worker_start(false));
        assert!(scheduler_state.mark_started());
        assert!(scheduler_state.has_started());
        assert_eq!(scheduler_state.plan_worker_start(), plan_callback_worker_start(true));
        assert!(!scheduler_state.mark_started());

        assert!(scheduler_state.acquire_dosage_queue_slot());
        assert_eq!(scheduler_state.dosage_queue_occupied_count(), 1);
        assert!(scheduler_state.has_available_dosage_queue_slot());
        assert!(scheduler_state.release_dosage_queue_slot());
        assert_eq!(scheduler_state.dosage_queue_occupied_count(), 0);

        assert!(scheduler_state.acquire_result_queue_slot());
        assert_eq!(scheduler_state.result_queue_occupied_count(), 1);
        assert!(scheduler_state.has_available_result_queue_slot());
        assert!(scheduler_state.release_result_queue_slot());
        assert_eq!(scheduler_state.result_queue_occupied_count(), 0);

        assert!(scheduler_state.acquire_result_in_flight_slot());
        assert_eq!(scheduler_state.result_in_flight_occupied_count(), 1);
        assert!(scheduler_state.has_available_result_in_flight_slot());
        assert!(scheduler_state.release_result_in_flight_slot());
        assert_eq!(scheduler_state.result_in_flight_occupied_count(), 0);

        assert!(scheduler_state.register_dosage_buffer(11));
        assert!(scheduler_state.owns_dosage_buffer(11));
        assert_eq!(scheduler_state.dosage_buffer_allocated_count(), 1);
        assert_eq!(scheduler_state.dosage_buffer_identifiers(), vec![11]);
        assert!(scheduler_state.has_available_dosage_buffer_slot());
        assert!(scheduler_state.discard_dosage_buffer(11));
        assert_eq!(scheduler_state.dosage_buffer_allocated_count(), 0);

        assert!(!scheduler_state.has_dosage_worker_error());
        assert!(!scheduler_state.has_result_worker_error());
        scheduler_state.record_dosage_worker_error("dosage failed");
        scheduler_state.record_result_worker_error("writer failed");
        assert_eq!(
            scheduler_state.dosage_worker_error_message(),
            Some("native pipeline callback worker failed: dosage failed"),
        );
        assert_eq!(
            scheduler_state.result_worker_error_message(),
            Some("native pipeline result writer worker failed: writer failed"),
        );
        assert!(scheduler_state.has_dosage_worker_error());
        assert!(scheduler_state.has_result_worker_error());
        assert!(scheduler_state.clear_dosage_worker_error());
        assert!(scheduler_state.clear_result_worker_error());
        assert_eq!(scheduler_state.dosage_worker_error_message(), None);
        assert_eq!(scheduler_state.result_worker_error_message(), None);

        assert!((scheduler_state.backpressure_poll_timeout_seconds() - 0.1).abs() < f64::EPSILON);
        assert_eq!(scheduler_state.plan_worker_finish(), expected_callback_worker_finish_plan(),);
        assert_eq!(scheduler_state.plan_worker_abort(), expected_callback_worker_abort_plan(),);

        assert_eq!(
            scheduler_state.plan_dosage_worker_join(None),
            CallbackWorkerJoinPlan { should_join: true, timeout_seconds: 60.0 },
        );
        assert_eq!(
            scheduler_state.plan_dosage_worker_stop(None, true),
            CallbackWorkerStopPlan { should_stop: true, timeout_seconds: 60.0 },
        );
        assert_eq!(
            scheduler_state.plan_dosage_worker_stop_poll(1.0, true),
            CallbackWorkerStopPollPlan { should_stop: true, poll_timeout_seconds: 0.1 },
        );
        scheduler_state.record_result_worker_error("writer failed");
        assert_eq!(
            scheduler_state.plan_result_worker_stop(None, true),
            CallbackWorkerStopPlan { should_stop: false, timeout_seconds: 60.0 },
        );
        assert_eq!(
            scheduler_state.plan_result_worker_stop_poll(1.0, true),
            CallbackWorkerStopPollPlan { should_stop: false, poll_timeout_seconds: 0.1 },
        );
        assert_eq!(
            scheduler_state.plan_result_worker_join(Some(0.25)),
            CallbackWorkerJoinPlan { should_join: true, timeout_seconds: 0.25 },
        );
    }

    #[test]
    fn plans_callback_scheduler_queue_observations() {
        let scheduler_state = CallbackSchedulerState::new(3, 2, Some(7), Some(8)).unwrap();

        assert_eq!(
            scheduler_state
                .plan_queue_operation_observation(DOSAGE_BUFFER_POOL_NAME, QUEUE_RETURN_OPERATION, 0.25, true)
                .unwrap(),
            CallbackQueueOperationObservationPlan {
                queue_name: DOSAGE_BUFFER_POOL_NAME.to_string(),
                operation_name: QUEUE_RETURN_OPERATION.to_string(),
                blocked_seconds: 0.25,
            },
        );
        assert_eq!(
            scheduler_state
                .plan_queue_backpressure_observation(DOSAGE_BUFFER_POOL_NAME, QUEUE_RETURN_OPERATION, 1, 2, 0.25, true)
                .unwrap(),
            CallbackQueueBackpressureObservation {
                queue_name: DOSAGE_BUFFER_POOL_NAME.to_string(),
                operation_name: QUEUE_RETURN_OPERATION.to_string(),
                queue_depth: 1,
                queue_capacity: 2,
                elapsed_seconds: 0.25,
                blocked_seconds: 0.25,
            },
        );
        assert_eq!(
            scheduler_state
                .plan_queue_stage_observation(DOSAGE_QUEUE_NAME, QUEUE_PRODUCER_BLOCKING_OPERATION, 0.5, true)
                .unwrap(),
            CallbackQueueStageObservationPlan {
                queue_name: DOSAGE_QUEUE_NAME.to_string(),
                operation_name: QUEUE_PRODUCER_BLOCKING_OPERATION.to_string(),
                stage_name: "callback_queue_producer_blocking".to_string(),
                blocked_seconds: 0.5,
            },
        );
        assert_eq!(
            scheduler_state
                .plan_queue_stage_backpressure_observation(
                    DOSAGE_QUEUE_NAME,
                    QUEUE_PRODUCER_BLOCKING_OPERATION,
                    3,
                    3,
                    0.5,
                    true,
                )
                .unwrap(),
            CallbackQueueStageBackpressureObservation {
                queue_name: DOSAGE_QUEUE_NAME.to_string(),
                operation_name: QUEUE_PRODUCER_BLOCKING_OPERATION.to_string(),
                stage_name: "callback_queue_producer_blocking".to_string(),
                queue_depth: 3,
                queue_capacity: 3,
                elapsed_seconds: 0.5,
                blocked_seconds: 0.5,
            },
        );
    }

    #[test]
    fn plans_callback_scheduler_current_queue_observations() {
        let mut scheduler_state = CallbackSchedulerState::new(3, 2, Some(7), Some(8)).unwrap();

        assert!(scheduler_state.acquire_dosage_queue_slot());
        assert_eq!(
            scheduler_state
                .plan_current_queue_stage_backpressure_observation(
                    DOSAGE_QUEUE_NAME,
                    QUEUE_PRODUCER_BLOCKING_OPERATION,
                    0.5,
                    true,
                )
                .unwrap(),
            CallbackQueueStageBackpressureObservation {
                queue_name: DOSAGE_QUEUE_NAME.to_string(),
                operation_name: QUEUE_PRODUCER_BLOCKING_OPERATION.to_string(),
                stage_name: "callback_queue_producer_blocking".to_string(),
                queue_depth: 1,
                queue_capacity: 3,
                elapsed_seconds: 0.5,
                blocked_seconds: 0.5,
            },
        );
        assert!(scheduler_state.acquire_result_in_flight_slot());
        assert_eq!(
            scheduler_state
                .plan_current_queue_backpressure_observation(
                    RESULT_IN_FLIGHT_SLOTS_NAME,
                    RESULT_SLOT_RELEASE_OPERATION,
                    0.25,
                    false,
                )
                .unwrap(),
            CallbackQueueBackpressureObservation {
                queue_name: RESULT_IN_FLIGHT_SLOTS_NAME.to_string(),
                operation_name: RESULT_SLOT_RELEASE_OPERATION.to_string(),
                queue_depth: 1,
                queue_capacity: 7,
                elapsed_seconds: 0.25,
                blocked_seconds: 0.0,
            },
        );
        assert_eq!(
            scheduler_state
                .plan_dosage_buffer_pool_backpressure_observation(QUEUE_REUSE_OPERATION, 4, 0.25, false)
                .unwrap(),
            CallbackQueueBackpressureObservation {
                queue_name: DOSAGE_BUFFER_POOL_NAME.to_string(),
                operation_name: QUEUE_REUSE_OPERATION.to_string(),
                queue_depth: 4,
                queue_capacity: 8,
                elapsed_seconds: 0.25,
                blocked_seconds: 0.0,
            },
        );
        assert_eq!(
            scheduler_state
                .plan_dosage_buffer_pool_stage_backpressure_observation(QUEUE_CONSUMER_WAIT_OPERATION, 2, 0.5, true,)
                .unwrap(),
            CallbackQueueStageBackpressureObservation {
                queue_name: DOSAGE_BUFFER_POOL_NAME.to_string(),
                operation_name: QUEUE_CONSUMER_WAIT_OPERATION.to_string(),
                stage_name: "dosage_buffer_pool_consumer_wait".to_string(),
                queue_depth: 2,
                queue_capacity: 8,
                elapsed_seconds: 0.5,
                blocked_seconds: 0.5,
            },
        );
    }

    #[test]
    fn plans_callback_scheduler_queue_put_and_get_attempts() {
        let mut scheduler_state = CallbackSchedulerState::new(1, 1, None, None).unwrap();

        assert_eq!(
            scheduler_state.plan_dosage_queue_put_attempt(0.25),
            CallbackQueuePutAttemptPlan {
                should_put: true,
                should_wait: false,
                wait_timeout_seconds: 0.0,
                queue_depth: 1,
                queue_capacity: 1,
            },
        );
        assert_eq!(
            scheduler_state.plan_dosage_queue_put_attempt(0.25),
            CallbackQueuePutAttemptPlan {
                should_put: false,
                should_wait: true,
                wait_timeout_seconds: 0.25,
                queue_depth: 1,
                queue_capacity: 1,
            },
        );
        assert_eq!(
            scheduler_state.plan_dosage_queue_get_attempt(true),
            CallbackQueueGetAttemptPlan {
                should_get: true,
                should_wait: false,
                has_release_error: false,
                wait_timeout_seconds: 0.0,
                queue_depth: 0,
                queue_capacity: 1,
            },
        );
        assert_eq!(
            scheduler_state.plan_dosage_queue_get_attempt(false),
            CallbackQueueGetAttemptPlan {
                should_get: false,
                should_wait: true,
                has_release_error: false,
                wait_timeout_seconds: 0.1,
                queue_depth: 0,
                queue_capacity: 1,
            },
        );
        assert_eq!(
            scheduler_state.plan_result_queue_get_attempt(true),
            CallbackQueueGetAttemptPlan {
                should_get: false,
                should_wait: false,
                has_release_error: true,
                wait_timeout_seconds: 0.0,
                queue_depth: 0,
                queue_capacity: 1,
            },
        );
        assert_eq!(
            scheduler_state.plan_result_queue_put_attempt(f64::NAN),
            CallbackQueuePutAttemptPlan {
                should_put: true,
                should_wait: false,
                wait_timeout_seconds: 0.0,
                queue_depth: 1,
                queue_capacity: 1,
            },
        );
        assert_eq!(
            scheduler_state.plan_result_queue_put_attempt(f64::NAN),
            CallbackQueuePutAttemptPlan {
                should_put: false,
                should_wait: false,
                wait_timeout_seconds: 0.0,
                queue_depth: 1,
                queue_capacity: 1,
            },
        );
    }

    #[test]
    fn plans_callback_scheduler_queue_put_observations() {
        let scheduler_state = CallbackSchedulerState::new(1, 1, None, None).unwrap();

        assert_eq!(
            scheduler_state.plan_dosage_queue_put_observation(true),
            CallbackQueuePutObservationPlan {
                queue_name: DOSAGE_QUEUE_NAME.to_string(),
                operation_name: QUEUE_PUT_OPERATION.to_string(),
                blocked: false,
                should_retry_put: false,
            },
        );
        assert_eq!(
            scheduler_state.plan_dosage_queue_put_observation(false),
            CallbackQueuePutObservationPlan {
                queue_name: DOSAGE_QUEUE_NAME.to_string(),
                operation_name: QUEUE_PRODUCER_BLOCKING_OPERATION.to_string(),
                blocked: true,
                should_retry_put: true,
            },
        );
        assert_eq!(
            scheduler_state.plan_result_queue_put_observation(true),
            CallbackQueuePutObservationPlan {
                queue_name: RESULT_QUEUE_NAME.to_string(),
                operation_name: QUEUE_PUT_OPERATION.to_string(),
                blocked: false,
                should_retry_put: false,
            },
        );
        assert_eq!(
            scheduler_state.plan_result_queue_put_observation(false),
            CallbackQueuePutObservationPlan {
                queue_name: RESULT_QUEUE_NAME.to_string(),
                operation_name: QUEUE_PRODUCER_BLOCKING_OPERATION.to_string(),
                blocked: true,
                should_retry_put: true,
            },
        );
    }

    #[test]
    fn plans_callback_scheduler_queue_get_observations() {
        let scheduler_state = CallbackSchedulerState::new(1, 1, None, None).unwrap();

        assert_eq!(
            scheduler_state.plan_dosage_queue_get_observation(),
            CallbackQueueGetObservationPlan {
                queue_name: DOSAGE_QUEUE_NAME.to_string(),
                operation_name: QUEUE_CONSUMER_WAIT_OPERATION.to_string(),
                blocked: true,
            },
        );
        assert_eq!(
            scheduler_state.plan_result_queue_get_observation(),
            CallbackQueueGetObservationPlan {
                queue_name: RESULT_QUEUE_NAME.to_string(),
                operation_name: QUEUE_CONSUMER_WAIT_OPERATION.to_string(),
                blocked: true,
            },
        );
    }

    #[test]
    fn plans_callback_scheduler_queue_backpressure_attempts() {
        let mut dosage_backpressure_scheduler_state = CallbackSchedulerState::new(1, 1, None, None).unwrap();
        assert_eq!(
            dosage_backpressure_scheduler_state.plan_dosage_queue_put_backpressure_attempt(),
            CallbackQueuePutAttemptPlan {
                should_put: true,
                should_wait: false,
                wait_timeout_seconds: 0.0,
                queue_depth: 1,
                queue_capacity: 1,
            },
        );
        assert_eq!(
            dosage_backpressure_scheduler_state.plan_dosage_queue_put_backpressure_attempt(),
            CallbackQueuePutAttemptPlan {
                should_put: false,
                should_wait: true,
                wait_timeout_seconds: 0.1,
                queue_depth: 1,
                queue_capacity: 1,
            },
        );

        let mut result_backpressure_scheduler_state = CallbackSchedulerState::new(1, 1, None, None).unwrap();
        assert_eq!(
            result_backpressure_scheduler_state.plan_result_queue_put_backpressure_attempt(),
            CallbackQueuePutAttemptPlan {
                should_put: true,
                should_wait: false,
                wait_timeout_seconds: 0.0,
                queue_depth: 1,
                queue_capacity: 1,
            },
        );
        assert_eq!(
            result_backpressure_scheduler_state.plan_result_queue_put_backpressure_attempt(),
            CallbackQueuePutAttemptPlan {
                should_put: false,
                should_wait: true,
                wait_timeout_seconds: 0.1,
                queue_depth: 1,
                queue_capacity: 1,
            },
        );
    }

    #[test]
    fn resolves_callback_worker_shutdown_timeouts() {
        assert_eq!(
            callback_worker_shutdown_timeouts(),
            CallbackWorkerShutdownTimeouts {
                dosage_worker_join_timeout_seconds: 60.0,
                result_worker_join_timeout_seconds: 60.0,
                graceful_dosage_worker_join_timeout_seconds: 300.0,
                graceful_result_worker_join_timeout_seconds: 300.0,
                worker_abort_stop_timeout_seconds: 1.0,
            },
        );
    }

    #[test]
    fn resolves_callback_worker_backpressure_poll_timeout_seconds() {
        assert!((callback_worker_backpressure_poll_timeout_seconds() - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn resolves_callback_worker_stop_poll_timeout_seconds() {
        assert!((resolve_callback_worker_stop_poll_timeout_seconds(1.0) - 0.1).abs() < f64::EPSILON);
        assert!((resolve_callback_worker_stop_poll_timeout_seconds(0.05) - 0.05).abs() < f64::EPSILON);
        assert!(resolve_callback_worker_stop_poll_timeout_seconds(0.0).abs() < f64::EPSILON);
        assert!(resolve_callback_worker_stop_poll_timeout_seconds(-1.0).abs() < f64::EPSILON);
        assert!(resolve_callback_worker_stop_poll_timeout_seconds(f64::NAN).abs() < f64::EPSILON);
    }

    #[test]
    fn resolves_callback_worker_stop_attempt_decision() {
        assert!(should_attempt_callback_worker_stop(true, false, true));
        assert!(!should_attempt_callback_worker_stop(false, false, true));
        assert!(!should_attempt_callback_worker_stop(true, true, true));
        assert!(!should_attempt_callback_worker_stop(true, false, false));
    }

    #[test]
    fn plans_callback_worker_join_policy() {
        assert_eq!(
            plan_dosage_callback_worker_join(None, true),
            CallbackWorkerJoinPlan { should_join: true, timeout_seconds: 60.0 },
        );
        assert_eq!(
            plan_result_callback_worker_join(Some(0.25), true),
            CallbackWorkerJoinPlan { should_join: true, timeout_seconds: 0.25 },
        );
        assert_eq!(
            plan_result_callback_worker_join(None, false),
            CallbackWorkerJoinPlan { should_join: false, timeout_seconds: 60.0 },
        );
    }

    #[test]
    fn plans_callback_worker_stop_policy() {
        assert_eq!(
            plan_dosage_callback_worker_stop(None, true, false, true),
            CallbackWorkerStopPlan { should_stop: true, timeout_seconds: 60.0 },
        );
        assert_eq!(
            plan_result_callback_worker_stop(Some(0.25), true, false, true),
            CallbackWorkerStopPlan { should_stop: true, timeout_seconds: 0.25 },
        );
        assert_eq!(
            plan_result_callback_worker_stop(None, true, true, true),
            CallbackWorkerStopPlan { should_stop: false, timeout_seconds: 60.0 },
        );
    }

    #[test]
    fn plans_callback_worker_finish_and_abort_policy() {
        let finish_plan = plan_callback_worker_finish();
        assert!(finish_plan.stop_dosage_worker());
        assert!(finish_plan.join_dosage_worker());
        assert!(finish_plan.stop_result_worker());
        assert!(finish_plan.join_result_worker());
        assert!(finish_plan.raise_worker_error());
        assert!(finish_plan.complete_progress());
        assert!(finish_plan.emit_binary_correction_summary());
        assert_eq!(finish_plan, expected_callback_worker_finish_plan());
        let abort_plan = plan_callback_worker_abort();
        assert!(abort_plan.stop_dosage_worker());
        assert!(abort_plan.stop_result_worker());
        assert_eq!(abort_plan, expected_callback_worker_abort_plan());
    }

    #[test]
    fn plans_callback_worker_stop_poll_policy() {
        assert_eq!(
            plan_callback_worker_stop_poll(1.0, true, false, true),
            CallbackWorkerStopPollPlan { should_stop: true, poll_timeout_seconds: 0.1 },
        );
        assert_eq!(
            plan_callback_worker_stop_poll(0.05, true, true, true),
            CallbackWorkerStopPollPlan { should_stop: false, poll_timeout_seconds: 0.05 },
        );
        assert_eq!(
            plan_callback_worker_stop_poll(-1.0, true, false, true),
            CallbackWorkerStopPollPlan { should_stop: true, poll_timeout_seconds: 0.0 },
        );
    }

    #[test]
    fn formats_callback_worker_failure_messages() {
        assert_eq!(
            format_dosage_callback_worker_error_message("dosage failed"),
            "native pipeline callback worker failed: dosage failed",
        );
        assert_eq!(
            format_result_callback_worker_error_message("writer failed"),
            "native pipeline result writer worker failed: writer failed",
        );
    }

    #[test]
    fn resolves_native_callback_queue_limits() {
        assert_eq!(
            resolve_native_callback_queue_limits(3, 1, None, None).unwrap(),
            NativeCallbackQueueLimits {
                dosage_queue_depth: 3,
                result_queue_depth: 3,
                result_in_flight_limit: 4,
                dosage_buffer_limit: 4,
            },
        );
        assert_eq!(
            resolve_native_callback_queue_limits(3, 2, Some(7), Some(8)).unwrap(),
            NativeCallbackQueueLimits {
                dosage_queue_depth: 3,
                result_queue_depth: 3,
                result_in_flight_limit: 7,
                dosage_buffer_limit: 8,
            },
        );
    }

    #[test]
    fn rejects_invalid_native_callback_queue_limits() {
        assert_eq!(
            resolve_native_callback_queue_limits(0, 1, None, None).unwrap_err(),
            ScheduleError::NonPositiveStagingDepth,
        );
        assert_eq!(
            resolve_native_callback_queue_limits(1, 0, None, None).unwrap_err(),
            ScheduleError::NonPositiveCallbackBatchSize,
        );
        assert_eq!(
            resolve_native_callback_queue_limits(1, 1, Some(0), None).unwrap_err(),
            ScheduleError::NonPositiveResultInFlightLimit,
        );
        assert_eq!(
            resolve_native_callback_queue_limits(1, 1, None, Some(0)).unwrap_err(),
            ScheduleError::NonPositiveDosageBufferLimit,
        );
        assert_eq!(
            resolve_native_callback_queue_limits(1, 3, None, Some(2)).unwrap_err(),
            ScheduleError::CallbackBatchSizeExceedsDosageBufferLimit { dosage_buffer_limit: 2 },
        );
    }

    #[test]
    fn resolves_writer_finish_thread_count() {
        assert_eq!(resolve_writer_finish_thread_count(0, 0).unwrap(), 0);
        assert_eq!(resolve_writer_finish_thread_count(-1, 0).unwrap(), 0);
        assert_eq!(resolve_writer_finish_thread_count(3, 1).unwrap(), 1);
        assert_eq!(resolve_writer_finish_thread_count(3, 2).unwrap(), 2);
        assert_eq!(resolve_writer_finish_thread_count(3, 5).unwrap(), 3);
    }

    #[test]
    fn plans_writer_finish_execution() {
        assert_eq!(
            plan_writer_finish_execution(0, 0).unwrap(),
            WriterFinishExecutionPlan { writer_session_count: 0, thread_count: 0 },
        );
        assert_eq!(
            plan_writer_finish_execution(1, 1).unwrap(),
            WriterFinishExecutionPlan { writer_session_count: 1, thread_count: 1 },
        );
        let parallel_plan = plan_writer_finish_execution(3, 2).unwrap();
        assert_eq!(parallel_plan, WriterFinishExecutionPlan { writer_session_count: 3, thread_count: 2 });
        assert!(parallel_plan.has_writer_sessions());
        assert!(parallel_plan.uses_parallel_finish());
    }

    #[test]
    fn rejects_invalid_writer_finish_thread_count_when_writers_exist() {
        assert_eq!(
            resolve_writer_finish_thread_count(1, 0).unwrap_err(),
            ScheduleError::NonPositiveWriterFinishThreadCount,
        );
        assert_eq!(plan_writer_finish_execution(1, 0).unwrap_err(), ScheduleError::NonPositiveWriterFinishThreadCount,);
    }

    #[test]
    fn plans_bgen_delivery_cleanup() {
        assert_eq!(
            plan_bgen_delivery_cleanup(BGEN_DELIVERY_CLEANUP_SUCCESS, false).unwrap(),
            build_bgen_delivery_cleanup_plan(&[
                BGEN_DELIVERY_CLEANUP_ACTION_DRAIN_CALLBACK,
                BGEN_DELIVERY_CLEANUP_ACTION_FINISH_WRITER_SESSIONS,
                BGEN_DELIVERY_CLEANUP_ACTION_WRITE_STAGE_TIMING_SNAPSHOT,
            ]),
        );
        assert_eq!(
            plan_bgen_delivery_cleanup(BGEN_DELIVERY_CLEANUP_INTERRUPTED, false).unwrap(),
            build_bgen_delivery_cleanup_plan(&[
                BGEN_DELIVERY_CLEANUP_ACTION_DRAIN_CALLBACK,
                BGEN_DELIVERY_CLEANUP_ACTION_FINISH_INTERRUPTED_WRITER_SESSIONS,
                BGEN_DELIVERY_CLEANUP_ACTION_WRITE_STAGE_TIMING_SNAPSHOT,
            ]),
        );
        assert_eq!(
            plan_bgen_delivery_cleanup(BGEN_DELIVERY_CLEANUP_INTERRUPTED, true).unwrap(),
            build_bgen_delivery_cleanup_plan(&[
                BGEN_DELIVERY_CLEANUP_ACTION_FINISH_INTERRUPTED_WRITER_SESSIONS,
                BGEN_DELIVERY_CLEANUP_ACTION_WRITE_STAGE_TIMING_SNAPSHOT,
            ]),
        );
        assert_eq!(
            plan_bgen_delivery_cleanup(BGEN_DELIVERY_CLEANUP_FAILURE, false).unwrap(),
            build_bgen_delivery_cleanup_plan(&[
                BGEN_DELIVERY_CLEANUP_ACTION_ABORT_CALLBACK,
                BGEN_DELIVERY_CLEANUP_ACTION_ABORT_WRITER_SESSIONS,
                BGEN_DELIVERY_CLEANUP_ACTION_WRITE_STAGE_TIMING_SNAPSHOT,
            ]),
        );
        assert_eq!(
            plan_bgen_delivery_cleanup(BGEN_DELIVERY_CLEANUP_INTERRUPTED_CLEANUP_FAILURE, false).unwrap(),
            build_bgen_delivery_cleanup_plan(&[
                BGEN_DELIVERY_CLEANUP_ACTION_ABORT_CALLBACK,
                BGEN_DELIVERY_CLEANUP_ACTION_ABORT_WRITER_SESSIONS,
                BGEN_DELIVERY_CLEANUP_ACTION_WRITE_STAGE_TIMING_SNAPSHOT,
            ]),
        );
    }

    #[test]
    fn rejects_unknown_bgen_delivery_cleanup_outcome() {
        assert_eq!(
            plan_bgen_delivery_cleanup("unknown", false).unwrap_err(),
            ScheduleError::UnsupportedBgenDeliveryCleanupOutcome { outcome: "unknown".to_string() },
        );
    }

    #[test]
    fn plans_single_trait_output_write_method() {
        assert_eq!(
            plan_single_trait_output_write(true, "float64").unwrap(),
            SingleTraitOutputWritePlan {
                method_name: REGENIE2_NATIVE_CHUNK_WRITE_F64_METHOD.to_string(),
                uses_float64_native_writer: true,
            },
        );
        assert_eq!(
            plan_single_trait_output_write(true, "float32").unwrap(),
            SingleTraitOutputWritePlan {
                method_name: REGENIE2_NATIVE_CHUNK_WRITE_METHOD.to_string(),
                uses_float64_native_writer: false,
            },
        );
        assert_eq!(
            plan_single_trait_output_write(false, "float64").unwrap(),
            SingleTraitOutputWritePlan {
                method_name: REGENIE2_NATIVE_CHUNK_WRITE_METHOD.to_string(),
                uses_float64_native_writer: false,
            },
        );
    }

    #[test]
    fn plans_multi_trait_output_write_method() {
        assert_eq!(
            plan_multi_trait_output_write(2, true, "float64").unwrap(),
            MultiTraitOutputWritePlan {
                active_trait_count: 2,
                use_native_multi_writer: true,
                uses_float64_native_writer: true,
            },
        );
        assert_eq!(
            plan_multi_trait_output_write(2, false, "float64").unwrap(),
            MultiTraitOutputWritePlan {
                active_trait_count: 2,
                use_native_multi_writer: false,
                uses_float64_native_writer: false,
            },
        );
        assert_eq!(
            plan_multi_trait_output_write(0, true, "float64").unwrap(),
            MultiTraitOutputWritePlan {
                active_trait_count: 0,
                use_native_multi_writer: false,
                uses_float64_native_writer: false,
            },
        );
    }

    #[test]
    fn rejects_invalid_output_statistic_dtype_for_output_write_plans() {
        assert_eq!(
            plan_single_trait_output_write(true, "float16").unwrap_err(),
            ScheduleError::UnsupportedOutputStatisticDtype { output_statistic_dtype: "float16".to_string() },
        );
        assert_eq!(
            plan_multi_trait_output_write(1, true, "float16").unwrap_err(),
            ScheduleError::UnsupportedOutputStatisticDtype { output_statistic_dtype: "float16".to_string() },
        );
    }

    #[test]
    fn plans_callback_queue_stage_observations() {
        assert_eq!(
            plan_callback_queue_stage_observation("dosage_queue", "put", 0.25, false).unwrap(),
            CallbackQueueStageObservationPlan {
                queue_name: "dosage_queue".to_string(),
                operation_name: "put".to_string(),
                stage_name: "callback_queue_put".to_string(),
                blocked_seconds: 0.0,
            },
        );
        assert_eq!(
            plan_callback_queue_stage_observation("result_in_flight_slots", "producer_blocking", 0.5, true).unwrap(),
            CallbackQueueStageObservationPlan {
                queue_name: "result_in_flight_slots".to_string(),
                operation_name: "producer_blocking".to_string(),
                stage_name: "result_in_flight_producer_blocking".to_string(),
                blocked_seconds: 0.5,
            },
        );
        assert_eq!(
            plan_callback_queue_stage_backpressure_observation("dosage_queue", "put", 2, 3, 0.25, false).unwrap(),
            CallbackQueueStageBackpressureObservation {
                queue_name: "dosage_queue".to_string(),
                operation_name: "put".to_string(),
                stage_name: "callback_queue_put".to_string(),
                queue_depth: 2,
                queue_capacity: 3,
                elapsed_seconds: 0.25,
                blocked_seconds: 0.0,
            },
        );
    }

    #[test]
    fn plans_callback_queue_operation_observations() {
        assert_eq!(
            plan_callback_queue_operation_observation("dosage_buffer_pool", "reuse", 0.25, false).unwrap(),
            CallbackQueueOperationObservationPlan {
                queue_name: "dosage_buffer_pool".to_string(),
                operation_name: "reuse".to_string(),
                blocked_seconds: 0.0,
            },
        );
        assert_eq!(
            plan_callback_queue_operation_observation("result_in_flight_slots", "release", 0.5, true).unwrap(),
            CallbackQueueOperationObservationPlan {
                queue_name: "result_in_flight_slots".to_string(),
                operation_name: "release".to_string(),
                blocked_seconds: 0.5,
            },
        );
        assert_eq!(
            plan_callback_queue_backpressure_observation("dosage_buffer_pool", "reuse", 1, 2, 0.25, false).unwrap(),
            CallbackQueueBackpressureObservation {
                queue_name: "dosage_buffer_pool".to_string(),
                operation_name: "reuse".to_string(),
                queue_depth: 1,
                queue_capacity: 2,
                elapsed_seconds: 0.25,
                blocked_seconds: 0.0,
            },
        );
    }

    #[test]
    fn rejects_unknown_callback_queue_stage_observations() {
        assert_eq!(
            plan_callback_queue_stage_observation("unknown_queue", "put", 0.25, false).unwrap_err(),
            ScheduleError::UnsupportedCallbackQueueStageOperation {
                queue_name: "unknown_queue".to_string(),
                operation_name: "put".to_string(),
            },
        );
    }

    #[test]
    fn rejects_unknown_callback_queue_operation_observations() {
        assert_eq!(
            plan_callback_queue_operation_observation("dosage_buffer_pool", "unknown_operation", 0.25, false)
                .unwrap_err(),
            ScheduleError::UnsupportedCallbackQueueOperation {
                queue_name: "dosage_buffer_pool".to_string(),
                operation_name: "unknown_operation".to_string(),
            },
        );
    }

    #[test]
    fn resolves_bgen_delivery_method_with_native_alignment_precedence() {
        assert_eq!(
            resolve_bgen_delivery_method(false, true, true),
            BgenDeliveryMethod::DosageNativeMultiAlignedSamples,
        );
        assert_eq!(resolve_bgen_delivery_method(false, false, true), BgenDeliveryMethod::DosageNativeAlignedSamples,);
        assert_eq!(resolve_bgen_delivery_method(false, false, false), BgenDeliveryMethod::DosageSampleIndices);
        assert_eq!(
            resolve_bgen_delivery_method(true, true, true),
            BgenDeliveryMethod::Packed8NativeMultiAlignedSamples,
        );
        assert_eq!(resolve_bgen_delivery_method(true, false, true), BgenDeliveryMethod::Packed8NativeAlignedSamples,);
        assert_eq!(resolve_bgen_delivery_method(true, false, false), BgenDeliveryMethod::Packed8SampleIndices,);
    }
}
