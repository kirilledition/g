use super::{
    CallbackQueueBackpressureObservation, CallbackQueueGetAttemptPlan, CallbackQueueGetObservationPlan,
    CallbackQueueOccupancyState, CallbackQueueOperationObservationPlan, CallbackQueuePutAttemptPlan,
    CallbackQueuePutObservationPlan, CallbackQueueStageBackpressureObservation, CallbackQueueStageObservationPlan,
    CallbackWorkerAbortPlan, CallbackWorkerErrorRaisePlan, CallbackWorkerErrorUpdatePlan, CallbackWorkerFinishPlan,
    CallbackWorkerJoinPlan, CallbackWorkerLifecycleState, CallbackWorkerStartAttemptPlan, CallbackWorkerStartPlan,
    CallbackWorkerStopPlan, CallbackWorkerStopPollPlan, DOSAGE_BUFFER_POOL_NAME, DOSAGE_QUEUE_NAME,
    DosageBufferAcquireAttemptPlan, DosageBufferDiscardAttemptPlan, DosageBufferPoolObservationPlan,
    DosageBufferPoolState, DosageBufferRegisterAttemptPlan, DosageBufferReturnAttemptPlan, DosageBufferReusePlan,
    DosageWorkDrainCompletionPlan, DosageWorkHandoffPlan, DosageWorkItemDispatchPlan, DosageWorkItemStageDurationPlan,
    QUEUE_ALLOCATE_OPERATION, QUEUE_CONSUMER_WAIT_OPERATION, QUEUE_DISCARD_OPERATION, QUEUE_RETURN_OPERATION,
    QUEUE_REUSE_OPERATION, RESULT_IN_FLIGHT_SLOTS_NAME, RESULT_QUEUE_NAME, ResultInFlightAcquireAttemptPlan,
    ResultInFlightAcquireObservationPlan, ResultInFlightReleaseAttemptPlan, ResultInFlightReleaseObservationPlan,
    ResultInFlightSlotState, ResultWriteDrainCompletionPlan, ResultWriteHandoffPlan, ResultWriteItemDispatchPlan,
    ResultWriteItemKind, ResultWriteItemResourceReleasePlan, ScheduleError, VariantMajorDosageBatchHandoffPlan,
    callback_worker_backpressure_poll_timeout_seconds, format_dosage_callback_worker_error_message,
    format_result_callback_worker_error_message, plan_callback_queue_backpressure_observation,
    plan_callback_queue_get_attempt, plan_callback_queue_get_observation, plan_callback_queue_operation_observation,
    plan_callback_queue_put_attempt, plan_callback_queue_put_observation,
    plan_callback_queue_stage_backpressure_observation, plan_callback_queue_stage_observation,
    plan_callback_worker_abort, plan_callback_worker_error_raise, plan_callback_worker_finish,
    plan_callback_worker_start, plan_callback_worker_start_attempt, plan_callback_worker_stop_poll,
    plan_dosage_buffer_acquire_attempt, plan_dosage_buffer_discard_attempt, plan_dosage_buffer_pool_observation,
    plan_dosage_buffer_register_attempt, plan_dosage_buffer_return_attempt, plan_dosage_buffer_reuse,
    plan_dosage_callback_worker_join, plan_dosage_callback_worker_stop, plan_dosage_work_handoff,
    plan_dosage_work_item_dispatch, plan_dosage_work_item_stage_duration, plan_result_callback_worker_join,
    plan_result_callback_worker_stop, plan_result_in_flight_slot_acquire_attempt,
    plan_result_in_flight_slot_acquire_observation, plan_result_in_flight_slot_release_attempt,
    plan_result_in_flight_slot_release_observation, plan_result_write_handoff, plan_result_write_item_dispatch,
    plan_result_write_item_dispatch_for_kinds, plan_variant_major_dosage_batch_handoff, update_callback_worker_error,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeCallbackQueueLimits {
    pub dosage_queue_depth: usize,
    pub result_queue_depth: usize,
    pub result_in_flight_limit: usize,
    pub dosage_buffer_limit: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CallbackBoundedResourceOccupancy {
    queue_depth: usize,
    queue_capacity: usize,
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
    pub fn plan_result_write_item_dispatch_for_kinds(
        &self,
        result_work_item_kind: ResultWriteItemKind,
        expected_result_work_item_kind: ResultWriteItemKind,
    ) -> ResultWriteItemDispatchPlan {
        plan_result_write_item_dispatch_for_kinds(result_work_item_kind, expected_result_work_item_kind)
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
