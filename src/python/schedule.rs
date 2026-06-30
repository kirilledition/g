//! PyO3 adapters for engine scheduling policy helpers.

use std::collections::BTreeSet;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use g_engine::schedule as native_schedule;

#[pyclass]
pub(crate) struct NativeCallbackQueueLimits {
    #[pyo3(get)]
    dosage_queue_depth: usize,
    #[pyo3(get)]
    result_queue_depth: usize,
    #[pyo3(get)]
    result_in_flight_limit: usize,
    #[pyo3(get)]
    dosage_buffer_limit: usize,
}

#[pyclass]
pub(crate) struct NativeCallbackSchedulerState {
    inner: native_schedule::CallbackSchedulerState,
}

#[pyclass]
pub(crate) struct NativeDosageBufferReusePlan {
    #[pyo3(get)]
    requires_slice: bool,
    #[pyo3(get)]
    slice_dimensions: Vec<usize>,
}

impl NativeDosageBufferReusePlan {
    pub(crate) fn requires_slice_value(&self) -> bool {
        self.requires_slice
    }

    pub(crate) fn slice_dimensions_value(&self) -> &[usize] {
        &self.slice_dimensions
    }
}

#[pyclass]
pub(crate) struct NativeVariantMajorDosageBatchHandoffPlan {
    #[pyo3(get)]
    chunk_count: usize,
}

#[pyclass]
pub(crate) struct NativeDosageWorkHandoffPlan {
    #[pyo3(get)]
    chunk_count: usize,
}

#[pyclass]
pub(crate) struct NativeGpuGenotypeFormatResolutionPlan {
    inner: native_schedule::GpuGenotypeFormatResolutionPlan,
}

#[pyclass]
pub(crate) struct NativeMultiTraitChunkWritePlan {
    inner: native_schedule::MultiTraitChunkWritePlan,
}

#[pyclass]
pub(crate) struct NativeWriterFinishExecutionPlan {
    inner: native_schedule::WriterFinishExecutionPlan,
}

#[pyclass]
pub(crate) struct NativeBgenDeliveryCleanupPlan {
    inner: native_schedule::BgenDeliveryCleanupPlan,
}

#[pyclass]
pub(crate) struct NativeBgenDeliveryInvocationPlan {
    inner: native_schedule::BgenDeliveryInvocationPlan,
}

#[pyclass]
pub(crate) struct NativeSingleTraitOutputWritePlan {
    inner: native_schedule::SingleTraitOutputWritePlan,
}

#[pyclass]
pub(crate) struct NativeMultiTraitOutputWritePlan {
    inner: native_schedule::MultiTraitOutputWritePlan,
}

#[pyclass]
pub(crate) struct NativeCallbackQueueOperationObservationPlan {
    inner: native_schedule::CallbackQueueOperationObservationPlan,
}

#[pyclass]
pub(crate) struct NativeCallbackQueueStageObservationPlan {
    inner: native_schedule::CallbackQueueStageObservationPlan,
}

#[pyclass]
pub(crate) struct NativeCallbackQueueBackpressureObservation {
    inner: native_schedule::CallbackQueueBackpressureObservation,
}

#[pyclass]
pub(crate) struct NativeCallbackQueueStageBackpressureObservation {
    inner: native_schedule::CallbackQueueStageBackpressureObservation,
}

#[pyclass]
pub(crate) struct NativeCallbackQueuePutAttemptPlan {
    inner: native_schedule::CallbackQueuePutAttemptPlan,
}

#[pyclass]
pub(crate) struct NativeCallbackQueuePutObservationPlan {
    inner: native_schedule::CallbackQueuePutObservationPlan,
}

#[pyclass]
pub(crate) struct NativeCallbackQueueGetAttemptPlan {
    inner: native_schedule::CallbackQueueGetAttemptPlan,
}

#[pyclass]
pub(crate) struct NativeCallbackQueueGetObservationPlan {
    inner: native_schedule::CallbackQueueGetObservationPlan,
}

#[pyclass]
pub(crate) struct NativeDosageBufferAcquireAttemptPlan {
    inner: native_schedule::DosageBufferAcquireAttemptPlan,
}

#[pyclass]
pub(crate) struct NativeDosageBufferRegisterAttemptPlan {
    inner: native_schedule::DosageBufferRegisterAttemptPlan,
}

#[pyclass]
pub(crate) struct NativeDosageBufferReturnAttemptPlan {
    inner: native_schedule::DosageBufferReturnAttemptPlan,
}

#[pyclass]
pub(crate) struct NativeDosageBufferDiscardAttemptPlan {
    inner: native_schedule::DosageBufferDiscardAttemptPlan,
}

#[pyclass]
pub(crate) struct NativeDosageBufferPoolObservationPlan {
    inner: native_schedule::DosageBufferPoolObservationPlan,
}

#[pyclass]
pub(crate) struct NativeDosageBufferPoolState {
    inner: native_schedule::DosageBufferPoolState,
}

#[pyclass]
pub(crate) struct NativeResultInFlightSlotState {
    inner: native_schedule::ResultInFlightSlotState,
}

#[pyclass]
pub(crate) struct NativeResultInFlightAcquireAttemptPlan {
    inner: native_schedule::ResultInFlightAcquireAttemptPlan,
}

#[pyclass]
pub(crate) struct NativeResultInFlightAcquireObservationPlan {
    inner: native_schedule::ResultInFlightAcquireObservationPlan,
}

#[pyclass]
pub(crate) struct NativeResultInFlightReleaseAttemptPlan {
    inner: native_schedule::ResultInFlightReleaseAttemptPlan,
}

#[pyclass]
pub(crate) struct NativeResultInFlightReleaseObservationPlan {
    inner: native_schedule::ResultInFlightReleaseObservationPlan,
}

#[pyclass]
pub(crate) struct NativeResultWriteItemResourceReleasePlan {
    inner: native_schedule::ResultWriteItemResourceReleasePlan,
}

#[pyclass]
pub(crate) struct NativeResultWriteHandoffPlan {
    inner: native_schedule::ResultWriteHandoffPlan,
}

#[pyclass]
pub(crate) struct NativeResultWriteDrainCompletionPlan {
    inner: native_schedule::ResultWriteDrainCompletionPlan,
}

#[pyclass]
pub(crate) struct NativeResultWriteItemDispatchPlan {
    inner: native_schedule::ResultWriteItemDispatchPlan,
}

#[pyclass]
pub(crate) struct NativeDosageWorkDrainCompletionPlan {
    inner: native_schedule::DosageWorkDrainCompletionPlan,
}

#[pyclass]
pub(crate) struct NativeDosageWorkItemDispatchPlan {
    inner: native_schedule::DosageWorkItemDispatchPlan,
}

#[pyclass]
pub(crate) struct NativeDosageWorkItemStageDurationPlan {
    inner: native_schedule::DosageWorkItemStageDurationPlan,
}

#[pyclass]
pub(crate) struct NativeCallbackWorkerLifecycleState {
    inner: native_schedule::CallbackWorkerLifecycleState,
}

#[pyclass]
pub(crate) struct NativeCallbackWorkerStartPlan {
    inner: native_schedule::CallbackWorkerStartPlan,
}

#[pyclass]
pub(crate) struct NativeCallbackWorkerStartAttemptPlan {
    inner: native_schedule::CallbackWorkerStartAttemptPlan,
}

#[pyclass]
pub(crate) struct NativeCallbackWorkerShutdownTimeouts {
    inner: native_schedule::CallbackWorkerShutdownTimeouts,
}

#[pyclass]
pub(crate) struct NativeCallbackWorkerJoinPlan {
    inner: native_schedule::CallbackWorkerJoinPlan,
}

#[pyclass]
pub(crate) struct NativeCallbackWorkerStopPlan {
    inner: native_schedule::CallbackWorkerStopPlan,
}

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct NativeCallbackWorkerFinishPlan {
    inner: native_schedule::CallbackWorkerFinishPlan,
}

#[pyclass]
pub(crate) struct NativeCallbackWorkerAbortPlan {
    inner: native_schedule::CallbackWorkerAbortPlan,
}

#[pyclass]
pub(crate) struct NativeCallbackWorkerStopPollPlan {
    inner: native_schedule::CallbackWorkerStopPollPlan,
}

#[pyclass]
pub(crate) struct NativeCallbackWorkerErrorRaisePlan {
    inner: native_schedule::CallbackWorkerErrorRaisePlan,
}

#[pyclass]
pub(crate) struct NativeCallbackWorkerErrorUpdatePlan {
    inner: native_schedule::CallbackWorkerErrorUpdatePlan,
}

#[pymethods]
impl NativeDosageBufferPoolState {
    #[new]
    fn new(buffer_limit: usize) -> Self {
        Self { inner: native_schedule::DosageBufferPoolState::new(buffer_limit) }
    }

    #[getter]
    fn buffer_limit(&self) -> usize {
        self.inner.buffer_limit()
    }

    #[getter]
    fn allocated_count(&self) -> usize {
        self.inner.allocated_count()
    }

    #[getter]
    fn buffer_identifiers(&self) -> Vec<usize> {
        self.inner.buffer_identifiers()
    }

    fn has_available_slot(&self) -> bool {
        self.inner.has_available_slot()
    }

    fn owns_buffer(&self, buffer_identifier: usize) -> bool {
        self.inner.owns_buffer(buffer_identifier)
    }

    fn register_buffer(&mut self, buffer_identifier: usize) -> bool {
        self.inner.register_buffer(buffer_identifier)
    }

    fn discard_buffer(&mut self, buffer_identifier: usize) -> bool {
        self.inner.discard_buffer(buffer_identifier)
    }
}

#[pymethods]
impl NativeResultInFlightSlotState {
    #[new]
    fn new(slot_limit: usize) -> Self {
        Self { inner: native_schedule::ResultInFlightSlotState::new(slot_limit) }
    }

    #[getter]
    fn slot_limit(&self) -> usize {
        self.inner.slot_limit()
    }

    #[getter]
    fn occupied_count(&self) -> usize {
        self.inner.occupied_count()
    }

    fn has_available_slot(&self) -> bool {
        self.inner.has_available_slot()
    }

    fn acquire_slot(&mut self) -> bool {
        self.inner.acquire_slot()
    }

    fn release_slot(&mut self) -> bool {
        self.inner.release_slot()
    }
}

#[pymethods]
impl NativeCallbackWorkerLifecycleState {
    #[new]
    fn new() -> Self {
        Self { inner: native_schedule::CallbackWorkerLifecycleState::new() }
    }

    #[getter]
    fn has_started(&self) -> bool {
        self.inner.has_started()
    }

    fn mark_started(&mut self) -> bool {
        self.inner.mark_started()
    }
}

#[pymethods]
impl NativeCallbackWorkerStartPlan {
    #[getter]
    fn start_actions(&self) -> Vec<String> {
        self.inner.start_actions.clone()
    }

    #[getter]
    fn should_start(&self) -> bool {
        self.inner.should_start()
    }

    #[getter]
    fn start_result_worker(&self) -> bool {
        self.inner.start_result_worker()
    }

    #[getter]
    fn start_dosage_worker(&self) -> bool {
        self.inner.start_dosage_worker()
    }
}

#[pymethods]
impl NativeCallbackWorkerStartAttemptPlan {
    #[getter]
    fn start_actions(&self) -> Vec<String> {
        self.inner.start_actions.clone()
    }

    #[getter]
    fn should_start(&self) -> bool {
        self.inner.should_start()
    }

    #[getter]
    fn start_result_worker(&self) -> bool {
        self.inner.start_result_worker()
    }

    #[getter]
    fn start_dosage_worker(&self) -> bool {
        self.inner.start_dosage_worker()
    }

    #[getter]
    fn has_marked_started(&self) -> bool {
        self.inner.has_marked_started
    }

    #[getter]
    fn has_start_error(&self) -> bool {
        self.inner.has_start_error
    }

    #[getter]
    fn error_message(&self) -> Option<String> {
        self.inner.error_message.clone()
    }
}

#[pymethods]
impl NativeCallbackSchedulerState {
    #[new]
    fn new(
        staging_depth: i64,
        native_callback_batch_size: i64,
        result_in_flight_limit: Option<i64>,
        dosage_buffer_limit: Option<i64>,
    ) -> PyResult<Self> {
        Self::from_limits(staging_depth, native_callback_batch_size, result_in_flight_limit, dosage_buffer_limit)
    }

    #[getter]
    fn native_callback_batch_size(&self) -> usize {
        self.inner.native_callback_batch_size()
    }

    #[getter]
    fn dosage_queue_depth(&self) -> usize {
        self.inner.dosage_queue_depth()
    }

    #[getter]
    fn dosage_queue_capacity(&self) -> usize {
        self.inner.dosage_queue_capacity()
    }

    #[getter]
    fn dosage_queue_occupied_count(&self) -> usize {
        self.inner.dosage_queue_occupied_count()
    }

    fn has_available_dosage_queue_slot(&self) -> bool {
        self.inner.has_available_dosage_queue_slot()
    }

    fn acquire_dosage_queue_slot(&mut self) -> bool {
        self.inner.acquire_dosage_queue_slot()
    }

    fn release_dosage_queue_slot(&mut self) -> bool {
        self.inner.release_dosage_queue_slot()
    }

    fn plan_dosage_queue_put_attempt(&mut self, wait_timeout_seconds: f64) -> NativeCallbackQueuePutAttemptPlan {
        self.inner.plan_dosage_queue_put_attempt(wait_timeout_seconds).into()
    }

    fn plan_dosage_queue_put_backpressure_attempt(&mut self) -> NativeCallbackQueuePutAttemptPlan {
        self.inner.plan_dosage_queue_put_backpressure_attempt().into()
    }

    fn plan_dosage_queue_put_observation(&self, queued: bool) -> NativeCallbackQueuePutObservationPlan {
        self.inner.plan_dosage_queue_put_observation(queued).into()
    }

    fn plan_dosage_queue_get_attempt(&mut self, has_queued_item: bool) -> NativeCallbackQueueGetAttemptPlan {
        self.inner.plan_dosage_queue_get_attempt(has_queued_item).into()
    }

    fn plan_dosage_queue_get_observation(&self) -> NativeCallbackQueueGetObservationPlan {
        self.inner.plan_dosage_queue_get_observation().into()
    }

    #[getter]
    fn result_queue_depth(&self) -> usize {
        self.inner.result_queue_depth()
    }

    #[getter]
    fn result_queue_capacity(&self) -> usize {
        self.inner.result_queue_capacity()
    }

    #[getter]
    fn result_queue_occupied_count(&self) -> usize {
        self.inner.result_queue_occupied_count()
    }

    fn has_available_result_queue_slot(&self) -> bool {
        self.inner.has_available_result_queue_slot()
    }

    fn acquire_result_queue_slot(&mut self) -> bool {
        self.inner.acquire_result_queue_slot()
    }

    fn release_result_queue_slot(&mut self) -> bool {
        self.inner.release_result_queue_slot()
    }

    fn plan_result_queue_put_attempt(&mut self, wait_timeout_seconds: f64) -> NativeCallbackQueuePutAttemptPlan {
        self.inner.plan_result_queue_put_attempt(wait_timeout_seconds).into()
    }

    fn plan_result_queue_put_backpressure_attempt(&mut self) -> NativeCallbackQueuePutAttemptPlan {
        self.inner.plan_result_queue_put_backpressure_attempt().into()
    }

    fn plan_result_queue_put_observation(&self, queued: bool) -> NativeCallbackQueuePutObservationPlan {
        self.inner.plan_result_queue_put_observation(queued).into()
    }

    fn plan_result_queue_get_attempt(&mut self, has_queued_item: bool) -> NativeCallbackQueueGetAttemptPlan {
        self.inner.plan_result_queue_get_attempt(has_queued_item).into()
    }

    fn plan_result_queue_get_observation(&self) -> NativeCallbackQueueGetObservationPlan {
        self.inner.plan_result_queue_get_observation().into()
    }

    #[getter]
    fn result_in_flight_limit(&self) -> usize {
        self.inner.result_in_flight_limit()
    }

    #[getter]
    fn dosage_buffer_limit(&self) -> usize {
        self.inner.dosage_buffer_limit()
    }

    #[getter]
    fn has_started(&self) -> bool {
        self.inner.has_started()
    }

    fn mark_started(&mut self) -> bool {
        self.inner.mark_started()
    }

    fn plan_worker_start(&self) -> NativeCallbackWorkerStartPlan {
        self.inner.plan_worker_start().into()
    }

    fn plan_worker_start_attempt(&mut self) -> NativeCallbackWorkerStartAttemptPlan {
        self.inner.plan_worker_start_attempt().into()
    }

    #[getter]
    fn result_in_flight_slot_limit(&self) -> usize {
        self.inner.result_in_flight_slot_limit()
    }

    #[getter]
    fn result_in_flight_occupied_count(&self) -> usize {
        self.inner.result_in_flight_occupied_count()
    }

    fn has_available_result_in_flight_slot(&self) -> bool {
        self.inner.has_available_result_in_flight_slot()
    }

    fn acquire_result_in_flight_slot(&mut self) -> bool {
        self.inner.acquire_result_in_flight_slot()
    }

    fn release_result_in_flight_slot(&mut self) -> bool {
        self.inner.release_result_in_flight_slot()
    }

    fn plan_result_in_flight_slot_acquire_attempt(
        &mut self,
        wait_timeout_seconds: f64,
    ) -> NativeResultInFlightAcquireAttemptPlan {
        self.inner.plan_result_in_flight_slot_acquire_attempt(wait_timeout_seconds).into()
    }

    fn plan_result_in_flight_slot_acquire_backpressure_attempt(&mut self) -> NativeResultInFlightAcquireAttemptPlan {
        self.inner.plan_result_in_flight_slot_acquire_backpressure_attempt().into()
    }

    #[allow(clippy::needless_pass_by_value)]
    fn plan_result_in_flight_slot_acquire_observation(
        &self,
        acquire_attempt_plan: PyRef<'_, NativeResultInFlightAcquireAttemptPlan>,
    ) -> NativeResultInFlightAcquireObservationPlan {
        self.inner.plan_result_in_flight_slot_acquire_observation(&acquire_attempt_plan.inner).into()
    }

    fn plan_result_in_flight_slot_release_attempt(&mut self) -> NativeResultInFlightReleaseAttemptPlan {
        self.inner.plan_result_in_flight_slot_release_attempt().into()
    }

    fn plan_result_in_flight_slot_release_observation(&self) -> NativeResultInFlightReleaseObservationPlan {
        self.inner.plan_result_in_flight_slot_release_observation().into()
    }

    fn plan_result_write_item_pre_write_resource_release(
        &self,
        has_host_dosage_buffer: bool,
    ) -> NativeResultWriteItemResourceReleasePlan {
        self.inner.plan_result_write_item_pre_write_resource_release(has_host_dosage_buffer).into()
    }

    #[allow(clippy::fn_params_excessive_bools)]
    fn plan_result_write_item_final_resource_release(
        &self,
        has_host_dosage_buffer: bool,
        has_released_host_dosage_buffer: bool,
        release_in_flight_slot: bool,
    ) -> NativeResultWriteItemResourceReleasePlan {
        self.inner
            .plan_result_write_item_final_resource_release(
                has_host_dosage_buffer,
                has_released_host_dosage_buffer,
                release_in_flight_slot,
            )
            .into()
    }

    fn plan_result_write_handoff(&self, has_result_work_item: bool) -> NativeResultWriteHandoffPlan {
        self.inner.plan_result_write_handoff(has_result_work_item).into()
    }

    fn plan_result_write_drain_completion(
        &self,
        has_result_work_item: bool,
        flush_binary_correction_diagnostics_on_stop: bool,
    ) -> NativeResultWriteDrainCompletionPlan {
        self.inner
            .plan_result_write_drain_completion(has_result_work_item, flush_binary_correction_diagnostics_on_stop)
            .into()
    }

    fn plan_result_write_item_dispatch(
        &self,
        result_work_item_kind: &str,
        expected_result_work_item_kind: &str,
    ) -> PyResult<NativeResultWriteItemDispatchPlan> {
        self.inner
            .plan_result_write_item_dispatch(result_work_item_kind, expected_result_work_item_kind)
            .map(Into::into)
            .map_err(|error| schedule_error_to_py(&error))
    }

    fn plan_dosage_work_drain_completion(&self, has_dosage_work_item: bool) -> NativeDosageWorkDrainCompletionPlan {
        self.inner.plan_dosage_work_drain_completion(has_dosage_work_item).into()
    }

    fn plan_dosage_work_item_dispatch(
        &self,
        dosage_work_item_kind: &str,
    ) -> PyResult<NativeDosageWorkItemDispatchPlan> {
        self.inner
            .plan_dosage_work_item_dispatch(dosage_work_item_kind)
            .map(Into::into)
            .map_err(|error| schedule_error_to_py(&error))
    }

    fn plan_dosage_work_item_stage_duration(
        &self,
        dosage_work_item_kind: &str,
        chunk_count: usize,
        elapsed_seconds: f64,
    ) -> PyResult<NativeDosageWorkItemStageDurationPlan> {
        self.inner
            .plan_dosage_work_item_stage_duration(dosage_work_item_kind, chunk_count, elapsed_seconds)
            .map(Into::into)
            .map_err(|error| schedule_error_to_py(&error))
    }

    #[getter]
    fn dosage_buffer_pool_limit(&self) -> usize {
        self.inner.dosage_buffer_pool_limit()
    }

    #[getter]
    fn dosage_buffer_allocated_count(&self) -> usize {
        self.inner.dosage_buffer_allocated_count()
    }

    #[getter]
    fn dosage_buffer_identifiers(&self) -> Vec<usize> {
        self.inner.dosage_buffer_identifiers()
    }

    fn has_available_dosage_buffer_slot(&self) -> bool {
        self.inner.has_available_dosage_buffer_slot()
    }

    fn owns_dosage_buffer(&self, buffer_identifier: usize) -> bool {
        self.inner.owns_dosage_buffer(buffer_identifier)
    }

    fn register_dosage_buffer(&mut self, buffer_identifier: usize) -> bool {
        self.inner.register_dosage_buffer(buffer_identifier)
    }

    fn discard_dosage_buffer(&mut self, buffer_identifier: usize) -> bool {
        self.inner.discard_dosage_buffer(buffer_identifier)
    }

    fn plan_dosage_buffer_acquire_attempt(
        &self,
        free_buffer_count: usize,
        wait_timeout_seconds: f64,
    ) -> NativeDosageBufferAcquireAttemptPlan {
        self.inner.plan_dosage_buffer_acquire_attempt(free_buffer_count, wait_timeout_seconds).into()
    }

    fn plan_dosage_buffer_acquire_backpressure_attempt(
        &self,
        free_buffer_count: usize,
    ) -> NativeDosageBufferAcquireAttemptPlan {
        self.inner.plan_dosage_buffer_acquire_backpressure_attempt(free_buffer_count).into()
    }

    fn plan_dosage_buffer_register_attempt(
        &mut self,
        buffer_identifier: usize,
    ) -> NativeDosageBufferRegisterAttemptPlan {
        self.inner.plan_dosage_buffer_register_attempt(buffer_identifier).into()
    }

    fn plan_dosage_buffer_return_attempt(&self, buffer_identifier: usize) -> NativeDosageBufferReturnAttemptPlan {
        self.inner.plan_dosage_buffer_return_attempt(buffer_identifier).into()
    }

    fn plan_dosage_buffer_discard_attempt(&mut self, buffer_identifier: usize) -> NativeDosageBufferDiscardAttemptPlan {
        self.inner.plan_dosage_buffer_discard_attempt(buffer_identifier).into()
    }

    fn plan_dosage_buffer_pool_reuse_observation(&self) -> NativeDosageBufferPoolObservationPlan {
        self.inner.plan_dosage_buffer_pool_reuse_observation().into()
    }

    fn plan_dosage_buffer_pool_return_observation(&self) -> NativeDosageBufferPoolObservationPlan {
        self.inner.plan_dosage_buffer_pool_return_observation().into()
    }

    fn plan_dosage_buffer_pool_allocate_observation(&self) -> NativeDosageBufferPoolObservationPlan {
        self.inner.plan_dosage_buffer_pool_allocate_observation().into()
    }

    fn plan_dosage_buffer_pool_discard_observation(&self) -> NativeDosageBufferPoolObservationPlan {
        self.inner.plan_dosage_buffer_pool_discard_observation().into()
    }

    fn plan_dosage_buffer_pool_consumer_wait_observation(&self) -> NativeDosageBufferPoolObservationPlan {
        self.inner.plan_dosage_buffer_pool_consumer_wait_observation().into()
    }

    #[allow(clippy::needless_pass_by_value)]
    fn plan_dosage_buffer_reuse(
        &self,
        buffered_shape: Vec<usize>,
        expected_shape: Vec<usize>,
    ) -> Option<NativeDosageBufferReusePlan> {
        self.inner.plan_dosage_buffer_reuse(&buffered_shape, &expected_shape).map(Into::into)
    }

    fn plan_variant_major_dosage_batch_handoff(
        &self,
        metadata_count: usize,
        genotype_matrix_by_variant_count: usize,
        chunk_stats_count: usize,
    ) -> PyResult<NativeVariantMajorDosageBatchHandoffPlan> {
        self.inner
            .plan_variant_major_dosage_batch_handoff(
                metadata_count,
                genotype_matrix_by_variant_count,
                chunk_stats_count,
            )
            .map(Into::into)
            .map_err(|error| schedule_error_to_py(&error))
    }

    fn plan_dosage_work_handoff(&self, chunk_count: usize) -> PyResult<NativeDosageWorkHandoffPlan> {
        self.inner.plan_dosage_work_handoff(chunk_count).map(Into::into).map_err(|error| schedule_error_to_py(&error))
    }

    #[getter]
    fn dosage_worker_error_message(&self) -> Option<String> {
        self.inner.dosage_worker_error_message().map(str::to_string)
    }

    #[getter]
    fn result_worker_error_message(&self) -> Option<String> {
        self.inner.result_worker_error_message().map(str::to_string)
    }

    #[getter]
    fn has_dosage_worker_error(&self) -> bool {
        self.inner.has_dosage_worker_error()
    }

    #[getter]
    fn has_result_worker_error(&self) -> bool {
        self.inner.has_result_worker_error()
    }

    fn record_dosage_worker_error(&mut self, error_message: &str) {
        self.inner.record_dosage_worker_error(error_message);
    }

    fn record_result_worker_error(&mut self, error_message: &str) {
        self.inner.record_result_worker_error(error_message);
    }

    fn update_dosage_worker_error(&mut self, error_message: Option<&str>) -> NativeCallbackWorkerErrorUpdatePlan {
        self.inner.update_dosage_worker_error(error_message).into()
    }

    fn update_result_worker_error(&mut self, error_message: Option<&str>) -> NativeCallbackWorkerErrorUpdatePlan {
        self.inner.update_result_worker_error(error_message).into()
    }

    fn clear_dosage_worker_error(&mut self) -> bool {
        self.inner.clear_dosage_worker_error()
    }

    fn clear_result_worker_error(&mut self) -> bool {
        self.inner.clear_result_worker_error()
    }

    #[getter]
    fn backpressure_poll_timeout_seconds(&self) -> f64 {
        self.inner.backpressure_poll_timeout_seconds()
    }

    fn plan_worker_finish(&self) -> NativeCallbackWorkerFinishPlan {
        self.inner.plan_worker_finish().into()
    }

    fn plan_worker_abort(&self) -> NativeCallbackWorkerAbortPlan {
        self.inner.plan_worker_abort().into()
    }

    fn plan_worker_error_raise(&self) -> NativeCallbackWorkerErrorRaisePlan {
        self.inner.plan_worker_error_raise().into()
    }

    #[allow(clippy::needless_pass_by_value)]
    fn plan_queue_operation_observation(
        &self,
        queue_name: String,
        operation_name: String,
        elapsed_seconds: f64,
        blocked: bool,
    ) -> PyResult<NativeCallbackQueueOperationObservationPlan> {
        self.inner
            .plan_queue_operation_observation(&queue_name, &operation_name, elapsed_seconds, blocked)
            .map(Into::into)
            .map_err(|error| schedule_error_to_py(&error))
    }

    #[allow(clippy::needless_pass_by_value)]
    fn plan_queue_backpressure_observation(
        &self,
        queue_name: String,
        operation_name: String,
        queue_depth: usize,
        queue_capacity: usize,
        elapsed_seconds: f64,
        blocked: bool,
    ) -> PyResult<NativeCallbackQueueBackpressureObservation> {
        self.inner
            .plan_queue_backpressure_observation(
                &queue_name,
                &operation_name,
                queue_depth,
                queue_capacity,
                elapsed_seconds,
                blocked,
            )
            .map(Into::into)
            .map_err(|error| schedule_error_to_py(&error))
    }

    #[allow(clippy::needless_pass_by_value)]
    fn plan_current_queue_backpressure_observation(
        &self,
        queue_name: String,
        operation_name: String,
        elapsed_seconds: f64,
        blocked: bool,
    ) -> PyResult<NativeCallbackQueueBackpressureObservation> {
        self.inner
            .plan_current_queue_backpressure_observation(&queue_name, &operation_name, elapsed_seconds, blocked)
            .map(Into::into)
            .map_err(|error| schedule_error_to_py(&error))
    }

    #[allow(clippy::needless_pass_by_value)]
    fn plan_dosage_buffer_pool_backpressure_observation(
        &self,
        operation_name: String,
        free_buffer_count: usize,
        elapsed_seconds: f64,
        blocked: bool,
    ) -> PyResult<NativeCallbackQueueBackpressureObservation> {
        self.inner
            .plan_dosage_buffer_pool_backpressure_observation(
                &operation_name,
                free_buffer_count,
                elapsed_seconds,
                blocked,
            )
            .map(Into::into)
            .map_err(|error| schedule_error_to_py(&error))
    }

    #[allow(clippy::needless_pass_by_value)]
    fn plan_queue_stage_observation(
        &self,
        queue_name: String,
        operation_name: String,
        elapsed_seconds: f64,
        blocked: bool,
    ) -> PyResult<NativeCallbackQueueStageObservationPlan> {
        self.inner
            .plan_queue_stage_observation(&queue_name, &operation_name, elapsed_seconds, blocked)
            .map(Into::into)
            .map_err(|error| schedule_error_to_py(&error))
    }

    #[allow(clippy::needless_pass_by_value)]
    fn plan_queue_stage_backpressure_observation(
        &self,
        queue_name: String,
        operation_name: String,
        queue_depth: usize,
        queue_capacity: usize,
        elapsed_seconds: f64,
        blocked: bool,
    ) -> PyResult<NativeCallbackQueueStageBackpressureObservation> {
        self.inner
            .plan_queue_stage_backpressure_observation(
                &queue_name,
                &operation_name,
                queue_depth,
                queue_capacity,
                elapsed_seconds,
                blocked,
            )
            .map(Into::into)
            .map_err(|error| schedule_error_to_py(&error))
    }

    #[allow(clippy::needless_pass_by_value)]
    fn plan_current_queue_stage_backpressure_observation(
        &self,
        queue_name: String,
        operation_name: String,
        elapsed_seconds: f64,
        blocked: bool,
    ) -> PyResult<NativeCallbackQueueStageBackpressureObservation> {
        self.inner
            .plan_current_queue_stage_backpressure_observation(&queue_name, &operation_name, elapsed_seconds, blocked)
            .map(Into::into)
            .map_err(|error| schedule_error_to_py(&error))
    }

    #[allow(clippy::needless_pass_by_value)]
    fn plan_dosage_buffer_pool_stage_backpressure_observation(
        &self,
        operation_name: String,
        free_buffer_count: usize,
        elapsed_seconds: f64,
        blocked: bool,
    ) -> PyResult<NativeCallbackQueueStageBackpressureObservation> {
        self.inner
            .plan_dosage_buffer_pool_stage_backpressure_observation(
                &operation_name,
                free_buffer_count,
                elapsed_seconds,
                blocked,
            )
            .map(Into::into)
            .map_err(|error| schedule_error_to_py(&error))
    }

    fn plan_dosage_worker_join(&self, timeout_seconds: Option<f64>) -> NativeCallbackWorkerJoinPlan {
        self.inner.plan_dosage_worker_join(timeout_seconds).into()
    }

    fn plan_result_worker_join(&self, timeout_seconds: Option<f64>) -> NativeCallbackWorkerJoinPlan {
        self.inner.plan_result_worker_join(timeout_seconds).into()
    }

    fn plan_dosage_worker_stop(
        &self,
        timeout_seconds: Option<f64>,
        is_worker_alive: bool,
    ) -> NativeCallbackWorkerStopPlan {
        self.inner.plan_dosage_worker_stop(timeout_seconds, is_worker_alive).into()
    }

    fn plan_result_worker_stop(
        &self,
        timeout_seconds: Option<f64>,
        is_worker_alive: bool,
    ) -> NativeCallbackWorkerStopPlan {
        self.inner.plan_result_worker_stop(timeout_seconds, is_worker_alive).into()
    }

    fn plan_dosage_worker_stop_poll(
        &self,
        remaining_timeout_seconds: f64,
        is_worker_alive: bool,
    ) -> NativeCallbackWorkerStopPollPlan {
        self.inner.plan_dosage_worker_stop_poll(remaining_timeout_seconds, is_worker_alive).into()
    }

    fn plan_result_worker_stop_poll(
        &self,
        remaining_timeout_seconds: f64,
        is_worker_alive: bool,
    ) -> NativeCallbackWorkerStopPollPlan {
        self.inner.plan_result_worker_stop_poll(remaining_timeout_seconds, is_worker_alive).into()
    }
}

#[pymethods]
impl NativeCallbackWorkerShutdownTimeouts {
    #[getter]
    fn dosage_worker_join_timeout_seconds(&self) -> f64 {
        self.inner.dosage_worker_join_timeout_seconds
    }

    #[getter]
    fn result_worker_join_timeout_seconds(&self) -> f64 {
        self.inner.result_worker_join_timeout_seconds
    }

    #[getter]
    fn graceful_dosage_worker_join_timeout_seconds(&self) -> f64 {
        self.inner.graceful_dosage_worker_join_timeout_seconds
    }

    #[getter]
    fn graceful_result_worker_join_timeout_seconds(&self) -> f64 {
        self.inner.graceful_result_worker_join_timeout_seconds
    }

    #[getter]
    fn worker_abort_stop_timeout_seconds(&self) -> f64 {
        self.inner.worker_abort_stop_timeout_seconds
    }
}

#[pymethods]
impl NativeCallbackWorkerJoinPlan {
    #[getter]
    fn should_join(&self) -> bool {
        self.inner.should_join
    }

    #[getter]
    fn timeout_seconds(&self) -> f64 {
        self.inner.timeout_seconds
    }
}

#[pymethods]
impl NativeCallbackWorkerStopPlan {
    #[getter]
    fn should_stop(&self) -> bool {
        self.inner.should_stop
    }

    #[getter]
    fn timeout_seconds(&self) -> f64 {
        self.inner.timeout_seconds
    }
}

#[pymethods]
impl NativeCallbackWorkerFinishPlan {
    #[getter]
    fn finish_actions(&self) -> Vec<String> {
        self.inner.finish_actions.clone()
    }

    #[getter]
    fn stop_dosage_worker(&self) -> bool {
        self.inner.stop_dosage_worker()
    }

    #[getter]
    fn join_dosage_worker(&self) -> bool {
        self.inner.join_dosage_worker()
    }

    #[getter]
    fn stop_result_worker(&self) -> bool {
        self.inner.stop_result_worker()
    }

    #[getter]
    fn join_result_worker(&self) -> bool {
        self.inner.join_result_worker()
    }

    #[getter]
    fn raise_worker_error(&self) -> bool {
        self.inner.raise_worker_error()
    }

    #[getter]
    fn complete_progress(&self) -> bool {
        self.inner.complete_progress()
    }

    #[getter]
    fn emit_binary_correction_summary(&self) -> bool {
        self.inner.emit_binary_correction_summary()
    }

    #[getter]
    fn dosage_stop_timeout_seconds(&self) -> f64 {
        self.inner.dosage_stop_timeout_seconds
    }

    #[getter]
    fn dosage_join_timeout_seconds(&self) -> f64 {
        self.inner.dosage_join_timeout_seconds
    }

    #[getter]
    fn result_stop_timeout_seconds(&self) -> f64 {
        self.inner.result_stop_timeout_seconds
    }

    #[getter]
    fn result_join_timeout_seconds(&self) -> f64 {
        self.inner.result_join_timeout_seconds
    }
}

#[pymethods]
impl NativeCallbackWorkerAbortPlan {
    #[getter]
    fn abort_actions(&self) -> Vec<String> {
        self.inner.abort_actions.clone()
    }

    #[getter]
    fn stop_dosage_worker(&self) -> bool {
        self.inner.stop_dosage_worker()
    }

    #[getter]
    fn stop_result_worker(&self) -> bool {
        self.inner.stop_result_worker()
    }

    #[getter]
    fn dosage_stop_timeout_seconds(&self) -> f64 {
        self.inner.dosage_stop_timeout_seconds
    }

    #[getter]
    fn result_stop_timeout_seconds(&self) -> f64 {
        self.inner.result_stop_timeout_seconds
    }
}

#[pymethods]
impl NativeCallbackWorkerStopPollPlan {
    #[getter]
    fn should_stop(&self) -> bool {
        self.inner.should_stop
    }

    #[getter]
    fn poll_timeout_seconds(&self) -> f64 {
        self.inner.poll_timeout_seconds
    }
}

#[pymethods]
impl NativeCallbackWorkerErrorRaisePlan {
    #[getter]
    fn should_raise(&self) -> bool {
        self.inner.should_raise
    }

    #[getter]
    fn raise_dosage_worker_error(&self) -> bool {
        self.inner.raise_dosage_worker_error
    }

    #[getter]
    fn raise_result_worker_error(&self) -> bool {
        self.inner.raise_result_worker_error
    }

    #[getter]
    fn error_message(&self) -> Option<String> {
        self.inner.error_message.clone()
    }
}

#[pymethods]
impl NativeCallbackWorkerErrorUpdatePlan {
    #[getter]
    fn had_error(&self) -> bool {
        self.inner.had_error
    }

    #[getter]
    fn has_error(&self) -> bool {
        self.inner.has_error
    }

    #[getter]
    fn error_message(&self) -> Option<String> {
        self.inner.error_message.clone()
    }
}

#[pymethods]
impl NativeMultiTraitChunkWritePlan {
    #[getter]
    fn active_trait_indices(&self) -> Vec<usize> {
        self.inner.active_trait_indices.clone()
    }

    #[getter]
    fn total_trait_count(&self) -> usize {
        self.inner.total_trait_count
    }

    #[getter]
    fn active_trait_count(&self) -> usize {
        self.inner.active_trait_count()
    }

    #[getter]
    fn all_traits_committed(&self) -> bool {
        self.inner.all_traits_committed()
    }
}

#[pymethods]
impl NativeWriterFinishExecutionPlan {
    #[getter]
    fn writer_session_count(&self) -> usize {
        self.inner.writer_session_count
    }

    #[getter]
    fn thread_count(&self) -> usize {
        self.inner.thread_count
    }

    #[getter]
    fn has_writer_sessions(&self) -> bool {
        self.inner.has_writer_sessions()
    }

    #[getter]
    fn uses_parallel_finish(&self) -> bool {
        self.inner.uses_parallel_finish()
    }
}

#[pymethods]
impl NativeBgenDeliveryCleanupPlan {
    #[getter]
    fn cleanup_actions(&self) -> Vec<String> {
        self.inner.cleanup_actions.clone()
    }

    #[getter]
    fn drain_callback(&self) -> bool {
        self.inner.drain_callback()
    }

    #[getter]
    fn finish_writer_sessions(&self) -> bool {
        self.inner.finish_writer_sessions()
    }

    #[getter]
    fn finish_interrupted_writer_sessions(&self) -> bool {
        self.inner.finish_interrupted_writer_sessions()
    }

    #[getter]
    fn abort_callback(&self) -> bool {
        self.inner.abort_callback()
    }

    #[getter]
    fn abort_writer_sessions(&self) -> bool {
        self.inner.abort_writer_sessions()
    }

    #[getter]
    fn write_stage_timing_snapshot(&self) -> bool {
        self.inner.write_stage_timing_snapshot()
    }
}

#[pymethods]
impl NativeBgenDeliveryInvocationPlan {
    #[getter]
    fn delivery_method(&self) -> &str {
        self.inner.delivery_method.as_value()
    }

    #[getter]
    fn callback_batch_size(&self) -> usize {
        self.inner.callback_batch_size
    }
}

#[pymethods]
impl NativeSingleTraitOutputWritePlan {
    #[getter]
    fn method_name(&self) -> &str {
        &self.inner.method_name
    }

    #[getter]
    fn uses_float64_native_writer(&self) -> bool {
        self.inner.uses_float64_native_writer
    }
}

#[pymethods]
impl NativeMultiTraitOutputWritePlan {
    #[getter]
    fn active_trait_count(&self) -> usize {
        self.inner.active_trait_count
    }

    #[getter]
    fn use_native_multi_writer(&self) -> bool {
        self.inner.use_native_multi_writer
    }

    #[getter]
    fn uses_float64_native_writer(&self) -> bool {
        self.inner.uses_float64_native_writer
    }
}

#[pymethods]
impl NativeCallbackQueueOperationObservationPlan {
    #[getter]
    fn queue_name(&self) -> &str {
        &self.inner.queue_name
    }

    #[getter]
    fn operation_name(&self) -> &str {
        &self.inner.operation_name
    }

    #[getter]
    fn blocked_seconds(&self) -> f64 {
        self.inner.blocked_seconds
    }
}

#[pymethods]
impl NativeCallbackQueueBackpressureObservation {
    #[getter]
    fn queue_name(&self) -> &str {
        &self.inner.queue_name
    }

    #[getter]
    fn operation_name(&self) -> &str {
        &self.inner.operation_name
    }

    #[getter]
    fn queue_depth(&self) -> usize {
        self.inner.queue_depth
    }

    #[getter]
    fn queue_capacity(&self) -> usize {
        self.inner.queue_capacity
    }

    #[getter]
    fn elapsed_seconds(&self) -> f64 {
        self.inner.elapsed_seconds
    }

    #[getter]
    fn blocked_seconds(&self) -> f64 {
        self.inner.blocked_seconds
    }
}

#[pymethods]
impl NativeCallbackQueueStageObservationPlan {
    #[getter]
    fn queue_name(&self) -> &str {
        &self.inner.queue_name
    }

    #[getter]
    fn operation_name(&self) -> &str {
        &self.inner.operation_name
    }

    #[getter]
    fn stage_name(&self) -> &str {
        &self.inner.stage_name
    }

    #[getter]
    fn blocked_seconds(&self) -> f64 {
        self.inner.blocked_seconds
    }
}

#[pymethods]
impl NativeCallbackQueueStageBackpressureObservation {
    #[getter]
    fn queue_name(&self) -> &str {
        &self.inner.queue_name
    }

    #[getter]
    fn operation_name(&self) -> &str {
        &self.inner.operation_name
    }

    #[getter]
    fn stage_name(&self) -> &str {
        &self.inner.stage_name
    }

    #[getter]
    fn queue_depth(&self) -> usize {
        self.inner.queue_depth
    }

    #[getter]
    fn queue_capacity(&self) -> usize {
        self.inner.queue_capacity
    }

    #[getter]
    fn elapsed_seconds(&self) -> f64 {
        self.inner.elapsed_seconds
    }

    #[getter]
    fn blocked_seconds(&self) -> f64 {
        self.inner.blocked_seconds
    }
}

#[pymethods]
impl NativeCallbackQueuePutAttemptPlan {
    #[getter]
    fn should_put(&self) -> bool {
        self.inner.should_put
    }

    #[getter]
    fn should_wait(&self) -> bool {
        self.inner.should_wait
    }

    #[getter]
    fn wait_timeout_seconds(&self) -> f64 {
        self.inner.wait_timeout_seconds
    }

    #[getter]
    fn queue_depth(&self) -> usize {
        self.inner.queue_depth
    }

    #[getter]
    fn queue_capacity(&self) -> usize {
        self.inner.queue_capacity
    }
}

#[pymethods]
impl NativeCallbackQueuePutObservationPlan {
    #[getter]
    fn queue_name(&self) -> &str {
        &self.inner.queue_name
    }

    #[getter]
    fn operation_name(&self) -> &str {
        &self.inner.operation_name
    }

    #[getter]
    fn blocked(&self) -> bool {
        self.inner.blocked
    }

    #[getter]
    fn should_retry_put(&self) -> bool {
        self.inner.should_retry_put
    }
}

#[pymethods]
impl NativeCallbackQueueGetAttemptPlan {
    #[getter]
    fn should_get(&self) -> bool {
        self.inner.should_get
    }

    #[getter]
    fn should_wait(&self) -> bool {
        self.inner.should_wait
    }

    #[getter]
    fn has_release_error(&self) -> bool {
        self.inner.has_release_error
    }

    #[getter]
    fn wait_timeout_seconds(&self) -> f64 {
        self.inner.wait_timeout_seconds
    }

    #[getter]
    fn queue_depth(&self) -> usize {
        self.inner.queue_depth
    }

    #[getter]
    fn queue_capacity(&self) -> usize {
        self.inner.queue_capacity
    }
}

#[pymethods]
impl NativeCallbackQueueGetObservationPlan {
    #[getter]
    fn queue_name(&self) -> &str {
        &self.inner.queue_name
    }

    #[getter]
    fn operation_name(&self) -> &str {
        &self.inner.operation_name
    }

    #[getter]
    fn blocked(&self) -> bool {
        self.inner.blocked
    }
}

#[pymethods]
impl NativeDosageBufferAcquireAttemptPlan {
    #[getter]
    fn should_take_free_buffer(&self) -> bool {
        self.inner.should_take_free_buffer
    }

    #[getter]
    fn should_allocate(&self) -> bool {
        self.inner.should_allocate
    }

    #[getter]
    fn should_wait(&self) -> bool {
        self.inner.should_wait
    }

    #[getter]
    fn wait_timeout_seconds(&self) -> f64 {
        self.inner.wait_timeout_seconds
    }

    #[getter]
    fn free_buffer_count(&self) -> usize {
        self.inner.free_buffer_count
    }

    #[getter]
    fn allocated_count(&self) -> usize {
        self.inner.allocated_count
    }

    #[getter]
    fn buffer_limit(&self) -> usize {
        self.inner.buffer_limit
    }
}

#[pymethods]
impl NativeDosageBufferRegisterAttemptPlan {
    #[getter]
    fn should_register(&self) -> bool {
        self.inner.should_register
    }

    #[getter]
    fn has_registration_error(&self) -> bool {
        self.inner.has_registration_error
    }

    #[getter]
    fn allocated_count(&self) -> usize {
        self.inner.allocated_count
    }

    #[getter]
    fn buffer_limit(&self) -> usize {
        self.inner.buffer_limit
    }
}

#[pymethods]
impl NativeDosageBufferReturnAttemptPlan {
    #[getter]
    fn should_return(&self) -> bool {
        self.inner.should_return
    }

    #[getter]
    fn allocated_count(&self) -> usize {
        self.inner.allocated_count
    }

    #[getter]
    fn buffer_limit(&self) -> usize {
        self.inner.buffer_limit
    }
}

#[pymethods]
impl NativeDosageBufferDiscardAttemptPlan {
    #[getter]
    fn should_discard(&self) -> bool {
        self.inner.should_discard
    }

    #[getter]
    fn allocated_count(&self) -> usize {
        self.inner.allocated_count
    }

    #[getter]
    fn buffer_limit(&self) -> usize {
        self.inner.buffer_limit
    }
}

#[pymethods]
impl NativeDosageBufferPoolObservationPlan {
    #[getter]
    fn operation_name(&self) -> &str {
        &self.inner.operation_name
    }

    #[getter]
    fn blocked(&self) -> bool {
        self.inner.blocked
    }
}

#[pymethods]
impl NativeResultInFlightAcquireAttemptPlan {
    #[getter]
    fn should_acquire(&self) -> bool {
        self.inner.should_acquire
    }

    #[getter]
    fn should_wait(&self) -> bool {
        self.inner.should_wait
    }

    #[getter]
    fn wait_timeout_seconds(&self) -> f64 {
        self.inner.wait_timeout_seconds
    }

    #[getter]
    fn occupied_count(&self) -> usize {
        self.inner.occupied_count
    }

    #[getter]
    fn slot_limit(&self) -> usize {
        self.inner.slot_limit
    }
}

impl NativeResultInFlightAcquireObservationPlan {
    pub(crate) fn resource_name_value(&self) -> &str {
        &self.inner.resource_name
    }

    pub(crate) fn operation_name_value(&self) -> &str {
        &self.inner.operation_name
    }

    pub(crate) fn blocked_value(&self) -> bool {
        self.inner.blocked
    }
}

#[pymethods]
impl NativeResultInFlightAcquireObservationPlan {
    #[getter]
    fn resource_name(&self) -> &str {
        &self.inner.resource_name
    }

    #[getter]
    fn operation_name(&self) -> &str {
        &self.inner.operation_name
    }

    #[getter]
    fn blocked(&self) -> bool {
        self.inner.blocked
    }

    #[getter]
    fn should_retry_acquisition(&self) -> bool {
        self.inner.should_retry_acquisition
    }
}

#[pymethods]
impl NativeResultInFlightReleaseAttemptPlan {
    #[getter]
    fn should_release(&self) -> bool {
        self.inner.should_release
    }

    #[getter]
    fn has_release_error(&self) -> bool {
        self.inner.has_release_error
    }

    #[getter]
    fn occupied_count(&self) -> usize {
        self.inner.occupied_count
    }

    #[getter]
    fn slot_limit(&self) -> usize {
        self.inner.slot_limit
    }
}

#[pymethods]
impl NativeResultInFlightReleaseObservationPlan {
    #[getter]
    fn resource_name(&self) -> &str {
        &self.inner.resource_name
    }

    #[getter]
    fn operation_name(&self) -> &str {
        &self.inner.operation_name
    }

    #[getter]
    fn blocked(&self) -> bool {
        self.inner.blocked
    }
}

#[pymethods]
impl NativeResultWriteItemResourceReleasePlan {
    #[getter]
    fn should_release_host_buffer(&self) -> bool {
        self.inner.should_release_host_buffer
    }

    #[getter]
    fn should_release_result_in_flight_slot(&self) -> bool {
        self.inner.should_release_result_in_flight_slot
    }
}

#[pymethods]
impl NativeResultWriteHandoffPlan {
    #[getter]
    fn should_enqueue(&self) -> bool {
        self.inner.should_enqueue
    }

    #[getter]
    fn has_result_work_item(&self) -> bool {
        self.inner.has_result_work_item
    }

    #[getter]
    fn is_stop_signal(&self) -> bool {
        self.inner.is_stop_signal
    }
}

#[pymethods]
impl NativeResultWriteDrainCompletionPlan {
    #[getter]
    fn should_stop(&self) -> bool {
        self.inner.should_stop
    }

    #[getter]
    fn should_flush_binary_correction_diagnostics(&self) -> bool {
        self.inner.should_flush_binary_correction_diagnostics
    }
}

#[pymethods]
impl NativeResultWriteItemDispatchPlan {
    #[getter]
    fn result_work_item_kind(&self) -> String {
        self.inner.result_work_item_kind.clone()
    }

    #[getter]
    fn expected_result_work_item_kind(&self) -> String {
        self.inner.expected_result_work_item_kind.clone()
    }

    #[getter]
    fn should_process_result_write_item(&self) -> bool {
        self.inner.should_process_result_write_item
    }

    #[getter]
    fn should_process_multi_result_write_item(&self) -> bool {
        self.inner.should_process_multi_result_write_item
    }

    #[getter]
    fn has_dispatch_error(&self) -> bool {
        self.inner.has_dispatch_error
    }

    #[getter]
    fn error_message(&self) -> Option<String> {
        self.inner.error_message.clone()
    }
}

#[pymethods]
impl NativeDosageWorkDrainCompletionPlan {
    #[getter]
    fn should_stop(&self) -> bool {
        self.inner.should_stop
    }
}

#[pymethods]
impl NativeDosageWorkItemDispatchPlan {
    #[getter]
    fn dosage_work_item_kind(&self) -> String {
        self.inner.dosage_work_item_kind.clone()
    }

    #[getter]
    fn should_process_sample_major_dosage(&self) -> bool {
        self.inner.should_process_sample_major_dosage()
    }

    #[getter]
    fn should_process_variant_major_dosage(&self) -> bool {
        self.inner.should_process_variant_major_dosage()
    }

    #[getter]
    fn should_process_variant_major_dosage_batch(&self) -> bool {
        self.inner.should_process_variant_major_dosage_batch()
    }

    #[getter]
    fn should_process_variant_major_packed8_probability_pair(&self) -> bool {
        self.inner.should_process_variant_major_packed8_probability_pair()
    }

    #[getter]
    fn has_dispatch_error(&self) -> bool {
        self.inner.has_dispatch_error()
    }

    #[getter]
    fn error_message(&self) -> Option<String> {
        self.inner.error_message.clone()
    }
}

#[pymethods]
impl NativeDosageWorkItemStageDurationPlan {
    #[getter]
    fn chunk_count(&self) -> usize {
        self.inner.chunk_count
    }

    #[getter]
    fn duration_per_chunk(&self) -> f64 {
        self.inner.duration_per_chunk
    }
}

#[pymethods]
impl NativeGpuGenotypeFormatResolutionPlan {
    #[getter]
    fn requested_gpu_genotype_format(&self) -> &str {
        &self.inner.requested_gpu_genotype_format
    }

    #[getter]
    fn resolved_gpu_genotype_format(&self) -> Option<&str> {
        self.inner.resolved_gpu_genotype_format.as_deref()
    }

    #[getter]
    fn resolution_reason(&self) -> Option<&str> {
        self.inner.resolution_reason.as_deref()
    }

    #[getter]
    fn fallback_error(&self) -> Option<&str> {
        self.inner.fallback_error.as_deref()
    }

    #[getter]
    fn requires_trusted_validation(&self) -> bool {
        self.inner.requires_trusted_validation
    }

    #[getter]
    fn is_resolved(&self) -> bool {
        self.inner.is_resolved()
    }

    #[getter]
    fn should_log_auto_resolution(&self) -> bool {
        self.inner.should_log_auto_resolution()
    }
}

impl NativeCallbackSchedulerState {
    pub(crate) fn from_limits(
        staging_depth: i64,
        native_callback_batch_size: i64,
        result_in_flight_limit: Option<i64>,
        dosage_buffer_limit: Option<i64>,
    ) -> PyResult<Self> {
        native_schedule::CallbackSchedulerState::new(
            staging_depth,
            native_callback_batch_size,
            result_in_flight_limit,
            dosage_buffer_limit,
        )
        .map(|inner| Self { inner })
        .map_err(|error| schedule_error_to_py(&error))
    }

    pub(crate) fn has_started_value(&self) -> bool {
        self.inner.has_started()
    }

    pub(crate) fn native_callback_batch_size_value(&self) -> usize {
        self.inner.native_callback_batch_size()
    }

    pub(crate) fn dosage_queue_depth_value(&self) -> usize {
        self.inner.dosage_queue_depth()
    }

    pub(crate) fn result_queue_depth_value(&self) -> usize {
        self.inner.result_queue_depth()
    }

    pub(crate) fn result_in_flight_limit_value(&self) -> usize {
        self.inner.result_in_flight_limit()
    }

    pub(crate) fn dosage_buffer_limit_value(&self) -> usize {
        self.inner.dosage_buffer_limit()
    }

    pub(crate) fn dosage_queue_occupied_count_value(&self) -> usize {
        self.inner.dosage_queue_occupied_count()
    }

    pub(crate) fn result_queue_occupied_count_value(&self) -> usize {
        self.inner.result_queue_occupied_count()
    }

    pub(crate) fn result_in_flight_occupied_count_value(&self) -> usize {
        self.inner.result_in_flight_occupied_count()
    }

    pub(crate) fn dosage_buffer_allocated_count_value(&self) -> usize {
        self.inner.dosage_buffer_allocated_count()
    }

    pub(crate) fn dosage_buffer_identifiers_value(&self) -> Vec<usize> {
        self.inner.dosage_buffer_identifiers()
    }

    pub(crate) fn plan_dosage_queue_put_observation_value(
        &self,
        queued: bool,
    ) -> NativeCallbackQueuePutObservationPlan {
        self.inner.plan_dosage_queue_put_observation(queued).into()
    }

    pub(crate) fn plan_dosage_queue_get_observation_value(&self) -> NativeCallbackQueueGetObservationPlan {
        self.inner.plan_dosage_queue_get_observation().into()
    }

    pub(crate) fn plan_result_queue_put_observation_value(
        &self,
        queued: bool,
    ) -> NativeCallbackQueuePutObservationPlan {
        self.inner.plan_result_queue_put_observation(queued).into()
    }

    pub(crate) fn plan_result_queue_get_observation_value(&self) -> NativeCallbackQueueGetObservationPlan {
        self.inner.plan_result_queue_get_observation().into()
    }

    pub(crate) fn plan_worker_start_attempt_value(&mut self) -> NativeCallbackWorkerStartAttemptPlan {
        self.inner.plan_worker_start_attempt().into()
    }

    pub(crate) fn plan_dosage_worker_join_value(&self, timeout_seconds: Option<f64>) -> NativeCallbackWorkerJoinPlan {
        self.inner.plan_dosage_worker_join(timeout_seconds).into()
    }

    pub(crate) fn plan_result_worker_join_value(&self, timeout_seconds: Option<f64>) -> NativeCallbackWorkerJoinPlan {
        self.inner.plan_result_worker_join(timeout_seconds).into()
    }

    pub(crate) fn plan_dosage_worker_stop_value(
        &self,
        timeout_seconds: Option<f64>,
        is_worker_alive: bool,
    ) -> NativeCallbackWorkerStopPlan {
        self.inner.plan_dosage_worker_stop(timeout_seconds, is_worker_alive).into()
    }

    pub(crate) fn plan_result_worker_stop_value(
        &self,
        timeout_seconds: Option<f64>,
        is_worker_alive: bool,
    ) -> NativeCallbackWorkerStopPlan {
        self.inner.plan_result_worker_stop(timeout_seconds, is_worker_alive).into()
    }

    pub(crate) fn plan_dosage_worker_stop_poll_value(
        &self,
        remaining_timeout_seconds: f64,
        is_worker_alive: bool,
    ) -> NativeCallbackWorkerStopPollPlan {
        self.inner.plan_dosage_worker_stop_poll(remaining_timeout_seconds, is_worker_alive).into()
    }

    pub(crate) fn plan_result_worker_stop_poll_value(
        &self,
        remaining_timeout_seconds: f64,
        is_worker_alive: bool,
    ) -> NativeCallbackWorkerStopPollPlan {
        self.inner.plan_result_worker_stop_poll(remaining_timeout_seconds, is_worker_alive).into()
    }

    pub(crate) fn plan_result_in_flight_slot_acquire_backpressure_attempt_value(
        &mut self,
    ) -> NativeResultInFlightAcquireAttemptPlan {
        self.inner.plan_result_in_flight_slot_acquire_backpressure_attempt().into()
    }

    pub(crate) fn plan_result_in_flight_slot_acquire_observation_value(
        &self,
        acquire_attempt_plan: &NativeResultInFlightAcquireAttemptPlan,
    ) -> NativeResultInFlightAcquireObservationPlan {
        self.inner.plan_result_in_flight_slot_acquire_observation(&acquire_attempt_plan.inner).into()
    }

    pub(crate) fn plan_result_in_flight_slot_release_attempt_value(
        &mut self,
    ) -> NativeResultInFlightReleaseAttemptPlan {
        self.inner.plan_result_in_flight_slot_release_attempt().into()
    }

    pub(crate) fn plan_result_in_flight_slot_release_observation_value(
        &self,
    ) -> NativeResultInFlightReleaseObservationPlan {
        self.inner.plan_result_in_flight_slot_release_observation().into()
    }

    pub(crate) fn plan_worker_finish_value(&self) -> NativeCallbackWorkerFinishPlan {
        self.inner.plan_worker_finish().into()
    }

    pub(crate) fn plan_worker_abort_value(&self) -> NativeCallbackWorkerAbortPlan {
        self.inner.plan_worker_abort().into()
    }

    pub(crate) fn plan_worker_error_raise_value(&self) -> NativeCallbackWorkerErrorRaisePlan {
        self.inner.plan_worker_error_raise().into()
    }

    pub(crate) fn update_dosage_worker_error_value(
        &mut self,
        error_message: Option<&str>,
    ) -> NativeCallbackWorkerErrorUpdatePlan {
        self.inner.update_dosage_worker_error(error_message).into()
    }

    pub(crate) fn update_result_worker_error_value(
        &mut self,
        error_message: Option<&str>,
    ) -> NativeCallbackWorkerErrorUpdatePlan {
        self.inner.update_result_worker_error(error_message).into()
    }

    pub(crate) fn plan_result_write_item_pre_write_resource_release_value(
        &self,
        has_host_dosage_buffer: bool,
    ) -> NativeResultWriteItemResourceReleasePlan {
        self.inner.plan_result_write_item_pre_write_resource_release(has_host_dosage_buffer).into()
    }

    pub(crate) fn plan_result_write_item_final_resource_release_value(
        &self,
        has_host_dosage_buffer: bool,
        has_released_host_dosage_buffer: bool,
        release_in_flight_slot: bool,
    ) -> NativeResultWriteItemResourceReleasePlan {
        self.inner
            .plan_result_write_item_final_resource_release(
                has_host_dosage_buffer,
                has_released_host_dosage_buffer,
                release_in_flight_slot,
            )
            .into()
    }

    pub(crate) fn plan_dosage_buffer_acquire_backpressure_attempt_value(
        &self,
        free_buffer_count: usize,
    ) -> NativeDosageBufferAcquireAttemptPlan {
        self.inner.plan_dosage_buffer_acquire_backpressure_attempt(free_buffer_count).into()
    }

    pub(crate) fn plan_dosage_buffer_register_attempt_value(
        &mut self,
        buffer_identifier: usize,
    ) -> NativeDosageBufferRegisterAttemptPlan {
        self.inner.plan_dosage_buffer_register_attempt(buffer_identifier).into()
    }

    pub(crate) fn plan_dosage_buffer_return_attempt_value(
        &self,
        buffer_identifier: usize,
    ) -> NativeDosageBufferReturnAttemptPlan {
        self.inner.plan_dosage_buffer_return_attempt(buffer_identifier).into()
    }

    pub(crate) fn plan_dosage_buffer_discard_attempt_value(
        &mut self,
        buffer_identifier: usize,
    ) -> NativeDosageBufferDiscardAttemptPlan {
        self.inner.plan_dosage_buffer_discard_attempt(buffer_identifier).into()
    }

    pub(crate) fn plan_dosage_buffer_reuse_value(
        &self,
        buffered_shape: &[usize],
        expected_shape: &[usize],
    ) -> Option<NativeDosageBufferReusePlan> {
        self.inner.plan_dosage_buffer_reuse(buffered_shape, expected_shape).map(Into::into)
    }

    pub(crate) fn plan_dosage_queue_put_attempt_value(
        &mut self,
        wait_timeout_seconds: f64,
    ) -> NativeCallbackQueuePutAttemptPlan {
        self.inner.plan_dosage_queue_put_attempt(wait_timeout_seconds).into()
    }

    pub(crate) fn plan_dosage_queue_put_backpressure_attempt_value(&mut self) -> NativeCallbackQueuePutAttemptPlan {
        self.inner.plan_dosage_queue_put_backpressure_attempt().into()
    }

    pub(crate) fn release_dosage_queue_slot_value(&mut self) -> bool {
        self.inner.release_dosage_queue_slot()
    }

    pub(crate) fn acquire_dosage_queue_slot_value(&mut self) -> bool {
        self.inner.acquire_dosage_queue_slot()
    }

    pub(crate) fn plan_dosage_queue_get_attempt_value(
        &mut self,
        has_queued_item: bool,
    ) -> NativeCallbackQueueGetAttemptPlan {
        self.inner.plan_dosage_queue_get_attempt(has_queued_item).into()
    }

    pub(crate) fn plan_result_write_handoff_value(&self, has_result_work_item: bool) -> NativeResultWriteHandoffPlan {
        self.inner.plan_result_write_handoff(has_result_work_item).into()
    }

    pub(crate) fn plan_result_write_drain_completion_value(
        &self,
        has_result_work_item: bool,
        flush_binary_correction_diagnostics_on_stop: bool,
    ) -> NativeResultWriteDrainCompletionPlan {
        self.inner
            .plan_result_write_drain_completion(has_result_work_item, flush_binary_correction_diagnostics_on_stop)
            .into()
    }

    pub(crate) fn plan_result_write_item_dispatch_value(
        &self,
        result_work_item_kind: &str,
        expected_result_work_item_kind: &str,
    ) -> PyResult<NativeResultWriteItemDispatchPlan> {
        self.inner
            .plan_result_write_item_dispatch(result_work_item_kind, expected_result_work_item_kind)
            .map(Into::into)
            .map_err(|error| schedule_error_to_py(&error))
    }

    pub(crate) fn plan_result_queue_put_attempt_value(
        &mut self,
        wait_timeout_seconds: f64,
    ) -> NativeCallbackQueuePutAttemptPlan {
        self.inner.plan_result_queue_put_attempt(wait_timeout_seconds).into()
    }

    pub(crate) fn plan_result_queue_put_backpressure_attempt_value(&mut self) -> NativeCallbackQueuePutAttemptPlan {
        self.inner.plan_result_queue_put_backpressure_attempt().into()
    }

    pub(crate) fn release_result_queue_slot_value(&mut self) -> bool {
        self.inner.release_result_queue_slot()
    }

    pub(crate) fn acquire_result_queue_slot_value(&mut self) -> bool {
        self.inner.acquire_result_queue_slot()
    }

    pub(crate) fn plan_result_queue_get_attempt_value(
        &mut self,
        has_queued_item: bool,
    ) -> NativeCallbackQueueGetAttemptPlan {
        self.inner.plan_result_queue_get_attempt(has_queued_item).into()
    }

    pub(crate) fn plan_variant_major_dosage_batch_handoff_value(
        &self,
        metadata_count: usize,
        genotype_matrix_by_variant_count: usize,
        chunk_stats_count: usize,
    ) -> PyResult<NativeVariantMajorDosageBatchHandoffPlan> {
        self.inner
            .plan_variant_major_dosage_batch_handoff(
                metadata_count,
                genotype_matrix_by_variant_count,
                chunk_stats_count,
            )
            .map(Into::into)
            .map_err(|error| schedule_error_to_py(&error))
    }

    pub(crate) fn plan_dosage_work_handoff_value(&self, chunk_count: usize) -> PyResult<NativeDosageWorkHandoffPlan> {
        self.inner.plan_dosage_work_handoff(chunk_count).map(Into::into).map_err(|error| schedule_error_to_py(&error))
    }

    pub(crate) fn plan_dosage_work_drain_completion_value(
        &self,
        has_dosage_work_item: bool,
    ) -> NativeDosageWorkDrainCompletionPlan {
        self.inner.plan_dosage_work_drain_completion(has_dosage_work_item).into()
    }

    pub(crate) fn plan_dosage_work_item_dispatch_value(
        &self,
        dosage_work_item_kind: &str,
    ) -> PyResult<NativeDosageWorkItemDispatchPlan> {
        self.inner
            .plan_dosage_work_item_dispatch(dosage_work_item_kind)
            .map(Into::into)
            .map_err(|error| schedule_error_to_py(&error))
    }

    pub(crate) fn plan_dosage_work_item_stage_duration_value(
        &self,
        dosage_work_item_kind: &str,
        chunk_count: usize,
        elapsed_seconds: f64,
    ) -> PyResult<NativeDosageWorkItemStageDurationPlan> {
        self.inner
            .plan_dosage_work_item_stage_duration(dosage_work_item_kind, chunk_count, elapsed_seconds)
            .map(Into::into)
            .map_err(|error| schedule_error_to_py(&error))
    }

    pub(crate) fn plan_current_queue_backpressure_observation_value(
        &self,
        queue_name: &str,
        operation_name: &str,
        elapsed_seconds: f64,
        blocked: bool,
    ) -> PyResult<NativeCallbackQueueBackpressureObservation> {
        self.inner
            .plan_current_queue_backpressure_observation(queue_name, operation_name, elapsed_seconds, blocked)
            .map(Into::into)
            .map_err(|error| schedule_error_to_py(&error))
    }

    pub(crate) fn plan_current_queue_stage_backpressure_observation_value(
        &self,
        queue_name: &str,
        operation_name: &str,
        elapsed_seconds: f64,
        blocked: bool,
    ) -> PyResult<NativeCallbackQueueStageBackpressureObservation> {
        self.inner
            .plan_current_queue_stage_backpressure_observation(queue_name, operation_name, elapsed_seconds, blocked)
            .map(Into::into)
            .map_err(|error| schedule_error_to_py(&error))
    }

    pub(crate) fn plan_dosage_buffer_pool_reuse_observation_value(&self) -> NativeDosageBufferPoolObservationPlan {
        self.inner.plan_dosage_buffer_pool_reuse_observation().into()
    }

    pub(crate) fn plan_dosage_buffer_pool_return_observation_value(&self) -> NativeDosageBufferPoolObservationPlan {
        self.inner.plan_dosage_buffer_pool_return_observation().into()
    }

    pub(crate) fn plan_dosage_buffer_pool_allocate_observation_value(&self) -> NativeDosageBufferPoolObservationPlan {
        self.inner.plan_dosage_buffer_pool_allocate_observation().into()
    }

    pub(crate) fn plan_dosage_buffer_pool_discard_observation_value(&self) -> NativeDosageBufferPoolObservationPlan {
        self.inner.plan_dosage_buffer_pool_discard_observation().into()
    }

    pub(crate) fn plan_dosage_buffer_pool_consumer_wait_observation_value(
        &self,
    ) -> NativeDosageBufferPoolObservationPlan {
        self.inner.plan_dosage_buffer_pool_consumer_wait_observation().into()
    }

    pub(crate) fn plan_dosage_buffer_pool_backpressure_observation_value(
        &self,
        operation_name: &str,
        free_buffer_count: usize,
        elapsed_seconds: f64,
        blocked: bool,
    ) -> PyResult<NativeCallbackQueueBackpressureObservation> {
        self.inner
            .plan_dosage_buffer_pool_backpressure_observation(
                operation_name,
                free_buffer_count,
                elapsed_seconds,
                blocked,
            )
            .map(Into::into)
            .map_err(|error| schedule_error_to_py(&error))
    }

    pub(crate) fn plan_dosage_buffer_pool_stage_backpressure_observation_value(
        &self,
        operation_name: &str,
        free_buffer_count: usize,
        elapsed_seconds: f64,
        blocked: bool,
    ) -> PyResult<NativeCallbackQueueStageBackpressureObservation> {
        self.inner
            .plan_dosage_buffer_pool_stage_backpressure_observation(
                operation_name,
                free_buffer_count,
                elapsed_seconds,
                blocked,
            )
            .map(Into::into)
            .map_err(|error| schedule_error_to_py(&error))
    }
}

impl NativeCallbackWorkerStartAttemptPlan {
    pub(crate) fn has_start_error_value(&self) -> bool {
        self.inner.has_start_error
    }

    pub(crate) fn should_start_result_worker(&self) -> bool {
        self.inner.start_result_worker()
    }

    pub(crate) fn should_start_dosage_worker(&self) -> bool {
        self.inner.start_dosage_worker()
    }
}

impl NativeCallbackWorkerJoinPlan {
    pub(crate) fn should_join_value(&self) -> bool {
        self.inner.should_join
    }

    pub(crate) fn timeout_seconds_value(&self) -> f64 {
        self.inner.timeout_seconds
    }
}

impl NativeCallbackWorkerStopPlan {
    pub(crate) fn should_stop_value(&self) -> bool {
        self.inner.should_stop
    }

    pub(crate) fn timeout_seconds_value(&self) -> f64 {
        self.inner.timeout_seconds
    }
}

impl NativeCallbackWorkerStopPollPlan {
    pub(crate) fn should_stop_value(&self) -> bool {
        self.inner.should_stop
    }

    pub(crate) fn poll_timeout_seconds_value(&self) -> f64 {
        self.inner.poll_timeout_seconds
    }
}

impl NativeCallbackWorkerFinishPlan {
    pub(crate) fn stop_dosage_worker_value(&self) -> bool {
        self.inner.stop_dosage_worker()
    }

    pub(crate) fn join_dosage_worker_value(&self) -> bool {
        self.inner.join_dosage_worker()
    }

    pub(crate) fn stop_result_worker_value(&self) -> bool {
        self.inner.stop_result_worker()
    }

    pub(crate) fn join_result_worker_value(&self) -> bool {
        self.inner.join_result_worker()
    }

    pub(crate) fn raise_worker_error_value(&self) -> bool {
        self.inner.raise_worker_error()
    }

    pub(crate) fn complete_progress_value(&self) -> bool {
        self.inner.complete_progress()
    }

    pub(crate) fn emit_binary_correction_summary_value(&self) -> bool {
        self.inner.emit_binary_correction_summary()
    }

    pub(crate) fn dosage_stop_timeout_seconds_value(&self) -> f64 {
        self.inner.dosage_stop_timeout_seconds
    }

    pub(crate) fn dosage_join_timeout_seconds_value(&self) -> f64 {
        self.inner.dosage_join_timeout_seconds
    }

    pub(crate) fn result_stop_timeout_seconds_value(&self) -> f64 {
        self.inner.result_stop_timeout_seconds
    }

    pub(crate) fn result_join_timeout_seconds_value(&self) -> f64 {
        self.inner.result_join_timeout_seconds
    }
}

impl NativeCallbackWorkerAbortPlan {
    pub(crate) fn stop_dosage_worker_value(&self) -> bool {
        self.inner.stop_dosage_worker()
    }

    pub(crate) fn stop_result_worker_value(&self) -> bool {
        self.inner.stop_result_worker()
    }

    pub(crate) fn dosage_stop_timeout_seconds_value(&self) -> f64 {
        self.inner.dosage_stop_timeout_seconds
    }

    pub(crate) fn result_stop_timeout_seconds_value(&self) -> f64 {
        self.inner.result_stop_timeout_seconds
    }
}

impl NativeResultInFlightAcquireAttemptPlan {
    pub(crate) fn should_acquire_value(&self) -> bool {
        self.inner.should_acquire
    }

    pub(crate) fn should_wait_value(&self) -> bool {
        self.inner.should_wait
    }

    pub(crate) fn wait_timeout_seconds_value(&self) -> f64 {
        self.inner.wait_timeout_seconds
    }
}

impl NativeDosageBufferPoolObservationPlan {
    pub(crate) fn operation_name_value(&self) -> &str {
        &self.inner.operation_name
    }

    pub(crate) fn blocked_value(&self) -> bool {
        self.inner.blocked
    }
}

impl NativeCallbackQueuePutObservationPlan {
    pub(crate) fn queue_name_value(&self) -> &str {
        &self.inner.queue_name
    }

    pub(crate) fn operation_name_value(&self) -> &str {
        &self.inner.operation_name
    }

    pub(crate) fn blocked_value(&self) -> bool {
        self.inner.blocked
    }
}

impl NativeCallbackQueueGetObservationPlan {
    pub(crate) fn queue_name_value(&self) -> &str {
        &self.inner.queue_name
    }

    pub(crate) fn operation_name_value(&self) -> &str {
        &self.inner.operation_name
    }

    pub(crate) fn blocked_value(&self) -> bool {
        self.inner.blocked
    }
}

impl NativeResultInFlightReleaseAttemptPlan {
    pub(crate) fn has_release_error_value(&self) -> bool {
        self.inner.has_release_error
    }
}

impl NativeResultInFlightReleaseObservationPlan {
    pub(crate) fn resource_name_value(&self) -> &str {
        &self.inner.resource_name
    }

    pub(crate) fn operation_name_value(&self) -> &str {
        &self.inner.operation_name
    }

    pub(crate) fn blocked_value(&self) -> bool {
        self.inner.blocked
    }
}

impl NativeResultWriteItemResourceReleasePlan {
    pub(crate) fn should_release_host_buffer_value(&self) -> bool {
        self.inner.should_release_host_buffer
    }

    pub(crate) fn should_release_result_in_flight_slot_value(&self) -> bool {
        self.inner.should_release_result_in_flight_slot
    }
}

impl NativeDosageBufferAcquireAttemptPlan {
    pub(crate) fn should_take_free_buffer_value(&self) -> bool {
        self.inner.should_take_free_buffer
    }

    pub(crate) fn should_allocate_value(&self) -> bool {
        self.inner.should_allocate
    }

    pub(crate) fn should_wait_value(&self) -> bool {
        self.inner.should_wait
    }

    pub(crate) fn wait_timeout_seconds_value(&self) -> f64 {
        self.inner.wait_timeout_seconds
    }
}

impl NativeDosageBufferRegisterAttemptPlan {
    pub(crate) fn has_registration_error_value(&self) -> bool {
        self.inner.has_registration_error
    }
}

impl NativeDosageBufferReturnAttemptPlan {
    pub(crate) fn should_return_value(&self) -> bool {
        self.inner.should_return
    }
}

impl NativeDosageBufferDiscardAttemptPlan {
    pub(crate) fn should_discard_value(&self) -> bool {
        self.inner.should_discard
    }
}

impl NativeCallbackQueuePutAttemptPlan {
    pub(crate) fn should_put_value(&self) -> bool {
        self.inner.should_put
    }

    pub(crate) fn should_wait_value(&self) -> bool {
        self.inner.should_wait
    }

    pub(crate) fn wait_timeout_seconds_value(&self) -> f64 {
        self.inner.wait_timeout_seconds
    }
}

impl NativeCallbackQueueGetAttemptPlan {
    pub(crate) fn should_get_value(&self) -> bool {
        self.inner.should_get
    }

    pub(crate) fn should_wait_value(&self) -> bool {
        self.inner.should_wait
    }

    pub(crate) fn has_release_error_value(&self) -> bool {
        self.inner.has_release_error
    }

    pub(crate) fn wait_timeout_seconds_value(&self) -> f64 {
        self.inner.wait_timeout_seconds
    }
}

impl NativeResultWriteHandoffPlan {
    pub(crate) fn should_enqueue_value(&self) -> bool {
        self.inner.should_enqueue
    }

    pub(crate) fn has_result_work_item_value(&self) -> bool {
        self.inner.has_result_work_item
    }
}

impl NativeResultWriteItemDispatchPlan {
    pub(crate) fn has_dispatch_error_value(&self) -> bool {
        self.inner.has_dispatch_error
    }

    pub(crate) fn error_message_value(&self) -> Option<&str> {
        self.inner.error_message.as_deref()
    }
}

impl NativeDosageWorkItemDispatchPlan {
    pub(crate) fn has_dispatch_error_value(&self) -> bool {
        self.inner.has_dispatch_error()
    }

    pub(crate) fn error_message_value(&self) -> Option<&str> {
        self.inner.error_message.as_deref()
    }
}

impl From<native_schedule::CallbackWorkerShutdownTimeouts> for NativeCallbackWorkerShutdownTimeouts {
    fn from(worker_shutdown_timeouts: native_schedule::CallbackWorkerShutdownTimeouts) -> Self {
        Self { inner: worker_shutdown_timeouts }
    }
}

impl From<native_schedule::CallbackWorkerStartPlan> for NativeCallbackWorkerStartPlan {
    fn from(start_plan: native_schedule::CallbackWorkerStartPlan) -> Self {
        Self { inner: start_plan }
    }
}

impl From<native_schedule::CallbackWorkerStartAttemptPlan> for NativeCallbackWorkerStartAttemptPlan {
    fn from(start_attempt_plan: native_schedule::CallbackWorkerStartAttemptPlan) -> Self {
        Self { inner: start_attempt_plan }
    }
}

impl From<native_schedule::CallbackWorkerJoinPlan> for NativeCallbackWorkerJoinPlan {
    fn from(join_plan: native_schedule::CallbackWorkerJoinPlan) -> Self {
        Self { inner: join_plan }
    }
}

impl From<native_schedule::CallbackWorkerStopPlan> for NativeCallbackWorkerStopPlan {
    fn from(stop_plan: native_schedule::CallbackWorkerStopPlan) -> Self {
        Self { inner: stop_plan }
    }
}

impl From<native_schedule::CallbackWorkerFinishPlan> for NativeCallbackWorkerFinishPlan {
    fn from(finish_plan: native_schedule::CallbackWorkerFinishPlan) -> Self {
        Self { inner: finish_plan }
    }
}

impl From<native_schedule::CallbackWorkerAbortPlan> for NativeCallbackWorkerAbortPlan {
    fn from(abort_plan: native_schedule::CallbackWorkerAbortPlan) -> Self {
        Self { inner: abort_plan }
    }
}

impl From<native_schedule::CallbackWorkerStopPollPlan> for NativeCallbackWorkerStopPollPlan {
    fn from(stop_poll_plan: native_schedule::CallbackWorkerStopPollPlan) -> Self {
        Self { inner: stop_poll_plan }
    }
}

impl From<native_schedule::CallbackWorkerErrorRaisePlan> for NativeCallbackWorkerErrorRaisePlan {
    fn from(error_raise_plan: native_schedule::CallbackWorkerErrorRaisePlan) -> Self {
        Self { inner: error_raise_plan }
    }
}

impl From<native_schedule::CallbackWorkerErrorUpdatePlan> for NativeCallbackWorkerErrorUpdatePlan {
    fn from(error_update_plan: native_schedule::CallbackWorkerErrorUpdatePlan) -> Self {
        Self { inner: error_update_plan }
    }
}

impl From<native_schedule::NativeCallbackQueueLimits> for NativeCallbackQueueLimits {
    fn from(queue_limits: native_schedule::NativeCallbackQueueLimits) -> Self {
        Self {
            dosage_queue_depth: queue_limits.dosage_queue_depth,
            result_queue_depth: queue_limits.result_queue_depth,
            result_in_flight_limit: queue_limits.result_in_flight_limit,
            dosage_buffer_limit: queue_limits.dosage_buffer_limit,
        }
    }
}

impl From<native_schedule::DosageBufferReusePlan> for NativeDosageBufferReusePlan {
    fn from(reuse_plan: native_schedule::DosageBufferReusePlan) -> Self {
        Self { requires_slice: reuse_plan.requires_slice, slice_dimensions: reuse_plan.slice_dimensions }
    }
}

impl From<native_schedule::VariantMajorDosageBatchHandoffPlan> for NativeVariantMajorDosageBatchHandoffPlan {
    fn from(batch_handoff_plan: native_schedule::VariantMajorDosageBatchHandoffPlan) -> Self {
        Self { chunk_count: batch_handoff_plan.chunk_count }
    }
}

impl From<native_schedule::DosageWorkHandoffPlan> for NativeDosageWorkHandoffPlan {
    fn from(handoff_plan: native_schedule::DosageWorkHandoffPlan) -> Self {
        Self { chunk_count: handoff_plan.chunk_count }
    }
}

impl From<native_schedule::GpuGenotypeFormatResolutionPlan> for NativeGpuGenotypeFormatResolutionPlan {
    fn from(resolution_plan: native_schedule::GpuGenotypeFormatResolutionPlan) -> Self {
        Self { inner: resolution_plan }
    }
}

impl From<native_schedule::MultiTraitChunkWritePlan> for NativeMultiTraitChunkWritePlan {
    fn from(write_plan: native_schedule::MultiTraitChunkWritePlan) -> Self {
        Self { inner: write_plan }
    }
}

impl From<native_schedule::WriterFinishExecutionPlan> for NativeWriterFinishExecutionPlan {
    fn from(finish_plan: native_schedule::WriterFinishExecutionPlan) -> Self {
        Self { inner: finish_plan }
    }
}

impl From<native_schedule::BgenDeliveryCleanupPlan> for NativeBgenDeliveryCleanupPlan {
    fn from(cleanup_plan: native_schedule::BgenDeliveryCleanupPlan) -> Self {
        Self { inner: cleanup_plan }
    }
}

impl From<native_schedule::BgenDeliveryInvocationPlan> for NativeBgenDeliveryInvocationPlan {
    fn from(invocation_plan: native_schedule::BgenDeliveryInvocationPlan) -> Self {
        Self { inner: invocation_plan }
    }
}

impl From<native_schedule::SingleTraitOutputWritePlan> for NativeSingleTraitOutputWritePlan {
    fn from(write_plan: native_schedule::SingleTraitOutputWritePlan) -> Self {
        Self { inner: write_plan }
    }
}

impl From<native_schedule::MultiTraitOutputWritePlan> for NativeMultiTraitOutputWritePlan {
    fn from(write_plan: native_schedule::MultiTraitOutputWritePlan) -> Self {
        Self { inner: write_plan }
    }
}

impl From<native_schedule::CallbackQueueOperationObservationPlan> for NativeCallbackQueueOperationObservationPlan {
    fn from(observation_plan: native_schedule::CallbackQueueOperationObservationPlan) -> Self {
        Self { inner: observation_plan }
    }
}

impl From<native_schedule::CallbackQueueBackpressureObservation> for NativeCallbackQueueBackpressureObservation {
    fn from(observation: native_schedule::CallbackQueueBackpressureObservation) -> Self {
        Self { inner: observation }
    }
}

impl From<native_schedule::CallbackQueueStageObservationPlan> for NativeCallbackQueueStageObservationPlan {
    fn from(observation_plan: native_schedule::CallbackQueueStageObservationPlan) -> Self {
        Self { inner: observation_plan }
    }
}

impl From<native_schedule::CallbackQueueStageBackpressureObservation>
    for NativeCallbackQueueStageBackpressureObservation
{
    fn from(observation: native_schedule::CallbackQueueStageBackpressureObservation) -> Self {
        Self { inner: observation }
    }
}

impl From<native_schedule::CallbackQueuePutAttemptPlan> for NativeCallbackQueuePutAttemptPlan {
    fn from(put_attempt_plan: native_schedule::CallbackQueuePutAttemptPlan) -> Self {
        Self { inner: put_attempt_plan }
    }
}

impl From<native_schedule::CallbackQueuePutObservationPlan> for NativeCallbackQueuePutObservationPlan {
    fn from(put_observation_plan: native_schedule::CallbackQueuePutObservationPlan) -> Self {
        Self { inner: put_observation_plan }
    }
}

impl From<native_schedule::CallbackQueueGetAttemptPlan> for NativeCallbackQueueGetAttemptPlan {
    fn from(get_attempt_plan: native_schedule::CallbackQueueGetAttemptPlan) -> Self {
        Self { inner: get_attempt_plan }
    }
}

impl From<native_schedule::CallbackQueueGetObservationPlan> for NativeCallbackQueueGetObservationPlan {
    fn from(get_observation_plan: native_schedule::CallbackQueueGetObservationPlan) -> Self {
        Self { inner: get_observation_plan }
    }
}

impl From<native_schedule::DosageBufferAcquireAttemptPlan> for NativeDosageBufferAcquireAttemptPlan {
    fn from(acquire_attempt_plan: native_schedule::DosageBufferAcquireAttemptPlan) -> Self {
        Self { inner: acquire_attempt_plan }
    }
}

impl From<native_schedule::DosageBufferRegisterAttemptPlan> for NativeDosageBufferRegisterAttemptPlan {
    fn from(register_attempt_plan: native_schedule::DosageBufferRegisterAttemptPlan) -> Self {
        Self { inner: register_attempt_plan }
    }
}

impl From<native_schedule::DosageBufferReturnAttemptPlan> for NativeDosageBufferReturnAttemptPlan {
    fn from(return_attempt_plan: native_schedule::DosageBufferReturnAttemptPlan) -> Self {
        Self { inner: return_attempt_plan }
    }
}

impl From<native_schedule::DosageBufferDiscardAttemptPlan> for NativeDosageBufferDiscardAttemptPlan {
    fn from(discard_attempt_plan: native_schedule::DosageBufferDiscardAttemptPlan) -> Self {
        Self { inner: discard_attempt_plan }
    }
}

impl From<native_schedule::DosageBufferPoolObservationPlan> for NativeDosageBufferPoolObservationPlan {
    fn from(pool_observation_plan: native_schedule::DosageBufferPoolObservationPlan) -> Self {
        Self { inner: pool_observation_plan }
    }
}

impl From<native_schedule::ResultInFlightAcquireAttemptPlan> for NativeResultInFlightAcquireAttemptPlan {
    fn from(acquire_attempt_plan: native_schedule::ResultInFlightAcquireAttemptPlan) -> Self {
        Self { inner: acquire_attempt_plan }
    }
}

impl From<native_schedule::ResultInFlightAcquireObservationPlan> for NativeResultInFlightAcquireObservationPlan {
    fn from(acquire_observation_plan: native_schedule::ResultInFlightAcquireObservationPlan) -> Self {
        Self { inner: acquire_observation_plan }
    }
}

impl From<native_schedule::ResultInFlightReleaseAttemptPlan> for NativeResultInFlightReleaseAttemptPlan {
    fn from(release_attempt_plan: native_schedule::ResultInFlightReleaseAttemptPlan) -> Self {
        Self { inner: release_attempt_plan }
    }
}

impl From<native_schedule::ResultInFlightReleaseObservationPlan> for NativeResultInFlightReleaseObservationPlan {
    fn from(release_observation_plan: native_schedule::ResultInFlightReleaseObservationPlan) -> Self {
        Self { inner: release_observation_plan }
    }
}

impl From<native_schedule::ResultWriteItemResourceReleasePlan> for NativeResultWriteItemResourceReleasePlan {
    fn from(resource_release_plan: native_schedule::ResultWriteItemResourceReleasePlan) -> Self {
        Self { inner: resource_release_plan }
    }
}

impl From<native_schedule::ResultWriteHandoffPlan> for NativeResultWriteHandoffPlan {
    fn from(handoff_plan: native_schedule::ResultWriteHandoffPlan) -> Self {
        Self { inner: handoff_plan }
    }
}

impl From<native_schedule::ResultWriteDrainCompletionPlan> for NativeResultWriteDrainCompletionPlan {
    fn from(drain_completion_plan: native_schedule::ResultWriteDrainCompletionPlan) -> Self {
        Self { inner: drain_completion_plan }
    }
}

impl From<native_schedule::ResultWriteItemDispatchPlan> for NativeResultWriteItemDispatchPlan {
    fn from(dispatch_plan: native_schedule::ResultWriteItemDispatchPlan) -> Self {
        Self { inner: dispatch_plan }
    }
}

impl From<native_schedule::DosageWorkDrainCompletionPlan> for NativeDosageWorkDrainCompletionPlan {
    fn from(drain_completion_plan: native_schedule::DosageWorkDrainCompletionPlan) -> Self {
        Self { inner: drain_completion_plan }
    }
}

impl From<native_schedule::DosageWorkItemDispatchPlan> for NativeDosageWorkItemDispatchPlan {
    fn from(dispatch_plan: native_schedule::DosageWorkItemDispatchPlan) -> Self {
        Self { inner: dispatch_plan }
    }
}

impl From<native_schedule::DosageWorkItemStageDurationPlan> for NativeDosageWorkItemStageDurationPlan {
    fn from(stage_duration_plan: native_schedule::DosageWorkItemStageDurationPlan) -> Self {
        Self { inner: stage_duration_plan }
    }
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn intersect_committed_chunk_identifier_sets(
    committed_chunk_identifier_sets: Vec<Vec<usize>>,
) -> Vec<usize> {
    let native_committed_chunk_identifier_sets: Vec<BTreeSet<usize>> = committed_chunk_identifier_sets
        .into_iter()
        .map(|chunk_identifiers| chunk_identifiers.into_iter().collect())
        .collect();
    native_schedule::intersect_committed_chunk_identifier_sets(&native_committed_chunk_identifier_sets)
        .into_iter()
        .collect()
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn resolve_manifest_gpu_genotype_format(
    resume: bool,
    manifest_gpu_genotype_format: Option<String>,
    association_backend_genotype_format: Option<String>,
) -> Option<String> {
    native_schedule::resolve_manifest_gpu_genotype_format(
        resume,
        manifest_gpu_genotype_format.as_deref(),
        association_backend_genotype_format.as_deref(),
    )
    .map(str::to_string)
}

#[pyfunction]
pub(crate) fn resolve_effective_trusted_no_missing_diploid(
    requested_trusted_no_missing_diploid: bool,
    variant_major_packed8_probability_pairs: bool,
) -> bool {
    native_schedule::resolve_effective_trusted_no_missing_diploid(
        requested_trusted_no_missing_diploid,
        variant_major_packed8_probability_pairs,
    )
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn plan_gpu_genotype_format_auto_to_dosage(
    requested_gpu_genotype_format: String,
    resolution_reason: String,
) -> PyResult<NativeGpuGenotypeFormatResolutionPlan> {
    native_schedule::plan_gpu_genotype_format_auto_to_dosage(&requested_gpu_genotype_format, &resolution_reason)
        .map(Into::into)
        .map_err(|error| schedule_error_to_py(&error))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn plan_single_trait_binary_gpu_genotype_format_resolution(
    requested_gpu_genotype_format: String,
    manifest_gpu_genotype_format: Option<String>,
    association_backend_genotype_format: Option<String>,
    resume: bool,
    jax_device: String,
) -> PyResult<NativeGpuGenotypeFormatResolutionPlan> {
    native_schedule::plan_single_trait_binary_gpu_genotype_format_resolution(
        &requested_gpu_genotype_format,
        manifest_gpu_genotype_format.as_deref(),
        association_backend_genotype_format.as_deref(),
        resume,
        &jax_device,
    )
    .map(Into::into)
    .map_err(|error| schedule_error_to_py(&error))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn plan_auto_gpu_genotype_format_after_trusted_validation(
    fallback_error: Option<String>,
) -> NativeGpuGenotypeFormatResolutionPlan {
    native_schedule::plan_auto_gpu_genotype_format_after_trusted_validation(fallback_error.as_deref()).into()
}

#[pyfunction]
pub(crate) fn resolve_delivery_callback_batch_size(
    callback_batch_size: Option<i64>,
    variant_major_packed8_probability_pairs: bool,
) -> PyResult<usize> {
    native_schedule::resolve_delivery_callback_batch_size(callback_batch_size, variant_major_packed8_probability_pairs)
        .map_err(|error| schedule_error_to_py(&error))
}

#[pyfunction]
pub(crate) fn resolve_grouped_union_callback_batch_size(native_callback_batch_size: i64) -> PyResult<usize> {
    native_schedule::resolve_grouped_union_callback_batch_size(native_callback_batch_size)
        .map_err(|error| schedule_error_to_py(&error))
}

#[pyfunction]
pub(crate) fn resolve_native_callback_queue_limits(
    staging_depth: i64,
    native_callback_batch_size: i64,
    result_in_flight_limit: Option<i64>,
    dosage_buffer_limit: Option<i64>,
) -> PyResult<NativeCallbackQueueLimits> {
    native_schedule::resolve_native_callback_queue_limits(
        staging_depth,
        native_callback_batch_size,
        result_in_flight_limit,
        dosage_buffer_limit,
    )
    .map(Into::into)
    .map_err(|error| schedule_error_to_py(&error))
}

#[pyfunction]
pub(crate) fn resolve_native_callback_worker_shutdown_timeouts() -> NativeCallbackWorkerShutdownTimeouts {
    native_schedule::callback_worker_shutdown_timeouts().into()
}

#[pyfunction]
pub(crate) fn plan_dosage_callback_worker_join(
    timeout_seconds: Option<f64>,
    has_started: bool,
) -> NativeCallbackWorkerJoinPlan {
    native_schedule::plan_dosage_callback_worker_join(timeout_seconds, has_started).into()
}

#[pyfunction]
pub(crate) fn plan_result_callback_worker_join(
    timeout_seconds: Option<f64>,
    has_started: bool,
) -> NativeCallbackWorkerJoinPlan {
    native_schedule::plan_result_callback_worker_join(timeout_seconds, has_started).into()
}

#[pyfunction]
pub(crate) fn plan_dosage_callback_worker_stop(
    timeout_seconds: Option<f64>,
    has_started: bool,
    has_worker_error: bool,
    is_worker_alive: bool,
) -> NativeCallbackWorkerStopPlan {
    native_schedule::plan_dosage_callback_worker_stop(timeout_seconds, has_started, has_worker_error, is_worker_alive)
        .into()
}

#[pyfunction]
pub(crate) fn plan_result_callback_worker_stop(
    timeout_seconds: Option<f64>,
    has_started: bool,
    has_worker_error: bool,
    is_worker_alive: bool,
) -> NativeCallbackWorkerStopPlan {
    native_schedule::plan_result_callback_worker_stop(timeout_seconds, has_started, has_worker_error, is_worker_alive)
        .into()
}

#[pyfunction]
pub(crate) fn plan_callback_worker_finish() -> NativeCallbackWorkerFinishPlan {
    native_schedule::plan_callback_worker_finish().into()
}

#[pyfunction]
pub(crate) fn plan_callback_worker_abort() -> NativeCallbackWorkerAbortPlan {
    native_schedule::plan_callback_worker_abort().into()
}

#[pyfunction]
pub(crate) fn plan_callback_worker_start(has_started: bool) -> NativeCallbackWorkerStartPlan {
    native_schedule::plan_callback_worker_start(has_started).into()
}

#[pyfunction]
pub(crate) fn plan_callback_worker_stop_poll(
    remaining_timeout_seconds: f64,
    has_started: bool,
    has_worker_error: bool,
    is_worker_alive: bool,
) -> NativeCallbackWorkerStopPollPlan {
    native_schedule::plan_callback_worker_stop_poll(
        remaining_timeout_seconds,
        has_started,
        has_worker_error,
        is_worker_alive,
    )
    .into()
}

#[pyfunction]
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn format_dosage_callback_worker_error_message(error_message: String) -> String {
    native_schedule::format_dosage_callback_worker_error_message(&error_message)
}

#[pyfunction]
#[must_use]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn format_result_callback_worker_error_message(error_message: String) -> String {
    native_schedule::format_result_callback_worker_error_message(&error_message)
}

#[pyfunction]
pub(crate) fn resolve_callback_worker_backpressure_poll_timeout_seconds() -> f64 {
    native_schedule::callback_worker_backpressure_poll_timeout_seconds()
}

#[pyfunction]
pub(crate) fn resolve_callback_worker_stop_poll_timeout_seconds(remaining_timeout_seconds: f64) -> f64 {
    native_schedule::resolve_callback_worker_stop_poll_timeout_seconds(remaining_timeout_seconds)
}

#[pyfunction]
pub(crate) fn should_attempt_callback_worker_stop(
    has_started: bool,
    has_worker_error: bool,
    is_worker_alive: bool,
) -> bool {
    native_schedule::should_attempt_callback_worker_stop(has_started, has_worker_error, is_worker_alive)
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn plan_dosage_buffer_reuse(
    buffered_shape: Vec<usize>,
    expected_shape: Vec<usize>,
) -> Option<NativeDosageBufferReusePlan> {
    native_schedule::plan_dosage_buffer_reuse(&buffered_shape, &expected_shape).map(Into::into)
}

#[pyfunction]
pub(crate) fn plan_variant_major_dosage_batch_handoff(
    metadata_count: usize,
    genotype_matrix_by_variant_count: usize,
    chunk_stats_count: usize,
) -> PyResult<NativeVariantMajorDosageBatchHandoffPlan> {
    native_schedule::plan_variant_major_dosage_batch_handoff(
        metadata_count,
        genotype_matrix_by_variant_count,
        chunk_stats_count,
    )
    .map(Into::into)
    .map_err(|error| schedule_error_to_py(&error))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn plan_multi_trait_chunk_write(
    writer_session_count: usize,
    chunk_identifier: usize,
    committed_chunk_identifier_sets: Vec<Vec<usize>>,
) -> PyResult<NativeMultiTraitChunkWritePlan> {
    let native_committed_chunk_identifier_sets: Vec<BTreeSet<usize>> = committed_chunk_identifier_sets
        .into_iter()
        .map(|chunk_identifiers| chunk_identifiers.into_iter().collect())
        .collect();
    native_schedule::plan_multi_trait_chunk_write(
        writer_session_count,
        chunk_identifier,
        &native_committed_chunk_identifier_sets,
    )
    .map(Into::into)
    .map_err(|error| schedule_error_to_py(&error))
}

#[pyfunction]
pub(crate) fn resolve_bgen_delivery_method_value(
    variant_major_packed8_probability_pairs: bool,
    has_native_multi_aligned_sample_data: bool,
    has_native_aligned_sample_data: bool,
) -> String {
    native_schedule::resolve_bgen_delivery_method(
        variant_major_packed8_probability_pairs,
        has_native_multi_aligned_sample_data,
        has_native_aligned_sample_data,
    )
    .as_value()
    .to_string()
}

#[pyfunction]
pub(crate) fn resolve_writer_finish_thread_count(
    writer_session_count: i64,
    requested_thread_count: i64,
) -> PyResult<usize> {
    native_schedule::resolve_writer_finish_thread_count(writer_session_count, requested_thread_count)
        .map_err(|error| schedule_error_to_py(&error))
}

#[pyfunction]
pub(crate) fn plan_writer_finish_execution(
    writer_session_count: i64,
    requested_thread_count: i64,
) -> PyResult<NativeWriterFinishExecutionPlan> {
    native_schedule::plan_writer_finish_execution(writer_session_count, requested_thread_count)
        .map(Into::into)
        .map_err(|error| schedule_error_to_py(&error))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn plan_bgen_delivery_cleanup(
    cleanup_outcome: String,
    callback_finished: bool,
) -> PyResult<NativeBgenDeliveryCleanupPlan> {
    native_schedule::plan_bgen_delivery_cleanup(&cleanup_outcome, callback_finished)
        .map(Into::into)
        .map_err(|error| schedule_error_to_py(&error))
}

#[pyfunction]
pub(crate) fn plan_bgen_delivery_invocation(
    callback_batch_size: Option<i64>,
    variant_major_packed8_probability_pairs: bool,
    has_native_multi_aligned_sample_data: bool,
    has_native_aligned_sample_data: bool,
) -> PyResult<NativeBgenDeliveryInvocationPlan> {
    native_schedule::plan_bgen_delivery_invocation(
        callback_batch_size,
        variant_major_packed8_probability_pairs,
        has_native_multi_aligned_sample_data,
        has_native_aligned_sample_data,
    )
    .map(Into::into)
    .map_err(|error| schedule_error_to_py(&error))
}

#[pyfunction]
pub(crate) fn plan_dosage_work_handoff(chunk_count: usize) -> PyResult<NativeDosageWorkHandoffPlan> {
    native_schedule::plan_dosage_work_handoff(chunk_count).map(Into::into).map_err(|error| schedule_error_to_py(&error))
}

#[pyfunction]
pub(crate) fn plan_result_write_handoff(has_result_work_item: bool) -> NativeResultWriteHandoffPlan {
    native_schedule::plan_result_write_handoff(has_result_work_item).into()
}

#[pyfunction]
pub(crate) fn plan_result_write_item_dispatch(
    result_work_item_kind: &str,
    expected_result_work_item_kind: &str,
) -> PyResult<NativeResultWriteItemDispatchPlan> {
    native_schedule::plan_result_write_item_dispatch(result_work_item_kind, expected_result_work_item_kind)
        .map(Into::into)
        .map_err(|error| schedule_error_to_py(&error))
}

#[pyfunction]
pub(crate) fn plan_dosage_work_item_dispatch(
    dosage_work_item_kind: &str,
) -> PyResult<NativeDosageWorkItemDispatchPlan> {
    native_schedule::plan_dosage_work_item_dispatch(dosage_work_item_kind)
        .map(Into::into)
        .map_err(|error| schedule_error_to_py(&error))
}

#[pyfunction]
pub(crate) fn plan_dosage_work_item_stage_duration(
    dosage_work_item_kind: &str,
    chunk_count: usize,
    elapsed_seconds: f64,
) -> PyResult<NativeDosageWorkItemStageDurationPlan> {
    native_schedule::plan_dosage_work_item_stage_duration(dosage_work_item_kind, chunk_count, elapsed_seconds)
        .map(Into::into)
        .map_err(|error| schedule_error_to_py(&error))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn plan_single_trait_output_write(
    is_native_writer_session: bool,
    output_statistic_dtype: String,
) -> PyResult<NativeSingleTraitOutputWritePlan> {
    native_schedule::plan_single_trait_output_write(is_native_writer_session, &output_statistic_dtype)
        .map(Into::into)
        .map_err(|error| schedule_error_to_py(&error))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn plan_multi_trait_output_write(
    active_trait_count: usize,
    all_writer_sessions_native: bool,
    output_statistic_dtype: String,
) -> PyResult<NativeMultiTraitOutputWritePlan> {
    native_schedule::plan_multi_trait_output_write(
        active_trait_count,
        all_writer_sessions_native,
        &output_statistic_dtype,
    )
    .map(Into::into)
    .map_err(|error| schedule_error_to_py(&error))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn plan_callback_queue_operation_observation(
    queue_name: String,
    operation_name: String,
    elapsed_seconds: f64,
    blocked: bool,
) -> PyResult<NativeCallbackQueueOperationObservationPlan> {
    native_schedule::plan_callback_queue_operation_observation(&queue_name, &operation_name, elapsed_seconds, blocked)
        .map(Into::into)
        .map_err(|error| schedule_error_to_py(&error))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn plan_callback_queue_backpressure_observation(
    queue_name: String,
    operation_name: String,
    queue_depth: usize,
    queue_capacity: usize,
    elapsed_seconds: f64,
    blocked: bool,
) -> PyResult<NativeCallbackQueueBackpressureObservation> {
    native_schedule::plan_callback_queue_backpressure_observation(
        &queue_name,
        &operation_name,
        queue_depth,
        queue_capacity,
        elapsed_seconds,
        blocked,
    )
    .map(Into::into)
    .map_err(|error| schedule_error_to_py(&error))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn plan_callback_queue_stage_observation(
    queue_name: String,
    operation_name: String,
    elapsed_seconds: f64,
    blocked: bool,
) -> PyResult<NativeCallbackQueueStageObservationPlan> {
    native_schedule::plan_callback_queue_stage_observation(&queue_name, &operation_name, elapsed_seconds, blocked)
        .map(Into::into)
        .map_err(|error| schedule_error_to_py(&error))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn plan_callback_queue_stage_backpressure_observation(
    queue_name: String,
    operation_name: String,
    queue_depth: usize,
    queue_capacity: usize,
    elapsed_seconds: f64,
    blocked: bool,
) -> PyResult<NativeCallbackQueueStageBackpressureObservation> {
    native_schedule::plan_callback_queue_stage_backpressure_observation(
        &queue_name,
        &operation_name,
        queue_depth,
        queue_capacity,
        elapsed_seconds,
        blocked,
    )
    .map(Into::into)
    .map_err(|error| schedule_error_to_py(&error))
}

fn schedule_error_to_py(error: &native_schedule::ScheduleError) -> PyErr {
    PyValueError::new_err(error.to_string())
}
