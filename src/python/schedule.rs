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

#[pyclass]
pub(crate) struct NativeVariantMajorDosageBatchHandoffPlan {
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
pub(crate) struct NativeDosageBufferPoolState {
    inner: native_schedule::DosageBufferPoolState,
}

#[pyclass]
pub(crate) struct NativeResultInFlightSlotState {
    inner: native_schedule::ResultInFlightSlotState,
}

#[pyclass]
pub(crate) struct NativeCallbackWorkerLifecycleState {
    inner: native_schedule::CallbackWorkerLifecycleState,
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

#[pyclass]
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
impl NativeCallbackSchedulerState {
    #[new]
    fn new(
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

impl From<native_schedule::CallbackWorkerShutdownTimeouts> for NativeCallbackWorkerShutdownTimeouts {
    fn from(worker_shutdown_timeouts: native_schedule::CallbackWorkerShutdownTimeouts) -> Self {
        Self { inner: worker_shutdown_timeouts }
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

impl From<native_schedule::CallbackQueueStageObservationPlan> for NativeCallbackQueueStageObservationPlan {
    fn from(observation_plan: native_schedule::CallbackQueueStageObservationPlan) -> Self {
        Self { inner: observation_plan }
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

fn schedule_error_to_py(error: &native_schedule::ScheduleError) -> PyErr {
    PyValueError::new_err(error.to_string())
}
