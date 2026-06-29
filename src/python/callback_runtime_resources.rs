//! PyO3 owner for callback runtime native resources.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use super::callback_progress::NativeCallbackProgressState;
use super::callback_queue::{
    NativeCallbackObjectQueue, NativeCallbackObjectQueueGetResult, NativeCallbackWaitSignal, NativeCallbackWorkerThread,
};
use super::callback_summary::NativeBinaryCorrectionSummary;
use super::schedule::{
    NativeCallbackQueueBackpressureObservation, NativeCallbackQueueGetObservationPlan,
    NativeCallbackQueuePutObservationPlan, NativeCallbackQueueStageBackpressureObservation,
    NativeCallbackSchedulerState, NativeCallbackWorkerAbortPlan, NativeCallbackWorkerErrorRaisePlan,
    NativeCallbackWorkerErrorUpdatePlan, NativeCallbackWorkerStartAttemptPlan, NativeDosageBufferPoolObservationPlan,
    NativeDosageBufferReturnAttemptPlan, NativeDosageBufferReusePlan, NativeDosageWorkDrainCompletionPlan,
    NativeDosageWorkHandoffPlan, NativeDosageWorkItemDispatchPlan, NativeDosageWorkItemStageDurationPlan,
    NativeResultInFlightAcquireObservationPlan, NativeResultInFlightReleaseObservationPlan,
    NativeResultWriteDrainCompletionPlan, NativeResultWriteHandoffPlan, NativeResultWriteItemDispatchPlan,
    NativeResultWriteItemResourceReleasePlan, NativeVariantMajorDosageBatchHandoffPlan,
};

#[pyclass]
pub(crate) struct NativeCallbackRuntimeResources {
    callback_scheduler_state: Py<NativeCallbackSchedulerState>,
    progress_state: Py<NativeCallbackProgressState>,
    result_in_flight_slot_signal: Py<NativeCallbackWaitSignal>,
    dosage_buffer_pool_signal: Py<NativeCallbackWaitSignal>,
    dosage_queue: Py<NativeCallbackObjectQueue>,
    result_queue: Py<NativeCallbackObjectQueue>,
    free_dosage_buffers: Py<NativeCallbackObjectQueue>,
    binary_correction_summary: Py<NativeBinaryCorrectionSummary>,
    worker_thread: Py<NativeCallbackWorkerThread>,
    result_worker_thread: Py<NativeCallbackWorkerThread>,
    worker_start_lock: Mutex<()>,
}

#[pyclass]
pub(crate) struct NativeDosageBufferAcquireResult {
    dosage_buffer: Option<Py<PyAny>>,
    should_allocate: bool,
    free_buffer_count: usize,
    waited: bool,
}

#[pyclass]
pub(crate) struct NativeCallbackWorkerFinishLifecycleResult {
    shutdown_worker_name: Option<String>,
    shutdown_timeout_seconds: Option<f64>,
    raise_worker_error: bool,
    complete_progress: bool,
    emit_binary_correction_summary: bool,
}

#[pyclass]
pub(crate) struct NativeResultWorkItemResourceReleaseResult {
    released_host_buffer: bool,
    free_buffer_count: Option<usize>,
    released_result_in_flight_slot: bool,
    result_in_flight_resource_name: Option<String>,
    result_in_flight_operation_name: Option<String>,
    result_in_flight_blocked: Option<bool>,
}

#[pymethods]
impl NativeCallbackRuntimeResources {
    #[new]
    #[pyo3(signature = (
        *,
        worker_name,
        dosage_worker_target,
        result_worker_target,
        staging_depth,
        native_callback_batch_size,
        result_in_flight_limit = None,
        dosage_buffer_limit = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        worker_name: String,
        dosage_worker_target: &Bound<'_, PyAny>,
        result_worker_target: &Bound<'_, PyAny>,
        staging_depth: i64,
        native_callback_batch_size: i64,
        result_in_flight_limit: Option<i64>,
        dosage_buffer_limit: Option<i64>,
    ) -> PyResult<Self> {
        let callback_scheduler_state = NativeCallbackSchedulerState::from_limits(
            staging_depth,
            native_callback_batch_size,
            result_in_flight_limit,
            dosage_buffer_limit,
        )?;
        let dosage_queue_depth = callback_scheduler_state.dosage_queue_depth_value();
        let result_queue_depth = callback_scheduler_state.result_queue_depth_value();
        let dosage_buffer_limit = callback_scheduler_state.dosage_buffer_limit_value();
        let result_worker_name = format!("{worker_name}-writer");

        Ok(Self {
            callback_scheduler_state: Py::new(py, callback_scheduler_state)?,
            progress_state: Py::new(py, NativeCallbackProgressState::new_state())?,
            result_in_flight_slot_signal: Py::new(py, NativeCallbackWaitSignal::new_signal())?,
            dosage_buffer_pool_signal: Py::new(py, NativeCallbackWaitSignal::new_signal())?,
            dosage_queue: Py::new(py, NativeCallbackObjectQueue::with_capacity(dosage_queue_depth)?)?,
            result_queue: Py::new(py, NativeCallbackObjectQueue::with_capacity(result_queue_depth)?)?,
            free_dosage_buffers: Py::new(py, NativeCallbackObjectQueue::with_capacity(dosage_buffer_limit)?)?,
            binary_correction_summary: Py::new(py, NativeBinaryCorrectionSummary::new_summary())?,
            worker_thread: Py::new(
                py,
                NativeCallbackWorkerThread::from_target(py, dosage_worker_target, worker_name, true)?,
            )?,
            result_worker_thread: Py::new(
                py,
                NativeCallbackWorkerThread::from_target(py, result_worker_target, result_worker_name, true)?,
            )?,
            worker_start_lock: Mutex::new(()),
        })
    }

    #[getter]
    fn callback_scheduler_state(&self, py: Python<'_>) -> Py<NativeCallbackSchedulerState> {
        self.callback_scheduler_state.clone_ref(py)
    }

    #[getter]
    fn progress_state(&self, py: Python<'_>) -> Py<NativeCallbackProgressState> {
        self.progress_state.clone_ref(py)
    }

    #[getter]
    fn result_in_flight_slot_signal(&self, py: Python<'_>) -> Py<NativeCallbackWaitSignal> {
        self.result_in_flight_slot_signal.clone_ref(py)
    }

    #[getter]
    fn dosage_buffer_pool_signal(&self, py: Python<'_>) -> Py<NativeCallbackWaitSignal> {
        self.dosage_buffer_pool_signal.clone_ref(py)
    }

    #[getter]
    fn dosage_queue(&self, py: Python<'_>) -> Py<NativeCallbackObjectQueue> {
        self.dosage_queue.clone_ref(py)
    }

    #[getter]
    fn result_queue(&self, py: Python<'_>) -> Py<NativeCallbackObjectQueue> {
        self.result_queue.clone_ref(py)
    }

    #[getter]
    fn free_dosage_buffers(&self, py: Python<'_>) -> Py<NativeCallbackObjectQueue> {
        self.free_dosage_buffers.clone_ref(py)
    }

    #[getter]
    fn binary_correction_summary(&self, py: Python<'_>) -> Py<NativeBinaryCorrectionSummary> {
        self.binary_correction_summary.clone_ref(py)
    }

    #[getter]
    fn worker_thread(&self, py: Python<'_>) -> Py<NativeCallbackWorkerThread> {
        self.worker_thread.clone_ref(py)
    }

    #[getter]
    fn result_worker_thread(&self, py: Python<'_>) -> Py<NativeCallbackWorkerThread> {
        self.result_worker_thread.clone_ref(py)
    }

    #[getter]
    fn has_started(&self, py: Python<'_>) -> bool {
        self.callback_scheduler_state.bind(py).borrow().has_started_value()
    }

    #[getter]
    fn native_callback_batch_size(&self, py: Python<'_>) -> usize {
        self.callback_scheduler_state.bind(py).borrow().native_callback_batch_size_value()
    }

    #[getter]
    fn dosage_queue_depth(&self, py: Python<'_>) -> usize {
        self.callback_scheduler_state.bind(py).borrow().dosage_queue_depth_value()
    }

    #[getter]
    fn result_queue_depth(&self, py: Python<'_>) -> usize {
        self.callback_scheduler_state.bind(py).borrow().result_queue_depth_value()
    }

    #[getter]
    fn result_in_flight_limit(&self, py: Python<'_>) -> usize {
        self.callback_scheduler_state.bind(py).borrow().result_in_flight_limit_value()
    }

    #[getter]
    fn dosage_buffer_limit(&self, py: Python<'_>) -> usize {
        self.callback_scheduler_state.bind(py).borrow().dosage_buffer_limit_value()
    }

    #[getter]
    fn dosage_queue_occupied_count(&self, py: Python<'_>) -> usize {
        self.callback_scheduler_state.bind(py).borrow().dosage_queue_occupied_count_value()
    }

    #[getter]
    fn result_queue_occupied_count(&self, py: Python<'_>) -> usize {
        self.callback_scheduler_state.bind(py).borrow().result_queue_occupied_count_value()
    }

    #[getter]
    fn result_in_flight_occupied_count(&self, py: Python<'_>) -> usize {
        self.callback_scheduler_state.bind(py).borrow().result_in_flight_occupied_count_value()
    }

    #[getter]
    fn dosage_buffer_allocated_count(&self, py: Python<'_>) -> usize {
        self.callback_scheduler_state.bind(py).borrow().dosage_buffer_allocated_count_value()
    }

    #[getter]
    fn dosage_buffer_identifiers(&self, py: Python<'_>) -> Vec<usize> {
        self.callback_scheduler_state.bind(py).borrow().dosage_buffer_identifiers_value()
    }

    fn start_workers(&self, py: Python<'_>) -> PyResult<NativeCallbackWorkerStartAttemptPlan> {
        let _start_guard = self.worker_start_lock.lock().map_err(|_| {
            PyRuntimeError::new_err("native callback worker start lock was poisoned during worker startup")
        })?;
        let start_attempt_plan = {
            let mut scheduler_state = self.callback_scheduler_state.bind(py).borrow_mut();
            scheduler_state.plan_worker_start_attempt_value()
        };
        if start_attempt_plan.has_start_error_value() {
            return Ok(start_attempt_plan);
        }
        if start_attempt_plan.should_start_result_worker() {
            self.result_worker_thread.bind(py).borrow().start_thread(py)?;
        }
        if start_attempt_plan.should_start_dosage_worker() {
            self.worker_thread.bind(py).borrow().start_thread(py)?;
        }
        Ok(start_attempt_plan)
    }

    fn stop_dosage_worker(&self, py: Python<'_>, timeout_seconds: Option<f64>) -> PyResult<Option<f64>> {
        let is_worker_alive = self.worker_thread.bind(py).borrow().is_thread_alive(py)?;
        let stop_plan = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            scheduler_state.plan_dosage_worker_stop_value(timeout_seconds, is_worker_alive)
        };
        if !stop_plan.should_stop_value() {
            return Ok(None);
        }
        let stop_deadline = Instant::now() + normalize_timeout_duration(stop_plan.timeout_seconds_value());
        while Instant::now() < stop_deadline {
            let remaining_seconds = remaining_timeout_seconds(stop_deadline);
            let is_worker_alive = self.worker_thread.bind(py).borrow().is_thread_alive(py)?;
            let stop_poll_plan = {
                let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
                scheduler_state.plan_dosage_worker_stop_poll_value(remaining_seconds, is_worker_alive)
            };
            if !stop_poll_plan.should_stop_value() {
                return Ok(None);
            }
            let stop_signal = py.None();
            if self.try_put_dosage_work_item(py, stop_signal.bind(py), stop_poll_plan.poll_timeout_seconds_value())? {
                return Ok(None);
            }
        }
        Ok(Some(stop_plan.timeout_seconds_value()))
    }

    fn join_dosage_worker(&self, py: Python<'_>, timeout_seconds: Option<f64>) -> PyResult<Option<f64>> {
        let join_plan = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            scheduler_state.plan_dosage_worker_join_value(timeout_seconds)
        };
        if !join_plan.should_join_value() {
            return Ok(None);
        }
        self.worker_thread.bind(py).borrow().join_thread(py, Some(join_plan.timeout_seconds_value()))?;
        if self.worker_thread.bind(py).borrow().is_thread_alive(py)? {
            return Ok(Some(join_plan.timeout_seconds_value()));
        }
        Ok(None)
    }

    fn stop_result_worker(&self, py: Python<'_>, timeout_seconds: Option<f64>) -> PyResult<Option<f64>> {
        let is_worker_alive = self.result_worker_thread.bind(py).borrow().is_thread_alive(py)?;
        let stop_plan = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            scheduler_state.plan_result_worker_stop_value(timeout_seconds, is_worker_alive)
        };
        if !stop_plan.should_stop_value() {
            return Ok(None);
        }
        let stop_deadline = Instant::now() + normalize_timeout_duration(stop_plan.timeout_seconds_value());
        while Instant::now() < stop_deadline {
            let remaining_seconds = remaining_timeout_seconds(stop_deadline);
            let is_worker_alive = self.result_worker_thread.bind(py).borrow().is_thread_alive(py)?;
            let stop_poll_plan = {
                let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
                scheduler_state.plan_result_worker_stop_poll_value(remaining_seconds, is_worker_alive)
            };
            if !stop_poll_plan.should_stop_value() {
                return Ok(None);
            }
            let stop_signal = py.None();
            if self.try_put_result_write_item(py, stop_signal.bind(py), stop_poll_plan.poll_timeout_seconds_value())? {
                return Ok(None);
            }
        }
        Ok(Some(stop_plan.timeout_seconds_value()))
    }

    fn join_result_worker(&self, py: Python<'_>, timeout_seconds: Option<f64>) -> PyResult<Option<f64>> {
        let join_plan = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            scheduler_state.plan_result_worker_join_value(timeout_seconds)
        };
        if !join_plan.should_join_value() {
            return Ok(None);
        }
        self.result_worker_thread.bind(py).borrow().join_thread(py, Some(join_plan.timeout_seconds_value()))?;
        if self.result_worker_thread.bind(py).borrow().is_thread_alive(py)? {
            return Ok(Some(join_plan.timeout_seconds_value()));
        }
        Ok(None)
    }

    fn finish_worker_lifecycle(&self, py: Python<'_>) -> PyResult<NativeCallbackWorkerFinishLifecycleResult> {
        let finish_plan = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            scheduler_state.plan_worker_finish_value()
        };
        let mut finish_result = NativeCallbackWorkerFinishLifecycleResult::from_finish_plan(&finish_plan);
        if finish_plan.stop_dosage_worker_value() {
            let timeout_seconds = self.stop_dosage_worker(py, Some(finish_plan.dosage_stop_timeout_seconds_value()))?;
            if let Some(timeout_seconds) = timeout_seconds {
                finish_result.record_shutdown_timeout(
                    self.worker_thread.bind(py).borrow().name_value().to_owned(),
                    timeout_seconds,
                );
                return Ok(finish_result);
            }
        }
        if finish_plan.join_dosage_worker_value() {
            let timeout_seconds = self.join_dosage_worker(py, Some(finish_plan.dosage_join_timeout_seconds_value()))?;
            if let Some(timeout_seconds) = timeout_seconds {
                finish_result.record_shutdown_timeout(
                    self.worker_thread.bind(py).borrow().name_value().to_owned(),
                    timeout_seconds,
                );
                return Ok(finish_result);
            }
        }
        if finish_plan.stop_result_worker_value() {
            let timeout_seconds = self.stop_result_worker(py, Some(finish_plan.result_stop_timeout_seconds_value()))?;
            if let Some(timeout_seconds) = timeout_seconds {
                finish_result.record_shutdown_timeout(
                    self.result_worker_thread.bind(py).borrow().name_value().to_owned(),
                    timeout_seconds,
                );
                return Ok(finish_result);
            }
        }
        if finish_plan.join_result_worker_value() {
            let timeout_seconds = self.join_result_worker(py, Some(finish_plan.result_join_timeout_seconds_value()))?;
            if let Some(timeout_seconds) = timeout_seconds {
                finish_result.record_shutdown_timeout(
                    self.result_worker_thread.bind(py).borrow().name_value().to_owned(),
                    timeout_seconds,
                );
                return Ok(finish_result);
            }
        }
        Ok(finish_result)
    }

    fn abort_worker_lifecycle(&self, py: Python<'_>) -> PyResult<NativeCallbackWorkerAbortPlan> {
        let abort_plan = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            scheduler_state.plan_worker_abort_value()
        };
        if abort_plan.stop_dosage_worker_value() {
            let _ = self.stop_dosage_worker(py, Some(abort_plan.dosage_stop_timeout_seconds_value()))?;
        }
        if abort_plan.stop_result_worker_value() {
            let _ = self.stop_result_worker(py, Some(abort_plan.result_stop_timeout_seconds_value()))?;
        }
        Ok(abort_plan)
    }

    fn plan_worker_error_raise(&self, py: Python<'_>) -> NativeCallbackWorkerErrorRaisePlan {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_worker_error_raise_value()
    }

    fn update_dosage_worker_error(
        &self,
        py: Python<'_>,
        error_message: Option<&str>,
    ) -> NativeCallbackWorkerErrorUpdatePlan {
        let mut scheduler_state = self.callback_scheduler_state.bind(py).borrow_mut();
        scheduler_state.update_dosage_worker_error_value(error_message)
    }

    fn update_result_worker_error(
        &self,
        py: Python<'_>,
        error_message: Option<&str>,
    ) -> NativeCallbackWorkerErrorUpdatePlan {
        let mut scheduler_state = self.callback_scheduler_state.bind(py).borrow_mut();
        scheduler_state.update_result_worker_error_value(error_message)
    }

    fn acquire_result_in_flight_slot_with_backpressure_timeout(
        &self,
        py: Python<'_>,
    ) -> PyResult<NativeResultInFlightAcquireObservationPlan> {
        let observed_generation = self.result_in_flight_slot_signal.bind(py).borrow().generation_value()?;
        let (attempt_plan, observation_plan) = {
            let mut scheduler_state = self.callback_scheduler_state.bind(py).borrow_mut();
            let attempt_plan = scheduler_state.plan_result_in_flight_slot_acquire_backpressure_attempt_value();
            let observation_plan = scheduler_state.plan_result_in_flight_slot_acquire_observation_value(&attempt_plan);
            (attempt_plan, observation_plan)
        };
        if !attempt_plan.should_acquire_value() && attempt_plan.should_wait_value() {
            self.result_in_flight_slot_signal.bind(py).borrow().wait_for_change_value(
                py,
                observed_generation,
                attempt_plan.wait_timeout_seconds_value(),
            )?;
        }
        Ok(observation_plan)
    }

    fn release_result_in_flight_slot(&self, py: Python<'_>) -> PyResult<NativeResultInFlightReleaseObservationPlan> {
        let release_plan = {
            let mut scheduler_state = self.callback_scheduler_state.bind(py).borrow_mut();
            scheduler_state.plan_result_in_flight_slot_release_attempt_value()
        };
        if release_plan.has_release_error_value() {
            return Err(PyRuntimeError::new_err("Native result in-flight slot state has no occupied slot to release."));
        }
        self.result_in_flight_slot_signal.bind(py).borrow().notify_waiters_value()?;
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        Ok(scheduler_state.plan_result_in_flight_slot_release_observation_value())
    }

    fn release_result_work_item_pre_write_resources(
        &self,
        py: Python<'_>,
        host_dosage_buffer_identifier: Option<usize>,
        host_dosage_buffer: &Bound<'_, PyAny>,
    ) -> PyResult<NativeResultWorkItemResourceReleaseResult> {
        let has_host_dosage_buffer = !host_dosage_buffer.is_none();
        let resource_release_plan = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            scheduler_state.plan_result_write_item_pre_write_resource_release_value(has_host_dosage_buffer)
        };
        self.release_result_work_item_resources_with_plan(
            py,
            &resource_release_plan,
            host_dosage_buffer_identifier,
            host_dosage_buffer,
        )
    }

    fn release_result_work_item_final_resources(
        &self,
        py: Python<'_>,
        host_dosage_buffer_identifier: Option<usize>,
        host_dosage_buffer: &Bound<'_, PyAny>,
        has_released_host_dosage_buffer: bool,
        release_in_flight_slot: bool,
    ) -> PyResult<NativeResultWorkItemResourceReleaseResult> {
        let has_host_dosage_buffer = !host_dosage_buffer.is_none();
        let resource_release_plan = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            scheduler_state.plan_result_write_item_final_resource_release_value(
                has_host_dosage_buffer,
                has_released_host_dosage_buffer,
                release_in_flight_slot,
            )
        };
        self.release_result_work_item_resources_with_plan(
            py,
            &resource_release_plan,
            host_dosage_buffer_identifier,
            host_dosage_buffer,
        )
    }

    fn acquire_dosage_buffer_with_backpressure_timeout(
        &self,
        py: Python<'_>,
    ) -> PyResult<NativeDosageBufferAcquireResult> {
        let observed_generation = self.dosage_buffer_pool_signal.bind(py).borrow().generation_value()?;
        let free_buffer_count = self.free_dosage_buffers.bind(py).borrow().occupied_count_value()?;
        let acquire_plan = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            scheduler_state.plan_dosage_buffer_acquire_backpressure_attempt_value(free_buffer_count)
        };
        if acquire_plan.should_take_free_buffer_value() {
            let get_result = self.free_dosage_buffers.bind(py).borrow().get_item(py, 0.0)?;
            if !get_result.has_item_value() {
                return Err(PyRuntimeError::new_err(
                    "Native dosage-buffer free queue was empty after scheduler selected reuse.",
                ));
            }
            let free_buffer_count = self.free_dosage_buffers.bind(py).borrow().occupied_count_value()?;
            return Ok(NativeDosageBufferAcquireResult {
                dosage_buffer: get_result.into_item_value(),
                should_allocate: false,
                free_buffer_count,
                waited: false,
            });
        }
        if acquire_plan.should_allocate_value() {
            let free_buffer_count = self.free_dosage_buffers.bind(py).borrow().occupied_count_value()?;
            return Ok(NativeDosageBufferAcquireResult {
                dosage_buffer: None,
                should_allocate: true,
                free_buffer_count,
                waited: false,
            });
        }
        if acquire_plan.should_wait_value() {
            self.dosage_buffer_pool_signal.bind(py).borrow().wait_for_change_value(
                py,
                observed_generation,
                acquire_plan.wait_timeout_seconds_value(),
            )?;
            let free_buffer_count = self.free_dosage_buffers.bind(py).borrow().occupied_count_value()?;
            return Ok(NativeDosageBufferAcquireResult {
                dosage_buffer: None,
                should_allocate: false,
                free_buffer_count,
                waited: true,
            });
        }
        let free_buffer_count = self.free_dosage_buffers.bind(py).borrow().occupied_count_value()?;
        Ok(NativeDosageBufferAcquireResult {
            dosage_buffer: None,
            should_allocate: false,
            free_buffer_count,
            waited: false,
        })
    }

    fn register_dosage_buffer(&self, py: Python<'_>, buffer_identifier: usize) -> PyResult<usize> {
        let register_plan = {
            let mut scheduler_state = self.callback_scheduler_state.bind(py).borrow_mut();
            scheduler_state.plan_dosage_buffer_register_attempt_value(buffer_identifier)
        };
        if register_plan.has_registration_error_value() {
            return Err(PyRuntimeError::new_err("Native dosage-buffer pool has no available slot for allocation."));
        }
        self.free_dosage_buffers.bind(py).borrow().occupied_count_value()
    }

    fn return_dosage_buffer(
        &self,
        py: Python<'_>,
        buffer_identifier: usize,
        dosage_buffer: &Bound<'_, PyAny>,
    ) -> PyResult<Option<usize>> {
        let return_plan = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            scheduler_state.plan_dosage_buffer_return_attempt_value(buffer_identifier)
        };
        if !return_plan.should_return_value() {
            return Ok(None);
        }
        let queued = self.free_dosage_buffers.bind(py).borrow().put_item(py, dosage_buffer.clone().unbind(), 0.0)?;
        if !queued {
            return Err(PyRuntimeError::new_err("Native dosage-buffer free queue had no slot for returned buffer."));
        }
        let free_buffer_count = self.free_dosage_buffers.bind(py).borrow().occupied_count_value()?;
        self.dosage_buffer_pool_signal.bind(py).borrow().notify_waiters_value()?;
        Ok(Some(free_buffer_count))
    }

    fn discard_dosage_buffer(&self, py: Python<'_>, buffer_identifier: usize) -> PyResult<Option<usize>> {
        let discard_plan = {
            let mut scheduler_state = self.callback_scheduler_state.bind(py).borrow_mut();
            scheduler_state.plan_dosage_buffer_discard_attempt_value(buffer_identifier)
        };
        if !discard_plan.should_discard_value() {
            return Ok(None);
        }
        let free_buffer_count = self.free_dosage_buffers.bind(py).borrow().occupied_count_value()?;
        self.dosage_buffer_pool_signal.bind(py).borrow().notify_waiters_value()?;
        Ok(Some(free_buffer_count))
    }

    fn plan_dosage_buffer_return_attempt(
        &self,
        py: Python<'_>,
        buffer_identifier: usize,
    ) -> NativeDosageBufferReturnAttemptPlan {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_dosage_buffer_return_attempt_value(buffer_identifier)
    }

    #[allow(clippy::needless_pass_by_value)]
    fn plan_dosage_buffer_reuse(
        &self,
        py: Python<'_>,
        buffered_shape: Vec<usize>,
        expected_shape: Vec<usize>,
    ) -> Option<NativeDosageBufferReusePlan> {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_dosage_buffer_reuse_value(&buffered_shape, &expected_shape)
    }

    fn try_put_dosage_work_item(
        &self,
        py: Python<'_>,
        work_item: &Bound<'_, PyAny>,
        timeout_seconds: f64,
    ) -> PyResult<bool> {
        let deadline = Instant::now() + normalize_timeout_duration(timeout_seconds);
        loop {
            let attempt_plan = {
                let mut scheduler_state = self.callback_scheduler_state.bind(py).borrow_mut();
                scheduler_state.plan_dosage_queue_put_attempt_value(remaining_timeout_seconds(deadline))
            };
            if attempt_plan.should_put_value() {
                return self.put_dosage_work_item_after_slot_acquisition(py, work_item);
            }
            if !attempt_plan.should_wait_value() {
                return Ok(false);
            }
            self.dosage_queue
                .bind(py)
                .borrow()
                .wait_for_available_slot_value(py, attempt_plan.wait_timeout_seconds_value())?;
        }
    }

    fn try_put_dosage_work_item_with_backpressure_timeout(
        &self,
        py: Python<'_>,
        work_item: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let mut deadline = None;
        loop {
            let attempt_plan = {
                let mut scheduler_state = self.callback_scheduler_state.bind(py).borrow_mut();
                if let Some(deadline) = deadline {
                    scheduler_state.plan_dosage_queue_put_attempt_value(remaining_timeout_seconds(deadline))
                } else {
                    let attempt_plan = scheduler_state.plan_dosage_queue_put_backpressure_attempt_value();
                    if attempt_plan.should_wait_value() {
                        deadline = Some(
                            Instant::now() + normalize_timeout_duration(attempt_plan.wait_timeout_seconds_value()),
                        );
                    }
                    attempt_plan
                }
            };
            if attempt_plan.should_put_value() {
                return self.put_dosage_work_item_after_slot_acquisition(py, work_item);
            }
            if !attempt_plan.should_wait_value() {
                return Ok(false);
            }
            self.dosage_queue
                .bind(py)
                .borrow()
                .wait_for_available_slot_value(py, attempt_plan.wait_timeout_seconds_value())?;
        }
    }

    fn get_dosage_work_item(&self, py: Python<'_>) -> PyResult<NativeCallbackObjectQueueGetResult> {
        loop {
            let has_queued_item = self.dosage_queue.bind(py).borrow().has_queued_item_value()?;
            let get_plan = {
                let mut scheduler_state = self.callback_scheduler_state.bind(py).borrow_mut();
                scheduler_state.plan_dosage_queue_get_attempt_value(has_queued_item)
            };
            if get_plan.has_release_error_value() {
                return Err(PyRuntimeError::new_err("Native dosage-queue state has no occupied slot to release."));
            }
            if get_plan.should_get_value() {
                let get_result = self.dosage_queue.bind(py).borrow().get_item(py, 0.0)?;
                if !get_result.has_item_value() {
                    let reacquired_slot = {
                        let mut scheduler_state = self.callback_scheduler_state.bind(py).borrow_mut();
                        scheduler_state.acquire_dosage_queue_slot_value()
                    };
                    if !reacquired_slot {
                        return Err(PyRuntimeError::new_err(
                            "Native dosage queue storage was empty after scheduler slot release.",
                        ));
                    }
                    return Err(PyRuntimeError::new_err(
                        "Native dosage queue storage had no queued item after scheduler selected get.",
                    ));
                }
                return Ok(get_result);
            }
            if get_plan.should_wait_value() {
                self.dosage_queue
                    .bind(py)
                    .borrow()
                    .wait_for_queued_item_value(py, get_plan.wait_timeout_seconds_value())?;
            }
        }
    }

    fn plan_dosage_work_drain_completion(
        &self,
        py: Python<'_>,
        has_dosage_work_item: bool,
    ) -> NativeDosageWorkDrainCompletionPlan {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_dosage_work_drain_completion_value(has_dosage_work_item)
    }

    fn plan_validated_dosage_work_item_dispatch(
        &self,
        py: Python<'_>,
        dosage_work_item_kind: &str,
    ) -> PyResult<NativeDosageWorkItemDispatchPlan> {
        let dispatch_plan = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            scheduler_state.plan_dosage_work_item_dispatch_value(dosage_work_item_kind)?
        };
        if !dispatch_plan.has_dispatch_error_value() {
            return Ok(dispatch_plan);
        }
        let error_message = dispatch_plan
            .error_message_value()
            .unwrap_or("Native dosage work dispatch plan omitted the error message.");
        Err(PyRuntimeError::new_err(error_message.to_owned()))
    }

    fn plan_dosage_work_item_stage_duration(
        &self,
        py: Python<'_>,
        dosage_work_item_kind: &str,
        chunk_count: usize,
        elapsed_seconds: f64,
    ) -> PyResult<NativeDosageWorkItemStageDurationPlan> {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_dosage_work_item_stage_duration_value(dosage_work_item_kind, chunk_count, elapsed_seconds)
    }

    fn plan_current_queue_backpressure_observation(
        &self,
        py: Python<'_>,
        queue_name: &str,
        operation_name: &str,
        elapsed_seconds: f64,
        blocked: bool,
    ) -> PyResult<NativeCallbackQueueBackpressureObservation> {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_current_queue_backpressure_observation_value(
            queue_name,
            operation_name,
            elapsed_seconds,
            blocked,
        )
    }

    fn plan_current_queue_stage_backpressure_observation(
        &self,
        py: Python<'_>,
        queue_name: &str,
        operation_name: &str,
        elapsed_seconds: f64,
        blocked: bool,
    ) -> PyResult<NativeCallbackQueueStageBackpressureObservation> {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_current_queue_stage_backpressure_observation_value(
            queue_name,
            operation_name,
            elapsed_seconds,
            blocked,
        )
    }

    fn plan_dosage_queue_put_observation(&self, py: Python<'_>, queued: bool) -> NativeCallbackQueuePutObservationPlan {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_dosage_queue_put_observation_value(queued)
    }

    fn plan_dosage_queue_get_observation(&self, py: Python<'_>) -> NativeCallbackQueueGetObservationPlan {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_dosage_queue_get_observation_value()
    }

    fn plan_result_queue_put_observation(&self, py: Python<'_>, queued: bool) -> NativeCallbackQueuePutObservationPlan {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_result_queue_put_observation_value(queued)
    }

    fn plan_result_queue_get_observation(&self, py: Python<'_>) -> NativeCallbackQueueGetObservationPlan {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_result_queue_get_observation_value()
    }

    fn plan_dosage_buffer_pool_reuse_observation(&self, py: Python<'_>) -> NativeDosageBufferPoolObservationPlan {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_dosage_buffer_pool_reuse_observation_value()
    }

    fn plan_dosage_buffer_pool_return_observation(&self, py: Python<'_>) -> NativeDosageBufferPoolObservationPlan {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_dosage_buffer_pool_return_observation_value()
    }

    fn plan_dosage_buffer_pool_allocate_observation(&self, py: Python<'_>) -> NativeDosageBufferPoolObservationPlan {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_dosage_buffer_pool_allocate_observation_value()
    }

    fn plan_dosage_buffer_pool_discard_observation(&self, py: Python<'_>) -> NativeDosageBufferPoolObservationPlan {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_dosage_buffer_pool_discard_observation_value()
    }

    fn plan_dosage_buffer_pool_consumer_wait_observation(
        &self,
        py: Python<'_>,
    ) -> NativeDosageBufferPoolObservationPlan {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_dosage_buffer_pool_consumer_wait_observation_value()
    }

    fn plan_dosage_buffer_pool_backpressure_observation(
        &self,
        py: Python<'_>,
        operation_name: &str,
        free_buffer_count: usize,
        elapsed_seconds: f64,
        blocked: bool,
    ) -> PyResult<NativeCallbackQueueBackpressureObservation> {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_dosage_buffer_pool_backpressure_observation_value(
            operation_name,
            free_buffer_count,
            elapsed_seconds,
            blocked,
        )
    }

    fn plan_dosage_buffer_pool_stage_backpressure_observation(
        &self,
        py: Python<'_>,
        operation_name: &str,
        free_buffer_count: usize,
        elapsed_seconds: f64,
        blocked: bool,
    ) -> PyResult<NativeCallbackQueueStageBackpressureObservation> {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_dosage_buffer_pool_stage_backpressure_observation_value(
            operation_name,
            free_buffer_count,
            elapsed_seconds,
            blocked,
        )
    }

    fn plan_variant_major_dosage_batch_handoff(
        &self,
        py: Python<'_>,
        metadata_count: usize,
        genotype_matrix_by_variant_count: usize,
        chunk_stats_count: usize,
    ) -> PyResult<NativeVariantMajorDosageBatchHandoffPlan> {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_variant_major_dosage_batch_handoff_value(
            metadata_count,
            genotype_matrix_by_variant_count,
            chunk_stats_count,
        )
    }

    fn plan_dosage_work_handoff(&self, py: Python<'_>, chunk_count: usize) -> PyResult<NativeDosageWorkHandoffPlan> {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state.plan_dosage_work_handoff_value(chunk_count)
    }

    fn try_put_result_write_item(
        &self,
        py: Python<'_>,
        work_item: &Bound<'_, PyAny>,
        timeout_seconds: f64,
    ) -> PyResult<bool> {
        let handoff_plan = self.plan_result_write_handoff(py, work_item)?;
        let deadline = Instant::now() + normalize_timeout_duration(timeout_seconds);
        loop {
            let attempt_plan = {
                let mut scheduler_state = self.callback_scheduler_state.bind(py).borrow_mut();
                scheduler_state.plan_result_queue_put_attempt_value(remaining_timeout_seconds(deadline))
            };
            if attempt_plan.should_put_value() && handoff_plan.should_enqueue_value() {
                return self.put_result_write_item_after_slot_acquisition(py, work_item);
            }
            if !attempt_plan.should_wait_value() {
                return Ok(false);
            }
            self.result_queue
                .bind(py)
                .borrow()
                .wait_for_available_slot_value(py, attempt_plan.wait_timeout_seconds_value())?;
        }
    }

    fn try_put_result_write_item_with_backpressure_timeout(
        &self,
        py: Python<'_>,
        work_item: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let handoff_plan = self.plan_result_write_handoff(py, work_item)?;
        let mut deadline = None;
        loop {
            let attempt_plan = {
                let mut scheduler_state = self.callback_scheduler_state.bind(py).borrow_mut();
                if let Some(deadline) = deadline {
                    scheduler_state.plan_result_queue_put_attempt_value(remaining_timeout_seconds(deadline))
                } else {
                    let attempt_plan = scheduler_state.plan_result_queue_put_backpressure_attempt_value();
                    if attempt_plan.should_wait_value() {
                        deadline = Some(
                            Instant::now() + normalize_timeout_duration(attempt_plan.wait_timeout_seconds_value()),
                        );
                    }
                    attempt_plan
                }
            };
            if attempt_plan.should_put_value() && handoff_plan.should_enqueue_value() {
                return self.put_result_write_item_after_slot_acquisition(py, work_item);
            }
            if !attempt_plan.should_wait_value() {
                return Ok(false);
            }
            self.result_queue
                .bind(py)
                .borrow()
                .wait_for_available_slot_value(py, attempt_plan.wait_timeout_seconds_value())?;
        }
    }

    fn get_result_write_item(&self, py: Python<'_>) -> PyResult<NativeCallbackObjectQueueGetResult> {
        loop {
            let has_queued_item = self.result_queue.bind(py).borrow().has_queued_item_value()?;
            let get_plan = {
                let mut scheduler_state = self.callback_scheduler_state.bind(py).borrow_mut();
                scheduler_state.plan_result_queue_get_attempt_value(has_queued_item)
            };
            if get_plan.has_release_error_value() {
                return Err(PyRuntimeError::new_err("Native result-queue state has no occupied slot to release."));
            }
            if get_plan.should_get_value() {
                let get_result = self.result_queue.bind(py).borrow().get_item(py, 0.0)?;
                if !get_result.has_item_value() {
                    let reacquired_slot = {
                        let mut scheduler_state = self.callback_scheduler_state.bind(py).borrow_mut();
                        scheduler_state.acquire_result_queue_slot_value()
                    };
                    if !reacquired_slot {
                        return Err(PyRuntimeError::new_err(
                            "Native result queue storage was empty after scheduler slot release.",
                        ));
                    }
                    return Err(PyRuntimeError::new_err(
                        "Native result queue storage had no queued item after scheduler selected get.",
                    ));
                }
                return Ok(get_result);
            }
            if get_plan.should_wait_value() {
                self.result_queue
                    .bind(py)
                    .borrow()
                    .wait_for_queued_item_value(py, get_plan.wait_timeout_seconds_value())?;
            }
        }
    }

    fn plan_result_write_drain_completion(
        &self,
        py: Python<'_>,
        has_result_work_item: bool,
        flush_binary_correction_diagnostics_on_stop: bool,
    ) -> NativeResultWriteDrainCompletionPlan {
        let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
        scheduler_state
            .plan_result_write_drain_completion_value(has_result_work_item, flush_binary_correction_diagnostics_on_stop)
    }

    fn plan_validated_result_write_item_dispatch(
        &self,
        py: Python<'_>,
        result_work_item_kind: &str,
        expected_result_work_item_kind: &str,
    ) -> PyResult<NativeResultWriteItemDispatchPlan> {
        let dispatch_plan = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            scheduler_state
                .plan_result_write_item_dispatch_value(result_work_item_kind, expected_result_work_item_kind)?
        };
        if !dispatch_plan.has_dispatch_error_value() {
            return Ok(dispatch_plan);
        }
        let error_message = dispatch_plan
            .error_message_value()
            .unwrap_or("Native result write dispatch plan omitted the error message.");
        Err(PyRuntimeError::new_err(error_message.to_owned()))
    }
}

#[pymethods]
impl NativeResultWorkItemResourceReleaseResult {
    #[getter]
    fn released_host_buffer(&self) -> bool {
        self.released_host_buffer
    }

    #[getter]
    fn free_buffer_count(&self) -> Option<usize> {
        self.free_buffer_count
    }

    #[getter]
    fn released_result_in_flight_slot(&self) -> bool {
        self.released_result_in_flight_slot
    }

    #[getter]
    fn result_in_flight_resource_name(&self) -> Option<&str> {
        self.result_in_flight_resource_name.as_deref()
    }

    #[getter]
    fn result_in_flight_operation_name(&self) -> Option<&str> {
        self.result_in_flight_operation_name.as_deref()
    }

    #[getter]
    fn result_in_flight_blocked(&self) -> Option<bool> {
        self.result_in_flight_blocked
    }
}

impl NativeResultWorkItemResourceReleaseResult {
    fn empty() -> Self {
        Self {
            released_host_buffer: false,
            free_buffer_count: None,
            released_result_in_flight_slot: false,
            result_in_flight_resource_name: None,
            result_in_flight_operation_name: None,
            result_in_flight_blocked: None,
        }
    }

    fn record_result_in_flight_release(
        &mut self,
        release_observation_plan: &NativeResultInFlightReleaseObservationPlan,
    ) {
        self.released_result_in_flight_slot = true;
        self.result_in_flight_resource_name = Some(release_observation_plan.resource_name_value().to_owned());
        self.result_in_flight_operation_name = Some(release_observation_plan.operation_name_value().to_owned());
        self.result_in_flight_blocked = Some(release_observation_plan.blocked_value());
    }
}

#[pymethods]
impl NativeCallbackWorkerFinishLifecycleResult {
    #[getter]
    fn has_shutdown_timeout(&self) -> bool {
        self.shutdown_timeout_seconds.is_some()
    }

    #[getter]
    fn shutdown_worker_name(&self) -> Option<&str> {
        self.shutdown_worker_name.as_deref()
    }

    #[getter]
    fn shutdown_timeout_seconds(&self) -> Option<f64> {
        self.shutdown_timeout_seconds
    }

    #[getter]
    fn raise_worker_error(&self) -> bool {
        self.raise_worker_error
    }

    #[getter]
    fn complete_progress(&self) -> bool {
        self.complete_progress
    }

    #[getter]
    fn emit_binary_correction_summary(&self) -> bool {
        self.emit_binary_correction_summary
    }
}

impl NativeCallbackWorkerFinishLifecycleResult {
    fn from_finish_plan(finish_plan: &super::schedule::NativeCallbackWorkerFinishPlan) -> Self {
        Self {
            shutdown_worker_name: None,
            shutdown_timeout_seconds: None,
            raise_worker_error: finish_plan.raise_worker_error_value(),
            complete_progress: finish_plan.complete_progress_value(),
            emit_binary_correction_summary: finish_plan.emit_binary_correction_summary_value(),
        }
    }

    fn record_shutdown_timeout(&mut self, worker_name: String, timeout_seconds: f64) {
        self.shutdown_worker_name = Some(worker_name);
        self.shutdown_timeout_seconds = Some(timeout_seconds);
    }
}

#[pymethods]
impl NativeDosageBufferAcquireResult {
    #[getter]
    fn dosage_buffer(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.dosage_buffer.as_ref().map(|dosage_buffer| dosage_buffer.clone_ref(py))
    }

    #[getter]
    fn should_allocate(&self) -> bool {
        self.should_allocate
    }

    #[getter]
    fn free_buffer_count(&self) -> usize {
        self.free_buffer_count
    }

    #[getter]
    fn waited(&self) -> bool {
        self.waited
    }
}

impl NativeCallbackRuntimeResources {
    fn put_dosage_work_item_after_slot_acquisition(
        &self,
        py: Python<'_>,
        work_item: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let queued = self.dosage_queue.bind(py).borrow().put_item(py, work_item.clone().unbind(), 0.0)?;
        if queued {
            return Ok(true);
        }
        let released_slot = {
            let mut scheduler_state = self.callback_scheduler_state.bind(py).borrow_mut();
            scheduler_state.release_dosage_queue_slot_value()
        };
        if !released_slot {
            return Err(PyRuntimeError::new_err(
                "Native dosage queue storage rejected a put after scheduler slot acquisition.",
            ));
        }
        Err(PyRuntimeError::new_err("Native dosage queue storage had no slot after scheduler selected put."))
    }

    fn put_result_write_item_after_slot_acquisition(
        &self,
        py: Python<'_>,
        work_item: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let queued = self.result_queue.bind(py).borrow().put_item(py, work_item.clone().unbind(), 0.0)?;
        if queued {
            return Ok(true);
        }
        let released_slot = {
            let mut scheduler_state = self.callback_scheduler_state.bind(py).borrow_mut();
            scheduler_state.release_result_queue_slot_value()
        };
        if !released_slot {
            return Err(PyRuntimeError::new_err(
                "Native result queue storage rejected a put after scheduler slot acquisition.",
            ));
        }
        Err(PyRuntimeError::new_err("Native result queue storage had no slot after scheduler selected put."))
    }

    fn plan_result_write_handoff(
        &self,
        py: Python<'_>,
        work_item: &Bound<'_, PyAny>,
    ) -> PyResult<NativeResultWriteHandoffPlan> {
        let has_result_work_item = !work_item.is_none();
        let handoff_plan = {
            let scheduler_state = self.callback_scheduler_state.bind(py).borrow();
            scheduler_state.plan_result_write_handoff_value(has_result_work_item)
        };
        if handoff_plan.has_result_work_item_value() != has_result_work_item {
            return Err(PyRuntimeError::new_err(
                "Native result write handoff plan disagrees with the queued result item.",
            ));
        }
        Ok(handoff_plan)
    }

    fn release_result_work_item_resources_with_plan(
        &self,
        py: Python<'_>,
        resource_release_plan: &NativeResultWriteItemResourceReleasePlan,
        host_dosage_buffer_identifier: Option<usize>,
        host_dosage_buffer: &Bound<'_, PyAny>,
    ) -> PyResult<NativeResultWorkItemResourceReleaseResult> {
        let mut release_result = NativeResultWorkItemResourceReleaseResult::empty();
        if resource_release_plan.should_release_host_buffer_value() {
            if host_dosage_buffer.is_none() {
                return Err(PyRuntimeError::new_err(
                    "Native result work item resource release plan selected a missing host buffer.",
                ));
            }
            let Some(buffer_identifier) = host_dosage_buffer_identifier else {
                return Err(PyRuntimeError::new_err(
                    "Native result work item resource release plan selected a missing host buffer identifier.",
                ));
            };
            release_result.released_host_buffer = true;
            release_result.free_buffer_count = self.return_dosage_buffer(py, buffer_identifier, host_dosage_buffer)?;
        }
        if resource_release_plan.should_release_result_in_flight_slot_value() {
            let release_observation_plan = self.release_result_in_flight_slot(py)?;
            release_result.record_result_in_flight_release(&release_observation_plan);
        }
        Ok(release_result)
    }
}

fn normalize_timeout_duration(timeout_seconds: f64) -> Duration {
    if timeout_seconds.is_finite() && timeout_seconds > 0.0 {
        Duration::try_from_secs_f64(timeout_seconds).unwrap_or(Duration::MAX)
    } else {
        Duration::ZERO
    }
}

fn remaining_timeout_seconds(deadline: Instant) -> f64 {
    deadline.saturating_duration_since(Instant::now()).as_secs_f64()
}
