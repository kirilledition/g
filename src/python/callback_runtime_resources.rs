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
    NativeCallbackSchedulerState, NativeCallbackWorkerStartAttemptPlan, NativeDosageWorkDrainCompletionPlan,
    NativeDosageWorkItemDispatchPlan, NativeResultWriteDrainCompletionPlan, NativeResultWriteHandoffPlan,
    NativeResultWriteItemDispatchPlan,
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
