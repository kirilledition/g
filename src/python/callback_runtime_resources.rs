//! PyO3 owner for callback runtime native resources.

use std::sync::Mutex;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use super::callback_progress::NativeCallbackProgressState;
use super::callback_queue::{NativeCallbackObjectQueue, NativeCallbackWaitSignal, NativeCallbackWorkerThread};
use super::callback_summary::NativeBinaryCorrectionSummary;
use super::schedule::{NativeCallbackSchedulerState, NativeCallbackWorkerStartAttemptPlan};

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
}
