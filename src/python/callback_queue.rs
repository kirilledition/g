//! PyO3 adapters for native callback queue primitives.

use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

#[pyclass]
pub(crate) struct NativeCallbackObjectQueue {
    queue: Mutex<g_engine::BoundedCallbackQueue<Py<PyAny>>>,
    condition: Condvar,
}

#[pyclass]
pub(crate) struct NativeCallbackObjectQueueGetResult {
    item: Option<Py<PyAny>>,
}

#[pymethods]
impl NativeCallbackObjectQueue {
    #[new]
    fn new(capacity: usize) -> PyResult<Self> {
        let Some(queue) = g_engine::BoundedCallbackQueue::new(capacity) else {
            return Err(PyValueError::new_err("native callback object queue capacity must be positive"));
        };
        Ok(Self { queue: Mutex::new(queue), condition: Condvar::new() })
    }

    #[getter]
    fn capacity(&self) -> PyResult<usize> {
        Ok(self.lock_queue()?.capacity())
    }

    #[getter]
    fn occupied_count(&self) -> PyResult<usize> {
        Ok(self.lock_queue()?.occupied_count())
    }

    #[getter]
    fn has_available_slot(&self) -> PyResult<bool> {
        Ok(self.lock_queue()?.has_available_slot())
    }

    #[getter]
    fn has_queued_item(&self) -> PyResult<bool> {
        Ok(self.lock_queue()?.has_queued_item())
    }

    fn put(&self, py: Python<'_>, item: Py<PyAny>, timeout_seconds: f64) -> PyResult<bool> {
        py.detach(|| self.put_without_gil(item, timeout_seconds))
    }

    fn get(&self, py: Python<'_>, timeout_seconds: f64) -> PyResult<NativeCallbackObjectQueueGetResult> {
        py.detach(|| self.get_without_gil(timeout_seconds))
    }

    fn wait_for_available_slot(&self, py: Python<'_>, timeout_seconds: f64) -> PyResult<bool> {
        py.detach(|| {
            self.wait_until_without_gil(
                timeout_seconds,
                g_engine::BoundedCallbackQueue::has_available_slot,
                "native callback object queue lock was poisoned during available-slot wait",
            )
        })
    }

    fn wait_for_queued_item(&self, py: Python<'_>, timeout_seconds: f64) -> PyResult<bool> {
        py.detach(|| {
            self.wait_until_without_gil(
                timeout_seconds,
                g_engine::BoundedCallbackQueue::has_queued_item,
                "native callback object queue lock was poisoned during queued-item wait",
            )
        })
    }
}

impl NativeCallbackObjectQueue {
    fn lock_queue(&self) -> PyResult<MutexGuard<'_, g_engine::BoundedCallbackQueue<Py<PyAny>>>> {
        self.queue.lock().map_err(|_| PyRuntimeError::new_err("native callback object queue lock was poisoned"))
    }

    fn put_without_gil(&self, item: Py<PyAny>, timeout_seconds: f64) -> PyResult<bool> {
        let timeout_duration = normalize_timeout_duration(timeout_seconds);
        let deadline = Instant::now().checked_add(timeout_duration);
        let mut pending_item = Some(item);
        let mut queue = self.lock_queue()?;

        loop {
            let item_to_queue = pending_item
                .take()
                .ok_or_else(|| PyRuntimeError::new_err("native callback object queue item was consumed"))?;
            match queue.try_push(item_to_queue) {
                Ok(()) => {
                    self.condition.notify_one();
                    return Ok(true);
                }
                Err(returned_item) => {
                    pending_item = Some(returned_item);
                }
            }

            if timeout_duration.is_zero() {
                return Ok(false);
            }
            queue = if let Some(deadline) = deadline {
                let remaining_timeout = deadline.saturating_duration_since(Instant::now());
                if remaining_timeout.is_zero() {
                    return Ok(false);
                }
                let (next_queue, _) = self.condition.wait_timeout(queue, remaining_timeout).map_err(|_| {
                    PyRuntimeError::new_err("native callback object queue lock was poisoned during put wait")
                })?;
                next_queue
            } else {
                self.condition.wait(queue).map_err(|_| {
                    PyRuntimeError::new_err("native callback object queue lock was poisoned during put wait")
                })?
            };
        }
    }

    fn get_without_gil(&self, timeout_seconds: f64) -> PyResult<NativeCallbackObjectQueueGetResult> {
        let timeout_duration = normalize_timeout_duration(timeout_seconds);
        let deadline = Instant::now().checked_add(timeout_duration);
        let mut queue = self.lock_queue()?;

        loop {
            if let Some(item) = queue.pop() {
                self.condition.notify_one();
                return Ok(NativeCallbackObjectQueueGetResult { item: Some(item) });
            }
            if timeout_duration.is_zero() {
                return Ok(NativeCallbackObjectQueueGetResult { item: None });
            }
            queue = if let Some(deadline) = deadline {
                let remaining_timeout = deadline.saturating_duration_since(Instant::now());
                if remaining_timeout.is_zero() {
                    return Ok(NativeCallbackObjectQueueGetResult { item: None });
                }
                let (next_queue, _) = self.condition.wait_timeout(queue, remaining_timeout).map_err(|_| {
                    PyRuntimeError::new_err("native callback object queue lock was poisoned during get wait")
                })?;
                next_queue
            } else {
                self.condition.wait(queue).map_err(|_| {
                    PyRuntimeError::new_err("native callback object queue lock was poisoned during get wait")
                })?
            };
        }
    }

    fn wait_until_without_gil(
        &self,
        timeout_seconds: f64,
        queue_condition: fn(&g_engine::BoundedCallbackQueue<Py<PyAny>>) -> bool,
        wait_error_message: &'static str,
    ) -> PyResult<bool> {
        let timeout_duration = normalize_timeout_duration(timeout_seconds);
        let deadline = Instant::now().checked_add(timeout_duration);
        let mut queue = self.lock_queue()?;

        loop {
            if queue_condition(&queue) {
                return Ok(true);
            }
            if timeout_duration.is_zero() {
                return Ok(false);
            }
            queue = if let Some(deadline) = deadline {
                let remaining_timeout = deadline.saturating_duration_since(Instant::now());
                if remaining_timeout.is_zero() {
                    return Ok(false);
                }
                let (next_queue, _) = self
                    .condition
                    .wait_timeout(queue, remaining_timeout)
                    .map_err(|_| PyRuntimeError::new_err(wait_error_message))?;
                next_queue
            } else {
                self.condition.wait(queue).map_err(|_| PyRuntimeError::new_err(wait_error_message))?
            };
        }
    }
}

#[pymethods]
impl NativeCallbackObjectQueueGetResult {
    #[getter]
    fn has_item(&self) -> bool {
        self.item.is_some()
    }

    #[getter]
    fn item(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.item.as_ref().map(|item| item.clone_ref(py))
    }
}

fn normalize_timeout_duration(timeout_seconds: f64) -> Duration {
    if timeout_seconds.is_finite() && timeout_seconds > 0.0 {
        Duration::try_from_secs_f64(timeout_seconds).unwrap_or(Duration::MAX)
    } else {
        Duration::ZERO
    }
}
