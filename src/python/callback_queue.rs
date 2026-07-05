//! PyO3 adapters for native callback queue primitives.

use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule};

#[pyclass]
pub(crate) struct NativeCallbackObjectQueue {
    queue: Mutex<g_engine::BoundedCallbackQueue<Py<PyAny>>>,
    condition: Condvar,
}

#[pyclass]
pub(crate) struct NativeCallbackObjectQueueGetResult {
    item: Option<Py<PyAny>>,
}

#[pyclass]
pub(crate) struct NativeCallbackWaitSignal {
    generation: Mutex<u64>,
    condition: Condvar,
}

#[pyclass]
pub(crate) struct NativeCallbackWorkerThread {
    thread: Py<PyAny>,
    name: String,
}

#[pymethods]
impl NativeCallbackObjectQueue {
    #[new]
    fn new(capacity: usize) -> PyResult<Self> {
        Self::with_capacity(capacity)
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
    pub(crate) fn with_capacity(capacity: usize) -> PyResult<Self> {
        let Some(queue) = g_engine::BoundedCallbackQueue::new(capacity) else {
            return Err(PyValueError::new_err("native callback object queue capacity must be positive"));
        };
        Ok(Self { queue: Mutex::new(queue), condition: Condvar::new() })
    }

    pub(crate) fn put_item(&self, py: Python<'_>, item: Py<PyAny>, timeout_seconds: f64) -> PyResult<bool> {
        py.detach(|| self.put_without_gil(item, timeout_seconds))
    }

    pub(crate) fn get_item(
        &self,
        py: Python<'_>,
        timeout_seconds: f64,
    ) -> PyResult<NativeCallbackObjectQueueGetResult> {
        py.detach(|| self.get_without_gil(timeout_seconds))
    }

    pub(crate) fn wait_for_available_slot_value(&self, py: Python<'_>, timeout_seconds: f64) -> PyResult<bool> {
        py.detach(|| {
            self.wait_until_without_gil(
                timeout_seconds,
                g_engine::BoundedCallbackQueue::has_available_slot,
                "native callback object queue lock was poisoned during available-slot wait",
            )
        })
    }

    pub(crate) fn wait_for_queued_item_value(&self, py: Python<'_>, timeout_seconds: f64) -> PyResult<bool> {
        py.detach(|| {
            self.wait_until_without_gil(
                timeout_seconds,
                g_engine::BoundedCallbackQueue::has_queued_item,
                "native callback object queue lock was poisoned during queued-item wait",
            )
        })
    }

    pub(crate) fn has_queued_item_value(&self) -> PyResult<bool> {
        Ok(self.lock_queue()?.has_queued_item())
    }

    pub(crate) fn occupied_count_value(&self) -> PyResult<usize> {
        Ok(self.lock_queue()?.occupied_count())
    }

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
impl NativeCallbackWaitSignal {
    #[new]
    fn new() -> Self {
        Self::new_signal()
    }

    #[getter]
    fn generation(&self) -> PyResult<u64> {
        Ok(*self.lock_generation()?)
    }

    fn notify_waiters(&self) -> PyResult<u64> {
        self.notify_waiters_value()
    }

    fn wait_for_change(&self, py: Python<'_>, observed_generation: u64, timeout_seconds: f64) -> PyResult<bool> {
        self.wait_for_change_value(py, observed_generation, timeout_seconds)
    }
}

impl NativeCallbackWaitSignal {
    pub(crate) fn new_signal() -> Self {
        Self { generation: Mutex::new(0), condition: Condvar::new() }
    }

    pub(crate) fn generation_value(&self) -> PyResult<u64> {
        Ok(*self.lock_generation()?)
    }

    pub(crate) fn notify_waiters_value(&self) -> PyResult<u64> {
        let mut generation = self.lock_generation()?;
        *generation = generation.wrapping_add(1);
        let next_generation = *generation;
        self.condition.notify_all();
        Ok(next_generation)
    }

    pub(crate) fn wait_for_change_value(
        &self,
        py: Python<'_>,
        observed_generation: u64,
        timeout_seconds: f64,
    ) -> PyResult<bool> {
        py.detach(|| self.wait_for_change_without_gil(observed_generation, timeout_seconds))
    }

    fn lock_generation(&self) -> PyResult<MutexGuard<'_, u64>> {
        self.generation.lock().map_err(|_| PyRuntimeError::new_err("native callback wait signal lock was poisoned"))
    }

    fn wait_for_change_without_gil(&self, observed_generation: u64, timeout_seconds: f64) -> PyResult<bool> {
        let timeout_duration = normalize_timeout_duration(timeout_seconds);
        let deadline = Instant::now().checked_add(timeout_duration);
        let mut generation = self.lock_generation()?;

        loop {
            if *generation != observed_generation {
                return Ok(true);
            }
            if timeout_duration.is_zero() {
                return Ok(false);
            }
            generation = if let Some(deadline) = deadline {
                let remaining_timeout = deadline.saturating_duration_since(Instant::now());
                if remaining_timeout.is_zero() {
                    return Ok(false);
                }
                let (next_generation, _) =
                    self.condition.wait_timeout(generation, remaining_timeout).map_err(|_| {
                        PyRuntimeError::new_err("native callback wait signal lock was poisoned during wait")
                    })?;
                next_generation
            } else {
                self.condition
                    .wait(generation)
                    .map_err(|_| PyRuntimeError::new_err("native callback wait signal lock was poisoned during wait"))?
            };
        }
    }
}

#[pymethods]
impl NativeCallbackWorkerThread {
    #[new]
    #[pyo3(signature = (*, target, name, daemon = true))]
    fn new(py: Python<'_>, target: &Bound<'_, PyAny>, name: String, daemon: bool) -> PyResult<Self> {
        Self::from_target(py, target, name, daemon)
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    fn start(&self, py: Python<'_>) -> PyResult<()> {
        self.start_thread(py)
    }

    #[pyo3(signature = (timeout = None))]
    fn join(&self, py: Python<'_>, timeout: Option<f64>) -> PyResult<()> {
        self.join_thread(py, timeout)
    }

    fn is_alive(&self, py: Python<'_>) -> PyResult<bool> {
        self.is_thread_alive(py)
    }
}

impl NativeCallbackWorkerThread {
    pub(crate) fn from_target(py: Python<'_>, target: &Bound<'_, PyAny>, name: String, daemon: bool) -> PyResult<Self> {
        let threading_module = PyModule::import(py, "threading")?;
        let keyword_arguments = PyDict::new(py);
        keyword_arguments.set_item("target", target)?;
        keyword_arguments.set_item("name", name.as_str())?;
        keyword_arguments.set_item("daemon", daemon)?;
        let thread = threading_module.getattr("Thread")?.call((), Some(&keyword_arguments))?.unbind();
        Ok(Self { thread, name })
    }

    pub(crate) fn start_thread(&self, py: Python<'_>) -> PyResult<()> {
        self.thread.bind(py).call_method0("start")?;
        Ok(())
    }

    pub(crate) fn name_value(&self) -> &str {
        &self.name
    }

    pub(crate) fn join_thread(&self, py: Python<'_>, timeout: Option<f64>) -> PyResult<()> {
        match timeout {
            Some(timeout_seconds) => {
                let keyword_arguments = PyDict::new(py);
                keyword_arguments.set_item("timeout", timeout_seconds)?;
                self.thread.bind(py).call_method("join", (), Some(&keyword_arguments))?;
            }
            None => {
                self.thread.bind(py).call_method0("join")?;
            }
        }
        Ok(())
    }

    pub(crate) fn is_thread_alive(&self, py: Python<'_>) -> PyResult<bool> {
        self.thread.bind(py).call_method0("is_alive")?.extract()
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

impl NativeCallbackObjectQueueGetResult {
    pub(crate) fn has_item_value(&self) -> bool {
        self.item.is_some()
    }

    pub(crate) fn into_item_value(self) -> Option<Py<PyAny>> {
        self.item
    }
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = module;
    Ok(())
}

fn normalize_timeout_duration(timeout_seconds: f64) -> Duration {
    if timeout_seconds.is_finite() && timeout_seconds > 0.0 {
        Duration::try_from_secs_f64(timeout_seconds).unwrap_or(Duration::MAX)
    } else {
        Duration::ZERO
    }
}
