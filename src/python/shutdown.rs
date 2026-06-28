//! PyO3 adapters for deterministic graceful-shutdown signal helpers.

use std::sync::{Mutex, MutexGuard};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use g_runtime::shutdown as native_shutdown;

#[pyclass]
pub(crate) struct NativeShutdownController {
    state: Mutex<native_shutdown::ShutdownControllerState>,
}

#[pyclass]
pub(crate) struct NativeSecondSignalExceptionPlan {
    inner: native_shutdown::SecondSignalExceptionPlan,
}

#[pymethods]
impl NativeSecondSignalExceptionPlan {
    #[getter]
    fn raise_keyboard_interrupt(&self) -> bool {
        self.inner.raise_keyboard_interrupt
    }

    #[getter]
    fn exit_code(&self) -> i32 {
        self.inner.exit_code
    }
}

#[pymethods]
impl NativeShutdownController {
    #[new]
    fn new() -> Self {
        Self { state: Mutex::new(native_shutdown::ShutdownControllerState::default()) }
    }

    fn reset(&self) -> PyResult<()> {
        self.lock_state()?.reset();
        Ok(())
    }

    fn requested_signal_payload<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let state = self.lock_state()?;
        state.requested_signal.as_ref().map(|payload| shutdown_signal_payload_to_dict(py, payload)).transpose()
    }

    fn request_shutdown_payload<'py>(&self, py: Python<'py>, signal_number: i32) -> PyResult<Bound<'py, PyDict>> {
        let decision = self.lock_state()?.request_shutdown(signal_number).map_err(PyValueError::new_err)?;
        let python_payload = PyDict::new(py);
        python_payload.set_item("action", decision.action.as_str())?;
        python_payload.set_item("signal", shutdown_signal_payload_to_dict(py, &decision.signal)?)?;
        Ok(python_payload)
    }
}

impl NativeShutdownController {
    fn lock_state(&self) -> PyResult<MutexGuard<'_, native_shutdown::ShutdownControllerState>> {
        self.state.lock().map_err(|_| PyValueError::new_err("Shutdown controller mutex was poisoned."))
    }
}

#[pyfunction]
pub(crate) fn build_shutdown_signal_payload<'py>(py: Python<'py>, signal_number: i32) -> PyResult<Bound<'py, PyDict>> {
    let payload = native_shutdown::build_shutdown_signal(signal_number).map_err(PyValueError::new_err)?;
    shutdown_signal_payload_to_dict(py, &payload)
}

#[pyfunction]
pub(crate) fn plan_second_signal_exception(signal_number: i32) -> PyResult<NativeSecondSignalExceptionPlan> {
    let plan = native_shutdown::plan_second_signal_exception(signal_number).map_err(PyValueError::new_err)?;
    Ok(NativeSecondSignalExceptionPlan { inner: plan })
}

fn shutdown_signal_payload_to_dict<'py>(
    py: Python<'py>,
    payload: &native_shutdown::ShutdownSignalPayload,
) -> PyResult<Bound<'py, PyDict>> {
    let python_payload = PyDict::new(py);
    python_payload.set_item("number", payload.number)?;
    python_payload.set_item("name", &payload.name)?;
    python_payload.set_item("exit_code", payload.exit_code)?;
    Ok(python_payload)
}
