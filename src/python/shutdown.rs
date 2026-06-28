//! PyO3 adapters for deterministic graceful-shutdown signal helpers.

use std::sync::{Mutex, MutexGuard};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use g_runtime::shutdown as native_shutdown;

#[pyclass]
pub(crate) struct NativeShutdownController {
    controller: Mutex<native_shutdown::ShutdownController>,
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
    #[allow(clippy::needless_pass_by_value)]
    fn new(handled_signal_numbers: Vec<i32>) -> PyResult<Self> {
        Ok(Self {
            controller: Mutex::new(
                native_shutdown::ShutdownController::new(&handled_signal_numbers).map_err(PyValueError::new_err)?,
            ),
        })
    }

    fn reset(&self) -> PyResult<()> {
        self.lock_controller()?.reset();
        Ok(())
    }

    #[getter]
    fn handlers_installed(&self) -> PyResult<bool> {
        Ok(self.lock_controller()?.handlers_installed())
    }

    fn requested_signal_payload<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let controller = self.lock_controller()?;
        controller.requested_signal().map(|payload| shutdown_signal_payload_to_dict(py, payload)).transpose()
    }

    fn request_shutdown_payload<'py>(&self, py: Python<'py>, signal_number: i32) -> PyResult<Bound<'py, PyDict>> {
        let decision = self.lock_controller()?.request_shutdown(signal_number).map_err(PyValueError::new_err)?;
        let python_payload = PyDict::new(py);
        python_payload.set_item("action", decision.action.as_str())?;
        python_payload.set_item("signal", shutdown_signal_payload_to_dict(py, &decision.signal)?)?;
        Ok(python_payload)
    }

    fn handler_install_plan_payload<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let plan = self.lock_controller()?.begin_handler_install();
        let python_payload = PyDict::new(py);
        python_payload.set_item("handled_signals", shutdown_signal_payloads_to_tuple(py, &plan.handled_signals)?)?;
        Ok(python_payload)
    }

    fn mark_handlers_installed(&self) -> PyResult<()> {
        self.lock_controller()?.mark_handlers_installed();
        Ok(())
    }

    fn handler_restore_plan_payload<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let plan = self.lock_controller()?.plan_handler_restore();
        let python_payload = PyDict::new(py);
        python_payload.set_item("should_restore", plan.should_restore)?;
        python_payload.set_item("handled_signals", shutdown_signal_payloads_to_tuple(py, &plan.handled_signals)?)?;
        Ok(python_payload)
    }

    fn mark_handlers_restored(&self) -> PyResult<()> {
        self.lock_controller()?.mark_handlers_restored();
        Ok(())
    }
}

impl NativeShutdownController {
    fn lock_controller(&self) -> PyResult<MutexGuard<'_, native_shutdown::ShutdownController>> {
        self.controller.lock().map_err(|_| PyValueError::new_err("Shutdown controller mutex was poisoned."))
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

fn shutdown_signal_payloads_to_tuple<'py>(
    py: Python<'py>,
    payloads: &[native_shutdown::ShutdownSignalPayload],
) -> PyResult<Bound<'py, PyTuple>> {
    let python_payloads =
        payloads.iter().map(|payload| shutdown_signal_payload_to_dict(py, payload)).collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(py, &python_payloads)
}
