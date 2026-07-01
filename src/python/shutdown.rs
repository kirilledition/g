//! PyO3 adapters for deterministic graceful-shutdown signal helpers.

use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard};

use pyo3::exceptions::{PyKeyboardInterrupt, PyRuntimeError, PySystemExit, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule, PyTuple};

use g_runtime::shutdown as native_shutdown;

#[pyclass]
pub(crate) struct NativeShutdownController {
    controller: Mutex<native_shutdown::ShutdownController>,
    previous_handlers: Mutex<BTreeMap<i32, Py<PyAny>>>,
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
            previous_handlers: Mutex::new(BTreeMap::new()),
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
        shutdown_request_decision_payload_to_dict(py, &decision)
    }

    fn request_shutdown_signal_or_raise_second_signal_payload<'py>(
        &self,
        py: Python<'py>,
        signal_number: i32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let decision = self.lock_controller()?.request_shutdown(signal_number).map_err(PyValueError::new_err)?;
        if decision.action == native_shutdown::ShutdownRequestAction::Force {
            self.restore_python_signal_handlers(py)?;
            return raise_second_signal_exception_from_plan(signal_number);
        }
        shutdown_signal_payload_to_dict(py, &decision.signal)
    }

    fn handler_install_plan_payload<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let plan = self.lock_controller()?.begin_handler_install();
        self.lock_previous_handlers()?.clear();
        let python_payload = PyDict::new(py);
        python_payload.set_item("handled_signals", shutdown_signal_payloads_to_tuple(py, &plan.handled_signals)?)?;
        Ok(python_payload)
    }

    fn mark_handlers_installed(&self) -> PyResult<()> {
        self.lock_controller()?.mark_handlers_installed();
        Ok(())
    }

    fn install_python_signal_handlers(&self, py: Python<'_>, handler: &Bound<'_, PyAny>) -> PyResult<()> {
        let plan = self.lock_controller()?.begin_handler_install();
        let signal_module = py.import("signal")?;
        let signal_class = signal_module.getattr("Signals")?;
        let mut previous_handlers = self.lock_previous_handlers()?;
        previous_handlers.clear();
        for signal_payload in &plan.handled_signals {
            let python_signal = signal_class.call1((signal_payload.number,))?;
            let previous_handler = signal_module.call_method1("getsignal", (&python_signal,))?;
            signal_module.call_method1("signal", (&python_signal, handler))?;
            previous_handlers.insert(signal_payload.number, previous_handler.unbind());
        }
        drop(previous_handlers);
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
        self.lock_previous_handlers()?.clear();
        Ok(())
    }

    fn restore_python_signal_handlers(&self, py: Python<'_>) -> PyResult<bool> {
        let plan = self.lock_controller()?.plan_handler_restore();
        if !plan.should_restore {
            return Ok(false);
        }
        let signal_module = py.import("signal")?;
        let signal_class = signal_module.getattr("Signals")?;
        let mut previous_handlers = self.lock_previous_handlers()?;
        for signal_payload in &plan.handled_signals {
            let python_signal = signal_class.call1((signal_payload.number,))?;
            let Some(previous_handler) = previous_handlers.get(&signal_payload.number) else {
                return Err(PyRuntimeError::new_err(format!("missing previous handler for {}", signal_payload.name)));
            };
            signal_module.call_method1("signal", (&python_signal, previous_handler.bind(py)))?;
        }
        previous_handlers.clear();
        drop(previous_handlers);
        self.lock_controller()?.mark_handlers_restored();
        Ok(true)
    }

    fn restore_python_signal_handlers_and_reset(&self, py: Python<'_>) -> PyResult<bool> {
        let restored_handlers = self.restore_python_signal_handlers(py)?;
        self.lock_controller()?.finish_handler_session();
        Ok(restored_handlers)
    }
}

impl NativeShutdownController {
    fn lock_controller(&self) -> PyResult<MutexGuard<'_, native_shutdown::ShutdownController>> {
        self.controller.lock().map_err(|_| PyValueError::new_err("Shutdown controller mutex was poisoned."))
    }

    fn lock_previous_handlers(&self) -> PyResult<MutexGuard<'_, BTreeMap<i32, Py<PyAny>>>> {
        self.previous_handlers
            .lock()
            .map_err(|_| PyValueError::new_err("Previous shutdown handlers mutex was poisoned."))
    }
}

#[pyfunction]
pub(crate) fn build_shutdown_signal_payload<'py>(py: Python<'py>, signal_number: i32) -> PyResult<Bound<'py, PyDict>> {
    let payload = native_shutdown::build_shutdown_signal(signal_number).map_err(PyValueError::new_err)?;
    shutdown_signal_payload_to_dict(py, &payload)
}

#[pyfunction]
pub(crate) fn default_shutdown_signal_numbers() -> Vec<i32> {
    native_shutdown::default_shutdown_signal_numbers()
}

#[pyfunction]
pub(crate) fn plan_second_signal_exception(signal_number: i32) -> PyResult<NativeSecondSignalExceptionPlan> {
    let plan = native_shutdown::plan_second_signal_exception(signal_number).map_err(PyValueError::new_err)?;
    Ok(NativeSecondSignalExceptionPlan { inner: plan })
}

#[pyfunction]
pub(crate) fn raise_second_signal_exception(signal_number: i32) -> PyResult<()> {
    raise_second_signal_exception_from_plan(signal_number)
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeSecondSignalExceptionPlan>()?;
    module.add_class::<NativeShutdownController>()?;
    module.add_function(wrap_pyfunction!(build_shutdown_signal_payload, module)?)?;
    module.add_function(wrap_pyfunction!(default_shutdown_signal_numbers, module)?)?;
    module.add_function(wrap_pyfunction!(plan_second_signal_exception, module)?)?;
    module.add_function(wrap_pyfunction!(raise_second_signal_exception, module)?)?;
    Ok(())
}

fn raise_second_signal_exception_from_plan<T>(signal_number: i32) -> PyResult<T> {
    let plan = native_shutdown::plan_second_signal_exception(signal_number).map_err(PyValueError::new_err)?;
    if plan.raise_keyboard_interrupt {
        return Err(PyKeyboardInterrupt::new_err(()));
    }
    Err(PySystemExit::new_err(plan.exit_code))
}

fn shutdown_request_decision_payload_to_dict<'py>(
    py: Python<'py>,
    decision: &native_shutdown::ShutdownRequestDecisionPayload,
) -> PyResult<Bound<'py, PyDict>> {
    let python_payload = PyDict::new(py);
    python_payload.set_item("action", decision.action.as_str())?;
    python_payload.set_item("signal", shutdown_signal_payload_to_dict(py, &decision.signal)?)?;
    Ok(python_payload)
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
