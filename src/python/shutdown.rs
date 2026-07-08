//! PyO3 adapters for deterministic graceful-shutdown signal helpers.

use std::sync::{Mutex, MutexGuard};

use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyKeyboardInterrupt, PyRuntimeError, PySystemExit, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};

use g_runtime::debug as native_shutdown;

#[pyclass]
pub(crate) struct NativeShutdownController {
    session: Mutex<native_shutdown::ShutdownHandlerSession<Py<PyAny>>>,
}

#[pyclass(skip_from_py_object)]
#[derive(Clone, Eq, PartialEq)]
pub(crate) struct NativeShutdownSignal {
    number: i32,
    name: String,
    exit_code: i32,
}

#[pymethods]
impl NativeShutdownController {
    #[new]
    #[pyo3(signature = (handled_signal_numbers=None))]
    #[allow(clippy::needless_pass_by_value)]
    fn new(handled_signal_numbers: Option<Vec<i32>>) -> PyResult<Self> {
        let resolved_signal_numbers =
            handled_signal_numbers.unwrap_or_else(native_shutdown::default_shutdown_signal_numbers);
        Ok(Self {
            session: Mutex::new(
                native_shutdown::ShutdownHandlerSession::new(&resolved_signal_numbers)
                    .map_err(|error| PyValueError::new_err(error.to_string()))?,
            ),
        })
    }

    fn reset(&self) -> PyResult<()> {
        self.lock_session()?.reset();
        Ok(())
    }

    #[getter]
    fn handlers_installed(&self) -> PyResult<bool> {
        Ok(self.lock_session()?.handlers_installed())
    }

    fn requested_signal(&self) -> PyResult<Option<NativeShutdownSignal>> {
        let session = self.lock_session()?;
        Ok(session.requested_signal().map(NativeShutdownSignal::from_native_signal))
    }

    fn request_shutdown_signal_or_raise_second_signal(
        &self,
        py: Python<'_>,
        signal_number: i32,
    ) -> PyResult<NativeShutdownSignal> {
        let decision = self
            .lock_session()?
            .request_shutdown(signal_number)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        if decision.action == native_shutdown::ShutdownRequestAction::Force {
            self.restore_python_signal_handlers(py)?;
            return raise_second_signal_exception_from_plan(signal_number);
        }
        Ok(NativeShutdownSignal::from_native_signal(&decision.signal))
    }

    fn mark_handlers_installed(&self) -> PyResult<()> {
        self.lock_session()?.mark_handlers_installed();
        Ok(())
    }

    fn install_python_signal_handlers(&self, py: Python<'_>, handler: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut session = self.lock_session()?;
        let plan = session.begin_handler_install();
        let signal_module = py.import("signal")?;
        let signal_class = signal_module.getattr("Signals")?;
        for signal_payload in &plan.handled_signals {
            let python_signal = signal_class.call1((signal_payload.number,))?;
            let previous_handler = signal_module.call_method1("getsignal", (&python_signal,))?;
            signal_module.call_method1("signal", (&python_signal, handler))?;
            session.record_previous_handler(signal_payload.number, previous_handler.unbind());
        }
        session.mark_handlers_installed();
        Ok(())
    }

    fn mark_handlers_restored(&self) -> PyResult<()> {
        self.lock_session()?.mark_handlers_restored();
        Ok(())
    }

    fn restore_python_signal_handlers(&self, py: Python<'_>) -> PyResult<bool> {
        let mut session = self.lock_session()?;
        let plan = session.plan_handler_restore();
        if !plan.should_restore {
            return Ok(false);
        }
        let signal_module = py.import("signal")?;
        let signal_class = signal_module.getattr("Signals")?;
        for signal_payload in &plan.handled_signals {
            let python_signal = signal_class.call1((signal_payload.number,))?;
            let Some(previous_handler) = session.previous_handler(signal_payload.number) else {
                return Err(PyRuntimeError::new_err(format!("missing previous handler for {}", signal_payload.name)));
            };
            signal_module.call_method1("signal", (&python_signal, previous_handler.bind(py)))?;
        }
        session.mark_handlers_restored();
        Ok(true)
    }

    fn restore_python_signal_handlers_and_reset(&self, py: Python<'_>) -> PyResult<bool> {
        let restored_handlers = self.restore_python_signal_handlers(py)?;
        self.lock_session()?.finish_handler_session();
        Ok(restored_handlers)
    }
}

impl NativeShutdownController {
    fn lock_session(&self) -> PyResult<MutexGuard<'_, native_shutdown::ShutdownHandlerSession<Py<PyAny>>>> {
        self.session.lock().map_err(|_| PyValueError::new_err("Shutdown handler session mutex was poisoned."))
    }
}

impl NativeShutdownSignal {
    fn from_native_signal(signal_payload: &native_shutdown::ShutdownSignalPayload) -> Self {
        Self { number: signal_payload.number, name: signal_payload.name.clone(), exit_code: signal_payload.exit_code }
    }
}

#[pymethods]
impl NativeShutdownSignal {
    #[new]
    fn new(number: i32, name: String, exit_code: i32) -> Self {
        Self { number, name, exit_code }
    }

    #[getter]
    fn number(&self) -> i32 {
        self.number
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    #[getter]
    fn exit_code(&self) -> i32 {
        self.exit_code
    }

    #[expect(clippy::needless_pass_by_value, reason = "PyO3 __richcmp__ requires owned PyRef extraction.")]
    fn __richcmp__(&self, other: PyRef<'_, Self>, operation: CompareOp) -> bool {
        match operation {
            CompareOp::Eq => self == &*other,
            CompareOp::Ne => self != &*other,
            CompareOp::Lt | CompareOp::Le | CompareOp::Gt | CompareOp::Ge => false,
        }
    }
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeShutdownController>()?;
    module.add_class::<NativeShutdownSignal>()?;
    Ok(())
}

fn raise_second_signal_exception_from_plan<T>(signal_number: i32) -> PyResult<T> {
    let plan = native_shutdown::plan_second_signal_exception(signal_number)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    if plan.raise_keyboard_interrupt {
        return Err(PyKeyboardInterrupt::new_err(()));
    }
    Err(PySystemExit::new_err(plan.exit_code))
}
