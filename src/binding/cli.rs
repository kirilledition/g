//! PyO3 adapter for the Rust-owned native CLI lifecycle.

use std::sync::Arc;

use g_runner as native_runner;
use pyo3::exceptions::{PyException, PyKeyboardInterrupt, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::binding::engine::{PyJaxBackend, create_jax_backend};
use crate::binding::{jax_runtime, logging};

pyo3::create_exception!(g, NativeSigtermRequested, PyException);
pyo3::create_exception!(g, NativeInterruptFlushed, PyException);

#[pyclass]
pub(crate) struct NativeCliRunResult {
    inner: native_runner::CliRunResult,
}

struct PythonRunHost;

impl native_runner::NativeRunHost for PythonRunHost {
    type Backend = PyJaxBackend;
    type Error = PyErr;

    fn install_python_logging(&mut self) -> PyResult<()> {
        Python::attach(logging::install_python_logging)
    }

    fn apply_jax_config_updates(&mut self, updates: &[native_runner::JaxRuntimeConfigUpdate<'_>]) -> PyResult<()> {
        Python::attach(|py| jax_runtime::apply_jax_config_updates(py, updates))
    }

    fn observe_jax_devices(&mut self) -> PyResult<Vec<native_runner::JaxDevice>> {
        Python::attach(jax_runtime::observe_jax_devices)
    }

    fn create_backend(&mut self, plan: native_runner::JaxAssociationBackendPlan<'_>) -> PyResult<Arc<Self::Backend>> {
        Python::attach(|py| create_jax_backend(py, plan).map(Arc::new))
    }

    #[allow(clippy::redundant_closure_for_method_calls)]
    fn check_interruption(&mut self) -> PyResult<()> {
        Python::attach(|py| py.check_signals())
    }

    fn sigterm_interruption_error(&mut self) -> Self::Error {
        NativeSigtermRequested::new_err("SIGTERM requested graceful shutdown.")
    }

    fn flushed_interruption_error(&mut self, error: Self::Error) -> Self::Error {
        Python::attach(|py| {
            if error.is_instance_of::<PyKeyboardInterrupt>(py) {
                NativeInterruptFlushed::new_err("SIGINT interrupted the run after resumable output was flushed.")
            } else {
                error
            }
        })
    }

    fn interruption_signal_name(error: &Self::Error) -> Option<&str> {
        Python::attach(|py| {
            if error.is_instance_of::<NativeSigtermRequested>(py) {
                Some("SIGTERM")
            } else if error.is_instance_of::<PyKeyboardInterrupt>(py) {
                Some("SIGINT")
            } else {
                None
            }
        })
    }

    fn interruption_kind(&mut self, error: &Self::Error) -> Option<native_runner::NativeRunInterruption> {
        Python::attach(|py| {
            if error.is_instance_of::<NativeInterruptFlushed>(py) {
                Some(native_runner::NativeRunInterruption::FlushedSigint)
            } else if error.is_instance_of::<NativeSigtermRequested>(py) {
                Some(native_runner::NativeRunInterruption::Sigterm)
            } else if error.is_instance_of::<PyKeyboardInterrupt>(py) {
                Some(native_runner::NativeRunInterruption::Sigint)
            } else {
                None
            }
        })
    }

    fn run_error(&mut self, message: String) -> Self::Error {
        PyRuntimeError::new_err(message)
    }

    fn failed_event(&mut self, error: &Self::Error) -> native_runner::NativeRunFailure {
        Python::attach(|py| {
            let event_payload = || -> PyResult<native_runner::NativeRunFailure> {
                let error = error.value(py);
                let error_type = error.get_type().name()?.to_string_lossy().into_owned();
                let error_message = error.str()?.to_string_lossy().into_owned();
                Ok(native_runner::NativeRunFailure { error_type, error_message })
            };
            event_payload().unwrap_or_else(|event_error| native_runner::NativeRunFailure {
                error_type: "PythonError".to_string(),
                error_message: event_error.to_string(),
            })
        })
    }

    fn current_thread_name(&mut self) -> PyResult<String> {
        Python::attach(|py| {
            PyModule::import(py, "threading")?.call_method0("current_thread")?.getattr("name")?.extract::<String>()
        })
    }

    fn detach<ResultValue, Operation>(operation: Operation) -> ResultValue
    where
        ResultValue: Send,
        Operation: FnOnce() -> ResultValue + Send,
    {
        Python::attach(|py| py.detach(operation))
    }
}

#[pymethods]
impl NativeCliRunResult {
    #[getter]
    fn exit_code(&self) -> i32 {
        self.inner.exit_code
    }

    #[getter]
    fn stdout_chunks<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.stdout_chunks)
    }

    #[getter]
    fn stderr_chunks<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.stderr_chunks)
    }
}

/// Execute the native CLI through the Python host adapter.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn run(arguments: Vec<String>) -> PyResult<NativeCliRunResult> {
    let mut host = PythonRunHost;
    native_runner::run_cli(&arguments, &mut host).map(|inner| NativeCliRunResult { inner })
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeCliRunResult>()?;
    module.add_function(wrap_pyfunction!(run, module)?)?;
    Ok(())
}
