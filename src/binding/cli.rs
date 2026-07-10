//! PyO3 adapter for the Rust-owned native CLI lifecycle.

use std::sync::Arc;

use g_engine as native_engine;
use g_runner as native_runner;
use pyo3::exceptions::{PyKeyboardInterrupt, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyTuple};

use crate::binding::engine::{JaxBackendConfig, PyJaxBackend};
use crate::binding::{errors, runtime, telemetry};

#[pyclass]
pub(crate) struct NativeCliRunResult {
    exit_code: i32,
    stdout_chunks: Vec<String>,
    stderr_chunks: Vec<String>,
}

struct PythonRunHost;

impl native_runner::NativeRunHost for PythonRunHost {
    type Backend = PyJaxBackend;
    type Error = PyErr;

    fn install_python_logging(&mut self) -> PyResult<()> {
        Python::attach(telemetry::logging::install_python_logging)
    }

    fn apply_jax_config_updates(&mut self, updates: &[native_runner::JaxRuntimeConfigUpdatePayload]) -> PyResult<()> {
        Python::attach(|py| runtime::jax_runtime::apply_jax_config_updates(py, updates))
    }

    fn observe_jax_devices(&mut self) -> PyResult<Vec<native_runner::JaxDeviceObservation>> {
        Python::attach(runtime::jax_runtime::observe_jax_devices)
    }

    fn create_backend(&mut self, settings: native_engine::JaxBackendSettings) -> PyResult<Arc<Self::Backend>> {
        Python::attach(|py| {
            let backend_config = Py::new(py, JaxBackendConfig::new(settings))?;
            let backend = PyModule::import(py, "g.jax_backend")?
                .getattr("JaxAssociationBackend")?
                .call1((backend_config,))?
                .unbind();
            Ok(Arc::new(PyJaxBackend::new(backend)))
        })
    }

    fn check_interruption(&mut self) -> PyResult<()> {
        Python::attach(runtime::check_process_signals)
    }

    fn interruption_signal_name(error: &Self::Error) -> Option<&str> {
        Python::attach(|py| {
            if runtime::is_sigterm_request(error, py) {
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
            if runtime::is_flushed_interrupt(error, py) {
                Some(native_runner::NativeRunInterruption::FlushedSigint)
            } else if runtime::is_sigterm_request(error, py) {
                Some(native_runner::NativeRunInterruption::Sigterm)
            } else if error.is_instance_of::<PyKeyboardInterrupt>(py) {
                Some(native_runner::NativeRunInterruption::Sigint)
            } else {
                None
            }
        })
    }

    fn convert_engine_error(
        &mut self,
        error: native_engine::CoordinatedRunError<
            <Self::Backend as native_engine::AssociationBackend>::Error,
            Self::Error,
        >,
    ) -> Self::Error {
        errors::convert_coordinated_run_error(error)
    }

    fn native_runtime_error(&mut self, message: String) -> Self::Error {
        PyRuntimeError::new_err(message)
    }

    fn failed_event(&mut self, error: &Self::Error) -> native_runner::RunFailedEventPayload {
        Python::attach(|py| {
            telemetry::run_events::run_failed_event_payload_from_error(error.value(py)).unwrap_or_else(|event_error| {
                native_runner::build_run_failed_event_payload("PythonError", &event_error.to_string())
            })
        })
    }

    fn current_thread_name(&mut self) -> PyResult<String> {
        telemetry::current_python_thread_name()
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
        self.exit_code
    }

    #[getter]
    fn stdout_chunks<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.stdout_chunks)
    }

    #[getter]
    fn stderr_chunks<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.stderr_chunks)
    }
}

/// Execute the native CLI through the Python host adapter.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn run(arguments: Vec<String>) -> PyResult<NativeCliRunResult> {
    let mut host = PythonRunHost;
    let result = native_runner::run_cli(&arguments, &mut host)?;
    Ok(NativeCliRunResult {
        exit_code: result.exit_code,
        stdout_chunks: result.stdout_chunks,
        stderr_chunks: result.stderr_chunks,
    })
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeCliRunResult>()?;
    module.add_function(wrap_pyfunction!(run, module)?)?;
    Ok(())
}
