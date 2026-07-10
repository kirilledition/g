//! Crate-private telemetry support for the native CLI and run engine.

pub(crate) mod logging;
pub(crate) mod run_events;

pub(crate) use crate::binding::errors;

pub(crate) fn current_python_thread_name() -> pyo3::PyResult<String> {
    pyo3::Python::attach(|py| {
        use pyo3::prelude::*;

        let threading_module = pyo3::types::PyModule::import(py, "threading")?;
        threading_module.call_method0("current_thread")?.getattr("name")?.extract::<String>()
    })
}
