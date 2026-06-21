//! PyO3 adapters for deterministic graceful-shutdown signal helpers.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::shutdown as native_shutdown;

#[pyfunction]
pub(crate) fn build_shutdown_signal_payload<'py>(py: Python<'py>, signal_number: i32) -> PyResult<Bound<'py, PyDict>> {
    let payload = native_shutdown::build_shutdown_signal(signal_number).map_err(PyValueError::new_err)?;
    let python_payload = PyDict::new(py);
    python_payload.set_item("number", payload.number)?;
    python_payload.set_item("name", payload.name)?;
    python_payload.set_item("exit_code", payload.exit_code)?;
    Ok(python_payload)
}
