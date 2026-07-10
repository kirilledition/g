use pyo3::exceptions::PyException;
use pyo3::prelude::*;

pub(crate) mod jax_runtime;

pyo3::create_exception!(g, NativeSigtermRequested, PyException);
pyo3::create_exception!(g, NativeInterruptFlushed, PyException);

pub(crate) fn check_process_signals(py: Python<'_>) -> PyResult<()> {
    py.check_signals()
}

pub(crate) fn is_sigterm_request(error: &PyErr, py: Python<'_>) -> bool {
    error.is_instance_of::<NativeSigtermRequested>(py)
}

pub(crate) fn is_flushed_interrupt(error: &PyErr, py: Python<'_>) -> bool {
    error.is_instance_of::<NativeInterruptFlushed>(py)
}

pub(crate) fn flushed_interrupt_error() -> PyErr {
    NativeInterruptFlushed::new_err("SIGINT interrupted the run after resumable output was flushed.")
}

pub(crate) fn sigterm_interrupt_error() -> PyErr {
    NativeSigtermRequested::new_err("SIGTERM requested graceful shutdown.")
}
