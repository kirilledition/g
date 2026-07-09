//! Debug and compatibility PyO3 bindings.

pub(crate) mod callback_diagnostics;
pub(crate) mod callback_progress;
pub(crate) mod callback_queue;
pub(crate) mod callback_runtime_resources;
pub(crate) mod callback_summary;
pub(crate) mod preflight;
pub(crate) mod schedule;

use pyo3::prelude::*;

pub(crate) use crate::binding::errors;
pub(crate) use crate::binding::telemetry::logging;

fn register_engine_internals(module: &Bound<'_, PyModule>) -> PyResult<()> {
    callback_summary::register_module(module);
    callback_progress::register_module(module)?;
    callback_queue::register_module(module);
    callback_runtime_resources::register_module(module)?;
    callback_diagnostics::register_module(module)?;
    schedule::register_module(module)?;
    Ok(())
}

fn register_preflight(module: &Bound<'_, PyModule>) -> PyResult<()> {
    preflight::register_module(module)
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    register_engine_internals(module)?;
    register_preflight(module)?;
    Ok(())
}

pub(crate) fn register_root_compatibility_aliases(module: &Bound<'_, PyModule>) -> PyResult<()> {
    register_module(module)
}
