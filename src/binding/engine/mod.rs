//! PyO3 bindings for `_core.engine`.

pub(crate) mod backend;
pub(crate) mod backend_delivery;
pub(crate) mod input;
pub(crate) mod run_engine;
pub(crate) mod run_lifecycle;

use pyo3::prelude::*;

pub(crate) use crate::binding::errors;
pub(crate) use crate::binding::output;
pub(crate) use crate::binding::runtime::timing;

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    backend::register_module(module)?;
    Ok(())
}
