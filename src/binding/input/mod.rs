//! PyO3 bindings for `_core.input`.

pub(crate) mod prediction_sources;
pub(crate) mod sample_alignment;

use pyo3::prelude::*;

pub(crate) use crate::binding::errors;

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    sample_alignment::register_module(module)?;
    prediction_sources::register_module(module)?;
    Ok(())
}
