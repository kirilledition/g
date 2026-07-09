//! Debug bindings intentionally empty after production callback runtime promotion.
//!
//! Production callback runtime, schedule, and queue adapters live under
//! `crate::binding::engine`.

use pyo3::prelude::*;

pub(crate) fn register_module(_module: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}

pub(crate) fn register_root_compatibility_aliases(_module: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
