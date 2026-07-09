//! PyO3 bindings for `_core.telemetry`.

pub(crate) mod logging;
pub(crate) mod run_events;
pub(crate) mod telemetry_policy;

use pyo3::prelude::*;

pub(crate) use crate::binding::debug::{callback_progress, schedule};
pub(crate) use crate::binding::errors;
pub(crate) use crate::binding::json_bridge;
pub(crate) use crate::binding::runtime::jax_runtime;

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    logging::register_module(module)?;
    telemetry_policy::register_module(module)?;
    run_events::register_module(module)?;
    Ok(())
}
