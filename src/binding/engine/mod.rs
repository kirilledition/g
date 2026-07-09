//! PyO3 bindings for `_core.engine`.

pub(crate) mod callback_diagnostics;
pub(crate) mod callback_progress;
pub(crate) mod callback_queue;
pub(crate) mod callback_runtime_resources;
pub(crate) mod callback_schedule;
pub(crate) mod callback_summary;
pub(crate) mod run_engine;
pub(crate) mod run_lifecycle;

use pyo3::prelude::*;

pub(crate) use crate::binding::config;
pub(crate) use crate::binding::errors;
pub(crate) use crate::binding::genotype;
pub(crate) use crate::binding::input::{prediction_sources, sample_alignment};
pub(crate) use crate::binding::json_bridge;
pub(crate) use crate::binding::output::{self, profile};
pub(crate) use crate::binding::runtime::{runtime_state, timing};
pub(crate) use callback_schedule as schedule;

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    callback_summary::register_module(module);
    callback_progress::register_module(module)?;
    callback_queue::register_module(module);
    callback_runtime_resources::register_module(module)?;
    callback_diagnostics::register_module(module)?;
    callback_schedule::register_module(module)?;
    run_engine::register_module(module)?;
    run_lifecycle::register_module(module)?;
    Ok(())
}
