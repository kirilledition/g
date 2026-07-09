//! PyO3 bindings for `_core.engine`.

pub(crate) mod run_engine;
pub(crate) mod run_lifecycle;

use pyo3::prelude::*;

pub(crate) use crate::binding::config;
pub(crate) use crate::binding::debug::{preflight, schedule};
pub(crate) use crate::binding::errors;
pub(crate) use crate::binding::genotype;
pub(crate) use crate::binding::input::{prediction_sources, sample_alignment};
pub(crate) use crate::binding::json_bridge;
pub(crate) use crate::binding::output::{self, profile};
pub(crate) use crate::binding::runtime::{runtime_state, timing};
pub(crate) use crate::binding::telemetry::run_events;

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    run_engine::register_module(module)?;
    run_lifecycle::register_module(module)?;
    Ok(())
}
