#![allow(clippy::elidable_lifetime_names)]
#![allow(clippy::fn_params_excessive_bools)]

use pyo3::prelude::*;

mod callback_diagnostics;
mod callback_progress;
mod callback_queue;
mod callback_runtime_resources;
mod callback_summary;
mod config;
mod errors;
mod genotype;
mod host_policy;
mod jax_runtime;
mod json_bridge;
mod logging;
mod output;
mod prediction_sources;
mod preflight;
mod preparation;
mod profile;
mod run_engine;
mod run_events;
mod run_metadata;
mod runtime;
mod runtime_state;
mod sample_alignment;
mod schedule;
mod shutdown;
mod telemetry_policy;
mod timing;

#[allow(clippy::missing_errors_doc)]
#[allow(clippy::too_many_lines)]
pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    config::register_module(module)?;
    genotype::register_module(module)?;
    sample_alignment::register_module(module)?;
    callback_summary::register_module(module)?;
    callback_progress::register_module(module)?;
    callback_queue::register_module(module)?;
    callback_runtime_resources::register_module(module)?;
    callback_diagnostics::register_module(module)?;
    schedule::register_module(module)?;
    preparation::register_module(module)?;
    jax_runtime::register_module(module)?;
    runtime_state::register_module(module)?;
    shutdown::register_module(module)?;
    timing::register_module(module)?;
    output::register_module(module)?;
    run_engine::register_module(module)?;
    prediction_sources::register_module(module)?;
    logging::register_module(module)?;
    telemetry_policy::register_module(module)?;
    run_events::register_module(module)?;
    run_metadata::register_module(module)?;
    preflight::register_module(module)?;
    host_policy::register_module(module)?;
    runtime::register_module(module)?;
    Ok(())
}
