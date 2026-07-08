#![allow(clippy::elidable_lifetime_names)]
#![allow(clippy::fn_params_excessive_bools)]

use pyo3::prelude::*;

use self::cli_driver::{
    NativeCliRunContext, NativeCliRunResult, NativeCliTelemetryPaths, NativeCliTelemetrySessionView,
    run_cli_with_python_backend,
};

mod callback_diagnostics;
mod callback_progress;
mod callback_queue;
mod callback_runtime_resources;
mod callback_summary;
mod cli_driver;
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
mod profile;
mod run_engine;
mod run_events;
mod run_lifecycle;
mod runtime;
mod runtime_state;
mod sample_alignment;
mod schedule;
mod shutdown;
mod telemetry_policy;
mod timing;

#[allow(clippy::missing_errors_doc)]
pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    register_config_domain(module)?;
    register_input_domain(module)?;
    register_engine_domain(module)?;
    register_runtime_domain(module)?;
    register_output_domain(module)?;
    register_domain_submodules(module)?;
    Ok(())
}

fn register_domain_submodules(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__path__", Vec::<String>::new())?;
    register_submodule(module, "cli", register_cli_driver_exports)?;
    register_submodule(module, "config", register_config_domain)?;
    register_submodule(module, "genotype", register_genotype_submodule)?;
    register_submodule(module, "input", register_input_submodule)?;
    register_submodule(module, "engine", register_engine_submodule)?;
    register_submodule(module, "runtime", register_runtime_submodule)?;
    register_submodule(module, "telemetry", register_telemetry_submodule)?;
    register_submodule(module, "output", register_output_domain)?;
    register_submodule(module, "debug", register_debug_submodule)?;
    Ok(())
}

fn register_submodule(
    module: &Bound<'_, PyModule>,
    name: &str,
    register: fn(&Bound<'_, PyModule>) -> PyResult<()>,
) -> PyResult<()> {
    let py = module.py();
    let module_name = module.name()?;
    let full_name = format!("{}.{}", module_name.to_str()?, name);
    let submodule = PyModule::new(py, &full_name)?;
    register(&submodule)?;
    module.add_submodule(&submodule)?;
    py.import("sys")?.getattr("modules")?.set_item(full_name, &submodule)?;
    Ok(())
}

fn register_config_domain(module: &Bound<'_, PyModule>) -> PyResult<()> {
    config::register_module(module)?;
    host_policy::register_module(module)?;
    Ok(())
}

fn register_input_domain(module: &Bound<'_, PyModule>) -> PyResult<()> {
    genotype::register_module(module)?;
    sample_alignment::register_module(module)?;
    prediction_sources::register_module(module)?;
    Ok(())
}

fn register_genotype_submodule(module: &Bound<'_, PyModule>) -> PyResult<()> {
    genotype::register_module(module)?;
    Ok(())
}

fn register_input_submodule(module: &Bound<'_, PyModule>) -> PyResult<()> {
    sample_alignment::register_module(module)?;
    prediction_sources::register_module(module)?;
    Ok(())
}

fn register_engine_domain(module: &Bound<'_, PyModule>) -> PyResult<()> {
    callback_summary::register_module(module);
    callback_progress::register_module(module)?;
    callback_queue::register_module(module);
    callback_runtime_resources::register_module(module)?;
    callback_diagnostics::register_module(module)?;
    schedule::register_module(module)?;
    run_engine::register_module(module)?;
    run_lifecycle::register_module(module)?;
    preflight::register_module(module)?;
    Ok(())
}

fn register_engine_submodule(module: &Bound<'_, PyModule>) -> PyResult<()> {
    run_engine::register_module(module)?;
    run_lifecycle::register_module(module)?;
    Ok(())
}

fn register_debug_submodule(module: &Bound<'_, PyModule>) -> PyResult<()> {
    callback_summary::register_module(module);
    callback_progress::register_module(module)?;
    callback_queue::register_module(module);
    callback_runtime_resources::register_module(module)?;
    callback_diagnostics::register_module(module)?;
    schedule::register_module(module)?;
    preflight::register_module(module)?;
    Ok(())
}

fn register_runtime_domain(module: &Bound<'_, PyModule>) -> PyResult<()> {
    jax_runtime::register_module(module)?;
    runtime_state::register_module(module)?;
    shutdown::register_module(module)?;
    timing::register_module(module)?;
    logging::register_module(module)?;
    telemetry_policy::register_module(module)?;
    run_events::register_module(module)?;
    register_cli_driver_exports(module)?;
    Ok(())
}

fn register_runtime_submodule(module: &Bound<'_, PyModule>) -> PyResult<()> {
    jax_runtime::register_module(module)?;
    runtime_state::register_module(module)?;
    shutdown::register_module(module)?;
    timing::register_module(module)?;
    Ok(())
}

fn register_telemetry_submodule(module: &Bound<'_, PyModule>) -> PyResult<()> {
    logging::register_module(module)?;
    telemetry_policy::register_module(module)?;
    run_events::register_module(module)?;
    Ok(())
}

fn register_output_domain(module: &Bound<'_, PyModule>) -> PyResult<()> {
    output::register_module(module)?;
    Ok(())
}

fn register_cli_driver_exports(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeCliRunResult>()?;
    module.add_class::<NativeCliTelemetryPaths>()?;
    module.add_class::<NativeCliTelemetrySessionView>()?;
    module.add_class::<NativeCliRunContext>()?;
    module.add_function(wrap_pyfunction!(run_cli_with_python_backend, module)?)?;
    Ok(())
}
