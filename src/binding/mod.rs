#![allow(clippy::elidable_lifetime_names)]
#![allow(clippy::fn_params_excessive_bools)]

use pyo3::prelude::*;

pub(crate) mod cli;
pub(crate) mod config;
pub(crate) mod convert;
pub(crate) mod engine;
pub(crate) mod errors;
pub(crate) mod genotype;
pub(crate) mod input;
pub(crate) mod output;
pub(crate) mod runtime;
pub(crate) mod telemetry;

pub(crate) use convert::json_bridge;
pub(crate) use runtime::runtime_state;
pub(crate) use telemetry::{logging, run_events, telemetry_policy};

#[allow(clippy::missing_errors_doc)]
pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    register_root_compatibility_aliases(module)?;
    register_domain_submodules(module)?;
    Ok(())
}

fn register_root_compatibility_aliases(module: &Bound<'_, PyModule>) -> PyResult<()> {
    config::register_domain(module)?;
    genotype::register_module(module)?;
    input::register_module(module)?;
    engine::register_module(module)?;
    runtime::register_module(module)?;
    telemetry::register_module(module)?;
    cli::register_module(module)?;
    output::register_module(module)?;
    Ok(())
}

fn register_domain_submodules(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__path__", Vec::<String>::new())?;
    register_submodule(module, "cli", cli::register_module)?;
    register_submodule(module, "config", config::register_domain)?;
    register_submodule(module, "genotype", genotype::register_module)?;
    register_submodule(module, "input", input::register_module)?;
    register_submodule(module, "engine", engine::register_module)?;
    register_submodule(module, "runtime", runtime::register_module)?;
    register_submodule(module, "telemetry", telemetry::register_module)?;
    register_submodule(module, "output", output::register_module)?;
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
