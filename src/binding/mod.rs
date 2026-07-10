#![allow(clippy::elidable_lifetime_names)]
#![allow(clippy::fn_params_excessive_bools)]

use pyo3::prelude::*;

pub(crate) mod cli;
pub(crate) mod engine;
pub(crate) mod errors;
pub(crate) mod runtime;
pub(crate) mod telemetry;

#[allow(clippy::missing_errors_doc)]
pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__path__", Vec::<String>::new())?;
    register_submodule(module, "cli", cli::register_module)?;
    register_submodule(module, "engine", engine::register_module)?;
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
