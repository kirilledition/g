use pyo3::prelude::*;

pub(crate) mod cli;
pub(crate) mod engine;
pub(crate) mod jax_runtime;
pub(crate) mod logging;

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__path__", Vec::<String>::new())?;
    let py = module.py();
    let module_name = module.name()?;
    let full_name = format!("{}.cli", module_name.to_str()?);
    let submodule = PyModule::new(py, &full_name)?;
    cli::register_module(&submodule)?;
    module.add_submodule(&submodule)?;
    py.import("sys")?.getattr("modules")?.set_item(full_name, &submodule)?;
    Ok(())
}
