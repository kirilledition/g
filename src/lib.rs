#![warn(clippy::pedantic)]

pub(crate) mod callback_summary;
pub use g_genotype as genotype;
pub(crate) mod host_policy;
pub mod interface;
pub mod output;
pub mod pipeline;
pub mod python;
pub mod regenie;
pub(crate) mod run_metadata;
pub(crate) mod runtime_policy;
pub mod sample;
pub(crate) mod shutdown;
pub(crate) mod telemetry_policy;
pub(crate) mod timing;
pub(crate) mod trusted_validation;

use pyo3::prelude::*;

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register_module(module)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::used_underscore_items)]
    fn pymodule_entrypoint_registers_core_symbols() -> PyResult<()> {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "_core_test")?;
            super::_core(&module)?;
            assert!(module.hasattr("plan_genotype_chunks")?);
            Ok(())
        })
    }
}
