#![warn(clippy::pedantic)]

pub use g_engine as engine;
pub use g_genotype as genotype;
pub use g_input as input;
pub use g_input::{regenie, sample};
pub use g_interface as interface;
pub use g_output as output;
pub use g_plan as plan;
pub use g_runtime as runtime;
pub mod python;
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
