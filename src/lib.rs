#![warn(clippy::pedantic)]

pub mod genotype;
pub mod native;
pub mod output;
pub mod pipeline;
#[cfg(feature = "python")]
pub mod python;
pub mod regenie;
pub mod sample;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register_module(module)
}

#[cfg(all(test, feature = "python"))]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::used_underscore_items)]
    fn pymodule_entrypoint_registers_core_symbols() -> PyResult<()> {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "_core_test")?;
            super::_core(&module)?;
            assert!(module.hasattr("hello_from_bin")?);
            Ok(())
        })
    }
}
