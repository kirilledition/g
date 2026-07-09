#![warn(clippy::pedantic)]

mod binding;

use pyo3::prelude::*;

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    binding::register_module(module)
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
            assert!(module.hasattr("summarize_variant_major_dosage_chunk_stats")?);
            Ok(())
        })
    }
}
