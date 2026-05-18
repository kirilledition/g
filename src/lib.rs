#![warn(clippy::pedantic)]

pub mod genotype;
pub mod native;
pub mod output;
pub mod pipeline;
#[cfg(feature = "python")]
pub mod python;
pub mod regenie;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register_module(module)
}
