#![warn(clippy::pedantic)]

pub mod genotype;
pub mod output;
pub mod pipeline;
pub mod python;
pub mod regenie;
pub mod sample;

use pyo3::prelude::*;

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register_module(module)
}
