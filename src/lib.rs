#![warn(clippy::pedantic)]

pub mod bgen;
pub mod genotype;
pub mod output;
pub mod pipeline;
pub mod python;

use pyo3::prelude::*;

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register_module(module)
}
