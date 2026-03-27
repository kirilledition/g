#![warn(clippy::pedantic)]

use pyo3::prelude::*;

#[pyfunction]
fn hello_from_bin() -> String {
    "Hello from g!".to_string()
}

/// Placeholder Python module implemented in Rust.
#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(hello_from_bin, module)?)?;
    Ok(())
}
