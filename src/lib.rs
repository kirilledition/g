#[cfg(not(target_pointer_width = "64"))]
compile_error!("g requires a 64-bit target.");

mod binding;

use pyo3::prelude::*;

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    binding::register_module(module)
}
