//! Feature-gated native support for focused CUDA tests and benchmarks.

use pyo3::prelude::*;

use crate::binding::engine;

#[pyfunction]
fn register_firth_components_ffi(py: Python<'_>) -> PyResult<String> {
    engine::require_firth_components_ffi_target(py).map(str::to_string)
}

#[pyfunction]
fn register_packed8_deflate_ffi(py: Python<'_>) -> PyResult<String> {
    engine::require_nvcomp_ffi_target(py).map(str::to_string)
}

#[pyfunction]
fn nvcomp_input_alignment(py: Python<'_>) -> PyResult<usize> {
    engine::require_nvcomp_input_alignment(py)
}

pub(super) fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let parent_name = parent.name()?;
    let full_name = format!("{}._testing", parent_name.to_str()?);
    let module = PyModule::new(py, &full_name)?;
    module.add_function(wrap_pyfunction!(register_firth_components_ffi, &module)?)?;
    module.add_function(wrap_pyfunction!(register_packed8_deflate_ffi, &module)?)?;
    module.add_function(wrap_pyfunction!(nvcomp_input_alignment, &module)?)?;
    parent.add_submodule(&module)?;
    py.import("sys")?.getattr("modules")?.set_item(full_name, &module)?;
    Ok(())
}
