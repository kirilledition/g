//! PyO3 adapters for runtime path construction policy.

use pyo3::prelude::*;
use pyo3::types::PyModule;

use g_runtime::runtime_paths as native_runtime_paths;

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn default_local_temporary_root_value() -> String {
    native_runtime_paths::default_local_temporary_root().to_string_lossy().into_owned()
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn default_local_cache_directory_value(directory_name: String) -> String {
    native_runtime_paths::default_local_cache_directory(&directory_name).to_string_lossy().into_owned()
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(default_local_cache_directory_value, module)?)?;
    module.add_function(wrap_pyfunction!(default_local_temporary_root_value, module)?)?;
    Ok(())
}
