//! PyO3 adapters for runtime path construction policy.

use std::path::Path;

use pyo3::prelude::*;

use g_runtime::runtime_paths as native_runtime_paths;

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn build_default_local_cache_directory_value(
    temporary_root: String,
    user_name: String,
    directory_name: String,
) -> String {
    let cache_directory = native_runtime_paths::build_default_local_cache_directory(
        Path::new(&temporary_root),
        &user_name,
        &directory_name,
    );
    cache_directory.to_string_lossy().into_owned()
}
