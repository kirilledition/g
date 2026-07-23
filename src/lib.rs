#[cfg(not(target_pointer_width = "64"))]
compile_error!("g requires a 64-bit target.");

mod binding;

use pyo3::prelude::*;

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__build_git_commit__", option_env!("GWAS_ENGINE_BUILD_GIT_COMMIT").unwrap_or("unavailable"))?;
    module.add(
        "__build_science_source_sha256__",
        option_env!("GWAS_ENGINE_BUILD_SCIENCE_SOURCE_SHA256").unwrap_or("unavailable"),
    )?;
    module.add("__build_source_clean__", option_env!("GWAS_ENGINE_BUILD_SOURCE_CLEAN") == Some("1"))?;
    module.add("__build_profile__", if cfg!(debug_assertions) { "dev" } else { "release" })?;
    module.add("__build_run_nonce__", option_env!("GWAS_ENGINE_BUILD_RUN_NONCE").unwrap_or("unavailable"))?;
    binding::register_module(module)
}
