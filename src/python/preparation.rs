//! PyO3 adapters for engine-owned run-preparation policies.

use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::runtime_state::NativeRuntimeCompatibilityToken;

#[pyfunction]
pub(crate) fn validate_pipeline_resume_compatibility(
    chunks_directories: Vec<String>,
    existing_manifest_json_values: Vec<Option<String>>,
    current_header_json_values: Vec<String>,
    resume_mode: &str,
) -> PyResult<()> {
    let chunks_directories = chunks_directories.into_iter().map(PathBuf::from).collect::<Vec<_>>();
    let native_resume_mode =
        g_output::OutputResumeMode::parse(resume_mode).map_err(|error| PyValueError::new_err(error.to_string()))?;
    g_engine::validate_pipeline_resume_compatibility(
        chunks_directories,
        existing_manifest_json_values,
        current_header_json_values,
        native_resume_mode,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn initialize_pipeline_output_runs(
    run_directories: Vec<String>,
    chunks_directories: Vec<String>,
    existing_manifest_json_values: Vec<Option<String>>,
    current_header_json_values: Vec<String>,
    resume: bool,
    resume_mode: &str,
    runtime_compatibility_token: PyRef<'_, NativeRuntimeCompatibilityToken>,
) -> PyResult<Vec<Vec<i64>>> {
    let _runtime_compatibility_token = runtime_compatibility_token.native_token();
    let run_directories = run_directories.into_iter().map(PathBuf::from).collect::<Vec<_>>();
    let chunks_directories = chunks_directories.into_iter().map(PathBuf::from).collect::<Vec<_>>();
    let native_resume_mode =
        g_output::OutputResumeMode::parse(resume_mode).map_err(|error| PyValueError::new_err(error.to_string()))?;
    g_engine::initialize_pipeline_output_runs(
        run_directories,
        chunks_directories,
        existing_manifest_json_values,
        current_header_json_values,
        resume,
        native_resume_mode,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))
}
