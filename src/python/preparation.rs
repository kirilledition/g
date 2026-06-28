//! PyO3 adapters for engine-owned run-preparation policies.

use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::runtime_state::NativeRuntimeCompatibilityToken;

#[pyclass]
pub(crate) struct NativePipelineOutputInitialization {
    initialization: g_engine::PipelineOutputInitialization,
}

#[pymethods]
impl NativePipelineOutputInitialization {
    #[getter]
    fn output_count(&self) -> usize {
        self.initialization.output_count()
    }

    fn committed_chunk_identifier_sets(&self) -> Vec<Vec<i64>> {
        self.initialization.committed_chunk_identifier_sets().to_vec()
    }

    fn committed_chunk_identifiers(&self, output_index: usize) -> PyResult<Vec<i64>> {
        self.initialization
            .committed_chunk_identifiers(output_index)
            .map(<[i64]>::to_vec)
            .ok_or_else(|| PyValueError::new_err(format!("Output index {output_index} is out of range.")))
    }
}

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
pub(crate) fn initialize_pipeline_output_run_batch(
    run_directories: Vec<String>,
    chunks_directories: Vec<String>,
    existing_manifest_json_values: Vec<Option<String>>,
    current_header_json_values: Vec<String>,
    resume: bool,
    resume_mode: &str,
    runtime_compatibility_token: PyRef<'_, NativeRuntimeCompatibilityToken>,
) -> PyResult<NativePipelineOutputInitialization> {
    let _runtime_compatibility_token = runtime_compatibility_token.native_token();
    let run_directories = run_directories.into_iter().map(PathBuf::from).collect::<Vec<_>>();
    let chunks_directories = chunks_directories.into_iter().map(PathBuf::from).collect::<Vec<_>>();
    let native_resume_mode =
        g_output::OutputResumeMode::parse(resume_mode).map_err(|error| PyValueError::new_err(error.to_string()))?;
    let initialization = g_engine::initialize_pipeline_output_run_batch(
        run_directories,
        chunks_directories,
        existing_manifest_json_values,
        current_header_json_values,
        resume,
        native_resume_mode,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(NativePipelineOutputInitialization { initialization })
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
