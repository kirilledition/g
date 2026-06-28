//! PyO3 adapters for engine-owned run-preparation policies.

use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::runtime_state::NativeRuntimeCompatibilityToken;

#[pyclass]
pub(crate) struct NativePipelineOutputPreparationBatch {
    batch: g_engine::PipelineOutputPreparationBatch,
}

#[pyclass]
pub(crate) struct NativePipelineOutputInitialization {
    initialization: g_engine::PipelineOutputInitialization,
}

#[pymethods]
impl NativePipelineOutputPreparationBatch {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        run_directories: Vec<String>,
        chunks_directories: Vec<String>,
        existing_manifest_json_values: Vec<Option<String>>,
        current_header_json_values: Vec<String>,
        resume: bool,
        resume_mode: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            batch: parse_pipeline_output_preparation_batch(
                run_directories,
                chunks_directories,
                existing_manifest_json_values,
                current_header_json_values,
                resume,
                resume_mode,
            )?,
        })
    }

    #[getter]
    fn output_count(&self) -> usize {
        self.batch.output_count()
    }

    #[getter]
    fn resume(&self) -> bool {
        self.batch.resume()
    }

    fn validate_resume_compatibility(&self) -> PyResult<()> {
        self.batch.validate_resume_compatibility().map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[allow(clippy::needless_pass_by_value)]
    fn initialize(
        &self,
        runtime_compatibility_token: PyRef<'_, NativeRuntimeCompatibilityToken>,
    ) -> PyResult<NativePipelineOutputInitialization> {
        let _runtime_compatibility_token = runtime_compatibility_token.native_token();
        let initialization = self.batch.initialize().map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(NativePipelineOutputInitialization { initialization })
    }
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
    let preparation_batch = parse_pipeline_output_preparation_batch(
        run_directories,
        chunks_directories,
        existing_manifest_json_values,
        current_header_json_values,
        resume,
        resume_mode,
    )?;
    let initialization = preparation_batch.initialize().map_err(|error| PyValueError::new_err(error.to_string()))?;
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

fn parse_pipeline_output_preparation_batch(
    run_directories: Vec<String>,
    chunks_directories: Vec<String>,
    existing_manifest_json_values: Vec<Option<String>>,
    current_header_json_values: Vec<String>,
    resume: bool,
    resume_mode: &str,
) -> PyResult<g_engine::PipelineOutputPreparationBatch> {
    let run_directories = run_directories.into_iter().map(PathBuf::from).collect::<Vec<_>>();
    let chunks_directories = chunks_directories.into_iter().map(PathBuf::from).collect::<Vec<_>>();
    let native_resume_mode =
        g_output::OutputResumeMode::parse(resume_mode).map_err(|error| PyValueError::new_err(error.to_string()))?;
    g_engine::PipelineOutputPreparationBatch::new(
        run_directories,
        chunks_directories,
        existing_manifest_json_values,
        current_header_json_values,
        resume,
        native_resume_mode,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))
}
