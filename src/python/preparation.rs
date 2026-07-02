//! PyO3 adapters for engine-owned run-preparation policies.

use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};

use super::{json_bridge, runtime_state::NativeRuntimeCompatibilityToken};

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
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn build_pipeline_output_preparation_batch_from_values(
    py: Python<'_>,
    run_directories: Vec<String>,
    chunks_directories: Vec<String>,
    existing_manifest_values: Vec<Py<PyAny>>,
    current_header_values: Vec<Py<PyAny>>,
    resume: bool,
    resume_mode: &str,
) -> PyResult<NativePipelineOutputPreparationBatch> {
    let existing_manifest_json_values = existing_manifest_values
        .into_iter()
        .map(|existing_manifest| {
            let existing_manifest = existing_manifest.bind(py);
            if existing_manifest.is_none() {
                Ok(None)
            } else {
                json_bridge::json_text_from_py_any(existing_manifest).map(Some)
            }
        })
        .collect::<PyResult<Vec<_>>>()?;
    let current_header_json_values = current_header_values
        .into_iter()
        .map(|current_header| json_bridge::json_text_from_py_any(current_header.bind(py)))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(NativePipelineOutputPreparationBatch {
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

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativePipelineOutputInitialization>()?;
    module.add_class::<NativePipelineOutputPreparationBatch>()?;
    module.add_function(wrap_pyfunction!(build_pipeline_output_preparation_batch_from_values, module)?)?;
    Ok(())
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
