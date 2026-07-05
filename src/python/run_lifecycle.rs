//! Coarse PyO3 boundary for Rust-owned run lifecycle state.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use g_interface as interface;
use g_output::{OutputFileFormat, OutputResumeMode, OutputWriterError};
use g_runtime::run_metadata as native_run_metadata;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule, PyTuple};

use super::config::{NativeRunRequest, RegenieConfig};
use super::json_bridge;
use super::run_events::NativeRunArtifacts;
use super::runtime_state::NativeRuntimeCompatibilityToken;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NativeRunLifecyclePhase {
    OutputsPrepared,
    Dispatching,
    Finalized,
}

impl NativeRunLifecyclePhase {
    const fn as_str(self) -> &'static str {
        match self {
            Self::OutputsPrepared => "outputs_prepared",
            Self::Dispatching => "dispatching",
            Self::Finalized => "finalized",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PreparedPhenotypeRunState {
    phenotype_name: String,
    run_directory: PathBuf,
    chunks_directory: PathBuf,
    existing_manifest_json: Option<String>,
    effective_config_path: PathBuf,
}

#[pyclass(name = "NativeRunLifecycleSession", skip_from_py_object)]
pub(crate) struct NativeRunLifecycleSession {
    config: interface::RegenieConfigData,
    run_request: g_plan::RunRequest,
    prepared_runs: Vec<PreparedPhenotypeRunState>,
    prepared_run_indices_by_name: BTreeMap<String, usize>,
    phase: Mutex<NativeRunLifecyclePhase>,
    initialized_metadata_phenotypes: Mutex<BTreeSet<String>>,
}

#[pyclass(name = "NativeRunLifecyclePhenotypeRun", skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct NativeRunLifecyclePhenotypeRun {
    phenotype_name: String,
    run_directory: String,
    chunks_directory: String,
    existing_manifest_json: Option<String>,
    effective_config_path: String,
}

#[pyclass(name = "NativeRunLifecycleOutputInitialization", skip_from_py_object)]
pub(crate) struct NativeRunLifecycleOutputInitialization {
    initialization: g_engine::PipelineOutputInitialization,
}

impl NativeRunLifecyclePhenotypeRun {
    fn from_state(state: &PreparedPhenotypeRunState) -> Self {
        Self {
            phenotype_name: state.phenotype_name.clone(),
            run_directory: state.run_directory.display().to_string(),
            chunks_directory: state.chunks_directory.display().to_string(),
            existing_manifest_json: state.existing_manifest_json.clone(),
            effective_config_path: state.effective_config_path.display().to_string(),
        }
    }
}

#[pymethods]
impl NativeRunLifecycleSession {
    #[new]
    fn new(
        py: Python<'_>,
        config: &RegenieConfig,
        runtime_compatibility_token: PyRef<'_, NativeRuntimeCompatibilityToken>,
    ) -> PyResult<Self> {
        let _runtime_compatibility_token = runtime_compatibility_token.native_token();
        let run_request = interface::compile_run_request(config.data())
            .map_err(|error| config_error_to_py("compile_run_request", error))?;
        let prepared_runs = py.detach(|| prepare_phenotype_runs(&run_request))?;
        let prepared_run_indices_by_name = prepared_runs
            .iter()
            .enumerate()
            .map(|(index, prepared_run)| (prepared_run.phenotype_name.clone(), index))
            .collect::<BTreeMap<_, _>>();
        Ok(Self {
            config: config.data().clone(),
            run_request,
            prepared_runs,
            prepared_run_indices_by_name,
            phase: Mutex::new(NativeRunLifecyclePhase::OutputsPrepared),
            initialized_metadata_phenotypes: Mutex::new(BTreeSet::new()),
        })
    }

    #[getter]
    fn phase(&self) -> PyResult<&'static str> {
        Ok(lock_phase(&self.phase)?.as_str())
    }

    #[getter]
    fn output_resume(&self) -> bool {
        self.run_request.output.resume
    }

    #[getter]
    fn run_request(&self) -> NativeRunRequest {
        NativeRunRequest::new(self.run_request.clone())
    }

    fn prepared_phenotype_runs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let prepared_runs = self
            .prepared_runs
            .iter()
            .map(|prepared_run| Py::new(py, NativeRunLifecyclePhenotypeRun::from_state(prepared_run)))
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, &prepared_runs)
    }

    #[allow(clippy::needless_pass_by_value)]
    fn prepared_phenotype_run(&self, phenotype_name: String) -> PyResult<NativeRunLifecyclePhenotypeRun> {
        Ok(NativeRunLifecyclePhenotypeRun::from_state(self.prepared_run_state(&phenotype_name)?))
    }

    fn mark_dispatch_started(&self) -> PyResult<()> {
        let mut phase = lock_phase(&self.phase)?;
        match *phase {
            NativeRunLifecyclePhase::OutputsPrepared => {
                *phase = NativeRunLifecyclePhase::Dispatching;
                Ok(())
            }
            NativeRunLifecyclePhase::Dispatching => Ok(()),
            NativeRunLifecyclePhase::Finalized => {
                Err(PyRuntimeError::new_err("Run lifecycle session cannot enter dispatch after finalization."))
            }
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    fn validate_output_resume_compatibility(
        &self,
        py: Python<'_>,
        phenotype_names: Vec<String>,
        current_headers: Vec<Py<PyAny>>,
    ) -> PyResult<()> {
        if !self.run_request.output.resume {
            return Ok(());
        }
        let preparation_batch = self.build_output_preparation_batch(py, &phenotype_names, current_headers)?;
        py.detach(|| preparation_batch.validate_resume_compatibility()).map_err(pipeline_resume_error_to_py)
    }

    #[allow(clippy::needless_pass_by_value)]
    fn initialize_output_runs(
        &self,
        py: Python<'_>,
        phenotype_names: Vec<String>,
        current_headers: Vec<Py<PyAny>>,
    ) -> PyResult<NativeRunLifecycleOutputInitialization> {
        self.ensure_not_finalized()?;
        let preparation_batch = self.build_output_preparation_batch(py, &phenotype_names, current_headers)?;
        let initialization = py.detach(|| preparation_batch.initialize()).map_err(pipeline_resume_error_to_py)?;
        self.write_initialized_metadata(&phenotype_names)?;
        Ok(NativeRunLifecycleOutputInitialization { initialization })
    }

    #[allow(clippy::needless_pass_by_value)]
    fn finalize_success(&self, final_output_paths: Vec<Option<String>>) -> PyResult<NativeRunArtifacts> {
        let phenotype_count = i64::try_from(self.prepared_runs.len())
            .map_err(|_| PyValueError::new_err("Phenotype count does not fit into int64 metadata."))?;
        let artifacts = native_run_metadata::build_execution_run_artifacts_from_sequences(
            native_run_metadata::ExecutionRunArtifactsSequenceInput {
                association_mode: self.run_request.association_mode.as_str().to_string(),
                phenotype_count,
                output_format: self.run_request.output.output_format.as_str().to_string(),
                output_run_directories: self
                    .prepared_runs
                    .iter()
                    .map(|prepared_run| prepared_run.run_directory.display().to_string())
                    .collect(),
                chunks_directories: self
                    .prepared_runs
                    .iter()
                    .map(|prepared_run| prepared_run.chunks_directory.display().to_string())
                    .collect(),
                effective_configs: self
                    .prepared_runs
                    .iter()
                    .map(|prepared_run| prepared_run.effective_config_path.display().to_string())
                    .collect(),
                phenotype_names: self
                    .prepared_runs
                    .iter()
                    .map(|prepared_run| prepared_run.phenotype_name.clone())
                    .collect(),
                final_output_paths,
            },
        )
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
        *lock_phase(&self.phase)? = NativeRunLifecyclePhase::Finalized;
        Ok(NativeRunArtifacts::new(artifacts))
    }
}

impl NativeRunLifecycleSession {
    fn ensure_not_finalized(&self) -> PyResult<()> {
        if *lock_phase(&self.phase)? == NativeRunLifecyclePhase::Finalized {
            return Err(PyRuntimeError::new_err("Run lifecycle session is finalized."));
        }
        Ok(())
    }

    fn prepared_run_state(&self, phenotype_name: &str) -> PyResult<&PreparedPhenotypeRunState> {
        let Some(index) = self.prepared_run_indices_by_name.get(phenotype_name) else {
            return Err(PyValueError::new_err(format!("Unknown prepared phenotype '{phenotype_name}'.")));
        };
        self.prepared_runs.get(*index).ok_or_else(|| {
            PyRuntimeError::new_err(format!("Prepared phenotype index for '{phenotype_name}' was inconsistent."))
        })
    }

    fn build_output_preparation_batch(
        &self,
        py: Python<'_>,
        phenotype_names: &[String],
        current_headers: Vec<Py<PyAny>>,
    ) -> PyResult<g_engine::PipelineOutputPreparationBatch> {
        if phenotype_names.len() != current_headers.len() {
            return Err(PyValueError::new_err(format!(
                "Output initialization input counts must match: phenotype_count={}, header_count={}.",
                phenotype_names.len(),
                current_headers.len()
            )));
        }
        let prepared_runs = phenotype_names
            .iter()
            .map(|phenotype_name| self.prepared_run_state(phenotype_name))
            .collect::<PyResult<Vec<_>>>()?;
        let current_header_json_values = current_headers
            .into_iter()
            .map(|current_header| json_bridge::json_text_from_py_any(current_header.bind(py)))
            .collect::<PyResult<Vec<_>>>()?;
        let resume_mode = OutputResumeMode::parse(self.run_request.output.resume_mode.as_str())
            .map_err(|error| output_writer_error_to_py(error, "parse_output_resume_mode"))?;
        g_engine::PipelineOutputPreparationBatch::new(
            prepared_runs.iter().map(|prepared_run| prepared_run.run_directory.clone()).collect(),
            prepared_runs.iter().map(|prepared_run| prepared_run.chunks_directory.clone()).collect(),
            prepared_runs.iter().map(|prepared_run| prepared_run.existing_manifest_json.clone()).collect(),
            current_header_json_values,
            self.run_request.output.resume,
            resume_mode,
        )
        .map_err(pipeline_resume_error_to_py)
    }

    fn write_initialized_metadata(&self, phenotype_names: &[String]) -> PyResult<()> {
        let mut initialized_metadata_phenotypes = lock_initialized_metadata(&self.initialized_metadata_phenotypes)?;
        for phenotype_name in phenotype_names {
            if initialized_metadata_phenotypes.contains(phenotype_name) {
                continue;
            }
            let prepared_run = self.prepared_run_state(phenotype_name)?;
            interface::write_toml(&self.config, &prepared_run.effective_config_path)
                .map_err(|error| config_error_to_py("write_toml", error))?;
            extend_run_manifest_metadata(&self.run_request, prepared_run)?;
            initialized_metadata_phenotypes.insert(phenotype_name.clone());
        }
        Ok(())
    }
}

#[pymethods]
impl NativeRunLifecyclePhenotypeRun {
    #[getter]
    fn phenotype_name(&self) -> &str {
        &self.phenotype_name
    }

    #[getter]
    fn run_directory(&self) -> &str {
        &self.run_directory
    }

    #[getter]
    fn chunks_directory(&self) -> &str {
        &self.chunks_directory
    }

    #[getter]
    fn effective_config_path(&self) -> &str {
        &self.effective_config_path
    }

    fn existing_manifest_payload(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self.existing_manifest_json.as_deref() {
            Some(existing_manifest_json) => {
                json_bridge::json_text_to_py_object(py, existing_manifest_json, "existing run manifest")
            }
            None => Ok(py.None()),
        }
    }
}

#[pymethods]
impl NativeRunLifecycleOutputInitialization {
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

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeRunLifecycleOutputInitialization>()?;
    module.add_class::<NativeRunLifecyclePhenotypeRun>()?;
    module.add_class::<NativeRunLifecycleSession>()?;
    Ok(())
}

fn prepare_phenotype_runs(run_request: &g_plan::RunRequest) -> PyResult<Vec<PreparedPhenotypeRunState>> {
    run_request
        .phenotype_runs
        .iter()
        .map(|phenotype_run| {
            let output_root = Path::new(&run_request.output.output_run_root).join(&phenotype_run.output_directory_name);
            let output_format = OutputFileFormat::parse(run_request.output.output_format.as_str())
                .map_err(|error| PyValueError::new_err(format!("Invalid output format: {error}")))?;
            let prepared_output_run = g_output::prepare_output_run(
                &output_root,
                run_request.association_mode.as_str(),
                output_format,
                run_request.output.resume,
            )
            .map_err(|error| output_writer_error_to_py(error, "prepare_output_run"))?;
            Ok(PreparedPhenotypeRunState {
                phenotype_name: phenotype_run.phenotype_name.clone(),
                run_directory: prepared_output_run.output_run_paths.run_directory.clone(),
                chunks_directory: prepared_output_run.output_run_paths.chunks_directory,
                existing_manifest_json: prepared_output_run.existing_manifest_json,
                effective_config_path: prepared_output_run.output_run_paths.run_directory.join("effective_config.toml"),
            })
        })
        .collect()
}

fn extend_run_manifest_metadata(
    run_request: &g_plan::RunRequest,
    prepared_run: &PreparedPhenotypeRunState,
) -> PyResult<()> {
    let extension = native_run_metadata::build_run_manifest_extension(native_run_metadata::RunManifestExtensionInput {
        phenotype_name: prepared_run.phenotype_name.clone(),
        effective_config: prepared_run.effective_config_path.display().to_string(),
        output_format: run_request.output.output_format.as_str().to_string(),
        device: run_request.compute.device.as_str().to_string(),
        staging_depth: i64::from(run_request.compute.staging_depth),
        native_callback_batch_size: i64::from(run_request.compute.native_callback_batch_size),
        threads: run_request.trait_request.thread_count.map(i64::from),
        writer_threads: i64::from(run_request.output.writer_thread_count),
        writer_queue_depth: i64::from(run_request.output.writer_queue_depth),
        chunks_per_arrow_file: i64::from(run_request.output.chunks_per_arrow_file),
        arrow_compression: run_request.output.arrow_compression.as_str().to_string(),
        parquet_compression: run_request.output.parquet_compression.as_str().to_string(),
        output_statistic_dtype: run_request.output.output_statistic_dtype.as_str().to_string(),
        bgen_decode_tile_variant_count: i64::from(run_request.compute.bgen_decode_tile_variant_count),
        trusted_no_missing_diploid: run_request.compute.trusted_no_missing_diploid,
        trusted_bgen_validation_mode: run_request.compute.trusted_bgen_validation_mode.as_str().to_string(),
    });
    let command = serde_json::to_value(&extension.command).map_err(|error| PyValueError::new_err(error.to_string()))?;
    let runtime = serde_json::to_value(&extension.runtime).map_err(|error| PyValueError::new_err(error.to_string()))?;
    g_output::extend_run_manifest_metadata(&prepared_run.run_directory, command, runtime)
        .map_err(|error| output_writer_error_to_py(error, "extend_run_manifest_metadata"))
}

fn lock_phase(phase: &Mutex<NativeRunLifecyclePhase>) -> PyResult<MutexGuard<'_, NativeRunLifecyclePhase>> {
    phase.lock().map_err(|_| PyRuntimeError::new_err("Run lifecycle phase mutex was poisoned."))
}

fn lock_initialized_metadata(
    initialized_metadata_phenotypes: &Mutex<BTreeSet<String>>,
) -> PyResult<MutexGuard<'_, BTreeSet<String>>> {
    initialized_metadata_phenotypes
        .lock()
        .map_err(|_| PyRuntimeError::new_err("Run lifecycle metadata mutex was poisoned."))
}

fn config_error_to_py(operation: &str, error: interface::ConfigError) -> PyErr {
    tracing::warn!(
        target: "g.python.run_lifecycle",
        g_event = "native_run_lifecycle_config_error",
        operation = operation,
        error_message = %error,
        "Native run lifecycle config error."
    );
    PyValueError::new_err(error.to_string())
}

fn pipeline_resume_error_to_py(error: g_engine::PipelineResumeCompatibilityError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn output_writer_error_to_py(error: OutputWriterError, operation: &str) -> PyErr {
    let (error_kind, message) = match &error {
        OutputWriterError::InvalidInput(message) => ("invalid_input", message.clone()),
        OutputWriterError::Runtime(message) => ("runtime", message.clone()),
    };
    tracing::warn!(
        target: "g.python.run_lifecycle",
        g_event = "native_run_lifecycle_output_error",
        operation = operation,
        error_kind = error_kind,
        error_message = %message,
        "Native run lifecycle output error."
    );
    match error {
        OutputWriterError::InvalidInput(message) => PyValueError::new_err(message),
        OutputWriterError::Runtime(message) => PyRuntimeError::new_err(message),
    }
}
