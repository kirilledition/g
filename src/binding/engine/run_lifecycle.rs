//! Coarse PyO3 boundary for Rust-owned run lifecycle state.

#![allow(clippy::needless_pass_by_value)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

use g_interface as interface;
use g_output::ManifestFileFingerprintCache;
use g_output::OutputFileFormat;
use g_runtime as native_run_metadata;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyTuple};

use super::config::{NativeRunRequest, RegenieConfig};
use super::errors;
use super::json_bridge;
use super::output;
use super::run_events::{self, NativeRunArtifacts};
use super::runtime_state::NativeRuntimeCompatibilityToken;
use super::schedule::NativeMultiTraitChunkWritePlanner;
use super::timing::NativeStageTimingRecorder;

pub(crate) type NativeOutputRuntimeGroupInput = (
    Vec<String>,
    Vec<String>,
    i64,
    String,
    Option<String>,
    Option<Vec<i64>>,
    Option<Vec<String>>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
);

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
    manifest_fingerprint_cache: Mutex<ManifestFileFingerprintCache>,
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

#[pyclass(name = "NativePreparedOutputBundle", skip_from_py_object)]
pub(crate) struct NativePreparedOutputBundle {
    initialization: g_engine::PipelineOutputInitialization,
    writer_sessions: Vec<Py<output::OutputWriterSession>>,
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
        Self::from_config(py, config, &runtime_compatibility_token)
    }

    #[getter]
    fn phase(&self) -> PyResult<&'static str> {
        self.phase_label()
    }

    #[getter]
    fn output_resume(&self) -> bool {
        self.output_resume_value()
    }

    #[getter]
    fn run_request(&self) -> NativeRunRequest {
        self.run_request_handle()
    }

    fn prepared_phenotype_runs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        self.prepared_phenotype_runs_tuple(py)
    }

    #[allow(clippy::needless_pass_by_value)]
    fn prepared_phenotype_run(&self, phenotype_name: String) -> PyResult<NativeRunLifecyclePhenotypeRun> {
        self.prepared_phenotype_run_handle(phenotype_name)
    }

    fn mark_dispatch_started(&self) -> PyResult<()> {
        self.mark_dispatch_started_internal()
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    fn prepare_output_bundles_from_runtime_plan<'py>(
        &self,
        py: Python<'py>,
        output_groups: Vec<NativeOutputRuntimeGroupInput>,
        variant_count: i64,
        effective_trusted_no_missing_diploid: bool,
        sample_key_mode: String,
        binary_kernel_config_json: Option<String>,
        requested_gpu_genotype_format: String,
        gpu_genotype_format: String,
        score_dtype: String,
        firth_dtype: String,
        stage_timing_recorder: Option<PyRef<'py, NativeStageTimingRecorder>>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        self.prepare_output_bundles_from_runtime_plan_internal(
            py,
            output_groups,
            variant_count,
            effective_trusted_no_missing_diploid,
            sample_key_mode,
            binary_kernel_config_json,
            requested_gpu_genotype_format,
            gpu_genotype_format,
            score_dtype,
            firth_dtype,
            stage_timing_recorder.as_deref(),
        )
    }

    #[allow(clippy::needless_pass_by_value)]
    fn finalize_success(&self, final_output_paths: Vec<Option<String>>) -> PyResult<NativeRunArtifacts> {
        self.finalize_success_artifacts(final_output_paths)
    }
}

impl NativeRunLifecycleSession {
    pub(crate) fn from_config(
        py: Python<'_>,
        config: &RegenieConfig,
        runtime_compatibility_token: &NativeRuntimeCompatibilityToken,
    ) -> PyResult<Self> {
        let _runtime_compatibility_token = runtime_compatibility_token.native_token();
        let run_request = interface::compile_run_request(config.data())
            .map_err(|error| errors::convert_config_error("compile_run_request", &error))?;
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
            manifest_fingerprint_cache: Mutex::new(ManifestFileFingerprintCache::new()),
        })
    }

    pub(crate) fn phase_label(&self) -> PyResult<&'static str> {
        Ok(lock_phase(&self.phase)?.as_str())
    }

    pub(crate) fn output_resume_value(&self) -> bool {
        self.run_request.output.resume
    }

    pub(crate) fn run_request_handle(&self) -> NativeRunRequest {
        NativeRunRequest::new(self.run_request.clone())
    }

    pub(crate) fn prepared_phenotype_runs_tuple<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let prepared_runs = self
            .prepared_runs
            .iter()
            .map(|prepared_run| Py::new(py, NativeRunLifecyclePhenotypeRun::from_state(prepared_run)))
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, &prepared_runs)
    }

    pub(crate) fn prepared_phenotype_run_handle(
        &self,
        phenotype_name: String,
    ) -> PyResult<NativeRunLifecyclePhenotypeRun> {
        Ok(NativeRunLifecyclePhenotypeRun::from_state(self.prepared_run_state(&phenotype_name)?))
    }

    pub(crate) fn mark_dispatch_started_internal(&self) -> PyResult<()> {
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

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    pub(crate) fn prepare_output_bundles_from_runtime_plan_internal<'py>(
        &self,
        py: Python<'py>,
        output_groups: Vec<NativeOutputRuntimeGroupInput>,
        variant_count: i64,
        effective_trusted_no_missing_diploid: bool,
        sample_key_mode: String,
        binary_kernel_config_json: Option<String>,
        requested_gpu_genotype_format: String,
        gpu_genotype_format: String,
        score_dtype: String,
        firth_dtype: String,
        stage_timing_recorder: Option<&NativeStageTimingRecorder>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let output_bundles = self.prepare_output_bundle_objects_from_runtime_plan_internal(
            py,
            output_groups,
            variant_count,
            effective_trusted_no_missing_diploid,
            sample_key_mode,
            binary_kernel_config_json,
            requested_gpu_genotype_format,
            gpu_genotype_format,
            score_dtype,
            firth_dtype,
            stage_timing_recorder,
        )?;
        PyTuple::new(py, &output_bundles)
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    pub(crate) fn prepare_output_bundle_objects_from_runtime_plan_internal(
        &self,
        py: Python<'_>,
        output_groups: Vec<NativeOutputRuntimeGroupInput>,
        variant_count: i64,
        effective_trusted_no_missing_diploid: bool,
        sample_key_mode: String,
        binary_kernel_config_json: Option<String>,
        requested_gpu_genotype_format: String,
        gpu_genotype_format: String,
        score_dtype: String,
        firth_dtype: String,
        stage_timing_recorder: Option<&NativeStageTimingRecorder>,
    ) -> PyResult<Vec<Py<NativePreparedOutputBundle>>> {
        self.ensure_not_finalized()?;
        let runtime_plan = g_engine::RuntimeOutputPlan {
            variant_count,
            effective_trusted_no_missing_diploid,
            sample_key_mode,
            binary_kernel_config_json,
            requested_gpu_genotype_format,
            gpu_genotype_format,
            score_dtype,
            firth_dtype,
        };
        let output_preparation_groups = {
            let prepared_runs = self
                .prepared_runs
                .iter()
                .map(|prepared_run| g_engine::RuntimeOutputPreparedRun {
                    phenotype_name: prepared_run.phenotype_name.clone(),
                    run_directory: prepared_run.run_directory.clone(),
                    chunks_directory: prepared_run.chunks_directory.clone(),
                    existing_manifest_json: prepared_run.existing_manifest_json.clone(),
                })
                .collect::<Vec<_>>();
            let mut fingerprint_cache = lock_manifest_fingerprint_cache(&self.manifest_fingerprint_cache)?;
            output_groups
                .into_iter()
                .map(runtime_output_group_from_input)
                .map(|output_group| {
                    output_group.and_then(|group| {
                        g_engine::build_runtime_output_preparation_group(
                            &self.run_request,
                            &prepared_runs,
                            group,
                            &runtime_plan,
                            &mut fingerprint_cache,
                        )
                        .map_err(|error| errors::convert_pipeline_output_preparation_error(&error))
                    })
                })
                .collect::<PyResult<Vec<_>>>()?
        };
        if self.run_request.output.resume {
            validate_output_resume_compatibility_for_groups(py, &output_preparation_groups)?;
        }
        let writer_preparation_start_time = Instant::now();
        let collect_stage_timings =
            stage_timing_recorder.map_or(Ok(false), NativeStageTimingRecorder::should_collect_exact_stage_timings)?;
        let output_bundles = output_preparation_groups
            .into_iter()
            .map(|output_preparation_group| {
                self.prepare_native_output_bundle(py, output_preparation_group, collect_stage_timings)
            })
            .collect::<PyResult<Vec<_>>>()?;
        if let Some(recorder) = stage_timing_recorder {
            recorder.record_stage_duration(
                "output_writer_preparation",
                writer_preparation_start_time.elapsed().as_secs_f64(),
            )?;
        }
        Ok(output_bundles)
    }

    pub(crate) fn finalize_success_artifacts(
        &self,
        final_output_paths: Vec<Option<String>>,
    ) -> PyResult<NativeRunArtifacts> {
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

    fn prepare_native_output_bundle(
        &self,
        py: Python<'_>,
        output_preparation_group: g_engine::RuntimeOutputPreparationGroup,
        collect_stage_timings: bool,
    ) -> PyResult<Py<NativePreparedOutputBundle>> {
        let initialization = py
            .detach(|| output_preparation_group.preparation_batch.initialize())
            .map_err(|error| errors::convert_pipeline_resume_compatibility_error(&error))?;
        self.write_initialized_metadata(&output_preparation_group.phenotype_names)?;
        if self.run_request.output.resume {
            let payloads = g_engine::build_output_resume_committed_chunk_diagnostic_payloads(&initialization)
                .map_err(|error| errors::convert_pipeline_output_preparation_error(&error))?;
            for payload in payloads {
                run_events::emit_run_diagnostic_event_payload(&payload)?;
            }
        }
        let writer_sessions =
            self.create_output_writer_sessions(py, &output_preparation_group.phenotype_names, collect_stage_timings)?;
        Py::new(py, NativePreparedOutputBundle { initialization, writer_sessions })
    }

    fn create_output_writer_sessions(
        &self,
        py: Python<'_>,
        phenotype_names: &[String],
        collect_stage_timings: bool,
    ) -> PyResult<Vec<Py<output::OutputWriterSession>>> {
        let prepared_runs = phenotype_names
            .iter()
            .map(|phenotype_name| self.prepared_run_state(phenotype_name))
            .collect::<PyResult<Vec<_>>>()?;
        output::create_output_writer_session_batch(
            py,
            prepared_runs.iter().map(|prepared_run| prepared_run.run_directory.display().to_string()).collect(),
            prepared_runs.iter().map(|prepared_run| prepared_run.chunks_directory.display().to_string()).collect(),
            self.run_request.association_mode.as_str(),
            u32_value_as_usize(self.run_request.output.writer_thread_count, "writer_thread_count")?,
            u32_value_as_usize(self.run_request.output.writer_queue_depth, "writer_queue_depth")?,
            self.run_request.output.output_format.as_str(),
            self.run_request.output.output_statistic_dtype.as_str(),
            self.run_request.output.finalize_parquet,
            u32_value_as_usize(self.run_request.output.chunks_per_arrow_file, "chunks_per_arrow_file")?,
            self.run_request.output.arrow_compression.as_str(),
            self.run_request.output.parquet_compression.as_str(),
            collect_stage_timings,
        )
    }

    fn write_initialized_metadata(&self, phenotype_names: &[String]) -> PyResult<()> {
        let mut initialized_metadata_phenotypes = lock_initialized_metadata(&self.initialized_metadata_phenotypes)?;
        for phenotype_name in phenotype_names {
            if initialized_metadata_phenotypes.contains(phenotype_name) {
                continue;
            }
            let prepared_run = self.prepared_run_state(phenotype_name)?;
            interface::write_toml(&self.config, &prepared_run.effective_config_path)
                .map_err(|error| errors::convert_config_error("write_toml", &error))?;
            extend_run_manifest_metadata(&self.run_request, prepared_run)?;
            initialized_metadata_phenotypes.insert(phenotype_name.clone());
        }
        Ok(())
    }
}

impl NativePreparedOutputBundle {
    pub(crate) fn writer_session_handle(
        &self,
        py: Python<'_>,
        output_index: usize,
    ) -> PyResult<Py<output::OutputWriterSession>> {
        self.writer_sessions
            .get(output_index)
            .map(|session| session.clone_ref(py))
            .ok_or_else(|| PyValueError::new_err(format!("Output index {output_index} is out of range.")))
    }

    pub(crate) fn committed_chunk_identifiers_usize(&self, output_index: usize) -> PyResult<Vec<usize>> {
        self.initialization
            .committed_chunk_identifiers(output_index)
            .ok_or_else(|| PyValueError::new_err(format!("Output index {output_index} is out of range.")))?
            .iter()
            .map(|identifier| {
                usize::try_from(*identifier).map_err(|_| {
                    PyValueError::new_err(format!("Committed chunk identifier {identifier} is out of range."))
                })
            })
            .collect()
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
impl NativePreparedOutputBundle {
    #[getter]
    fn writer_sessions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.writer_sessions)
    }

    #[getter]
    fn output_count(&self) -> usize {
        self.initialization.output_count()
    }

    fn committed_chunk_counts(&self) -> Vec<usize> {
        self.initialization.committed_chunk_counts()
    }

    fn committed_chunk_identifiers(&self, output_index: usize) -> PyResult<Vec<i64>> {
        self.initialization
            .committed_chunk_identifiers(output_index)
            .map(<[i64]>::to_vec)
            .ok_or_else(|| PyValueError::new_err(format!("Output index {output_index} is out of range.")))
    }

    fn shared_committed_chunk_identifiers(&self) -> Vec<i64> {
        self.initialization.shared_committed_chunk_identifiers()
    }

    fn shared_committed_chunk_identifiers_with(
        &self,
        other_bundles: Vec<PyRef<'_, NativePreparedOutputBundle>>,
    ) -> Vec<i64> {
        let mut initializations = Vec::with_capacity(other_bundles.len() + 1);
        initializations.push(&self.initialization);
        for other_bundle in &other_bundles {
            initializations.push(&other_bundle.initialization);
        }
        g_engine::PipelineOutputInitialization::shared_committed_chunk_identifiers_across(initializations)
    }

    fn multi_trait_chunk_write_planner(&self) -> PyResult<NativeMultiTraitChunkWritePlanner> {
        NativeMultiTraitChunkWritePlanner::from_i64_committed_chunk_identifier_sets(
            self.writer_sessions.len(),
            self.initialization.committed_chunk_identifier_sets(),
        )
    }
}

pub(crate) fn register_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativePreparedOutputBundle>()?;
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
                .map_err(|error| errors::convert_output_error("parse_output_format", error))?;
            let prepared_output_run = g_output::prepare_output_run(
                &output_root,
                run_request.association_mode.as_str(),
                output_format,
                run_request.output.resume,
            )
            .map_err(|error| errors::convert_output_error("prepare_output_run", error))?;
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
        .map_err(|error| errors::convert_output_error("extend_run_manifest_metadata", error))
}

fn lock_phase(phase: &Mutex<NativeRunLifecyclePhase>) -> PyResult<MutexGuard<'_, NativeRunLifecyclePhase>> {
    phase.lock().map_err(|_| PyRuntimeError::new_err("Run lifecycle phase mutex was poisoned."))
}

fn runtime_output_group_from_input(input: NativeOutputRuntimeGroupInput) -> PyResult<g_engine::RuntimeOutputGroup> {
    let (
        phenotype_names,
        covariate_names,
        sample_count,
        output_sample_mode,
        phenotype_compute_group_mode,
        phenotype_compute_group_indices,
        phenotype_compute_group_names,
        phenotype_compute_group_sample_mode,
        sample_set_fingerprint,
        covariate_design_fingerprint,
        prediction_alignment_fingerprint,
    ) = input;
    g_engine::RuntimeOutputGroup::from_input(g_engine::RuntimeOutputGroupInput {
        phenotype_names,
        covariate_names,
        sample_count,
        output_sample_mode,
        phenotype_compute_group_mode,
        phenotype_compute_group_indices,
        phenotype_compute_group_names,
        phenotype_compute_group_sample_mode,
        sample_set_fingerprint,
        covariate_design_fingerprint,
        prediction_alignment_fingerprint,
    })
    .map_err(|error| errors::convert_pipeline_output_preparation_error(&error))
}

fn lock_manifest_fingerprint_cache(
    manifest_fingerprint_cache: &Mutex<ManifestFileFingerprintCache>,
) -> PyResult<MutexGuard<'_, ManifestFileFingerprintCache>> {
    manifest_fingerprint_cache
        .lock()
        .map_err(|_| PyRuntimeError::new_err("Manifest fingerprint cache mutex was poisoned."))
}

fn validate_output_resume_compatibility_for_groups(
    py: Python<'_>,
    output_preparation_groups: &[g_engine::RuntimeOutputPreparationGroup],
) -> PyResult<()> {
    for output_preparation_group in output_preparation_groups {
        py.detach(|| output_preparation_group.preparation_batch.validate_resume_compatibility())
            .map_err(|error| errors::convert_pipeline_resume_compatibility_error(&error))?;
    }
    Ok(())
}

fn u32_value_as_usize(value: u32, field_name: &str) -> PyResult<usize> {
    usize::try_from(value).map_err(|_| PyValueError::new_err(format!("{field_name} does not fit into usize.")))
}

fn lock_initialized_metadata(
    initialized_metadata_phenotypes: &Mutex<BTreeSet<String>>,
) -> PyResult<MutexGuard<'_, BTreeSet<String>>> {
    initialized_metadata_phenotypes
        .lock()
        .map_err(|_| PyRuntimeError::new_err("Run lifecycle metadata mutex was poisoned."))
}
