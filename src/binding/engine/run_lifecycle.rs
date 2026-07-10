//! Coarse PyO3 boundary for Rust-owned run lifecycle state.

#![allow(clippy::needless_pass_by_value)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use g_interface as interface;
use g_output::ManifestFileFingerprintCache;
use g_output::OutputFileFormat;
use g_runtime as native_run_metadata;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use super::errors;
use super::output;
use super::timing::NativeStageTimingRecorder;
use crate::binding::telemetry::run_events::{self, NativeRunArtifacts};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NativeRunLifecyclePhase {
    OutputsPrepared,
    Dispatching,
    Finalized,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PreparedPhenotypeRunState {
    phenotype_name: String,
    run_directory: PathBuf,
    chunks_directory: PathBuf,
    existing_manifest_json: Option<String>,
    effective_config_path: PathBuf,
}

pub(crate) struct NativeRunLifecycleSession {
    config: interface::RegenieConfigData,
    run_request: g_plan::RunRequest,
    prepared_runs: Vec<PreparedPhenotypeRunState>,
    prepared_run_indices_by_name: BTreeMap<String, usize>,
    phase: Mutex<NativeRunLifecyclePhase>,
    initialized_metadata_phenotypes: Mutex<BTreeSet<String>>,
    manifest_fingerprint_cache: Mutex<ManifestFileFingerprintCache>,
}

pub(crate) struct NativePreparedOutputBundle {
    initialization: g_engine::PipelineOutputInitialization,
    writer_sessions: Vec<Arc<output::OutputWriterSession>>,
}

impl NativeRunLifecycleSession {
    pub(crate) fn config_data(&self) -> &interface::RegenieConfigData {
        &self.config
    }

    pub(crate) fn from_config(py: Python<'_>, config: &interface::RegenieConfigData) -> PyResult<Self> {
        let run_request = interface::compile_run_request(config)
            .map_err(|error| errors::convert_config_error("compile_run_request", &error))?;
        let prepared_runs = py.detach(|| prepare_phenotype_runs(&run_request))?;
        let prepared_run_indices_by_name = prepared_runs
            .iter()
            .enumerate()
            .map(|(index, prepared_run)| (prepared_run.phenotype_name.clone(), index))
            .collect::<BTreeMap<_, _>>();
        Ok(Self {
            config: config.clone(),
            run_request,
            prepared_runs,
            prepared_run_indices_by_name,
            phase: Mutex::new(NativeRunLifecyclePhase::OutputsPrepared),
            initialized_metadata_phenotypes: Mutex::new(BTreeSet::new()),
            manifest_fingerprint_cache: Mutex::new(ManifestFileFingerprintCache::new()),
        })
    }

    pub(crate) fn output_resume_value(&self) -> bool {
        self.run_request.output.resume
    }

    pub(crate) fn run_request_data(&self) -> &g_plan::RunRequest {
        &self.run_request
    }

    pub(crate) fn prepared_run_existing_manifest_json(&self, phenotype_name: &str) -> PyResult<Option<String>> {
        Ok(self.prepared_run_state(phenotype_name)?.existing_manifest_json.clone())
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
    pub(crate) fn prepare_output_bundles_from_runtime_plan_internal(
        &self,
        py: Python<'_>,
        output_groups: Vec<g_engine::RuntimeOutputGroupInput>,
        variant_count: i64,
        effective_trusted_no_missing_diploid: bool,
        sample_key_mode: String,
        binary_kernel_config_json: Option<String>,
        requested_gpu_genotype_format: String,
        gpu_genotype_format: String,
        score_dtype: String,
        firth_dtype: String,
        stage_timing_recorder: Option<&NativeStageTimingRecorder>,
    ) -> PyResult<Vec<NativePreparedOutputBundle>> {
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
                .map(|input| {
                    g_engine::RuntimeOutputGroup::from_input(input)
                        .map_err(|error| errors::convert_pipeline_output_preparation_error(&error))
                })
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
        Ok(artifacts)
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
    ) -> PyResult<NativePreparedOutputBundle> {
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
            self.create_output_writer_sessions(&output_preparation_group.phenotype_names, collect_stage_timings)?;
        Ok(NativePreparedOutputBundle { initialization, writer_sessions })
    }

    fn create_output_writer_sessions(
        &self,
        phenotype_names: &[String],
        collect_stage_timings: bool,
    ) -> PyResult<Vec<Arc<output::OutputWriterSession>>> {
        let prepared_runs = phenotype_names
            .iter()
            .map(|phenotype_name| self.prepared_run_state(phenotype_name))
            .collect::<PyResult<Vec<_>>>()?;
        output::create_output_writer_session_batch(
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
    pub(crate) fn writer_session_handle(&self, output_index: usize) -> PyResult<Arc<output::OutputWriterSession>> {
        self.writer_sessions
            .get(output_index)
            .map(Arc::clone)
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

    pub(crate) fn writer_session_handles(&self) -> Vec<Arc<output::OutputWriterSession>> {
        self.writer_sessions.iter().map(Arc::clone).collect()
    }
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
