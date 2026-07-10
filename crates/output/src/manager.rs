//! Run-scoped output lifecycle ownership.

use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Serialize;

use crate::error::{OutputError, OutputResult};
use crate::manifest::{
    CurrentRunManifestHeaderInput, ManifestFileFingerprintCache, OutputRunPaths, PreparedOutputRun,
    build_current_run_manifest_header_value_with_cache, extend_run_manifest_metadata, initialize_output_run,
    prepare_output_run, read_run_manifest_gpu_genotype_format_from_text, validate_output_run_resume_compatibility,
};
use crate::session::{
    OutputWriterSession, create_output_writer_sessions, finish_interrupted_output_writer_sessions,
    finish_output_writer_sessions,
};
const COMMAND_INTERFACE: &str = "g regenie";

/// Writer and resume state consumed by one association group.
pub struct OutputDeliveryState {
    pub writer_sessions: Vec<Arc<OutputWriterSession>>,
    pub committed_chunk_identifier_sets: Vec<Arc<BTreeSet<usize>>>,
}

/// Final artifacts for one phenotype output.
pub struct CompletedOutputRun {
    pub run_directory: PathBuf,
    pub parts_directory: PathBuf,
}

struct ManagedOutputRun {
    phenotype_name: String,
    paths: OutputRunPaths,
    existing_manifest_json: Option<String>,
    effective_config_path: PathBuf,
    committed_chunk_identifiers: Arc<BTreeSet<usize>>,
}

/// One owner for every output run and writer session in an association run.
pub struct OutputManager {
    run_plan: Arc<g_plan::RunPlan>,
    effective_config_toml: String,
    runs: Vec<ManagedOutputRun>,
    run_indices_by_phenotype: BTreeMap<String, usize>,
    writer_sessions: Option<Vec<Arc<OutputWriterSession>>>,
    terminal: bool,
}

impl OutputManager {
    /// Open output directories and existing manifests without initializing writers.
    ///
    /// # Errors
    ///
    /// Returns an error when output names are duplicated, paths cannot be prepared,
    /// or a non-resume output directory is already populated.
    pub fn open(run_plan: Arc<g_plan::RunPlan>, effective_config_toml: String) -> OutputResult<Self> {
        let mut runs = Vec::with_capacity(run_plan.phenotype_runs.len());
        let mut run_indices_by_phenotype = BTreeMap::new();
        for phenotype_run in &run_plan.phenotype_runs {
            let output_root = Path::new(&run_plan.output.output_run_root).join(&phenotype_run.output_directory_name);
            let PreparedOutputRun { output_run_paths, existing_manifest_json } =
                prepare_output_run(&output_root, run_plan.association_mode.as_str(), run_plan.output.resume)?;
            let output_index = runs.len();
            if run_indices_by_phenotype.insert(phenotype_run.phenotype_name.clone(), output_index).is_some() {
                return Err(OutputError::InvalidInput(format!(
                    "Duplicate phenotype output name '{}'.",
                    phenotype_run.phenotype_name
                )));
            }
            let effective_config_path = output_run_paths.run_directory.join("effective_config.toml");
            runs.push(ManagedOutputRun {
                phenotype_name: phenotype_run.phenotype_name.clone(),
                paths: output_run_paths,
                existing_manifest_json,
                effective_config_path,
                committed_chunk_identifiers: Arc::new(BTreeSet::new()),
            });
        }
        Ok(Self {
            run_plan,
            effective_config_toml,
            runs,
            run_indices_by_phenotype,
            writer_sessions: None,
            terminal: false,
        })
    }

    /// Return the concrete GPU genotype format recorded by an existing manifest.
    ///
    /// # Errors
    ///
    /// Returns an error when the phenotype was not planned.
    pub fn existing_manifest_gpu_genotype_format(
        &self,
        phenotype_name: &str,
    ) -> OutputResult<Option<g_plan::GpuGenotypeFormat>> {
        self.run(phenotype_name)?
            .existing_manifest_json
            .as_deref()
            .map(read_run_manifest_gpu_genotype_format_from_text)
            .transpose()
    }

    /// Initialize every output manifest and the shared bounded writer pool.
    ///
    /// # Errors
    ///
    /// Returns an error when headers do not cover the planned phenotypes exactly,
    /// resume validation fails, metadata cannot be written, or writer creation fails.
    pub fn initialize(
        &mut self,
        current_header_inputs: Vec<CurrentRunManifestHeaderInput>,
        collect_stage_timings: bool,
    ) -> OutputResult<()> {
        if self.writer_sessions.is_some() {
            return Err(OutputError::InvalidInput("Output manager is already initialized.".to_string()));
        }
        let mut headers_by_phenotype = BTreeMap::new();
        let mut fingerprint_cache = ManifestFileFingerprintCache::new();
        for current_header_input in current_header_inputs {
            let phenotype_name = current_header_input.phenotype_name.clone();
            let current_header = build_current_run_manifest_header_value_with_cache(
                &self.run_plan,
                current_header_input,
                &mut fingerprint_cache,
            )?;
            match headers_by_phenotype.entry(phenotype_name) {
                Entry::Vacant(entry) => {
                    entry.insert(current_header);
                }
                Entry::Occupied(entry) => {
                    return Err(OutputError::InvalidInput(format!(
                        "Duplicate output initialization for phenotype '{}'.",
                        entry.key()
                    )));
                }
            }
        }
        if headers_by_phenotype.len() != self.runs.len() {
            return Err(OutputError::InvalidInput(format!(
                "Output initialization count {} does not match planned phenotype count {}.",
                headers_by_phenotype.len(),
                self.runs.len()
            )));
        }
        let resume_mode = self.run_plan.output.resume_mode;
        if self.run_plan.output.resume {
            for run in &self.runs {
                let header = headers_by_phenotype.get(&run.phenotype_name).ok_or_else(|| {
                    OutputError::InvalidInput(format!(
                        "Missing output initialization for phenotype '{}'.",
                        run.phenotype_name
                    ))
                })?;
                let manifest = run.existing_manifest_json.as_deref().ok_or_else(|| {
                    OutputError::InvalidInput(format!(
                        "Resume output for phenotype '{}' has no manifest.",
                        run.phenotype_name
                    ))
                })?;
                validate_output_run_resume_compatibility(&run.paths.parts_directory, manifest, header, resume_mode)?;
            }
        }
        for run in &mut self.runs {
            let current_header = headers_by_phenotype.remove(&run.phenotype_name).ok_or_else(|| {
                OutputError::InvalidInput(format!(
                    "Missing output initialization for phenotype '{}'.",
                    run.phenotype_name
                ))
            })?;
            let committed_chunk_identifiers = initialize_output_run(
                &run.paths.run_directory,
                &run.paths.parts_directory,
                run.existing_manifest_json.as_deref(),
                &current_header,
                self.run_plan.output.resume,
                resume_mode,
            )?;
            run.committed_chunk_identifiers = Arc::new(
                committed_chunk_identifiers
                    .into_iter()
                    .map(|identifier| {
                        usize::try_from(identifier).map_err(|_| {
                            OutputError::InvalidInput(format!(
                                "Committed chunk identifier {identifier} does not fit usize."
                            ))
                        })
                    })
                    .collect::<OutputResult<BTreeSet<_>>>()?,
            );
            std::fs::write(&run.effective_config_path, &self.effective_config_toml).map_err(OutputError::runtime)?;
            extend_manifest_from_plan(&self.run_plan, run)?;
        }
        let writer_sessions = create_output_writer_sessions(
            self.runs.iter().map(|run| run.paths.run_directory.clone()).collect(),
            self.runs.iter().map(|run| run.paths.parts_directory.clone()).collect(),
            &self.run_plan.output,
            collect_stage_timings,
        )?;
        self.writer_sessions = Some(writer_sessions.into_iter().map(Arc::new).collect());
        Ok(())
    }

    /// Borrow delivery state for phenotype names in trait order.
    ///
    /// # Errors
    ///
    /// Returns an error when a phenotype is unknown or writers are not initialized.
    pub fn delivery_state_for_phenotypes(&self, phenotype_names: &[String]) -> OutputResult<OutputDeliveryState> {
        let writer_sessions = self.writer_sessions.as_ref().ok_or_else(|| {
            OutputError::InvalidInput("Output manager must be initialized before delivery.".to_string())
        })?;
        let mut selected_writer_sessions = Vec::with_capacity(phenotype_names.len());
        let mut committed_chunk_identifier_sets = Vec::with_capacity(phenotype_names.len());
        for phenotype_name in phenotype_names {
            let output_index =
                self.run_indices_by_phenotype.get(phenotype_name).copied().ok_or_else(|| {
                    OutputError::InvalidInput(format!("Unknown planned phenotype '{phenotype_name}'."))
                })?;
            selected_writer_sessions.push(
                writer_sessions.get(output_index).map(Arc::clone).ok_or_else(|| {
                    OutputError::InvalidInput(format!("Output index {output_index} is out of range."))
                })?,
            );
            committed_chunk_identifier_sets.push(Arc::clone(
                &self
                    .runs
                    .get(output_index)
                    .ok_or_else(|| OutputError::InvalidInput(format!("Output index {output_index} is out of range.")))?
                    .committed_chunk_identifiers,
            ));
        }
        Ok(OutputDeliveryState { writer_sessions: selected_writer_sessions, committed_chunk_identifier_sets })
    }

    /// Finish every writer and return artifact paths in phenotype order.
    ///
    /// # Errors
    ///
    /// Returns an error when a writer fails.
    pub fn finish(mut self) -> OutputResult<Vec<CompletedOutputRun>> {
        let writer_sessions = self.writer_session_references()?;
        let thread_count = finish_thread_count(&self.run_plan.output, writer_sessions.len())?;
        finish_output_writer_sessions(&writer_sessions, thread_count)?;
        self.terminal = true;
        Ok(self.take_completion())
    }

    /// Flush every writer and mark manifests interrupted.
    ///
    /// # Errors
    ///
    /// Returns an error when interrupted output cannot be flushed.
    pub fn finish_interrupted(mut self, signal_name: &str) -> OutputResult<()> {
        let writer_sessions = self.writer_session_references()?;
        let thread_count = finish_thread_count(&self.run_plan.output, writer_sessions.len())?;
        finish_interrupted_output_writer_sessions(&writer_sessions, thread_count, signal_name)?;
        self.terminal = true;
        Ok(())
    }

    /// Abort every writer and discard pending chunks.
    ///
    /// # Errors
    ///
    /// Returns the first writer abort failure.
    pub fn abort(mut self) -> OutputResult<()> {
        let result = abort_writer_sessions(self.writer_sessions.as_deref().unwrap_or_default());
        self.terminal = true;
        result
    }

    fn run(&self, phenotype_name: &str) -> OutputResult<&ManagedOutputRun> {
        let output_index = self
            .run_indices_by_phenotype
            .get(phenotype_name)
            .ok_or_else(|| OutputError::InvalidInput(format!("Unknown planned phenotype '{phenotype_name}'.")))?;
        self.runs.get(*output_index).ok_or_else(|| {
            OutputError::Runtime(format!("Output index for phenotype '{phenotype_name}' is inconsistent."))
        })
    }

    fn writer_session_references(&self) -> OutputResult<Vec<&OutputWriterSession>> {
        let writer_sessions = self.writer_sessions.as_ref().ok_or_else(|| {
            OutputError::InvalidInput("Output manager must be initialized before completion.".to_string())
        })?;
        Ok(writer_sessions.iter().map(Arc::as_ref).collect())
    }

    fn take_completion(&mut self) -> Vec<CompletedOutputRun> {
        self.runs
            .drain(..)
            .map(|run| CompletedOutputRun {
                run_directory: run.paths.run_directory,
                parts_directory: run.paths.parts_directory,
            })
            .collect()
    }
}

impl Drop for OutputManager {
    fn drop(&mut self) {
        if !self.terminal {
            let _ = abort_writer_sessions(self.writer_sessions.as_deref().unwrap_or_default());
        }
    }
}

fn finish_thread_count(output_plan: &g_plan::OutputPlan, writer_session_count: usize) -> OutputResult<usize> {
    let requested_thread_count = usize::try_from(output_plan.writer_thread_count).map_err(OutputError::runtime)?;
    if requested_thread_count == 0 && writer_session_count != 0 {
        return Err(OutputError::InvalidInput("Writer finish thread count must be positive.".to_string()));
    }
    Ok(requested_thread_count.min(writer_session_count))
}

fn abort_writer_sessions(writer_sessions: &[Arc<OutputWriterSession>]) -> OutputResult<()> {
    let mut first_error = None;
    for writer_session in writer_sessions {
        if let Err(error) = writer_session.abort()
            && first_error.is_none()
        {
            first_error = Some(error);
        }
    }
    first_error.map_or(Ok(()), Err)
}

#[derive(Serialize)]
struct ManifestCommand<'a> {
    interface: &'static str,
    phenotype: &'a str,
    effective_config: String,
}

#[derive(Serialize)]
struct ManifestRuntime {
    device: &'static str,
    staging_depth: u32,
    cpu_threads: Option<u32>,
    writer_threads: u32,
    writer_queue_depth: u32,
    chunks_per_parquet_file: u32,
    parquet_compression: &'static str,
    bgen_decode_tile_variant_count: u32,
    trusted_no_missing_diploid: bool,
    trusted_bgen_validation_mode: &'static str,
}

fn extend_manifest_from_plan(run_plan: &g_plan::RunPlan, run: &ManagedOutputRun) -> OutputResult<()> {
    let command = serde_json::to_value(ManifestCommand {
        interface: COMMAND_INTERFACE,
        phenotype: &run.phenotype_name,
        effective_config: run.effective_config_path.display().to_string(),
    })
    .map_err(OutputError::runtime)?;
    let runtime = serde_json::to_value(ManifestRuntime {
        device: run_plan.compute.device.as_str(),
        staging_depth: run_plan.compute.staging_depth,
        cpu_threads: run_plan.compute.cpu_thread_count,
        writer_threads: run_plan.output.writer_thread_count,
        writer_queue_depth: run_plan.output.writer_queue_depth,
        chunks_per_parquet_file: run_plan.output.chunks_per_parquet_file,
        parquet_compression: "zstd",
        bgen_decode_tile_variant_count: run_plan.compute.bgen_decode_tile_variant_count,
        trusted_no_missing_diploid: run_plan.compute.trusted_no_missing_diploid,
        trusted_bgen_validation_mode: run_plan.compute.trusted_bgen_validation_mode.as_str(),
    })
    .map_err(OutputError::runtime)?;
    extend_run_manifest_metadata(&run.paths.run_directory, command, runtime)
}
