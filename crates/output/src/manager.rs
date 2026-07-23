//! Run-scoped output lifecycle ownership.

use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Serialize;

use crate::error::{OutputError, OutputResult};
use crate::manifest::{
    CurrentRunManifestHeaderInput, ManifestFileFingerprintCache, OutputRunPaths,
    build_current_run_manifest_header_value_with_cache, extend_run_manifest_metadata, initialize_output_run,
    inspect_output_run, read_run_manifest_gpu_genotype_format_from_text, reconcile_output_run_resume,
    resolve_output_run_paths,
};
use crate::session::{
    CreatedOutputWriterSessions, OutputWriterResourceOwner, OutputWriterSession, create_output_writer_sessions,
    finish_interrupted_output_writer_sessions, finish_output_writer_sessions, validate_output_writer_settings,
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
    writer_resource_owner: Option<OutputWriterResourceOwner>,
    terminal: bool,
}

impl OutputManager {
    /// Plan output paths and inspect existing manifests without mutating the filesystem.
    ///
    /// # Errors
    ///
    /// Returns an error when output names or paths collide, paths cannot be
    /// inspected, or a non-resume output directory is already populated.
    pub fn open(run_plan: Arc<g_plan::RunPlan>, effective_config_toml: String) -> OutputResult<Self> {
        let mut runs = Vec::with_capacity(run_plan.phenotype_runs.len());
        let mut run_indices_by_phenotype = BTreeMap::new();
        let mut phenotype_names_by_run_directory = BTreeMap::new();
        for phenotype_run in &run_plan.phenotype_runs {
            let output_root = Path::new(&run_plan.output.output_run_root).join(&phenotype_run.output_directory_name);
            let output_run_paths = resolve_output_run_paths(&output_root, run_plan.association_mode.as_str());
            let output_index = runs.len();
            if run_indices_by_phenotype.insert(phenotype_run.phenotype_name.clone(), output_index).is_some() {
                return Err(OutputError::InvalidInput(format!(
                    "Duplicate phenotype output name '{}'.",
                    phenotype_run.phenotype_name
                )));
            }
            if let Some(existing_phenotype_name) = phenotype_names_by_run_directory
                .insert(output_run_paths.run_directory.clone(), phenotype_run.phenotype_name.clone())
            {
                return Err(OutputError::InvalidInput(format!(
                    "Phenotype outputs '{existing_phenotype_name}' and '{}' resolve to the same run directory '{}'.",
                    phenotype_run.phenotype_name,
                    output_run_paths.run_directory.display()
                )));
            }
            let effective_config_path = output_run_paths.run_directory.join("effective_config.toml");
            runs.push(ManagedOutputRun {
                phenotype_name: phenotype_run.phenotype_name.clone(),
                paths: output_run_paths,
                existing_manifest_json: None,
                effective_config_path,
                committed_chunk_identifiers: Arc::new(BTreeSet::new()),
            });
        }
        for run in &mut runs {
            run.existing_manifest_json = inspect_output_run(&run.paths, run_plan.output.resume)?;
        }
        Ok(Self {
            run_plan,
            effective_config_toml,
            runs,
            run_indices_by_phenotype,
            writer_sessions: None,
            writer_resource_owner: None,
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
        planned_chunk_ranges: &[Range<usize>],
        collect_stage_timings: bool,
    ) -> OutputResult<()> {
        if self.writer_sessions.is_some() {
            return Err(OutputError::InvalidInput("Output manager is already initialized.".to_string()));
        }
        validate_output_writer_settings(&self.run_plan.output, self.runs.len())?;
        let mut headers_by_phenotype = BTreeMap::new();
        let mut fingerprint_cache = ManifestFileFingerprintCache::default();
        for current_header_input in current_header_inputs {
            let phenotype_name = current_header_input.phenotype_name.clone();
            let current_header = build_current_run_manifest_header_value_with_cache(
                &self.run_plan,
                &current_header_input,
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
        self.validate_run_manifest_header_coverage(&headers_by_phenotype)?;
        self.refresh_run_manifest_hints()?;
        let resumed_chunk_commits = if self.run_plan.output.resume {
            self.runs
                .iter()
                .map(|run| {
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
                    reconcile_output_run_resume(&run.paths.parts_directory, manifest, header, planned_chunk_ranges)
                        .map(Some)
                })
                .collect::<OutputResult<Vec<_>>>()?
        } else {
            std::iter::repeat_with(|| None).take(self.runs.len()).collect()
        };
        for run in &self.runs {
            std::fs::create_dir_all(&run.paths.parts_directory).map_err(OutputError::runtime)?;
        }
        for (run, resumed_chunk_commits) in self.runs.iter_mut().zip(resumed_chunk_commits) {
            let current_header = headers_by_phenotype.remove(&run.phenotype_name).ok_or_else(|| {
                OutputError::InvalidInput(format!(
                    "Missing output initialization for phenotype '{}'.",
                    run.phenotype_name
                ))
            })?;
            let committed_chunk_identifiers = initialize_output_run(
                &run.paths.run_directory,
                run.existing_manifest_json.as_deref(),
                &current_header,
                resumed_chunk_commits,
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
        let CreatedOutputWriterSessions { sessions, resource_owner } = create_output_writer_sessions(
            self.runs.iter().map(|run| run.paths.run_directory.clone()).collect(),
            self.runs.iter().map(|run| run.paths.parts_directory.clone()).collect(),
            &self.run_plan.output,
            collect_stage_timings,
        )?;
        self.writer_sessions = Some(sessions.into_iter().map(Arc::new).collect());
        self.writer_resource_owner = resource_owner;
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
        let finish_result = (|| {
            let writer_sessions = self.writer_session_references()?;
            let thread_count = finish_thread_count(&self.run_plan.output, writer_sessions.len())?;
            finish_output_writer_sessions(&writer_sessions, thread_count)
        })();
        let shutdown_result = self.shutdown_writer_resources();
        self.terminal = true;
        finish_result?;
        shutdown_result?;
        Ok(self.take_completion())
    }

    /// Flush every writer and mark manifests interrupted.
    ///
    /// # Errors
    ///
    /// Returns an error when interrupted output cannot be flushed.
    pub fn finish_interrupted(mut self, signal_name: &str) -> OutputResult<()> {
        let finish_result = (|| {
            let writer_sessions = self.writer_session_references()?;
            let thread_count = finish_thread_count(&self.run_plan.output, writer_sessions.len())?;
            finish_interrupted_output_writer_sessions(&writer_sessions, thread_count, signal_name)
        })();
        let shutdown_result = self.shutdown_writer_resources();
        self.terminal = true;
        finish_result.and(shutdown_result)
    }

    /// Abort every writer and discard pending chunks.
    ///
    /// # Errors
    ///
    /// Returns the first writer abort failure.
    pub fn abort(mut self) -> OutputResult<()> {
        let abort_result = abort_writer_sessions(self.writer_sessions.as_deref().unwrap_or_default());
        let shutdown_result = self.shutdown_writer_resources();
        self.terminal = true;
        abort_result.and(shutdown_result)
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

    fn refresh_run_manifest_hints(&mut self) -> OutputResult<()> {
        let refreshed_manifest_json = self
            .runs
            .iter()
            .map(|run| inspect_output_run(&run.paths, self.run_plan.output.resume))
            .collect::<OutputResult<Vec<_>>>()?;
        for (run, existing_manifest_json) in self.runs.iter_mut().zip(refreshed_manifest_json) {
            run.existing_manifest_json = existing_manifest_json;
        }
        Ok(())
    }

    fn validate_run_manifest_header_coverage(
        &self,
        headers_by_phenotype: &BTreeMap<String, serde_json::Value>,
    ) -> OutputResult<()> {
        if headers_by_phenotype.len() != self.runs.len() {
            return Err(OutputError::InvalidInput(format!(
                "Output initialization count {} does not match planned phenotype count {}.",
                headers_by_phenotype.len(),
                self.runs.len()
            )));
        }
        for run in &self.runs {
            if !headers_by_phenotype.contains_key(&run.phenotype_name) {
                return Err(OutputError::InvalidInput(format!(
                    "Missing output initialization for phenotype '{}'.",
                    run.phenotype_name
                )));
            }
        }
        Ok(())
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

    fn shutdown_writer_resources(&mut self) -> OutputResult<()> {
        self.writer_resource_owner.take().map_or(Ok(()), |mut resource_owner| resource_owner.shutdown_and_join())
    }
}

impl Drop for OutputManager {
    fn drop(&mut self) {
        if !self.terminal {
            let _ = abort_writer_sessions(self.writer_sessions.as_deref().unwrap_or_default());
            drop(self.writer_resource_owner.take());
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
    cpu_threads: Option<u32>,
    writer_threads: u32,
    writer_queue_depth: usize,
    chunks_per_parquet_file: usize,
    parquet_compression: &'static str,
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
        cpu_threads: run_plan.compute.cpu_thread_count,
        writer_threads: run_plan.output.writer_thread_count,
        writer_queue_depth: crate::WRITER_QUEUE_DEPTH,
        chunks_per_parquet_file: crate::CHUNKS_PER_PARQUET_FILE,
        parquet_compression: "zstd",
    })
    .map_err(OutputError::runtime)?;
    extend_run_manifest_metadata(&run.paths.run_directory, command, runtime)
}

#[cfg(test)]
mod tests {
    use super::finish_thread_count;

    fn output_plan(writer_thread_count: u32) -> g_plan::OutputPlan {
        g_plan::OutputPlan {
            output_run_root: "unused".to_string(),
            resume: false,
            recover_attempt: None,
            writer_thread_count,
        }
    }

    #[test]
    fn finish_thread_count_is_bounded_by_sessions_and_rejects_zero_with_work() {
        assert_eq!(finish_thread_count(&output_plan(8), 3).expect("thread count is valid"), 3);
        assert_eq!(finish_thread_count(&output_plan(2), 3).expect("thread count is valid"), 2);
        assert_eq!(finish_thread_count(&output_plan(0), 0).expect("empty output needs no threads"), 0);

        let error = finish_thread_count(&output_plan(0), 1).expect_err("zero threads with output is rejected");
        assert!(error.to_string().contains("must be positive"));
    }
}
