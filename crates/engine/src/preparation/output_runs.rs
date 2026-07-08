use std::path::PathBuf;

use g_output::OutputResumeMode;

use super::batch::PipelineOutputPreparationBatch;
use super::error::PipelineResumeCompatibilityError;
use super::initialization::PipelineOutputInitialization;

/// Initialize all output runs after validating every resume manifest.
///
/// # Errors
///
/// Returns an error when input counts differ, all-manifest resume validation fails, or any output initialization fails.
pub fn initialize_pipeline_output_runs(
    run_directories: Vec<PathBuf>,
    chunks_directories: Vec<PathBuf>,
    existing_manifest_json_values: Vec<Option<String>>,
    current_header_json_values: Vec<String>,
    resume: bool,
    resume_mode: OutputResumeMode,
) -> Result<Vec<Vec<i64>>, PipelineResumeCompatibilityError> {
    let preparation_batch = PipelineOutputPreparationBatch::new(
        run_directories,
        chunks_directories,
        existing_manifest_json_values,
        current_header_json_values,
        resume,
        resume_mode,
    )?;
    Ok(preparation_batch.initialize()?.committed_chunk_identifier_sets().to_vec())
}

/// Initialize all output runs and return a native result handle.
///
/// # Errors
///
/// Returns an error when input counts differ, all-manifest resume validation fails, or any output initialization fails.
pub fn initialize_pipeline_output_run_batch(
    run_directories: Vec<PathBuf>,
    chunks_directories: Vec<PathBuf>,
    existing_manifest_json_values: Vec<Option<String>>,
    current_header_json_values: Vec<String>,
    resume: bool,
    resume_mode: OutputResumeMode,
) -> Result<PipelineOutputInitialization, PipelineResumeCompatibilityError> {
    let preparation_batch = PipelineOutputPreparationBatch::new(
        run_directories,
        chunks_directories,
        existing_manifest_json_values,
        current_header_json_values,
        resume,
        resume_mode,
    )?;
    preparation_batch.initialize()
}

pub(super) fn initialize_pipeline_outputs_after_count_check(
    run_directories: Vec<PathBuf>,
    chunks_directories: Vec<PathBuf>,
    existing_manifest_json_values: Vec<Option<String>>,
    current_header_json_values: Vec<String>,
    resume: bool,
    resume_mode: OutputResumeMode,
) -> Result<Vec<Vec<i64>>, PipelineResumeCompatibilityError> {
    let mut committed_chunk_identifier_sets = Vec::with_capacity(run_directories.len());
    for (((run_directory, chunks_directory), existing_manifest_json), current_header_json) in run_directories
        .into_iter()
        .zip(chunks_directories)
        .zip(existing_manifest_json_values)
        .zip(current_header_json_values)
    {
        let initialized_output_run = g_output::initialize_output_run(
            &run_directory,
            &chunks_directory,
            existing_manifest_json.as_deref(),
            &current_header_json,
            resume,
            resume_mode,
        )?;
        committed_chunk_identifier_sets.push(initialized_output_run.committed_chunk_identifiers);
    }
    Ok(committed_chunk_identifier_sets)
}
