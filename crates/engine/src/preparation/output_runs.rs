use std::path::PathBuf;

use g_output::OutputResumeMode;

use super::error::PipelineResumeCompatibilityError;

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
        let committed_chunk_identifiers = g_output::initialize_output_run(
            &run_directory,
            &chunks_directory,
            existing_manifest_json.as_deref(),
            &current_header_json,
            resume,
            resume_mode,
        )?;
        committed_chunk_identifier_sets.push(committed_chunk_identifiers);
    }
    Ok(committed_chunk_identifier_sets)
}
