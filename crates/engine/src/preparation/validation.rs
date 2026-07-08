use std::path::PathBuf;

use g_output::OutputResumeMode;

use super::error::PipelineResumeCompatibilityError;

/// Validate all resume manifests before output initialization mutates any run directory.
///
/// # Errors
///
/// Returns an error when input counts differ, a resume manifest is missing, manifest compatibility fails, or strict
/// resume chunk validation fails.
pub fn validate_pipeline_resume_compatibility(
    chunks_directories: Vec<PathBuf>,
    existing_manifest_json_values: Vec<Option<String>>,
    current_header_json_values: Vec<String>,
    resume_mode: OutputResumeMode,
) -> Result<(), PipelineResumeCompatibilityError> {
    validate_pipeline_input_counts(
        chunks_directories.len(),
        existing_manifest_json_values.len(),
        current_header_json_values.len(),
    )?;

    validate_pipeline_resume_compatibility_after_count_check(
        chunks_directories,
        existing_manifest_json_values,
        current_header_json_values,
        resume_mode,
    )
}

pub(super) fn validate_pipeline_input_counts(
    chunks_directory_count: usize,
    manifest_count: usize,
    header_count: usize,
) -> Result<(), PipelineResumeCompatibilityError> {
    if chunks_directory_count != manifest_count || chunks_directory_count != header_count {
        return Err(PipelineResumeCompatibilityError::MismatchedInputCounts {
            chunks_directory_count,
            manifest_count,
            header_count,
        });
    }
    Ok(())
}

pub(super) fn validate_pipeline_output_directory_counts(
    run_directory_count: usize,
    chunks_directory_count: usize,
) -> Result<(), PipelineResumeCompatibilityError> {
    if run_directory_count != chunks_directory_count {
        return Err(PipelineResumeCompatibilityError::MismatchedOutputRunDirectoryCount {
            run_directory_count,
            chunks_directory_count,
        });
    }
    Ok(())
}

pub(super) fn validate_pipeline_resume_compatibility_after_count_check(
    chunks_directories: Vec<PathBuf>,
    existing_manifest_json_values: Vec<Option<String>>,
    current_header_json_values: Vec<String>,
    resume_mode: OutputResumeMode,
) -> Result<(), PipelineResumeCompatibilityError> {
    for ((chunks_directory, existing_manifest_json), current_header_json) in
        chunks_directories.into_iter().zip(existing_manifest_json_values).zip(current_header_json_values)
    {
        let Some(existing_manifest_json) = existing_manifest_json else {
            return Err(PipelineResumeCompatibilityError::MissingManifest);
        };
        g_output::validate_run_manifest_compatibility(&existing_manifest_json, &current_header_json)?;
        if resume_mode == OutputResumeMode::Strict {
            let _repaired_commits =
                g_output::admin::repair_strict_manifest_chunk_commits(&chunks_directory, &existing_manifest_json)?;
        }
    }
    Ok(())
}
