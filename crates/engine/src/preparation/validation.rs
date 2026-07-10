use std::path::PathBuf;

use g_output::{OutputResumeMode, validate_output_run_resume_compatibility};

use super::error::PipelineResumeCompatibilityError;

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
        validate_output_run_resume_compatibility(
            &chunks_directory,
            &existing_manifest_json,
            &current_header_json,
            resume_mode,
        )?;
    }
    Ok(())
}
