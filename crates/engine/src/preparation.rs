//! Native run-preparation policies.

use std::path::PathBuf;

use g_output::OutputResumeMode;

#[derive(Debug, thiserror::Error)]
pub enum PipelineResumeCompatibilityError {
    #[error(
        "Resume compatibility input counts must match: chunks_directory_count={chunks_directory_count}, \
         manifest_count={manifest_count}, header_count={header_count}."
    )]
    MismatchedInputCounts { chunks_directory_count: usize, manifest_count: usize, header_count: usize },
    #[error("Resume requires run_manifest.json.")]
    MissingManifest,
    #[error(transparent)]
    Output(#[from] g_output::OutputWriterError),
}

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
    let chunks_directory_count = chunks_directories.len();
    let manifest_count = existing_manifest_json_values.len();
    let header_count = current_header_json_values.len();
    if chunks_directory_count != manifest_count || chunks_directory_count != header_count {
        return Err(PipelineResumeCompatibilityError::MismatchedInputCounts {
            chunks_directory_count,
            manifest_count,
            header_count,
        });
    }

    for ((chunks_directory, existing_manifest_json), current_header_json) in
        chunks_directories.into_iter().zip(existing_manifest_json_values).zip(current_header_json_values)
    {
        let Some(existing_manifest_json) = existing_manifest_json else {
            return Err(PipelineResumeCompatibilityError::MissingManifest);
        };
        g_output::validate_run_manifest_compatibility(&existing_manifest_json, &current_header_json)?;
        if resume_mode == OutputResumeMode::Strict {
            let _repaired_commits =
                g_output::repair_strict_manifest_chunk_commits(&chunks_directory, &existing_manifest_json)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest_json(chunk_size: i64) -> String {
        format!(r#"{{"schema_version":7,"chunk_size":{chunk_size},"committed_chunks":[]}}"#)
    }

    #[test]
    fn validates_compatible_resume_manifests() {
        validate_pipeline_resume_compatibility(
            vec![PathBuf::from("first"), PathBuf::from("second")],
            vec![Some(manifest_json(32)), Some(manifest_json(64))],
            vec![manifest_json(32), manifest_json(64)],
            OutputResumeMode::Fast,
        )
        .unwrap();
    }

    #[test]
    fn rejects_mismatched_resume_input_counts() {
        let error = validate_pipeline_resume_compatibility(
            vec![PathBuf::from("first")],
            Vec::new(),
            vec![manifest_json(32)],
            OutputResumeMode::Fast,
        )
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "Resume compatibility input counts must match: chunks_directory_count=1, manifest_count=0, header_count=1.",
        );
    }

    #[test]
    fn rejects_missing_resume_manifest() {
        let error = validate_pipeline_resume_compatibility(
            vec![PathBuf::from("first")],
            vec![None],
            vec![manifest_json(32)],
            OutputResumeMode::Fast,
        )
        .unwrap_err();

        assert_eq!(error.to_string(), "Resume requires run_manifest.json.");
    }

    #[test]
    fn rejects_incompatible_resume_manifest_before_later_inputs() {
        let error = validate_pipeline_resume_compatibility(
            vec![PathBuf::from("first"), PathBuf::from("second")],
            vec![Some(manifest_json(32)), None],
            vec![manifest_json(64), manifest_json(64)],
            OutputResumeMode::Fast,
        )
        .unwrap_err();

        assert!(error.to_string().contains("chunk_size"));
    }

    #[test]
    fn validates_strict_resume_chunks_without_mutating_manifest() {
        let unique_suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("current time should be after Unix epoch")
            .as_nanos();
        let chunks_directory = std::env::temp_dir().join(format!("g-engine-resume-preflight-test-{unique_suffix}"));
        std::fs::create_dir_all(&chunks_directory).expect("test chunks directory should be created");

        validate_pipeline_resume_compatibility(
            vec![chunks_directory.clone()],
            vec![Some(manifest_json(32))],
            vec![manifest_json(32)],
            OutputResumeMode::Strict,
        )
        .unwrap();

        std::fs::remove_dir_all(chunks_directory).expect("test chunks directory should be removed");
    }
}
