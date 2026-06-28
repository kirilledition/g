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
    #[error(
        "Pipeline output run directory count must match chunks directory count: run_directory_count={run_directory_count}, \
         chunks_directory_count={chunks_directory_count}."
    )]
    MismatchedOutputRunDirectoryCount { run_directory_count: usize, chunks_directory_count: usize },
    #[error("Resume requires run_manifest.json.")]
    MissingManifest,
    #[error(transparent)]
    Output(#[from] g_output::OutputWriterError),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PipelineOutputInitialization {
    committed_chunk_identifier_sets: Vec<Vec<i64>>,
}

impl PipelineOutputInitialization {
    #[must_use]
    pub fn new(committed_chunk_identifier_sets: Vec<Vec<i64>>) -> Self {
        Self { committed_chunk_identifier_sets }
    }

    #[must_use]
    pub fn committed_chunk_identifier_sets(&self) -> &[Vec<i64>] {
        &self.committed_chunk_identifier_sets
    }

    #[must_use]
    pub fn committed_chunk_identifiers(&self, output_index: usize) -> Option<&[i64]> {
        self.committed_chunk_identifier_sets.get(output_index).map(Vec::as_slice)
    }

    #[must_use]
    pub fn output_count(&self) -> usize {
        self.committed_chunk_identifier_sets.len()
    }
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
    validate_pipeline_input_counts(
        chunks_directories.len(),
        existing_manifest_json_values.len(),
        current_header_json_values.len(),
    )?;
    if run_directories.len() != chunks_directories.len() {
        return Err(PipelineResumeCompatibilityError::MismatchedOutputRunDirectoryCount {
            run_directory_count: run_directories.len(),
            chunks_directory_count: chunks_directories.len(),
        });
    }

    if resume {
        validate_pipeline_resume_compatibility_after_count_check(
            chunks_directories.clone(),
            existing_manifest_json_values.clone(),
            current_header_json_values.clone(),
            resume_mode,
        )?;
    }

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
    let committed_chunk_identifier_sets = initialize_pipeline_output_runs(
        run_directories,
        chunks_directories,
        existing_manifest_json_values,
        current_header_json_values,
        resume,
        resume_mode,
    )?;
    Ok(PipelineOutputInitialization::new(committed_chunk_identifier_sets))
}

fn validate_pipeline_input_counts(
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

fn validate_pipeline_resume_compatibility_after_count_check(
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

    #[test]
    fn initializes_pipeline_outputs_after_resume_preflight() {
        let unique_suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("current time should be after Unix epoch")
            .as_nanos();
        let run_directory = std::env::temp_dir().join(format!("g-engine-output-init-test-{unique_suffix}"));
        let chunks_directory = run_directory.join("chunks");
        std::fs::create_dir_all(&chunks_directory).expect("test output directory should be created");

        let committed_chunks_manifest = r#"{"schema_version":7,"chunk_size":32,"committed_chunks":[{"chunk_identifier":2,"variant_start_index":2,"variant_stop_index":4,"row_count":2,"chunk_file_name":"chunk_2.arrow"}]}"#.to_string();
        let current_header = r#"{"schema_version":7,"chunk_size":32}"#.to_string();
        let committed_chunk_identifier_sets = initialize_pipeline_output_runs(
            vec![run_directory.clone()],
            vec![chunks_directory],
            vec![Some(committed_chunks_manifest)],
            vec![current_header],
            true,
            OutputResumeMode::Fast,
        )
        .unwrap();

        assert_eq!(committed_chunk_identifier_sets, vec![vec![2]]);
        assert!(run_directory.join("run_manifest.json").exists());
        std::fs::remove_dir_all(run_directory).expect("test output directory should be removed");
    }

    #[test]
    fn initializes_pipeline_output_batch_handle() {
        let unique_suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("current time should be after Unix epoch")
            .as_nanos();
        let run_directory = std::env::temp_dir().join(format!("g-engine-output-batch-test-{unique_suffix}"));
        let chunks_directory = run_directory.join("chunks");
        std::fs::create_dir_all(&chunks_directory).expect("test output directory should be created");

        let committed_chunks_manifest = r#"{"schema_version":7,"chunk_size":32,"committed_chunks":[{"chunk_identifier":2,"variant_start_index":2,"variant_stop_index":4,"row_count":2,"chunk_file_name":"chunk_2.arrow"}]}"#.to_string();
        let current_header = r#"{"schema_version":7,"chunk_size":32}"#.to_string();
        let initialization = initialize_pipeline_output_run_batch(
            vec![run_directory.clone()],
            vec![chunks_directory],
            vec![Some(committed_chunks_manifest)],
            vec![current_header],
            true,
            OutputResumeMode::Fast,
        )
        .unwrap();

        assert_eq!(initialization.output_count(), 1);
        assert_eq!(initialization.committed_chunk_identifier_sets(), &[vec![2]]);
        assert_eq!(initialization.committed_chunk_identifiers(0), Some([2].as_slice()));
        assert_eq!(initialization.committed_chunk_identifiers(1), None);

        std::fs::remove_dir_all(run_directory).expect("test output directory should be removed");
    }

    #[test]
    fn rejects_pipeline_output_initialization_mismatched_counts() {
        let error = initialize_pipeline_output_runs(
            vec![PathBuf::from("run")],
            Vec::new(),
            Vec::new(),
            Vec::new(),
            false,
            OutputResumeMode::Fast,
        )
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "Pipeline output run directory count must match chunks directory count: \
             run_directory_count=1, chunks_directory_count=0.",
        );
    }
}
