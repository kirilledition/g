use std::path::PathBuf;

use g_output::OutputResumeMode;

use super::error::PipelineResumeCompatibilityError;
use super::initialization::PipelineOutputInitialization;
use super::output_runs::initialize_pipeline_outputs_after_count_check;
use super::validation::{
    validate_pipeline_input_counts, validate_pipeline_output_directory_counts,
    validate_pipeline_resume_compatibility_after_count_check,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PipelineOutputPreparationBatch {
    run_directories: Vec<PathBuf>,
    chunks_directories: Vec<PathBuf>,
    existing_manifest_json_values: Vec<Option<String>>,
    current_header_json_values: Vec<String>,
    resume: bool,
    resume_mode: OutputResumeMode,
}

impl PipelineOutputPreparationBatch {
    /// Build a native output-preparation batch from per-output preparation inputs.
    ///
    /// # Errors
    ///
    /// Returns an error when per-output input counts are inconsistent.
    pub fn new(
        run_directories: Vec<PathBuf>,
        chunks_directories: Vec<PathBuf>,
        existing_manifest_json_values: Vec<Option<String>>,
        current_header_json_values: Vec<String>,
        resume: bool,
        resume_mode: OutputResumeMode,
    ) -> Result<Self, PipelineResumeCompatibilityError> {
        validate_pipeline_input_counts(
            chunks_directories.len(),
            existing_manifest_json_values.len(),
            current_header_json_values.len(),
        )?;
        validate_pipeline_output_directory_counts(run_directories.len(), chunks_directories.len())?;
        Ok(Self {
            run_directories,
            chunks_directories,
            existing_manifest_json_values,
            current_header_json_values,
            resume,
            resume_mode,
        })
    }

    #[must_use]
    pub fn output_count(&self) -> usize {
        self.run_directories.len()
    }

    #[must_use]
    pub const fn resume(&self) -> bool {
        self.resume
    }

    #[must_use]
    pub const fn resume_mode(&self) -> OutputResumeMode {
        self.resume_mode
    }

    /// Validate all resume manifests before output initialization mutates any run directory.
    ///
    /// # Errors
    ///
    /// Returns an error when a resume manifest is missing, manifest compatibility fails, or strict
    /// resume chunk validation fails.
    pub fn validate_resume_compatibility(&self) -> Result<(), PipelineResumeCompatibilityError> {
        validate_pipeline_resume_compatibility_after_count_check(
            self.chunks_directories.clone(),
            self.existing_manifest_json_values.clone(),
            self.current_header_json_values.clone(),
            self.resume_mode,
        )
    }

    /// Initialize all output runs after validating every resume manifest when resume is enabled.
    ///
    /// # Errors
    ///
    /// Returns an error when all-manifest resume validation fails or any output initialization fails.
    pub fn initialize(&self) -> Result<PipelineOutputInitialization, PipelineResumeCompatibilityError> {
        if self.resume {
            self.validate_resume_compatibility()?;
        }
        let committed_chunk_identifier_sets = initialize_pipeline_outputs_after_count_check(
            self.run_directories.clone(),
            self.chunks_directories.clone(),
            self.existing_manifest_json_values.clone(),
            self.current_header_json_values.clone(),
            self.resume,
            self.resume_mode,
        )?;
        Ok(PipelineOutputInitialization::new(committed_chunk_identifier_sets))
    }
}
