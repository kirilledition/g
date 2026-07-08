//! Native run-preparation policies.

mod batch;
mod error;
mod initialization;
mod output_runs;
mod validation;

pub use batch::PipelineOutputPreparationBatch;
pub use error::PipelineResumeCompatibilityError;
pub use initialization::PipelineOutputInitialization;
pub use output_runs::{initialize_pipeline_output_run_batch, initialize_pipeline_output_runs};
pub use validation::validate_pipeline_resume_compatibility;

#[cfg(test)]
use g_output::OutputResumeMode;
#[cfg(test)]
use std::path::PathBuf;

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
    fn pipeline_output_preparation_batch_validates_resume_and_initializes_outputs() {
        let unique_suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("current time should be after Unix epoch")
            .as_nanos();
        let run_directory = std::env::temp_dir().join(format!("g-engine-output-prep-batch-test-{unique_suffix}"));
        let chunks_directory = run_directory.join("chunks");
        std::fs::create_dir_all(&chunks_directory).expect("test output directory should be created");

        let committed_chunks_manifest = r#"{"schema_version":7,"chunk_size":32,"committed_chunks":[{"chunk_identifier":2,"variant_start_index":2,"variant_stop_index":4,"row_count":2,"chunk_file_name":"chunk_2.arrow"}]}"#.to_string();
        let current_header = r#"{"schema_version":7,"chunk_size":32}"#.to_string();
        let preparation_batch = PipelineOutputPreparationBatch::new(
            vec![run_directory.clone()],
            vec![chunks_directory],
            vec![Some(committed_chunks_manifest)],
            vec![current_header],
            true,
            OutputResumeMode::Fast,
        )
        .unwrap();

        preparation_batch.validate_resume_compatibility().unwrap();
        let initialization = preparation_batch.initialize().unwrap();

        assert_eq!(preparation_batch.output_count(), 1);
        assert!(preparation_batch.resume());
        assert_eq!(preparation_batch.resume_mode(), OutputResumeMode::Fast);
        assert_eq!(initialization.committed_chunk_identifier_sets(), &[vec![2]]);

        std::fs::remove_dir_all(run_directory).expect("test output directory should be removed");
    }

    #[test]
    fn rejects_pipeline_output_preparation_batch_mismatched_counts() {
        let error = PipelineOutputPreparationBatch::new(
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
