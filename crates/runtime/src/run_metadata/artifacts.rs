use crate::run_events::RunArtifactsPayload;

use super::error::RunMetadataError;
use super::types::{ExecutionRunArtifactsInput, ExecutionRunArtifactsSequenceInput, PhenotypeRunArtifactsInput};

pub(super) const OUTPUT_FORMAT_PARQUET: &str = "parquet";
pub(super) const OUTPUT_FORMAT_REGENIE: &str = "regenie";

#[must_use]
pub fn build_phenotype_run_artifacts(input: PhenotypeRunArtifactsInput) -> RunArtifactsPayload {
    let PhenotypeRunArtifactsInput {
        output_run_directory,
        chunks_directory,
        effective_config,
        phenotype_name,
        association_mode,
        phenotype_count,
        output_format,
        final_output_path,
    } = input;
    let final_dataset = if output_format == OUTPUT_FORMAT_PARQUET { Some(chunks_directory) } else { None };
    let (final_parquet, final_regenie) =
        if output_format == OUTPUT_FORMAT_REGENIE { (None, final_output_path) } else { (final_output_path, None) };
    RunArtifactsPayload {
        output_run_directory: Some(output_run_directory),
        final_dataset,
        final_parquet,
        final_regenie,
        effective_config: Some(effective_config),
        phenotype_artifacts: Vec::new(),
        phenotype_name: Some(phenotype_name),
        association_mode: Some(association_mode),
        phenotype_count: Some(phenotype_count),
        run_id: None,
    }
}

#[must_use]
pub fn build_execution_run_artifacts(input: ExecutionRunArtifactsInput) -> RunArtifactsPayload {
    let ExecutionRunArtifactsInput { association_mode, phenotype_count, phenotype_artifacts } = input;
    let mut finalized_phenotype_artifacts =
        phenotype_artifacts.into_iter().map(build_phenotype_run_artifacts).collect::<Vec<_>>();
    if finalized_phenotype_artifacts.len() == 1
        && let Some(phenotype_artifact) = finalized_phenotype_artifacts.pop()
    {
        return phenotype_artifact;
    }
    RunArtifactsPayload {
        output_run_directory: None,
        final_dataset: None,
        final_parquet: None,
        final_regenie: None,
        effective_config: None,
        phenotype_artifacts: finalized_phenotype_artifacts,
        phenotype_name: None,
        association_mode: Some(association_mode),
        phenotype_count: Some(phenotype_count),
        run_id: None,
    }
}

/// Build execution artifacts from parallel per-phenotype sequences.
///
/// # Errors
///
/// Returns an error when the supplied per-phenotype sequences have different lengths.
pub fn build_execution_run_artifacts_from_sequences(
    input: ExecutionRunArtifactsSequenceInput,
) -> Result<RunArtifactsPayload, RunMetadataError> {
    let ExecutionRunArtifactsSequenceInput {
        association_mode,
        phenotype_count,
        output_format,
        output_run_directories,
        chunks_directories,
        effective_configs,
        phenotype_names,
        final_output_paths,
    } = input;
    let phenotype_name_count = phenotype_names.len();
    let sequence_lengths = [
        output_run_directories.len(),
        chunks_directories.len(),
        effective_configs.len(),
        phenotype_name_count,
        final_output_paths.len(),
    ];
    if sequence_lengths.iter().any(|sequence_length| *sequence_length != phenotype_name_count) {
        return Err(RunMetadataError::ArtifactSequenceLengthMismatch {
            output_run_directory_count: output_run_directories.len(),
            chunks_directory_count: chunks_directories.len(),
            effective_config_count: effective_configs.len(),
            phenotype_name_count,
            final_output_path_count: final_output_paths.len(),
        });
    }
    let phenotype_artifacts = output_run_directories
        .into_iter()
        .zip(chunks_directories)
        .zip(effective_configs)
        .zip(phenotype_names)
        .zip(final_output_paths)
        .map(|((((output_run_directory, chunks_directory), effective_config), phenotype_name), final_output_path)| {
            PhenotypeRunArtifactsInput {
                output_run_directory,
                chunks_directory,
                effective_config,
                phenotype_name,
                association_mode: association_mode.clone(),
                phenotype_count,
                output_format: output_format.clone(),
                final_output_path,
            }
        })
        .collect();
    Ok(build_execution_run_artifacts(ExecutionRunArtifactsInput {
        association_mode,
        phenotype_count,
        phenotype_artifacts,
    }))
}

#[must_use]
pub fn build_multi_run_artifacts(association_mode: &str, phenotype_count: i64) -> RunArtifactsPayload {
    RunArtifactsPayload {
        output_run_directory: None,
        final_dataset: None,
        final_parquet: None,
        final_regenie: None,
        effective_config: None,
        phenotype_artifacts: Vec::new(),
        phenotype_name: None,
        association_mode: Some(association_mode.to_string()),
        phenotype_count: Some(phenotype_count),
        run_id: None,
    }
}
