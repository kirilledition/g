//! Deterministic run metadata and artifact payload construction.

mod artifacts;
mod error;
mod manifest;
mod types;

pub use crate::run_events::RunArtifactsPayload;
pub use artifacts::{
    build_execution_run_artifacts, build_execution_run_artifacts_from_sequences, build_multi_run_artifacts,
    build_phenotype_run_artifacts,
};
pub use error::RunMetadataError;
pub use manifest::build_run_manifest_extension;
pub use types::{
    ExecutionRunArtifactsInput, ExecutionRunArtifactsSequenceInput, PhenotypeRunArtifactsInput,
    RunManifestCommandPayload, RunManifestExtensionInput, RunManifestExtensionPayload, RunManifestRuntimePayload,
};

#[cfg(test)]
use artifacts::{OUTPUT_FORMAT_PARQUET, OUTPUT_FORMAT_REGENIE};
#[cfg(test)]
use manifest::COMMAND_INTERFACE;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_parquet_phenotype_artifacts() {
        let artifacts = build_phenotype_run_artifacts(PhenotypeRunArtifactsInput {
            output_run_directory: "run".to_string(),
            chunks_directory: "run/chunks".to_string(),
            effective_config: "config.toml".to_string(),
            phenotype_name: "height".to_string(),
            association_mode: "regenie2_linear".to_string(),
            phenotype_count: 1,
            output_format: OUTPUT_FORMAT_PARQUET.to_string(),
            final_output_path: Some("run/final.parquet".to_string()),
        });

        assert_eq!(artifacts.final_dataset.as_deref(), Some("run/chunks"));
        assert_eq!(artifacts.final_parquet.as_deref(), Some("run/final.parquet"));
        assert_eq!(artifacts.final_regenie, None);
        assert_eq!(artifacts.phenotype_name.as_deref(), Some("height"));
        assert!(artifacts.phenotype_artifacts.is_empty());
    }

    #[test]
    fn builds_execution_artifact_tree() {
        let artifacts = build_execution_run_artifacts(ExecutionRunArtifactsInput {
            association_mode: "regenie2_linear".to_string(),
            phenotype_count: 2,
            phenotype_artifacts: vec![
                PhenotypeRunArtifactsInput {
                    output_run_directory: "run/height".to_string(),
                    chunks_directory: "run/height/chunks".to_string(),
                    effective_config: "run/height/effective_config.toml".to_string(),
                    phenotype_name: "height".to_string(),
                    association_mode: "regenie2_linear".to_string(),
                    phenotype_count: 2,
                    output_format: OUTPUT_FORMAT_PARQUET.to_string(),
                    final_output_path: Some("run/height/final.parquet".to_string()),
                },
                PhenotypeRunArtifactsInput {
                    output_run_directory: "run/weight".to_string(),
                    chunks_directory: "run/weight/chunks".to_string(),
                    effective_config: "run/weight/effective_config.toml".to_string(),
                    phenotype_name: "weight".to_string(),
                    association_mode: "regenie2_linear".to_string(),
                    phenotype_count: 2,
                    output_format: OUTPUT_FORMAT_PARQUET.to_string(),
                    final_output_path: Some("run/weight/final.parquet".to_string()),
                },
            ],
        });

        assert_eq!(artifacts.output_run_directory, None);
        assert_eq!(artifacts.association_mode.as_deref(), Some("regenie2_linear"));
        assert_eq!(artifacts.phenotype_count, Some(2));
        assert_eq!(artifacts.phenotype_artifacts.len(), 2);
        assert_eq!(artifacts.phenotype_artifacts[1].phenotype_name.as_deref(), Some("weight"));
        assert_eq!(artifacts.phenotype_artifacts[1].final_parquet.as_deref(), Some("run/weight/final.parquet"));
    }

    #[test]
    fn builds_execution_artifact_tree_from_sequences() {
        let artifacts = build_execution_run_artifacts_from_sequences(ExecutionRunArtifactsSequenceInput {
            association_mode: "regenie2_linear".to_string(),
            phenotype_count: 2,
            output_format: OUTPUT_FORMAT_PARQUET.to_string(),
            output_run_directories: vec!["run/height".to_string(), "run/weight".to_string()],
            chunks_directories: vec!["run/height/chunks".to_string(), "run/weight/chunks".to_string()],
            effective_configs: vec![
                "run/height/effective_config.toml".to_string(),
                "run/weight/effective_config.toml".to_string(),
            ],
            phenotype_names: vec!["height".to_string(), "weight".to_string()],
            final_output_paths: vec![
                Some("run/height/final.parquet".to_string()),
                Some("run/weight/final.parquet".to_string()),
            ],
        })
        .unwrap();

        assert_eq!(artifacts.phenotype_artifacts.len(), 2);
        assert_eq!(artifacts.phenotype_artifacts[0].phenotype_name.as_deref(), Some("height"));
        assert_eq!(artifacts.phenotype_artifacts[1].final_parquet.as_deref(), Some("run/weight/final.parquet"));
    }

    #[test]
    fn rejects_mismatched_execution_artifact_sequences() {
        let error = build_execution_run_artifacts_from_sequences(ExecutionRunArtifactsSequenceInput {
            association_mode: "regenie2_linear".to_string(),
            phenotype_count: 2,
            output_format: OUTPUT_FORMAT_PARQUET.to_string(),
            output_run_directories: vec!["run/height".to_string()],
            chunks_directories: vec!["run/height/chunks".to_string(), "run/weight/chunks".to_string()],
            effective_configs: vec!["run/height/effective_config.toml".to_string()],
            phenotype_names: vec!["height".to_string(), "weight".to_string()],
            final_output_paths: vec![Some("run/height/final.parquet".to_string())],
        })
        .unwrap_err();

        assert_eq!(
            error,
            RunMetadataError::ArtifactSequenceLengthMismatch {
                output_run_directory_count: 1,
                chunks_directory_count: 2,
                effective_config_count: 1,
                phenotype_name_count: 2,
                final_output_path_count: 1,
            },
        );
        assert_eq!(error.to_string(), "execution artifact sequence lengths must match");
    }

    #[test]
    fn builds_single_execution_artifacts_without_wrapper() {
        let artifacts = build_execution_run_artifacts(ExecutionRunArtifactsInput {
            association_mode: "regenie2_linear".to_string(),
            phenotype_count: 1,
            phenotype_artifacts: vec![PhenotypeRunArtifactsInput {
                output_run_directory: "run/height".to_string(),
                chunks_directory: "run/height/chunks".to_string(),
                effective_config: "run/height/effective_config.toml".to_string(),
                phenotype_name: "height".to_string(),
                association_mode: "regenie2_linear".to_string(),
                phenotype_count: 1,
                output_format: OUTPUT_FORMAT_REGENIE.to_string(),
                final_output_path: Some("run/height.regenie".to_string()),
            }],
        });

        assert_eq!(artifacts.output_run_directory.as_deref(), Some("run/height"));
        assert_eq!(artifacts.final_regenie.as_deref(), Some("run/height.regenie"));
        assert!(artifacts.phenotype_artifacts.is_empty());
    }

    #[test]
    fn builds_run_manifest_extension_payload() {
        let extension = build_run_manifest_extension(RunManifestExtensionInput {
            phenotype_name: "height".to_string(),
            effective_config: "config.toml".to_string(),
            output_format: OUTPUT_FORMAT_REGENIE.to_string(),
            device: "cpu".to_string(),
            staging_depth: 2,
            native_callback_batch_size: 3,
            threads: Some(4),
            writer_threads: 5,
            writer_queue_depth: 6,
            chunks_per_arrow_file: 7,
            arrow_compression: "zstd".to_string(),
            parquet_compression: "snappy".to_string(),
            output_statistic_dtype: "float32".to_string(),
            bgen_decode_tile_variant_count: 8,
            trusted_no_missing_diploid: true,
            trusted_bgen_validation_mode: "strict".to_string(),
        });

        assert_eq!(extension.command.interface, COMMAND_INTERFACE);
        assert_eq!(extension.command.phenotype, "height");
        assert_eq!(extension.runtime.threads, Some(4));
        assert!(extension.runtime.trusted_no_missing_diploid);
    }
}
