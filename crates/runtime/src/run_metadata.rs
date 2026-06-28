//! Deterministic run metadata and artifact payload construction.

pub use crate::run_events::RunArtifactsPayload;

const COMMAND_INTERFACE: &str = "g regenie";
const OUTPUT_FORMAT_PARQUET: &str = "parquet";
const OUTPUT_FORMAT_REGENIE: &str = "regenie";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhenotypeRunArtifactsInput {
    pub output_run_directory: String,
    pub chunks_directory: String,
    pub effective_config: String,
    pub phenotype_name: String,
    pub association_mode: String,
    pub phenotype_count: i64,
    pub output_format: String,
    pub final_output_path: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionRunArtifactsInput {
    pub association_mode: String,
    pub phenotype_count: i64,
    pub phenotype_artifacts: Vec<PhenotypeRunArtifactsInput>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunManifestExtensionInput {
    pub phenotype_name: String,
    pub effective_config: String,
    pub output_format: String,
    pub device: String,
    pub staging_depth: i64,
    pub native_callback_batch_size: i64,
    pub threads: Option<i64>,
    pub writer_threads: i64,
    pub writer_queue_depth: i64,
    pub chunks_per_arrow_file: i64,
    pub arrow_compression: String,
    pub parquet_compression: String,
    pub output_statistic_dtype: String,
    pub bgen_decode_tile_variant_count: i64,
    pub trusted_no_missing_diploid: bool,
    pub trusted_bgen_validation_mode: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunManifestCommandPayload {
    pub interface: &'static str,
    pub phenotype: String,
    pub effective_config: String,
    pub output_format: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunManifestRuntimePayload {
    pub device: String,
    pub staging_depth: i64,
    pub native_callback_batch_size: i64,
    pub threads: Option<i64>,
    pub writer_threads: i64,
    pub writer_queue_depth: i64,
    pub chunks_per_arrow_file: i64,
    pub arrow_compression: String,
    pub parquet_compression: String,
    pub output_statistic_dtype: String,
    pub bgen_decode_tile_variant_count: i64,
    pub trusted_no_missing_diploid: bool,
    pub trusted_bgen_validation_mode: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunManifestExtensionPayload {
    pub command: RunManifestCommandPayload,
    pub runtime: RunManifestRuntimePayload,
}

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

#[must_use]
pub fn build_run_manifest_extension(input: RunManifestExtensionInput) -> RunManifestExtensionPayload {
    let RunManifestExtensionInput {
        phenotype_name,
        effective_config,
        output_format,
        device,
        staging_depth,
        native_callback_batch_size,
        threads,
        writer_threads,
        writer_queue_depth,
        chunks_per_arrow_file,
        arrow_compression,
        parquet_compression,
        output_statistic_dtype,
        bgen_decode_tile_variant_count,
        trusted_no_missing_diploid,
        trusted_bgen_validation_mode,
    } = input;
    RunManifestExtensionPayload {
        command: RunManifestCommandPayload {
            interface: COMMAND_INTERFACE,
            phenotype: phenotype_name,
            effective_config,
            output_format,
        },
        runtime: RunManifestRuntimePayload {
            device,
            staging_depth,
            native_callback_batch_size,
            threads,
            writer_threads,
            writer_queue_depth,
            chunks_per_arrow_file,
            arrow_compression,
            parquet_compression,
            output_statistic_dtype,
            bgen_decode_tile_variant_count,
            trusted_no_missing_diploid,
            trusted_bgen_validation_mode,
        },
    }
}

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
