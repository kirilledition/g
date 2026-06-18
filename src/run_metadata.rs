//! Deterministic run metadata and artifact payload construction.

const COMMAND_INTERFACE: &str = "g regenie";
const OUTPUT_FORMAT_PARQUET: &str = "parquet";
const OUTPUT_FORMAT_REGENIE: &str = "regenie";

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PhenotypeRunArtifactsInput {
    pub(crate) output_run_directory: String,
    pub(crate) chunks_directory: String,
    pub(crate) effective_config: String,
    pub(crate) phenotype_name: String,
    pub(crate) association_mode: String,
    pub(crate) phenotype_count: i64,
    pub(crate) output_format: String,
    pub(crate) final_output_path: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RunArtifactsPayload {
    pub(crate) output_run_directory: Option<String>,
    pub(crate) final_dataset: Option<String>,
    pub(crate) final_parquet: Option<String>,
    pub(crate) final_regenie: Option<String>,
    pub(crate) effective_config: Option<String>,
    pub(crate) phenotype_name: Option<String>,
    pub(crate) association_mode: Option<String>,
    pub(crate) phenotype_count: Option<i64>,
    pub(crate) run_id: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RunManifestExtensionInput {
    pub(crate) phenotype_name: String,
    pub(crate) effective_config: String,
    pub(crate) output_format: String,
    pub(crate) device: String,
    pub(crate) staging_depth: i64,
    pub(crate) native_callback_batch_size: i64,
    pub(crate) threads: Option<i64>,
    pub(crate) writer_threads: i64,
    pub(crate) writer_queue_depth: i64,
    pub(crate) chunks_per_arrow_file: i64,
    pub(crate) arrow_compression: String,
    pub(crate) parquet_compression: String,
    pub(crate) output_statistic_dtype: String,
    pub(crate) bgen_decode_tile_variant_count: i64,
    pub(crate) trusted_no_missing_diploid: bool,
    pub(crate) trusted_bgen_validation_mode: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RunManifestCommandPayload {
    pub(crate) interface: &'static str,
    pub(crate) phenotype: String,
    pub(crate) effective_config: String,
    pub(crate) output_format: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RunManifestRuntimePayload {
    pub(crate) device: String,
    pub(crate) staging_depth: i64,
    pub(crate) native_callback_batch_size: i64,
    pub(crate) threads: Option<i64>,
    pub(crate) writer_threads: i64,
    pub(crate) writer_queue_depth: i64,
    pub(crate) chunks_per_arrow_file: i64,
    pub(crate) arrow_compression: String,
    pub(crate) parquet_compression: String,
    pub(crate) output_statistic_dtype: String,
    pub(crate) bgen_decode_tile_variant_count: i64,
    pub(crate) trusted_no_missing_diploid: bool,
    pub(crate) trusted_bgen_validation_mode: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RunManifestExtensionPayload {
    pub(crate) command: RunManifestCommandPayload,
    pub(crate) runtime: RunManifestRuntimePayload,
}

pub(crate) fn build_phenotype_run_artifacts(input: PhenotypeRunArtifactsInput) -> RunArtifactsPayload {
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
        phenotype_name: Some(phenotype_name),
        association_mode: Some(association_mode),
        phenotype_count: Some(phenotype_count),
        run_id: None,
    }
}

pub(crate) fn build_multi_run_artifacts(association_mode: &str, phenotype_count: i64) -> RunArtifactsPayload {
    RunArtifactsPayload {
        output_run_directory: None,
        final_dataset: None,
        final_parquet: None,
        final_regenie: None,
        effective_config: None,
        phenotype_name: None,
        association_mode: Some(association_mode.to_string()),
        phenotype_count: Some(phenotype_count),
        run_id: None,
    }
}

pub(crate) fn build_run_manifest_extension(input: RunManifestExtensionInput) -> RunManifestExtensionPayload {
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
