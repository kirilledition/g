use serde::Serialize;

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
pub struct ExecutionRunArtifactsSequenceInput {
    pub association_mode: String,
    pub phenotype_count: i64,
    pub output_format: String,
    pub output_run_directories: Vec<String>,
    pub chunks_directories: Vec<String>,
    pub effective_configs: Vec<String>,
    pub phenotype_names: Vec<String>,
    pub final_output_paths: Vec<Option<String>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunManifestExtensionInput {
    pub phenotype_name: String,
    pub effective_config: String,
    pub output_format: String,
    pub device: String,
    pub staging_depth: i64,
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

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RunManifestCommandPayload {
    pub interface: &'static str,
    pub phenotype: String,
    pub effective_config: String,
    pub output_format: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RunManifestRuntimePayload {
    pub device: String,
    pub staging_depth: i64,
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

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RunManifestExtensionPayload {
    pub command: RunManifestCommandPayload,
    pub runtime: RunManifestRuntimePayload,
}
