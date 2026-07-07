#![allow(clippy::missing_errors_doc)]

use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::{Value, json};

use crate::error::OutputError;

use super::{
    JAX_MATMUL_PRECISION_WHEN_UNSET, ManifestFileFingerprintCache, OUTPUT_SCHEMA_VERSION, RESUME_POLICY,
    RUN_MANIFEST_SCHEMA_VERSION, build_manifest_value_sha256, manifest_file_fingerprint_to_value,
};

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
pub struct CurrentRunManifestHeaderInput {
    pub association_mode: String,
    pub association_backend_kind: String,
    pub bgen_path: PathBuf,
    pub sample_path: Option<PathBuf>,
    pub phenotype_path: PathBuf,
    pub phenotype_name: String,
    pub covariate_path: Option<PathBuf>,
    pub covariate_names: Vec<String>,
    pub prediction_list_path: PathBuf,
    pub prediction_loco_files_json: String,
    pub sample_count: i64,
    pub variant_count: i64,
    pub chunk_size: i64,
    pub variant_limit: Option<i64>,
    pub binary_correction_plan_method: String,
    pub binary_correction_plan_p_threshold: f64,
    pub binary_correction_plan_firth_se: bool,
    pub trusted_no_missing_diploid: bool,
    pub sample_key_mode: String,
    pub binary_kernel_config_json: Option<String>,
    pub bgen_decode_tile_variant_count: i64,
    pub trusted_bgen_validation_mode: String,
    pub jax_device: String,
    pub jax_enable_x64: bool,
    pub jax_matmul_precision: Option<String>,
    pub requested_gpu_genotype_format: String,
    pub gpu_genotype_format: String,
    pub score_dtype: String,
    pub firth_dtype: String,
    pub multi_phenotype_sample_mode: String,
    pub phenotype_compute_group_mode: Option<String>,
    pub phenotype_compute_group_indices: Option<Vec<u32>>,
    pub phenotype_compute_group_names: Option<Vec<String>>,
    pub phenotype_compute_group_sample_mode: Option<String>,
    pub sample_set_fingerprint: Option<String>,
    pub covariate_design_fingerprint: Option<String>,
    pub prediction_alignment_fingerprint: Option<String>,
    pub output_format: String,
    pub finalize_parquet: bool,
    pub writer_thread_count: i64,
    pub writer_queue_depth: i64,
    pub chunks_per_arrow_file: i64,
    pub arrow_compression: String,
    pub parquet_compression: String,
    pub output_statistic_dtype: String,
}

#[allow(clippy::too_many_lines)]
pub fn build_current_run_manifest_header_json(input: CurrentRunManifestHeaderInput) -> Result<String, OutputError> {
    let mut fingerprint_cache = ManifestFileFingerprintCache::new();
    build_current_run_manifest_header_json_with_cache(input, &mut fingerprint_cache)
}

#[allow(clippy::too_many_lines)]
pub fn build_current_run_manifest_header_json_with_cache(
    input: CurrentRunManifestHeaderInput,
    fingerprint_cache: &mut ManifestFileFingerprintCache,
) -> Result<String, OutputError> {
    let bgen_fingerprint =
        build_required_file_fingerprint_with_cache(fingerprint_cache, &input.bgen_path, false, "BGEN")?;
    let sample_fingerprint =
        build_optional_file_fingerprint_with_cache(fingerprint_cache, input.sample_path.as_deref(), true)?;
    let phenotype_file_fingerprint =
        build_required_file_fingerprint_with_cache(fingerprint_cache, &input.phenotype_path, true, "phenotype file")?;
    let covariate_file_fingerprint =
        build_optional_file_fingerprint_with_cache(fingerprint_cache, input.covariate_path.as_deref(), true)?;
    let prediction_list_fingerprint = build_required_file_fingerprint_with_cache(
        fingerprint_cache,
        &input.prediction_list_path,
        true,
        "prediction list",
    )?;
    let prediction_loco_files = serde_json::from_str::<Value>(&input.prediction_loco_files_json)
        .map_err(|error| OutputError::InvalidInput(error.to_string()))?;
    if !prediction_loco_files.is_array() {
        return Err(OutputError::InvalidInput("prediction_loco_files_json must contain a JSON array.".to_string()));
    }
    let prediction_inputs = json!({
        "prediction_list": prediction_list_fingerprint.clone(),
        "loco_files": prediction_loco_files,
    });
    let phenotype_compute_group_id = build_current_header_phenotype_compute_group_id(&input)?;
    let binary_correction_plan = json!({
        "method": input.binary_correction_plan_method,
        "p_threshold": input.binary_correction_plan_p_threshold,
        "firth_se": input.binary_correction_plan_firth_se,
    });
    let binary_kernel_config = match input.binary_kernel_config_json {
        Some(binary_kernel_config_json) => serde_json::from_str::<Value>(&binary_kernel_config_json)
            .map_err(|error| OutputError::InvalidInput(error.to_string()))?,
        None => Value::Null,
    };
    let association_backend = json!({
        "kind": input.association_backend_kind,
        "association_mode": input.association_mode,
        "device": input.jax_device,
        "genotype_format": input.gpu_genotype_format,
    });
    let jax_policy = json!({
        "device": input.jax_device,
        "enable_x64": input.jax_enable_x64,
        "matmul_precision": input.jax_matmul_precision.unwrap_or_else(|| JAX_MATMUL_PRECISION_WHEN_UNSET.to_string()),
    });
    let output_writer = json!({
        "output_format": input.output_format,
        "finalize_parquet": input.finalize_parquet,
        "writer_thread_count": input.writer_thread_count,
        "writer_queue_depth": input.writer_queue_depth,
        "chunks_per_arrow_file": input.chunks_per_arrow_file,
        "arrow_compression": input.arrow_compression,
        "parquet_compression": input.parquet_compression,
        "result_statistic_dtype": input.output_statistic_dtype,
    });
    let execution_plan = json!({
        "manifest_schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "association_mode": input.association_mode,
        "association_backend": association_backend,
        "bgen": bgen_fingerprint,
        "sample": sample_fingerprint,
        "phenotype_file": phenotype_file_fingerprint,
        "phenotype_name": input.phenotype_name,
        "covariate_file": covariate_file_fingerprint,
        "covariate_names": input.covariate_names,
        "prediction_list": prediction_list_fingerprint,
        "prediction_inputs": prediction_inputs,
        "sample_count": input.sample_count,
        "variant_count": input.variant_count,
        "chunk_size": input.chunk_size,
        "variant_limit": input.variant_limit,
        "binary_correction_plan": binary_correction_plan,
        "binary_kernel_config": binary_kernel_config,
        "trusted_no_missing_diploid": input.trusted_no_missing_diploid,
        "trusted_bgen_validation_mode": input.trusted_bgen_validation_mode,
        "sample_key_mode": input.sample_key_mode,
        "bgen_decode_tile_variant_count": input.bgen_decode_tile_variant_count,
        "jax_policy": jax_policy,
        "requested_gpu_genotype_format": input.requested_gpu_genotype_format,
        "gpu_genotype_format": input.gpu_genotype_format,
        "score_dtype": input.score_dtype,
        "firth_dtype": input.firth_dtype,
        "multi_phenotype_sample_mode": input.multi_phenotype_sample_mode,
        "phenotype_compute_group_id": phenotype_compute_group_id,
        "sample_set_fingerprint": input.sample_set_fingerprint,
        "covariate_design_fingerprint": input.covariate_design_fingerprint,
        "prediction_alignment_fingerprint": input.prediction_alignment_fingerprint,
        "output_writer": output_writer,
        "resume_policy": RESUME_POLICY,
    });
    let execution_plan_hash = build_manifest_value_sha256(&execution_plan)?;
    let current_header = json!({
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "association_mode": input.association_mode,
        "association_backend": execution_plan["association_backend"].clone(),
        "bgen": execution_plan["bgen"].clone(),
        "sample": execution_plan["sample"].clone(),
        "phenotype_file": execution_plan["phenotype_file"].clone(),
        "phenotype_name": input.phenotype_name,
        "covariate_file": execution_plan["covariate_file"].clone(),
        "covariate_names": execution_plan["covariate_names"].clone(),
        "prediction_list": execution_plan["prediction_list"].clone(),
        "prediction_inputs": execution_plan["prediction_inputs"].clone(),
        "sample_count": input.sample_count,
        "variant_count": input.variant_count,
        "chunk_size": input.chunk_size,
        "variant_limit": input.variant_limit,
        "binary_correction_plan": execution_plan["binary_correction_plan"].clone(),
        "binary_kernel_config": execution_plan["binary_kernel_config"].clone(),
        "trusted_no_missing_diploid": input.trusted_no_missing_diploid,
        "trusted_bgen_validation_mode": input.trusted_bgen_validation_mode,
        "sample_key_mode": input.sample_key_mode,
        "bgen_decode_tile_variant_count": input.bgen_decode_tile_variant_count,
        "jax_policy": execution_plan["jax_policy"].clone(),
        "requested_gpu_genotype_format": execution_plan["requested_gpu_genotype_format"].clone(),
        "gpu_genotype_format": input.gpu_genotype_format,
        "score_dtype": input.score_dtype,
        "firth_dtype": input.firth_dtype,
        "multi_phenotype_sample_mode": input.multi_phenotype_sample_mode,
        "phenotype_compute_group_id": execution_plan["phenotype_compute_group_id"].clone(),
        "sample_set_fingerprint": input.sample_set_fingerprint,
        "covariate_design_fingerprint": input.covariate_design_fingerprint,
        "prediction_alignment_fingerprint": input.prediction_alignment_fingerprint,
        "output_writer": execution_plan["output_writer"].clone(),
        "resume_policy": RESUME_POLICY,
        "execution_plan": execution_plan,
        "execution_plan_hash": execution_plan_hash,
    });
    serde_json::to_string(&current_header).map_err(OutputError::runtime)
}

fn build_current_header_phenotype_compute_group_id(
    input: &CurrentRunManifestHeaderInput,
) -> Result<Option<String>, OutputError> {
    let Some(group_mode) = input.phenotype_compute_group_mode.as_deref() else {
        return ensure_no_partial_current_header_phenotype_compute_group(input);
    };
    let phenotype_indices = input.phenotype_compute_group_indices.as_ref().ok_or_else(|| {
        OutputError::InvalidInput("phenotype_compute_group_indices is required with group mode.".to_string())
    })?;
    let phenotype_names = input.phenotype_compute_group_names.as_ref().ok_or_else(|| {
        OutputError::InvalidInput("phenotype_compute_group_names is required with group mode.".to_string())
    })?;
    let sample_mode = input.phenotype_compute_group_sample_mode.as_deref().ok_or_else(|| {
        OutputError::InvalidInput("phenotype_compute_group_sample_mode is required with group mode.".to_string())
    })?;
    if phenotype_indices.len() != phenotype_names.len() {
        return Err(OutputError::InvalidInput(
            "phenotype compute group indices and names must have the same length.".to_string(),
        ));
    }
    let phenotype_compute_group = g_plan::PhenotypeComputeGroup {
        group_mode: parse_current_header_phenotype_compute_group_mode(group_mode)?,
        phenotype_indices: phenotype_indices.clone(),
        phenotype_names: phenotype_names.clone(),
        sample_mode: parse_current_header_multi_phenotype_sample_mode(sample_mode)?,
        sample_set_fingerprint: input.sample_set_fingerprint.clone(),
        covariate_design_fingerprint: input.covariate_design_fingerprint.clone(),
        prediction_alignment_fingerprint: input.prediction_alignment_fingerprint.clone(),
    };
    Ok(Some(g_plan::build_phenotype_compute_group_id(&phenotype_compute_group)))
}

fn ensure_no_partial_current_header_phenotype_compute_group(
    input: &CurrentRunManifestHeaderInput,
) -> Result<Option<String>, OutputError> {
    if input.phenotype_compute_group_indices.is_some()
        || input.phenotype_compute_group_names.is_some()
        || input.phenotype_compute_group_sample_mode.is_some()
    {
        return Err(OutputError::InvalidInput(
            "phenotype compute group fields must be all set or all unset.".to_string(),
        ));
    }
    Ok(None)
}

fn parse_current_header_phenotype_compute_group_mode(
    group_mode: &str,
) -> Result<g_plan::PhenotypeComputeGroupMode, OutputError> {
    serde_json::from_value(Value::String(group_mode.to_string()))
        .map_err(|error| OutputError::InvalidInput(format!("Invalid phenotype compute group mode: {error}")))
}

fn parse_current_header_multi_phenotype_sample_mode(
    sample_mode: &str,
) -> Result<g_plan::MultiPhenotypeSampleMode, OutputError> {
    serde_json::from_value(Value::String(sample_mode.to_string()))
        .map_err(|error| OutputError::InvalidInput(format!("Invalid phenotype compute group sample mode: {error}")))
}

fn build_required_file_fingerprint_with_cache(
    fingerprint_cache: &mut ManifestFileFingerprintCache,
    path: &Path,
    include_content_hash: bool,
    role_name: &str,
) -> Result<Value, OutputError> {
    build_optional_file_fingerprint_with_cache(fingerprint_cache, Some(path), include_content_hash)?
        .ok_or_else(|| OutputError::InvalidInput(format!("{role_name} fingerprint is required.")))
}

fn build_optional_file_fingerprint_with_cache(
    fingerprint_cache: &mut ManifestFileFingerprintCache,
    path: Option<&Path>,
    include_content_hash: bool,
) -> Result<Option<Value>, OutputError> {
    let Some(file_path) = path else {
        return Ok(None);
    };
    fingerprint_cache
        .build_file_fingerprint(file_path, include_content_hash)
        .map(|fingerprint| Some(manifest_file_fingerprint_to_value(&fingerprint)))
}
