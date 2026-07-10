#![allow(clippy::missing_errors_doc)]

use std::path::Path;

use serde_json::{Value, json};

use crate::error::OutputError;

use super::{
    JAX_MATMUL_PRECISION_WHEN_UNSET, ManifestFileFingerprintCache, OUTPUT_SCHEMA_VERSION, RESUME_POLICY,
    RUN_MANIFEST_SCHEMA_VERSION, build_manifest_value_sha256, manifest_file_fingerprint_to_value,
};

pub struct CurrentRunManifestHeaderInput {
    pub phenotype_name: String,
    pub covariate_names: Vec<String>,
    pub prediction_loco_files: Vec<PredictionLocoFileFingerprint>,
    pub sample_count: usize,
    pub variant_count: usize,
    pub effective_trusted_no_missing_diploid: bool,
    pub resolved_gpu_genotype_format: g_plan::GpuGenotypeFormat,
    pub output_sample_mode: g_plan::MultiPhenotypeSampleMode,
    pub phenotype_compute_group_id: String,
    pub sample_set_fingerprint: Option<String>,
    pub covariate_design_fingerprint: Option<String>,
    pub prediction_alignment_fingerprint: Option<String>,
}

pub struct PredictionLocoFileFingerprint {
    pub phenotype_name: String,
    pub file_fingerprint: super::ManifestFileFingerprint,
}

#[allow(clippy::too_many_lines)]
pub(crate) fn build_current_run_manifest_header_value_with_cache(
    run_plan: &g_plan::RunPlan,
    input: CurrentRunManifestHeaderInput,
    fingerprint_cache: &mut ManifestFileFingerprintCache,
) -> Result<Value, OutputError> {
    let bgen_fingerprint = build_required_file_fingerprint_with_cache(
        fingerprint_cache,
        Path::new(&run_plan.input.bgen_path),
        false,
        "BGEN",
    )?;
    let sample_fingerprint = build_optional_file_fingerprint_with_cache(
        fingerprint_cache,
        run_plan.input.sample_path.as_deref().map(Path::new),
        true,
    )?;
    let phenotype_file_fingerprint = build_required_file_fingerprint_with_cache(
        fingerprint_cache,
        Path::new(&run_plan.input.phenotype_path),
        true,
        "phenotype file",
    )?;
    let covariate_file_fingerprint = build_optional_file_fingerprint_with_cache(
        fingerprint_cache,
        run_plan.input.covariate_path.as_deref().map(Path::new),
        true,
    )?;
    let prediction_list_fingerprint = build_required_file_fingerprint_with_cache(
        fingerprint_cache,
        Path::new(&run_plan.input.prediction_list_path),
        true,
        "prediction list",
    )?;
    let prediction_loco_files = prediction_loco_file_fingerprints_to_value(input.prediction_loco_files)?;
    let prediction_inputs = json!({
        "prediction_list": prediction_list_fingerprint,
        "loco_files": prediction_loco_files,
    });
    let binary_correction_plan = json!({
        "method": run_plan.correction.method.as_str(),
        "p_threshold": run_plan.correction.p_threshold.get(),
        "firth_se": run_plan.correction.firth_se,
    });
    let binary_kernel_config = match run_plan.association_mode {
        g_plan::AssociationMode::Regenie2Binary => {
            serde_json::to_value(&run_plan.compute.kernels).map_err(OutputError::runtime)?
        }
        g_plan::AssociationMode::Regenie2Linear => Value::Null,
    };
    let association_backend_kind = match input.resolved_gpu_genotype_format {
        g_plan::GpuGenotypeFormat::Dosage => "jax_dosage",
        g_plan::GpuGenotypeFormat::Packed8 => "jax_packed8",
        g_plan::GpuGenotypeFormat::Auto => {
            return Err(OutputError::InvalidInput(
                "Resolved GPU genotype format cannot remain auto during manifest construction.".to_string(),
            ));
        }
    };
    let association_backend = json!({
        "kind": association_backend_kind,
        "genotype_format": input.resolved_gpu_genotype_format.as_str(),
    });
    let jax_policy = json!({
        "device": run_plan.compute.device.as_str(),
        "enable_x64": true,
        "matmul_precision": run_plan.runtime.jax_matmul_precision.map_or(JAX_MATMUL_PRECISION_WHEN_UNSET, g_plan::JaxMatmulPrecision::as_str),
    });
    let output_writer = json!({
        "writer_thread_count": run_plan.output.writer_thread_count,
        "writer_queue_depth": run_plan.output.writer_queue_depth,
        "chunks_per_parquet_file": run_plan.output.chunks_per_parquet_file,
        "parquet_compression": run_plan.output.parquet_compression.as_str(),
        "result_statistic_dtype": run_plan.output.output_statistic_dtype.as_str(),
    });
    let sample_count = i64::try_from(input.sample_count)
        .map_err(|_| OutputError::InvalidInput("Sample count does not fit manifest int64.".to_string()))?;
    let variant_count = i64::try_from(input.variant_count)
        .map_err(|_| OutputError::InvalidInput("Variant count does not fit manifest int64.".to_string()))?;
    let execution_plan = json!({
        "association_mode": run_plan.association_mode.as_str(),
        "association_backend": association_backend,
        "bgen": bgen_fingerprint,
        "sample": sample_fingerprint,
        "phenotype_file": phenotype_file_fingerprint,
        "phenotype_name": input.phenotype_name,
        "covariate_file": covariate_file_fingerprint,
        "covariate_names": input.covariate_names,
        "prediction_inputs": prediction_inputs,
        "sample_count": sample_count,
        "variant_count": variant_count,
        "chunk_size": run_plan.analysis.chunk_size,
        "variant_limit": run_plan.compute.variant_limit,
        "binary_correction_plan": binary_correction_plan,
        "binary_kernel_config": binary_kernel_config,
        "trusted_no_missing_diploid": input.effective_trusted_no_missing_diploid,
        "trusted_bgen_validation_mode": run_plan.compute.trusted_bgen_validation_mode.as_str(),
        "sample_key_mode": run_plan.input.sample_key_mode.as_str(),
        "bgen_decode_tile_variant_count": run_plan.compute.bgen_decode_tile_variant_count,
        "jax_policy": jax_policy,
        "requested_gpu_genotype_format": run_plan.compute.requested_gpu_genotype_format.as_str(),
        "score_dtype": run_plan.compute.score_dtype.as_str(),
        "multi_phenotype_sample_mode": input.output_sample_mode.as_str(),
        "phenotype_compute_group_id": input.phenotype_compute_group_id,
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
        "execution_plan": execution_plan,
        "execution_plan_hash": execution_plan_hash,
    });
    Ok(current_header)
}

fn prediction_loco_file_fingerprints_to_value(
    fingerprints: Vec<PredictionLocoFileFingerprint>,
) -> Result<Value, OutputError> {
    let values = fingerprints
        .into_iter()
        .map(|fingerprint| {
            let content_sha256 = fingerprint.file_fingerprint.content_sha256.ok_or_else(|| {
                OutputError::Runtime("LOCO prediction file fingerprint must include a content hash.".to_string())
            })?;
            Ok(json!({
                "phenotype": fingerprint.phenotype_name,
                "path": fingerprint.file_fingerprint.path,
                "size": fingerprint.file_fingerprint.size,
                "mtime_ns": fingerprint.file_fingerprint.mtime_ns,
                "content_hash_algorithm": fingerprint.file_fingerprint.content_hash_algorithm,
                "content_sha256": content_sha256,
            }))
        })
        .collect::<Result<Vec<_>, OutputError>>()?;
    Ok(Value::Array(values))
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
