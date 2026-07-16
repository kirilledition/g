#![allow(clippy::missing_errors_doc)]

use std::path::Path;
use std::sync::Arc;

use serde_json::{Value, json};

use crate::error::OutputError;

use super::{
    ManifestFileFingerprintCache, OUTPUT_SCHEMA_VERSION, RESUME_POLICY, RUN_MANIFEST_SCHEMA_VERSION,
    build_manifest_value_sha256, manifest_file_fingerprint_to_value,
};

pub struct CurrentRunManifestHeaderInput {
    pub phenotype_name: String,
    pub bgen_source_identity: Arc<g_genotype_contracts::BgenSourceIdentity>,
    pub covariate_names: Arc<[String]>,
    pub prediction_loco_files: Arc<[PredictionLocoFileFingerprint]>,
    pub sample_count: usize,
    pub variant_count: usize,
    pub resolved_gpu_genotype_format: g_plan::GpuGenotypeFormat,
    pub sample_mode: g_plan::MultiPhenotypeSampleMode,
    pub phenotype_compute_group_id: Arc<str>,
    pub sample_set_fingerprint: Arc<str>,
    pub covariate_design_fingerprint: Arc<str>,
    pub phenotype_design_fingerprint: Arc<str>,
    pub prediction_alignment_fingerprint: Arc<str>,
}

#[derive(Clone)]
pub struct PredictionLocoFileFingerprint {
    pub(crate) phenotype_name: Arc<str>,
    pub(crate) file_fingerprint: Arc<super::ManifestFileFingerprint>,
}

#[allow(clippy::too_many_lines)]
pub(crate) fn build_current_run_manifest_header_value_with_cache(
    run_plan: &g_plan::RunPlan,
    input: &CurrentRunManifestHeaderInput,
    fingerprint_cache: &mut ManifestFileFingerprintCache,
) -> Result<Value, OutputError> {
    let bgen_fingerprint = bgen_source_identity_to_value(&input.bgen_source_identity);
    let sample_fingerprint = build_optional_file_fingerprint_with_cache(
        fingerprint_cache,
        Some(Path::new(&run_plan.input.sample_path)),
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
    let prediction_loco_files = prediction_loco_file_fingerprints_to_value(&input.prediction_loco_files)?;
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
    };
    let association_backend = json!({
        "kind": association_backend_kind,
        "genotype_format": input.resolved_gpu_genotype_format.as_str(),
    });
    let approximate_firth_pseudo_inner_policy = (run_plan.association_mode == g_plan::AssociationMode::Regenie2Binary
        && run_plan.correction.method == g_plan::BinaryFallbackMethod::FirthApproximate)
        .then_some("float32_elementwise_float64_reduction");
    let jax_policy = json!({
        "device": run_plan.compute.device.as_str(),
        "enable_x64": true,
        "matmul_precision": "float32",
        "approximate_firth_pseudo_inner_policy": approximate_firth_pseudo_inner_policy,
    });
    let output_writer = json!({
        "writer_thread_count": run_plan.output.writer_thread_count,
        "writer_queue_depth": crate::WRITER_QUEUE_DEPTH,
        "chunks_per_parquet_file": crate::CHUNKS_PER_PARQUET_FILE,
        "parquet_compression": "zstd",
        "parquet_writer_version": crate::writer::REGENIE_STEP2_PARQUET_WRITER_VERSION.as_num(),
        "parquet_write_batch_size": crate::writer::REGENIE_STEP2_PARQUET_WRITE_BATCH_SIZE,
        "parquet_max_row_group_size": crate::writer::REGENIE_STEP2_PARQUET_MAX_ROW_GROUP_SIZE,
        "parquet_float_column_encoding": crate::writer::REGENIE_STEP2_PARQUET_FLOAT_ENCODING.to_string(),
        "result_statistic_dtype": "float32",
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
        "covariate_names": input.covariate_names.as_ref(),
        "prediction_inputs": prediction_inputs,
        "sample_count": sample_count,
        "variant_count": variant_count,
        "chunk_size": run_plan.chunk_size,
        "binary_correction_plan": binary_correction_plan,
        "binary_kernel_config": binary_kernel_config,
        "jax_policy": jax_policy,
        "score_dtype": "float32",
        "multi_phenotype_sample_mode": input.sample_mode.as_str(),
        "phenotype_compute_group_id": input.phenotype_compute_group_id.as_ref(),
        "sample_set_fingerprint": input.sample_set_fingerprint.as_ref(),
        "covariate_design_fingerprint": input.covariate_design_fingerprint.as_ref(),
        "phenotype_design_fingerprint": input.phenotype_design_fingerprint.as_ref(),
        "prediction_alignment_fingerprint": input.prediction_alignment_fingerprint.as_ref(),
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

fn bgen_source_identity_to_value(identity: &g_genotype_contracts::BgenSourceIdentity) -> Value {
    let resolved_path = identity.canonical_path.as_ref().unwrap_or(&identity.configured_path);
    json!({
        "path": resolved_path.display().to_string(),
        "configured_path": identity.configured_path.display().to_string(),
        "size": identity.file_size,
        "mtime_ns": identity.modification_time_nanoseconds,
        "ctime_ns": identity.change_time_nanoseconds,
        "device": identity.device_identifier,
        "inode": identity.inode_identifier,
        "content_hash_algorithm": "opened-file-identity",
        "content_sha256": Value::Null,
    })
}

fn prediction_loco_file_fingerprints_to_value(
    fingerprints: &[PredictionLocoFileFingerprint],
) -> Result<Value, OutputError> {
    let values = fingerprints
        .iter()
        .map(|fingerprint| {
            let content_sha256 = fingerprint.file_fingerprint.content_sha256.as_deref().ok_or_else(|| {
                OutputError::Runtime("LOCO prediction file fingerprint must include a content hash.".to_string())
            })?;
            Ok(json!({
                "phenotype": fingerprint.phenotype_name.as_ref(),
                "path": &fingerprint.file_fingerprint.path,
                "size": fingerprint.file_fingerprint.size,
                "mtime_ns": fingerprint.file_fingerprint.mtime_ns,
                "content_hash_algorithm": &fingerprint.file_fingerprint.content_hash_algorithm,
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
        .map(|fingerprint| Some(manifest_file_fingerprint_to_value(fingerprint.as_ref())))
}
