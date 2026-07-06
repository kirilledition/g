use serde::Deserialize;
use serde_json::{Value, json};

use crate::error::OutputError;

use super::{
    JAX_MATMUL_PRECISION_WHEN_UNSET, OUTPUT_SCHEMA_VERSION, RESUME_POLICY, RUN_MANIFEST_SCHEMA_VERSION,
    build_manifest_value_sha256,
};

#[derive(Clone, Debug, Deserialize, PartialEq)]
struct CurrentRunManifestHeaderContract {
    association_mode: g_plan::AssociationMode,
    bgen: g_plan::ManifestFileFingerprint,
    #[serde(default)]
    sample: Option<g_plan::ManifestFileFingerprint>,
    phenotype_file: g_plan::ManifestFileFingerprint,
    phenotype_name: String,
    #[serde(default)]
    covariate_file: Option<g_plan::ManifestFileFingerprint>,
    covariate_names: Vec<String>,
    prediction_list: g_plan::ManifestFileFingerprint,
    prediction_inputs: g_plan::PredictionInputsIdentity,
    sample_count: i64,
    variant_count: i64,
    chunk_size: i64,
    #[serde(default)]
    variant_limit: Option<i64>,
    binary_correction_plan: g_plan::CorrectionPlan,
    #[serde(default)]
    binary_kernel_config: Option<Value>,
    trusted_no_missing_diploid: bool,
    trusted_bgen_validation_mode: g_plan::TrustedBgenValidationMode,
    sample_key_mode: g_plan::SampleKeyMode,
    bgen_decode_tile_variant_count: i64,
    jax_policy: CurrentJaxPolicyManifest,
    #[serde(default)]
    requested_gpu_genotype_format: Option<g_plan::GpuGenotypeFormat>,
    gpu_genotype_format: g_plan::GpuGenotypeFormat,
    score_dtype: g_plan::FloatingPointDtype,
    firth_dtype: g_plan::FloatingPointDtype,
    multi_phenotype_sample_mode: g_plan::PreparedSampleMode,
    #[serde(default)]
    phenotype_compute_group_id: Option<String>,
    #[serde(default)]
    sample_set_fingerprint: Option<String>,
    #[serde(default)]
    covariate_design_fingerprint: Option<String>,
    #[serde(default)]
    prediction_alignment_fingerprint: Option<String>,
    output_writer: CurrentOutputWriterManifest,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
struct CurrentJaxPolicyManifest {
    device: g_plan::Device,
    enable_x64: bool,
    matmul_precision: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
struct CurrentOutputWriterManifest {
    output_format: g_plan::OutputFormat,
    finalize_parquet: bool,
    writer_thread_count: i64,
    writer_queue_depth: i64,
    chunks_per_arrow_file: i64,
    arrow_compression: g_plan::ArrowCompression,
    parquet_compression: g_plan::ParquetCompression,
    result_statistic_dtype: g_plan::FloatingPointDtype,
}

impl CurrentRunManifestHeaderContract {
    fn into_prepared_run_plan_input(self) -> Result<g_plan::PreparedRunPlanInput, OutputError> {
        let requested_gpu_genotype_format = self.requested_gpu_genotype_format.unwrap_or(self.gpu_genotype_format);
        let phenotype_compute_group =
            self.phenotype_compute_group_id.map(|group_id| g_plan::PreparedPhenotypeComputeGroup {
                group_id,
                sample_set_fingerprint: self.sample_set_fingerprint,
                covariate_design_fingerprint: self.covariate_design_fingerprint,
                prediction_alignment_fingerprint: self.prediction_alignment_fingerprint,
            });
        Ok(g_plan::PreparedRunPlanInput {
            association_mode: self.association_mode,
            input_identity: g_plan::PreparedInputIdentity {
                bgen: self.bgen,
                sample: self.sample,
                phenotype_file: self.phenotype_file,
                covariate_file: self.covariate_file,
                prediction_list: self.prediction_list,
                prediction_inputs: self.prediction_inputs,
            },
            phenotype_name: self.phenotype_name,
            covariate_names: self.covariate_names,
            sample_count: self.sample_count,
            variant_count: self.variant_count,
            chunk_size: self.chunk_size,
            variant_limit: self.variant_limit,
            correction: self.binary_correction_plan,
            binary_kernel_config: self.binary_kernel_config,
            compute: g_plan::PreparedComputePlan {
                trusted_no_missing_diploid: self.trusted_no_missing_diploid,
                trusted_bgen_validation_mode: self.trusted_bgen_validation_mode,
                sample_key_mode: self.sample_key_mode,
                bgen_decode_tile_variant_count: self.bgen_decode_tile_variant_count,
                jax_policy: self.jax_policy.into_prepared_jax_policy()?,
                requested_gpu_genotype_format,
                resolved_gpu_genotype_format: self.gpu_genotype_format,
                score_dtype: self.score_dtype,
                firth_dtype: self.firth_dtype,
                sample_mode: self.multi_phenotype_sample_mode,
            },
            phenotype_compute_group,
            output_writer: self.output_writer.into_prepared_output_writer_plan(),
        })
    }
}

impl CurrentJaxPolicyManifest {
    fn into_prepared_jax_policy(self) -> Result<g_plan::JaxPolicyPlan, OutputError> {
        let matmul_precision = if self.matmul_precision == JAX_MATMUL_PRECISION_WHEN_UNSET {
            None
        } else {
            Some(parse_current_header_matmul_precision(self.matmul_precision)?)
        };
        Ok(g_plan::JaxPolicyPlan { device: self.device, enable_x64: self.enable_x64, matmul_precision })
    }
}

impl CurrentOutputWriterManifest {
    fn into_prepared_output_writer_plan(self) -> g_plan::PreparedOutputWriterPlan {
        g_plan::PreparedOutputWriterPlan {
            output_format: self.output_format,
            finalize_parquet: self.finalize_parquet,
            writer_thread_count: self.writer_thread_count,
            writer_queue_depth: self.writer_queue_depth,
            chunks_per_arrow_file: self.chunks_per_arrow_file,
            arrow_compression: self.arrow_compression,
            parquet_compression: self.parquet_compression,
            output_statistic_dtype: self.result_statistic_dtype,
        }
    }
}

fn parse_current_header_matmul_precision(matmul_precision: String) -> Result<g_plan::JaxMatmulPrecision, OutputError> {
    serde_json::from_value(Value::String(matmul_precision))
        .map_err(|error| OutputError::InvalidInput(format!("Invalid JAX matmul precision: {error}")))
}

pub fn build_prepared_run_manifest_header_json(
    prepared_run_plan: &g_plan::PreparedRunPlan,
) -> Result<String, OutputError> {
    let execution_plan = build_prepared_run_execution_plan(prepared_run_plan)?;
    let current_header = build_prepared_run_manifest_header(prepared_run_plan, &execution_plan)?;
    serde_json::to_string(&current_header).map_err(OutputError::runtime)
}

pub fn build_prepared_run_plan_from_current_header_json(
    current_header_json: &str,
) -> Result<g_plan::PreparedRunPlan, OutputError> {
    let current_header = serde_json::from_str::<CurrentRunManifestHeaderContract>(current_header_json)
        .map_err(|error| OutputError::InvalidInput(format!("Invalid current run manifest header JSON: {error}")))?;
    let prepared_run_plan_input = current_header.into_prepared_run_plan_input()?;
    g_plan::build_prepared_run_plan(prepared_run_plan_input)
        .map_err(|error| OutputError::InvalidInput(format!("Invalid prepared run plan input: {error}")))
}

pub fn build_prepared_run_plan_json_from_current_header_json(current_header_json: &str) -> Result<String, OutputError> {
    let prepared_run_plan = build_prepared_run_plan_from_current_header_json(current_header_json)?;
    serde_json::to_string(&prepared_run_plan).map_err(OutputError::runtime)
}

pub fn build_prepared_run_manifest_header_json_from_current_header_json(
    current_header_json: &str,
) -> Result<String, OutputError> {
    let prepared_run_plan = build_prepared_run_plan_from_current_header_json(current_header_json)?;
    build_prepared_run_manifest_header_json(&prepared_run_plan)
}

fn build_prepared_run_execution_plan(prepared_run_plan: &g_plan::PreparedRunPlan) -> Result<Value, OutputError> {
    let input_identity = &prepared_run_plan.input_identity;
    let phenotype_compute_group = prepared_run_plan.phenotype_compute_group.as_ref();
    let binary_kernel_config = prepared_run_plan.binary_kernel_config.clone().unwrap_or(Value::Null);
    let prediction_inputs = serde_json::to_value(&input_identity.prediction_inputs).map_err(OutputError::runtime)?;
    let binary_correction_plan = serde_json::to_value(&prepared_run_plan.correction).map_err(OutputError::runtime)?;
    Ok(json!({
        "manifest_schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "association_mode": prepared_run_plan.association_mode.as_str(),
        "association_backend": build_prepared_association_backend(prepared_run_plan),
        "bgen": &input_identity.bgen,
        "sample": &input_identity.sample,
        "phenotype_file": &input_identity.phenotype_file,
        "phenotype_name": prepared_run_plan.phenotype_name.as_str(),
        "covariate_file": &input_identity.covariate_file,
        "covariate_names": &prepared_run_plan.covariate_names,
        "prediction_list": &input_identity.prediction_list,
        "prediction_inputs": prediction_inputs,
        "sample_count": prepared_run_plan.sample_count,
        "variant_count": prepared_run_plan.variant_count,
        "chunk_size": prepared_run_plan.chunk_size,
        "variant_limit": prepared_run_plan.variant_limit,
        "binary_correction_plan": binary_correction_plan,
        "binary_kernel_config": binary_kernel_config,
        "trusted_no_missing_diploid": prepared_run_plan.compute.trusted_no_missing_diploid,
        "trusted_bgen_validation_mode": prepared_run_plan.compute.trusted_bgen_validation_mode.as_str(),
        "sample_key_mode": prepared_run_plan.compute.sample_key_mode.as_str(),
        "bgen_decode_tile_variant_count": prepared_run_plan.compute.bgen_decode_tile_variant_count,
        "jax_policy": build_prepared_jax_policy(prepared_run_plan),
        "requested_gpu_genotype_format": prepared_run_plan.compute.requested_gpu_genotype_format.as_str(),
        "gpu_genotype_format": prepared_run_plan.compute.resolved_gpu_genotype_format.as_str(),
        "score_dtype": prepared_run_plan.compute.score_dtype.as_str(),
        "firth_dtype": prepared_run_plan.compute.firth_dtype.as_str(),
        "multi_phenotype_sample_mode": prepared_run_plan.compute.sample_mode.as_str(),
        "phenotype_compute_group_id": phenotype_compute_group.map(|group| group.group_id.as_str()),
        "sample_set_fingerprint": phenotype_compute_group.and_then(|group| group.sample_set_fingerprint.as_deref()),
        "covariate_design_fingerprint": phenotype_compute_group
            .and_then(|group| group.covariate_design_fingerprint.as_deref()),
        "prediction_alignment_fingerprint": phenotype_compute_group
            .and_then(|group| group.prediction_alignment_fingerprint.as_deref()),
        "output_writer": build_prepared_output_writer(prepared_run_plan),
        "resume_policy": RESUME_POLICY,
    }))
}

fn build_prepared_run_manifest_header(
    prepared_run_plan: &g_plan::PreparedRunPlan,
    execution_plan: &Value,
) -> Result<Value, OutputError> {
    let phenotype_compute_group = prepared_run_plan.phenotype_compute_group.as_ref();
    let execution_plan_hash = build_manifest_value_sha256(execution_plan)?;
    Ok(json!({
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "association_mode": prepared_run_plan.association_mode.as_str(),
        "association_backend": execution_plan["association_backend"].clone(),
        "bgen": execution_plan["bgen"].clone(),
        "sample": execution_plan["sample"].clone(),
        "phenotype_file": execution_plan["phenotype_file"].clone(),
        "phenotype_name": prepared_run_plan.phenotype_name.as_str(),
        "covariate_file": execution_plan["covariate_file"].clone(),
        "covariate_names": execution_plan["covariate_names"].clone(),
        "prediction_list": execution_plan["prediction_list"].clone(),
        "prediction_inputs": execution_plan["prediction_inputs"].clone(),
        "sample_count": prepared_run_plan.sample_count,
        "variant_count": prepared_run_plan.variant_count,
        "chunk_size": prepared_run_plan.chunk_size,
        "variant_limit": prepared_run_plan.variant_limit,
        "binary_correction_plan": execution_plan["binary_correction_plan"].clone(),
        "binary_kernel_config": execution_plan["binary_kernel_config"].clone(),
        "trusted_no_missing_diploid": prepared_run_plan.compute.trusted_no_missing_diploid,
        "trusted_bgen_validation_mode": prepared_run_plan.compute.trusted_bgen_validation_mode.as_str(),
        "sample_key_mode": prepared_run_plan.compute.sample_key_mode.as_str(),
        "bgen_decode_tile_variant_count": prepared_run_plan.compute.bgen_decode_tile_variant_count,
        "jax_policy": execution_plan["jax_policy"].clone(),
        "requested_gpu_genotype_format": execution_plan["requested_gpu_genotype_format"].clone(),
        "gpu_genotype_format": prepared_run_plan.compute.resolved_gpu_genotype_format.as_str(),
        "score_dtype": prepared_run_plan.compute.score_dtype.as_str(),
        "firth_dtype": prepared_run_plan.compute.firth_dtype.as_str(),
        "multi_phenotype_sample_mode": prepared_run_plan.compute.sample_mode.as_str(),
        "phenotype_compute_group_id": phenotype_compute_group.map(|group| group.group_id.as_str()),
        "sample_set_fingerprint": phenotype_compute_group.and_then(|group| group.sample_set_fingerprint.as_deref()),
        "covariate_design_fingerprint": phenotype_compute_group
            .and_then(|group| group.covariate_design_fingerprint.as_deref()),
        "prediction_alignment_fingerprint": phenotype_compute_group
            .and_then(|group| group.prediction_alignment_fingerprint.as_deref()),
        "output_writer": execution_plan["output_writer"].clone(),
        "resume_policy": RESUME_POLICY,
        "execution_plan": execution_plan.clone(),
        "execution_plan_hash": execution_plan_hash,
    }))
}

fn build_prepared_association_backend(prepared_run_plan: &g_plan::PreparedRunPlan) -> Value {
    json!({
        "kind": prepared_run_plan.association_backend.kind.as_str(),
        "association_mode": prepared_run_plan.association_backend.association_mode.as_str(),
        "device": prepared_run_plan.association_backend.device.as_str(),
        "genotype_format": prepared_run_plan.association_backend.resolved_genotype_format.as_str(),
    })
}

fn build_prepared_jax_policy(prepared_run_plan: &g_plan::PreparedRunPlan) -> Value {
    json!({
        "device": prepared_run_plan.compute.jax_policy.device.as_str(),
        "enable_x64": prepared_run_plan.compute.jax_policy.enable_x64,
        "matmul_precision": prepared_run_plan.compute.jax_policy.matmul_precision.map_or(
            JAX_MATMUL_PRECISION_WHEN_UNSET,
            g_plan::JaxMatmulPrecision::as_str,
        ),
    })
}

fn build_prepared_output_writer(prepared_run_plan: &g_plan::PreparedRunPlan) -> Value {
    json!({
        "output_format": prepared_run_plan.output_writer.output_format.as_str(),
        "finalize_parquet": prepared_run_plan.output_writer.finalize_parquet,
        "writer_thread_count": prepared_run_plan.output_writer.writer_thread_count,
        "writer_queue_depth": prepared_run_plan.output_writer.writer_queue_depth,
        "chunks_per_arrow_file": prepared_run_plan.output_writer.chunks_per_arrow_file,
        "arrow_compression": prepared_run_plan.output_writer.arrow_compression.as_str(),
        "parquet_compression": prepared_run_plan.output_writer.parquet_compression.as_str(),
        "result_statistic_dtype": prepared_run_plan.output_writer.output_statistic_dtype.as_str(),
    })
}
