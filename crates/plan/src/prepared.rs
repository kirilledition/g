//! Canonical prepared-run planning contracts.

use std::error;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::enums::{
    ArrowCompression, AssociationMode, Device, FloatingPointDtype, GpuGenotypeFormat, JaxMatmulPrecision, OutputFormat,
    ParquetCompression, SampleKeyMode, TrustedBgenValidationMode,
};
use crate::request::CorrectionPlan;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum AssociationBackendKind {
    #[serde(rename = "jax_dosage")]
    JaxDosage,
    #[serde(rename = "jax_packed8")]
    JaxPacked8,
}

impl AssociationBackendKind {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::JaxDosage => "jax_dosage",
            Self::JaxPacked8 => "jax_packed8",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum PreparedSampleMode {
    #[serde(rename = "single-phenotype")]
    SinglePhenotype,
    #[serde(rename = "per-phenotype")]
    PerPhenotype,
    #[serde(rename = "complete-case")]
    CompleteCase,
}

impl PreparedSampleMode {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SinglePhenotype => "single-phenotype",
            Self::PerPhenotype => "per-phenotype",
            Self::CompleteCase => "complete-case",
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PreparedRunPlan {
    pub association_mode: AssociationMode,
    pub association_backend: AssociationBackendPlan,
    pub input_identity: PreparedInputIdentity,
    pub phenotype_name: String,
    pub covariate_names: Vec<String>,
    pub sample_count: i64,
    pub variant_count: i64,
    pub chunk_size: i64,
    pub variant_limit: Option<i64>,
    pub correction: CorrectionPlan,
    pub binary_kernel_config: Option<Value>,
    pub compute: PreparedComputePlan,
    pub phenotype_compute_group: Option<PreparedPhenotypeComputeGroup>,
    pub output_writer: PreparedOutputWriterPlan,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PreparedRunPlanInput {
    pub association_mode: AssociationMode,
    pub input_identity: PreparedInputIdentity,
    pub phenotype_name: String,
    pub covariate_names: Vec<String>,
    pub sample_count: i64,
    pub variant_count: i64,
    pub chunk_size: i64,
    pub variant_limit: Option<i64>,
    pub correction: CorrectionPlan,
    pub binary_kernel_config: Option<Value>,
    pub compute: PreparedComputePlan,
    pub phenotype_compute_group: Option<PreparedPhenotypeComputeGroup>,
    pub output_writer: PreparedOutputWriterPlan,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AssociationBackendPlan {
    pub kind: AssociationBackendKind,
    pub association_mode: AssociationMode,
    pub device: Device,
    pub resolved_genotype_format: GpuGenotypeFormat,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PreparedPlanError {
    UnresolvedGpuGenotypeFormat,
}

impl fmt::Display for PreparedPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnresolvedGpuGenotypeFormat => {
                write!(formatter, "resolved_gpu_genotype_format must be dosage or packed8, not auto")
            }
        }
    }
}

impl error::Error for PreparedPlanError {}

/// Build the prepared association backend from resolved execution inputs.
///
/// # Errors
///
/// Returns [`PreparedPlanError::UnresolvedGpuGenotypeFormat`] when
/// `resolved_genotype_format` is still `auto`.
pub fn plan_association_backend(
    association_mode: AssociationMode,
    device: Device,
    resolved_genotype_format: GpuGenotypeFormat,
) -> Result<AssociationBackendPlan, PreparedPlanError> {
    let kind = match resolved_genotype_format {
        GpuGenotypeFormat::Auto => return Err(PreparedPlanError::UnresolvedGpuGenotypeFormat),
        GpuGenotypeFormat::Dosage => AssociationBackendKind::JaxDosage,
        GpuGenotypeFormat::Packed8 => AssociationBackendKind::JaxPacked8,
    };
    Ok(AssociationBackendPlan { kind, association_mode, device, resolved_genotype_format })
}

/// Build the canonical prepared run plan from resolved preparation inputs.
///
/// # Errors
///
/// Returns [`PreparedPlanError::UnresolvedGpuGenotypeFormat`] when the input
/// compute plan still contains `resolved_gpu_genotype_format=auto`.
pub fn build_prepared_run_plan(input: PreparedRunPlanInput) -> Result<PreparedRunPlan, PreparedPlanError> {
    let association_backend = plan_association_backend(
        input.association_mode,
        input.compute.jax_policy.device,
        input.compute.resolved_gpu_genotype_format,
    )?;
    Ok(PreparedRunPlan {
        association_mode: input.association_mode,
        association_backend,
        input_identity: input.input_identity,
        phenotype_name: input.phenotype_name,
        covariate_names: input.covariate_names,
        sample_count: input.sample_count,
        variant_count: input.variant_count,
        chunk_size: input.chunk_size,
        variant_limit: input.variant_limit,
        correction: input.correction,
        binary_kernel_config: input.binary_kernel_config,
        compute: input.compute,
        phenotype_compute_group: input.phenotype_compute_group,
        output_writer: input.output_writer,
    })
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PreparedInputIdentity {
    pub bgen: ManifestFileFingerprint,
    pub sample: Option<ManifestFileFingerprint>,
    pub phenotype_file: ManifestFileFingerprint,
    pub covariate_file: Option<ManifestFileFingerprint>,
    pub prediction_list: ManifestFileFingerprint,
    pub prediction_inputs: PredictionInputsIdentity,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PredictionInputsIdentity {
    pub prediction_list: ManifestFileFingerprint,
    pub loco_files: Vec<PredictionLocoFileFingerprint>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ManifestFileFingerprint {
    pub path: String,
    pub size: u64,
    pub mtime_ns: i64,
    pub content_hash_algorithm: String,
    pub content_sha256: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PredictionLocoFileFingerprint {
    pub phenotype: String,
    pub path: String,
    pub size: u64,
    pub mtime_ns: i64,
    pub content_hash_algorithm: String,
    pub content_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PreparedComputePlan {
    pub trusted_no_missing_diploid: bool,
    pub trusted_bgen_validation_mode: TrustedBgenValidationMode,
    pub sample_key_mode: SampleKeyMode,
    pub bgen_decode_tile_variant_count: i64,
    pub jax_policy: JaxPolicyPlan,
    pub requested_gpu_genotype_format: GpuGenotypeFormat,
    pub resolved_gpu_genotype_format: GpuGenotypeFormat,
    pub score_dtype: FloatingPointDtype,
    pub firth_dtype: FloatingPointDtype,
    pub sample_mode: PreparedSampleMode,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct JaxPolicyPlan {
    pub device: Device,
    pub enable_x64: bool,
    pub matmul_precision: Option<JaxMatmulPrecision>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PreparedPhenotypeComputeGroup {
    pub group_id: String,
    pub sample_set_fingerprint: Option<String>,
    pub covariate_design_fingerprint: Option<String>,
    pub prediction_alignment_fingerprint: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PreparedOutputWriterPlan {
    pub output_format: OutputFormat,
    pub finalize_parquet: bool,
    pub writer_thread_count: i64,
    pub writer_queue_depth: i64,
    pub chunks_per_arrow_file: i64,
    pub arrow_compression: ArrowCompression,
    pub parquet_compression: ParquetCompression,
    pub output_statistic_dtype: FloatingPointDtype,
}
