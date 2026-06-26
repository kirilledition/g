//! Canonical prepared-run planning contracts.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::request::{
    ArrowCompression, AssociationMode, CorrectionPlan, Device, FloatingPointDtype, GpuGenotypeFormat,
    JaxMatmulPrecision, OutputFormat, ParquetCompression, SampleKeyMode, TrustedBgenValidationMode,
};

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

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AssociationBackendPlan {
    pub kind: AssociationBackendKind,
    pub association_mode: AssociationMode,
    pub device: Device,
    pub resolved_genotype_format: GpuGenotypeFormat,
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
