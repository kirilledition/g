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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::BinaryFallbackMethod;

    #[test]
    fn builds_prepared_backend_plan_from_resolved_genotype_format() {
        let dosage_plan = build_prepared_association_backend_plan(
            AssociationMode::Regenie2Linear,
            Device::Cpu,
            GpuGenotypeFormat::Dosage,
        )
        .unwrap();
        assert_eq!(dosage_plan.kind, AssociationBackendKind::JaxDosage);
        assert_eq!(dosage_plan.association_mode, AssociationMode::Regenie2Linear);
        assert_eq!(dosage_plan.device, Device::Cpu);
        assert_eq!(dosage_plan.resolved_genotype_format, GpuGenotypeFormat::Dosage);

        let packed_plan = build_prepared_association_backend_plan(
            AssociationMode::Regenie2Binary,
            Device::Gpu,
            GpuGenotypeFormat::Packed8,
        )
        .unwrap();
        assert_eq!(packed_plan.kind, AssociationBackendKind::JaxPacked8);
        assert_eq!(packed_plan.association_mode, AssociationMode::Regenie2Binary);
        assert_eq!(packed_plan.device, Device::Gpu);
        assert_eq!(packed_plan.resolved_genotype_format, GpuGenotypeFormat::Packed8);
    }

    #[test]
    fn rejects_unresolved_prepared_backend_format() {
        assert_eq!(
            build_prepared_association_backend_plan(
                AssociationMode::Regenie2Linear,
                Device::Gpu,
                GpuGenotypeFormat::Auto,
            ),
            Err(PreparedPlanError::UnresolvedGpuGenotypeFormat),
        );
    }

    #[test]
    fn builds_prepared_run_plan_from_input_contract() {
        let input = build_test_prepared_run_plan_input(GpuGenotypeFormat::Packed8);
        let prepared_run_plan = build_prepared_run_plan(input).unwrap();

        assert_eq!(prepared_run_plan.association_backend.kind, AssociationBackendKind::JaxPacked8);
        assert_eq!(prepared_run_plan.association_backend.device, Device::Gpu);
        assert_eq!(prepared_run_plan.association_backend.resolved_genotype_format, GpuGenotypeFormat::Packed8,);
        assert_eq!(prepared_run_plan.phenotype_name, "height");
        assert_eq!(prepared_run_plan.compute.requested_gpu_genotype_format, GpuGenotypeFormat::Auto);
        assert_eq!(prepared_run_plan.compute.resolved_gpu_genotype_format, GpuGenotypeFormat::Packed8);
    }

    fn build_test_prepared_run_plan_input(resolved_genotype_format: GpuGenotypeFormat) -> PreparedRunPlanInput {
        let prediction_list = build_test_file_fingerprint("prediction.list");
        PreparedRunPlanInput {
            association_mode: AssociationMode::Regenie2Linear,
            input_identity: PreparedInputIdentity {
                bgen: build_test_file_fingerprint("study.bgen"),
                sample: Some(build_test_file_fingerprint("study.sample")),
                phenotype_file: build_test_file_fingerprint("phenotypes.tsv"),
                covariate_file: None,
                prediction_list: prediction_list.clone(),
                prediction_inputs: PredictionInputsIdentity {
                    prediction_list,
                    loco_files: vec![PredictionLocoFileFingerprint {
                        phenotype: "height".to_string(),
                        path: "height.loco".to_string(),
                        size: 10,
                        mtime_ns: 20,
                        content_hash_algorithm: "sha256".to_string(),
                        content_sha256: "abcd".to_string(),
                    }],
                },
            },
            phenotype_name: "height".to_string(),
            covariate_names: vec!["age".to_string()],
            sample_count: 4,
            variant_count: 10,
            chunk_size: 2,
            variant_limit: None,
            correction: CorrectionPlan { method: BinaryFallbackMethod::ScoreOnly, p_threshold: 0.05, firth_se: false },
            binary_kernel_config: None,
            compute: PreparedComputePlan {
                trusted_no_missing_diploid: false,
                trusted_bgen_validation_mode: TrustedBgenValidationMode::CacheOnMiss,
                sample_key_mode: SampleKeyMode::Iid,
                bgen_decode_tile_variant_count: 64,
                jax_policy: JaxPolicyPlan { device: Device::Gpu, enable_x64: true, matmul_precision: None },
                requested_gpu_genotype_format: GpuGenotypeFormat::Auto,
                resolved_gpu_genotype_format: resolved_genotype_format,
                score_dtype: FloatingPointDtype::Float32,
                firth_dtype: FloatingPointDtype::Float64,
                sample_mode: PreparedSampleMode::SinglePhenotype,
            },
            phenotype_compute_group: None,
            output_writer: PreparedOutputWriterPlan {
                output_format: OutputFormat::Parquet,
                finalize_parquet: false,
                writer_thread_count: 1,
                writer_queue_depth: 1,
                chunks_per_arrow_file: 16,
                arrow_compression: ArrowCompression::Zstd,
                parquet_compression: ParquetCompression::None,
                output_statistic_dtype: FloatingPointDtype::Float32,
            },
        }
    }

    fn build_test_file_fingerprint(path: &str) -> ManifestFileFingerprint {
        ManifestFileFingerprint {
            path: path.to_string(),
            size: 10,
            mtime_ns: 20,
            content_hash_algorithm: "metadata-only".to_string(),
            content_sha256: None,
        }
    }
}
