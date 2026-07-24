//! Canonical run-planning contracts.

use serde::{Deserialize, Serialize};

use crate::enums::{
    AssociationMode, BinaryFallbackMethod, Device, MultiPhenotypeSampleMode, NullLogisticNonconvergencePolicy,
    PhenotypeComputeGroupMode, TelemetryMode,
};
use crate::numeric::{DosageThreshold, PositiveF32, PositiveF64, Probability, ProbabilityFloor, StepScale};

#[derive(Debug, Deserialize, PartialEq, Serialize)]
pub struct RunPlan {
    pub association_mode: AssociationMode,
    pub chunk_size: u32,
    pub input: InputPlan,
    pub compute: ComputePlan,
    pub correction: CorrectionPlan,
    pub output: OutputPlan,
    pub telemetry: TelemetryMode,
    pub phenotype_runs: Vec<PhenotypeRunPlan>,
}

#[derive(Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct InputPlan {
    pub bgen_path: String,
    pub bgen_content_sha256: Option<g_genotype_contracts::BgenContentSha256>,
    pub sample_path: String,
    pub phenotype_path: String,
    pub prediction_list_path: String,
    pub covariate_path: Option<String>,
    pub covariate_names: Vec<String>,
}

#[derive(Debug, Deserialize, PartialEq, Serialize)]
pub struct ComputePlan {
    pub device: Device,
    pub cpu_thread_count: Option<u32>,
    pub jax_cache_directory: Option<String>,
    pub multi_phenotype_sample_mode: MultiPhenotypeSampleMode,
    pub kernels: KernelPlan,
}

#[derive(Debug, Deserialize, PartialEq, Serialize)]
pub struct KernelPlan {
    pub linear: LinearKernelPlan,
    pub binary_null: BinaryNullKernelPlan,
    pub firth: FirthKernelPlan,
    pub null_firth: NullFirthKernelPlan,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct LinearKernelPlan {
    pub minimum_variance: PositiveF32,
    pub relative_variance_tolerance: PositiveF32,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct BinaryNullKernelPlan {
    pub maximum_iterations: u32,
    pub coefficient_tolerance: PositiveF32,
    pub nonconvergence_policy: NullLogisticNonconvergencePolicy,
    pub minimum_probability: ProbabilityFloor,
    pub minimum_variance: PositiveF32,
    pub relative_variance_tolerance: PositiveF32,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct FirthKernelPlan {
    pub batch_size: u32,
    pub candidate_capacity: u32,
    pub maximum_iterations: u32,
    pub gradient_tolerance: PositiveF64,
    pub maximum_step_size: PositiveF64,
    pub pseudo_maximum_iterations: u32,
    pub pseudo_inner_maximum_iterations: u32,
    pub line_search_maximum_attempts: u32,
    pub sparse_carrier_dosage_threshold: DosageThreshold,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct NullFirthKernelPlan {
    pub maximum_iterations: u32,
    pub gradient_tolerance: PositiveF64,
    pub maximum_step_size: PositiveF64,
    pub fallback_iteration_multiplier: u32,
    pub fallback_step_divisor: PositiveF64,
    pub line_search_maximum_attempts: u32,
    pub step_halving_scale: StepScale,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct CorrectionPlan {
    pub method: BinaryFallbackMethod,
    pub p_threshold: Probability,
    pub firth_se: bool,
}

#[derive(Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct OutputPlan {
    pub output_run_root: String,
    pub resume: bool,
    pub writer_thread_count: u32,
}

#[derive(Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct PhenotypeRunPlan {
    pub phenotype_name: String,
    pub output_directory_name: String,
}

#[derive(Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct PhenotypeComputeGroup {
    pub group_mode: PhenotypeComputeGroupMode,
    pub phenotype_indices: Vec<u32>,
    pub phenotype_names: Vec<String>,
    pub sample_mode: MultiPhenotypeSampleMode,
    pub sample_set_fingerprint: String,
    pub covariate_design_fingerprint: String,
    pub phenotype_design_fingerprint: String,
    pub prediction_alignment_fingerprint: String,
}
