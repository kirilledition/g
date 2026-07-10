//! Canonical run-planning contracts.

use serde::{Deserialize, Serialize};

use crate::enums::{
    AssociationMode, BinaryFallbackMethod, Device, GpuGenotypeFormat, JaxMatmulPrecision, MultiPhenotypeSampleMode,
    NullLogisticNonconvergencePolicy, PhenotypeComputeGroupMode, RegenieTraitType, ResumeMode, SampleKeyMode,
    TelemetryMode, TrustedBgenValidationMode,
};
use crate::numeric::{DosageThreshold, PositiveF32, PositiveF64, Probability, ProbabilityFloor, StepScale};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RunPlan {
    pub association_mode: AssociationMode,
    pub input: InputPlan,
    pub analysis: AnalysisPlan,
    pub compute: ComputePlan,
    pub correction: CorrectionPlan,
    pub output: OutputPlan,
    pub runtime: RuntimePlan,
    pub diagnostics: DiagnosticsPlan,
    pub phenotype_runs: Vec<PhenotypeRunPlan>,
    pub phenotype_compute_groups: Vec<PhenotypeComputeGroup>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct InputPlan {
    pub bgen_path: String,
    pub sample_path: String,
    pub phenotype_path: String,
    pub prediction_list_path: String,
    pub covariate_path: Option<String>,
    pub covariate_names: Vec<String>,
    pub sample_key_mode: SampleKeyMode,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct AnalysisPlan {
    pub trait_type: RegenieTraitType,
    pub chunk_size: u32,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ComputePlan {
    pub device: Device,
    pub cpu_thread_count: Option<u32>,
    pub staging_depth: u32,
    pub result_in_flight_limit: Option<u32>,
    pub variant_limit: Option<u32>,
    pub bgen_decode_tile_variant_count: u32,
    pub requested_gpu_genotype_format: GpuGenotypeFormat,
    pub trusted_no_missing_diploid: bool,
    pub trusted_bgen_validation_mode: TrustedBgenValidationMode,
    pub multi_phenotype_sample_mode: MultiPhenotypeSampleMode,
    pub kernels: KernelPlan,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
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
    pub coefficient_tolerance: PositiveF64,
    pub likelihood_tolerance: PositiveF64,
    pub maximum_step_size: PositiveF64,
    pub pseudo_maximum_iterations: u32,
    pub pseudo_inner_maximum_iterations: u32,
    pub newton_raphson_zero_start_iterations: u32,
    pub line_search_maximum_attempts: u32,
    pub step_halving_maximum_attempts: u32,
    pub initial_response_scale: PositiveF64,
    pub sparse_carrier_dosage_threshold: DosageThreshold,
    pub step_halving_scale: StepScale,
    pub use_block_math: bool,
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

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct OutputPlan {
    pub output_prefix: String,
    pub output_run_root: String,
    pub resume: bool,
    pub resume_mode: ResumeMode,
    pub writer_thread_count: u32,
    pub writer_queue_depth: u32,
    pub chunks_per_parquet_file: u32,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct RuntimePlan {
    pub jax_cache_directory: Option<String>,
    pub jax_matmul_precision: Option<JaxMatmulPrecision>,
    pub persistent_cache_enabled: bool,
    pub persistent_cache_min_entry_size_bytes: i64,
    pub persistent_cache_min_compile_time_seconds: u32,
    pub xla_autotune_cache_enabled: bool,
    pub transfer_guard_enabled: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[expect(clippy::struct_excessive_bools, reason = "Diagnostics flags are independent runtime policies.")]
pub struct DiagnosticsPlan {
    pub telemetry: TelemetryMode,
    pub log_directory: Option<String>,
    pub stage_timings_path: Option<String>,
    pub log_filter: String,
    pub log_file: Option<String>,
    pub log_to_stderr: bool,
    pub profile_summary_path: Option<String>,
    pub trace_file: Option<String>,
    pub trace_filter: String,
    pub trace_event_cap: u32,
    pub log_queue_size: u32,
    pub lossy_logging: bool,
    pub include_source_location: bool,
    pub include_span_events: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct PhenotypeRunPlan {
    pub phenotype_index: u32,
    pub phenotype_name: String,
    pub output_directory_name: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct PhenotypeComputeGroup {
    pub group_mode: PhenotypeComputeGroupMode,
    pub phenotype_indices: Vec<u32>,
    pub phenotype_names: Vec<String>,
    pub sample_mode: MultiPhenotypeSampleMode,
    pub sample_set_fingerprint: Option<String>,
    pub covariate_design_fingerprint: Option<String>,
    pub prediction_alignment_fingerprint: Option<String>,
}
