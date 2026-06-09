use std::num::NonZeroU32;

use serde::Serialize;

use super::domain::{
    ArrowCompressionValue, DeviceValue, FloatingPointDtypeValue, GpuGenotypeFormatValue, JaxMatmulPrecisionValue,
    MultiPhenotypeSampleModeValue, NullLogisticNonconvergencePolicyValue, OutputFormatValue, ParquetCompressionValue,
    RegenieTraitTypeValue, ResumeModeValue, SampleKeyModeValue, TelemetryModeValue, TrustedBgenValidationModeValue,
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct InputConfigData {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bgen: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pheno_file: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub pheno_columns: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub covar_file: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub covar_columns: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pred: Option<String>,
}

impl InputConfigData {
    pub(crate) fn is_empty(&self) -> bool {
        self.bgen.is_none()
            && self.sample.is_none()
            && self.pheno_file.is_none()
            && self.pheno_columns.is_empty()
            && self.covar_file.is_none()
            && self.covar_columns.is_empty()
            && self.pred.is_none()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct TraitConfigData {
    pub step: u8,
    pub trait_type: RegenieTraitTypeValue,
    pub bsize: NonZeroU32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threads: Option<NonZeroU32>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[expect(clippy::struct_excessive_bools, reason = "Runtime config mirrors public REGENIE boolean flags.")]
pub struct BinaryConfigData {
    pub firth: bool,
    pub approx: bool,
    #[serde(skip)]
    pub spa: bool,
    pub p_threshold: f32,
    pub firth_se: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[expect(clippy::struct_excessive_bools, reason = "Runtime config mirrors public g-specific boolean options.")]
pub struct GComputeConfigData {
    pub device: DeviceValue,
    pub staging_depth: NonZeroU32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variant_limit: Option<NonZeroU32>,
    pub trusted_no_missing_diploid: bool,
    pub trusted_bgen_validation_mode: TrustedBgenValidationModeValue,
    pub sample_key_mode: SampleKeyModeValue,
    pub multi_phenotype_sample_mode: MultiPhenotypeSampleModeValue,
    pub firth_batch_size: NonZeroU32,
    pub firth_candidate_capacity: NonZeroU32,
    pub binary_null_maximum_iterations: NonZeroU32,
    pub binary_null_coefficient_tolerance: f32,
    pub null_logistic_nonconvergence_policy: NullLogisticNonconvergencePolicyValue,
    pub binary_minimum_probability: f32,
    pub binary_minimum_variance: f32,
    pub binary_relative_variance_tolerance: f32,
    pub linear_minimum_variance: f32,
    pub linear_relative_variance_tolerance: f32,
    pub firth_maximum_iterations: NonZeroU32,
    pub firth_gradient_tolerance: f32,
    pub firth_coefficient_tolerance: f32,
    pub firth_likelihood_tolerance: f32,
    pub firth_maximum_step_size: f32,
    pub firth_pseudo_maximum_iterations: NonZeroU32,
    pub firth_pseudo_inner_maximum_iterations: NonZeroU32,
    pub firth_newton_raphson_zero_start_iterations: NonZeroU32,
    pub firth_line_search_maximum_attempts: NonZeroU32,
    pub firth_step_halving_maximum_attempts: NonZeroU32,
    pub firth_initial_response_scale: f32,
    pub firth_sparse_carrier_dosage_threshold: f32,
    pub firth_step_halving_scale: f32,
    pub null_firth_maximum_iterations: NonZeroU32,
    pub null_firth_gradient_tolerance: f32,
    pub null_firth_maximum_step_size: f32,
    pub null_firth_fallback_iteration_multiplier: NonZeroU32,
    pub null_firth_fallback_step_divisor: f32,
    pub null_firth_line_search_maximum_attempts: NonZeroU32,
    pub null_firth_step_halving_scale: f32,
    pub use_block_firth_math: bool,
    pub bgen_decode_tile_variant_count: NonZeroU32,
    pub gpu_genotype_format: GpuGenotypeFormatValue,
    pub score_dtype: FloatingPointDtypeValue,
    pub firth_dtype: FloatingPointDtypeValue,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jax_cache_dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jax_matmul_precision: Option<JaxMatmulPrecisionValue>,
    pub jax_persistent_cache: bool,
    pub jax_persistent_cache_min_entry_size_bytes: i64,
    pub jax_persistent_cache_min_compile_time_seconds: u32,
    pub jax_xla_autotune_cache: bool,
    pub jax_transfer_guard: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct GOutputConfigData {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub out: Option<String>,
    pub format: OutputFormatValue,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_run_directory: Option<String>,
    pub writer_threads: NonZeroU32,
    pub writer_queue_depth: NonZeroU32,
    pub chunks_per_arrow_file: NonZeroU32,
    pub arrow_compression: ArrowCompressionValue,
    pub parquet_compression: ParquetCompressionValue,
    pub resume: bool,
    pub resume_mode: ResumeModeValue,
    pub finalize_parquet: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[expect(clippy::struct_excessive_bools, reason = "Diagnostics config mirrors public g-specific boolean options.")]
pub struct GDiagnosticsConfigData {
    pub telemetry: TelemetryModeValue,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log_dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage_timings_json: Option<String>,
    pub log_filter: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log_file: Option<String>,
    pub log_stderr: bool,
    pub progress_interval_seconds: f32,
    pub progress_interval_chunks: NonZeroU32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profile_summary_json: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace_file: Option<String>,
    pub trace_filter: String,
    pub trace_event_cap: u32,
    pub log_queue_size: NonZeroU32,
    pub log_lossy: bool,
    pub include_source_location: bool,
    pub include_span_events: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RegenieConfigData {
    #[serde(skip_serializing_if = "InputConfigData::is_empty")]
    pub input: InputConfigData,
    #[serde(rename = "trait")]
    pub trait_config: TraitConfigData,
    pub binary: BinaryConfigData,
    #[serde(rename = "compute")]
    pub g_compute: GComputeConfigData,
    #[serde(rename = "output")]
    pub g_output: GOutputConfigData,
    #[serde(rename = "diagnostics")]
    pub g_diagnostics: GDiagnosticsConfigData,
    #[serde(skip)]
    pub is_validated: bool,
}
