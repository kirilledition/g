use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputConfigData {
    pub bgen: Option<String>,
    pub sample: Option<String>,
    pub pheno_file: Option<String>,
    pub pheno_columns: Vec<String>,
    pub covar_file: Option<String>,
    pub covar_columns: Vec<String>,
    pub pred: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraitConfigData {
    pub step: u8,
    pub trait_type: String,
    pub bsize: u32,
    pub threads: Option<u32>,
}

#[derive(Clone, Debug, PartialEq)]
#[expect(clippy::struct_excessive_bools, reason = "Runtime config mirrors public REGENIE boolean flags.")]
pub struct BinaryConfigData {
    pub firth: bool,
    pub approx: bool,
    pub spa: bool,
    pub p_threshold: f32,
    pub firth_se: bool,
}

#[derive(Clone, Debug, PartialEq)]
#[expect(clippy::struct_excessive_bools, reason = "Runtime config mirrors public g-specific boolean options.")]
pub struct GComputeConfigData {
    pub device: String,
    pub staging_depth: u32,
    pub variant_limit: Option<u32>,
    pub trusted_no_missing_diploid: bool,
    pub trusted_bgen_validation_mode: String,
    pub sample_key_mode: String,
    pub multi_phenotype_sample_mode: String,
    pub firth_batch_size: u32,
    pub firth_candidate_capacity: u32,
    pub binary_null_maximum_iterations: u32,
    pub binary_null_coefficient_tolerance: f32,
    pub null_logistic_nonconvergence_policy: String,
    pub binary_minimum_probability: f32,
    pub binary_minimum_variance: f32,
    pub binary_relative_variance_tolerance: f32,
    pub linear_minimum_variance: f32,
    pub linear_relative_variance_tolerance: f32,
    pub firth_maximum_iterations: u32,
    pub firth_gradient_tolerance: f32,
    pub firth_coefficient_tolerance: f32,
    pub firth_likelihood_tolerance: f32,
    pub firth_maximum_step_size: f32,
    pub firth_pseudo_maximum_iterations: u32,
    pub firth_pseudo_inner_maximum_iterations: u32,
    pub firth_newton_raphson_zero_start_iterations: u32,
    pub firth_line_search_maximum_attempts: u32,
    pub firth_step_halving_maximum_attempts: u32,
    pub firth_initial_response_scale: f32,
    pub firth_sparse_carrier_dosage_threshold: f32,
    pub firth_step_halving_scale: f32,
    pub null_firth_maximum_iterations: u32,
    pub null_firth_gradient_tolerance: f32,
    pub null_firth_maximum_step_size: f32,
    pub null_firth_fallback_iteration_multiplier: u32,
    pub null_firth_fallback_step_divisor: f32,
    pub null_firth_line_search_maximum_attempts: u32,
    pub null_firth_step_halving_scale: f32,
    pub use_block_firth_math: bool,
    pub bgen_decode_tile_variant_count: u32,
    pub gpu_genotype_format: String,
    pub score_dtype: String,
    pub firth_dtype: String,
    pub jax_cache_dir: Option<String>,
    pub jax_matmul_precision: Option<String>,
    pub jax_persistent_cache: bool,
    pub jax_persistent_cache_min_entry_size_bytes: i64,
    pub jax_persistent_cache_min_compile_time_seconds: u32,
    pub jax_xla_autotune_cache: bool,
    pub jax_transfer_guard: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GOutputConfigData {
    pub out: Option<String>,
    pub format: String,
    pub output_run_directory: Option<String>,
    pub writer_threads: u32,
    pub writer_queue_depth: u32,
    pub chunks_per_arrow_file: u32,
    pub arrow_compression: String,
    pub parquet_compression: String,
    pub resume: bool,
    pub resume_mode: String,
    pub finalize_parquet: bool,
}

#[derive(Clone, Debug, PartialEq)]
#[expect(clippy::struct_excessive_bools, reason = "Diagnostics config mirrors public g-specific boolean options.")]
pub struct GDiagnosticsConfigData {
    pub telemetry: String,
    pub log_dir: Option<String>,
    pub stage_timings_json: Option<String>,
    pub log_filter: String,
    pub log_file: Option<String>,
    pub log_stderr: bool,
    pub progress_interval_seconds: f32,
    pub progress_interval_chunks: u32,
    pub profile_summary_json: Option<String>,
    pub trace_file: Option<String>,
    pub trace_filter: String,
    pub trace_event_cap: u32,
    pub log_queue_size: u32,
    pub log_lossy: bool,
    pub include_source_location: bool,
    pub include_span_events: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RegenieConfigData {
    pub input: InputConfigData,
    pub trait_config: TraitConfigData,
    pub binary: BinaryConfigData,
    pub g_compute: GComputeConfigData,
    pub g_output: GOutputConfigData,
    pub g_diagnostics: GDiagnosticsConfigData,
    pub explicit_options: BTreeSet<String>,
    pub is_validated: bool,
}
