use std::num::NonZeroU32;

use clap::{ArgAction, Args, Parser};

use super::super::domain::{
    ArrowCompressionValue, DeviceValue, FloatingPointDtypeValue, GpuGenotypeFormatValue, JaxMatmulPrecisionValue,
    MultiPhenotypeSampleModeValue, NameList, NullLogisticNonconvergencePolicyValue, OutputFormatValue,
    ParquetCompressionValue, PositiveF32, Probability, ProbabilityFloor, ResumeModeValue, SampleKeyModeValue,
    TelemetryModeValue, TrustedBgenValidationModeValue,
};
use super::super::overlay::ConfigLayer;
use super::super::{ConfigError, ConfigResult};

#[derive(Clone, Debug)]
pub(crate) struct ParsedRegenieCli {
    pub(crate) config_path: Option<String>,
    pub(crate) cli_layer: ConfigLayer,
}

pub(crate) fn parse_regenie_cli(args: &[String], program_name: &'static str) -> ConfigResult<ParsedRegenieCli> {
    let mut clap_arguments = Vec::with_capacity(args.len() + 1);
    clap_arguments.push(program_name.to_string());
    clap_arguments.extend(args.iter().cloned());
    let parsed_cli = RegenieCli::try_parse_from(clap_arguments).map_err(|error| ConfigError::new(error.to_string()))?;
    let config_path = parsed_cli.config.clone();
    let cli_layer = parsed_cli.into_config_layer()?;
    Ok(ParsedRegenieCli { config_path, cli_layer })
}

#[derive(Clone, Debug, Parser)]
#[command(about = "Run a REGENIE-compatible step 2 association scan.", disable_version_flag = true)]
pub(crate) struct RegenieCli {
    #[arg(long = "config", help_heading = "Config")]
    pub(crate) config: Option<String>,
    #[command(flatten)]
    pub(crate) trait_options: TraitCli,
    #[command(flatten)]
    pub(crate) input: InputCli,
    #[command(flatten)]
    pub(crate) binary: BinaryCli,
    #[command(flatten)]
    pub(crate) output: OutputCli,
    #[command(flatten)]
    pub(crate) compute: ComputeCli,
    #[command(flatten)]
    pub(crate) diagnostics: DiagnosticsCli,
}

#[derive(Clone, Debug, Args)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "Typed Clap parser mirrors supported trait flags and hidden negative flags."
)]
pub(crate) struct TraitCli {
    #[arg(long = "step", help_heading = "Trait")]
    pub(crate) step: Option<u8>,
    #[arg(long = "qt", action = ArgAction::SetTrue, help_heading = "Trait")]
    pub(crate) qt: bool,
    #[arg(long = "no-qt", hide = true, action = ArgAction::SetTrue, help_heading = "Trait")]
    pub(crate) no_qt: bool,
    #[arg(long = "bt", action = ArgAction::SetTrue, help_heading = "Trait")]
    pub(crate) bt: bool,
    #[arg(long = "no-bt", hide = true, action = ArgAction::SetTrue, help_heading = "Trait")]
    pub(crate) no_bt: bool,
    #[arg(long = "bsize", help_heading = "Trait")]
    pub(crate) bsize: Option<NonZeroU32>,
    #[arg(long = "threads", help_heading = "Trait")]
    pub(crate) threads: Option<NonZeroU32>,
}

#[derive(Clone, Debug, Args)]
pub(crate) struct InputCli {
    #[arg(long = "bgen", help_heading = "Input")]
    pub(crate) bgen: Option<String>,
    #[arg(long = "sample", help_heading = "Input")]
    pub(crate) sample: Option<String>,
    #[arg(long = "phenoFile", help_heading = "Input")]
    pub(crate) pheno_file: Option<String>,
    #[arg(long = "phenoCol", action = ArgAction::Append, help_heading = "Input")]
    pub(crate) pheno_col: Vec<String>,
    #[arg(long = "phenoColList", help_heading = "Input")]
    pub(crate) pheno_col_list: Option<NameList>,
    #[arg(long = "covarFile", help_heading = "Input")]
    pub(crate) covar_file: Option<String>,
    #[arg(long = "covarCol", action = ArgAction::Append, help_heading = "Input")]
    pub(crate) covar_col: Vec<String>,
    #[arg(long = "covarColList", help_heading = "Input")]
    pub(crate) covar_col_list: Option<NameList>,
    #[arg(long = "pred", help_heading = "Input")]
    pub(crate) pred: Option<String>,
}

#[derive(Clone, Debug, Args)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "Typed Clap parser mirrors supported binary flags and hidden negative flags."
)]
pub(crate) struct BinaryCli {
    #[arg(long = "firth", action = ArgAction::SetTrue, help_heading = "Binary")]
    pub(crate) firth: bool,
    #[arg(long = "no-firth", hide = true, action = ArgAction::SetTrue, help_heading = "Binary")]
    pub(crate) no_firth: bool,
    #[arg(long = "approx", action = ArgAction::SetTrue, help_heading = "Binary")]
    pub(crate) approx: bool,
    #[arg(long = "no-approx", hide = true, action = ArgAction::SetTrue, help_heading = "Binary")]
    pub(crate) no_approx: bool,
    #[arg(long = "pThresh", help_heading = "Binary")]
    pub(crate) p_threshold: Option<Probability>,
    #[arg(long = "firth-se", action = ArgAction::SetTrue, help_heading = "Binary")]
    pub(crate) firth_se: bool,
    #[arg(long = "no-firth-se", hide = true, action = ArgAction::SetTrue, help_heading = "Binary")]
    pub(crate) no_firth_se: bool,
}

#[derive(Clone, Debug, Args)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "Typed Clap parser mirrors supported output flags and hidden negative flags."
)]
pub(crate) struct OutputCli {
    #[arg(long = "out", help_heading = "Output")]
    pub(crate) out: Option<String>,
    #[arg(long = "format", help_heading = "Output")]
    pub(crate) format: Option<OutputFormatValue>,
    #[arg(long = "output_run_directory", help_heading = "Output")]
    pub(crate) output_run_directory: Option<String>,
    #[arg(long = "writer_threads", help_heading = "Output")]
    pub(crate) writer_threads: Option<NonZeroU32>,
    #[arg(long = "writer_queue_depth", help_heading = "Output")]
    pub(crate) writer_queue_depth: Option<NonZeroU32>,
    #[arg(long = "chunks_per_arrow_file", help_heading = "Output")]
    pub(crate) chunks_per_arrow_file: Option<NonZeroU32>,
    #[arg(long = "arrow_compression", help_heading = "Output")]
    pub(crate) arrow_compression: Option<ArrowCompressionValue>,
    #[arg(long = "parquet_compression", help_heading = "Output")]
    pub(crate) parquet_compression: Option<ParquetCompressionValue>,
    #[arg(long = "output_statistic_dtype", help_heading = "Output")]
    pub(crate) output_statistic_dtype: Option<FloatingPointDtypeValue>,
    #[arg(long = "resume", action = ArgAction::SetTrue, help_heading = "Output")]
    pub(crate) resume: bool,
    #[arg(long = "no-resume", hide = true, action = ArgAction::SetTrue, help_heading = "Output")]
    pub(crate) no_resume: bool,
    #[arg(long = "resume_mode", help_heading = "Output")]
    pub(crate) resume_mode: Option<ResumeModeValue>,
    #[arg(long = "finalize_parquet", action = ArgAction::SetTrue, help_heading = "Output")]
    pub(crate) finalize_parquet: bool,
    #[arg(long = "no-finalize_parquet", hide = true, action = ArgAction::SetTrue, help_heading = "Output")]
    pub(crate) no_finalize_parquet: bool,
}

#[derive(Clone, Debug, Args)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "Typed Clap parser mirrors supported compute flags and hidden negative flags."
)]
pub(crate) struct ComputeCli {
    #[arg(long = "device", help_heading = "Compute")]
    pub(crate) device: Option<DeviceValue>,
    #[arg(long = "staging_depth", help_heading = "Compute")]
    pub(crate) staging_depth: Option<NonZeroU32>,
    #[arg(long = "native_callback_batch_size", help_heading = "Compute")]
    pub(crate) native_callback_batch_size: Option<NonZeroU32>,
    #[arg(long = "result_in_flight_limit", help_heading = "Compute")]
    pub(crate) result_in_flight_limit: Option<NonZeroU32>,
    #[arg(long = "dosage_buffer_limit", help_heading = "Compute")]
    pub(crate) dosage_buffer_limit: Option<NonZeroU32>,
    #[arg(long = "variant_limit", help_heading = "Compute")]
    pub(crate) variant_limit: Option<NonZeroU32>,
    #[arg(long = "trusted_no_missing_diploid", action = ArgAction::SetTrue, help_heading = "Compute")]
    pub(crate) trusted_no_missing_diploid: bool,
    #[arg(long = "no-trusted_no_missing_diploid", hide = true, action = ArgAction::SetTrue, help_heading = "Compute")]
    pub(crate) no_trusted_no_missing_diploid: bool,
    #[arg(long = "trusted_bgen_validation_mode", help_heading = "Compute")]
    pub(crate) trusted_bgen_validation_mode: Option<TrustedBgenValidationModeValue>,
    #[arg(long = "sample_key_mode", help_heading = "Compute")]
    pub(crate) sample_key_mode: Option<SampleKeyModeValue>,
    #[arg(long = "multi_phenotype_sample_mode", help_heading = "Compute")]
    pub(crate) multi_phenotype_sample_mode: Option<MultiPhenotypeSampleModeValue>,
    #[arg(long = "firth_batch_size", help_heading = "Compute")]
    pub(crate) firth_batch_size: Option<NonZeroU32>,
    #[arg(long = "firth_candidate_capacity", help_heading = "Compute")]
    pub(crate) firth_candidate_capacity: Option<NonZeroU32>,
    #[arg(long = "binary_null_maximum_iterations", help_heading = "Compute")]
    pub(crate) binary_null_maximum_iterations: Option<NonZeroU32>,
    #[arg(long = "binary_null_coefficient_tolerance", help_heading = "Compute")]
    pub(crate) binary_null_coefficient_tolerance: Option<PositiveF32>,
    #[arg(long = "null_logistic_nonconvergence_policy", help_heading = "Compute")]
    pub(crate) null_logistic_nonconvergence_policy: Option<NullLogisticNonconvergencePolicyValue>,
    #[arg(long = "binary_minimum_probability", help_heading = "Compute")]
    pub(crate) binary_minimum_probability: Option<ProbabilityFloor>,
    #[arg(long = "binary_minimum_variance", help_heading = "Compute")]
    pub(crate) binary_minimum_variance: Option<PositiveF32>,
    #[arg(long = "binary_relative_variance_tolerance", help_heading = "Compute")]
    pub(crate) binary_relative_variance_tolerance: Option<PositiveF32>,
    #[arg(long = "linear_minimum_variance", help_heading = "Compute")]
    pub(crate) linear_minimum_variance: Option<PositiveF32>,
    #[arg(long = "linear_relative_variance_tolerance", help_heading = "Compute")]
    pub(crate) linear_relative_variance_tolerance: Option<PositiveF32>,
    #[arg(long = "firth_maximum_iterations", help_heading = "Compute")]
    pub(crate) firth_maximum_iterations: Option<NonZeroU32>,
    #[arg(long = "firth_gradient_tolerance", help_heading = "Compute")]
    pub(crate) firth_gradient_tolerance: Option<PositiveF32>,
    #[arg(long = "firth_coefficient_tolerance", help_heading = "Compute")]
    pub(crate) firth_coefficient_tolerance: Option<PositiveF32>,
    #[arg(long = "firth_likelihood_tolerance", help_heading = "Compute")]
    pub(crate) firth_likelihood_tolerance: Option<PositiveF32>,
    #[arg(long = "firth_maximum_step_size", help_heading = "Compute")]
    pub(crate) firth_maximum_step_size: Option<PositiveF32>,
    #[arg(long = "firth_pseudo_maximum_iterations", help_heading = "Compute")]
    pub(crate) firth_pseudo_maximum_iterations: Option<NonZeroU32>,
    #[arg(long = "firth_pseudo_inner_maximum_iterations", help_heading = "Compute")]
    pub(crate) firth_pseudo_inner_maximum_iterations: Option<NonZeroU32>,
    #[arg(long = "firth_newton_raphson_zero_start_iterations", help_heading = "Compute")]
    pub(crate) firth_newton_raphson_zero_start_iterations: Option<NonZeroU32>,
    #[arg(long = "firth_line_search_maximum_attempts", help_heading = "Compute")]
    pub(crate) firth_line_search_maximum_attempts: Option<NonZeroU32>,
    #[arg(long = "firth_step_halving_maximum_attempts", help_heading = "Compute")]
    pub(crate) firth_step_halving_maximum_attempts: Option<NonZeroU32>,
    #[arg(long = "firth_initial_response_scale", help_heading = "Compute")]
    pub(crate) firth_initial_response_scale: Option<PositiveF32>,
    #[arg(long = "firth_sparse_carrier_dosage_threshold", help_heading = "Compute")]
    pub(crate) firth_sparse_carrier_dosage_threshold: Option<PositiveF32>,
    #[arg(long = "firth_step_halving_scale", help_heading = "Compute")]
    pub(crate) firth_step_halving_scale: Option<PositiveF32>,
    #[arg(long = "null_firth_maximum_iterations", help_heading = "Compute")]
    pub(crate) null_firth_maximum_iterations: Option<NonZeroU32>,
    #[arg(long = "null_firth_gradient_tolerance", help_heading = "Compute")]
    pub(crate) null_firth_gradient_tolerance: Option<PositiveF32>,
    #[arg(long = "null_firth_maximum_step_size", help_heading = "Compute")]
    pub(crate) null_firth_maximum_step_size: Option<PositiveF32>,
    #[arg(long = "null_firth_fallback_iteration_multiplier", help_heading = "Compute")]
    pub(crate) null_firth_fallback_iteration_multiplier: Option<NonZeroU32>,
    #[arg(long = "null_firth_fallback_step_divisor", help_heading = "Compute")]
    pub(crate) null_firth_fallback_step_divisor: Option<PositiveF32>,
    #[arg(long = "null_firth_line_search_maximum_attempts", help_heading = "Compute")]
    pub(crate) null_firth_line_search_maximum_attempts: Option<NonZeroU32>,
    #[arg(long = "null_firth_step_halving_scale", help_heading = "Compute")]
    pub(crate) null_firth_step_halving_scale: Option<PositiveF32>,
    #[arg(long = "use_block_firth_math", action = ArgAction::SetTrue, help_heading = "Compute")]
    pub(crate) use_block_firth_math: bool,
    #[arg(long = "no-use_block_firth_math", hide = true, action = ArgAction::SetTrue, help_heading = "Compute")]
    pub(crate) no_use_block_firth_math: bool,
    #[arg(long = "bgen_decode_tile_variant_count", help_heading = "Compute")]
    pub(crate) bgen_decode_tile_variant_count: Option<NonZeroU32>,
    #[arg(long = "gpu_genotype_format", help_heading = "Compute")]
    pub(crate) gpu_genotype_format: Option<GpuGenotypeFormatValue>,
    #[arg(long = "score_dtype", help_heading = "Compute")]
    pub(crate) score_dtype: Option<FloatingPointDtypeValue>,
    #[arg(long = "firth_dtype", help_heading = "Compute")]
    pub(crate) firth_dtype: Option<FloatingPointDtypeValue>,
    #[arg(long = "jax_cache_dir", help_heading = "Compute")]
    pub(crate) jax_cache_dir: Option<String>,
    #[arg(long = "jax_matmul_precision", help_heading = "Compute")]
    pub(crate) jax_matmul_precision: Option<JaxMatmulPrecisionValue>,
    #[arg(long = "jax_persistent_cache", action = ArgAction::SetTrue, help_heading = "Compute")]
    pub(crate) jax_persistent_cache: bool,
    #[arg(long = "no-jax_persistent_cache", hide = true, action = ArgAction::SetTrue, help_heading = "Compute")]
    pub(crate) no_jax_persistent_cache: bool,
    #[arg(long = "jax_persistent_cache_min_entry_size_bytes", allow_hyphen_values = true, help_heading = "Compute")]
    pub(crate) jax_persistent_cache_min_entry_size_bytes: Option<i64>,
    #[arg(long = "jax_persistent_cache_min_compile_time_seconds", help_heading = "Compute")]
    pub(crate) jax_persistent_cache_min_compile_time_seconds: Option<u32>,
    #[arg(long = "jax_xla_autotune_cache", action = ArgAction::SetTrue, help_heading = "Compute")]
    pub(crate) jax_xla_autotune_cache: bool,
    #[arg(long = "no-jax_xla_autotune_cache", hide = true, action = ArgAction::SetTrue, help_heading = "Compute")]
    pub(crate) no_jax_xla_autotune_cache: bool,
    #[arg(long = "jax_transfer_guard", action = ArgAction::SetTrue, help_heading = "Compute")]
    pub(crate) jax_transfer_guard: bool,
    #[arg(long = "no-jax_transfer_guard", hide = true, action = ArgAction::SetTrue, help_heading = "Compute")]
    pub(crate) no_jax_transfer_guard: bool,
}

#[derive(Clone, Debug, Args)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "Typed Clap parser mirrors supported diagnostics flags and hidden negative flags."
)]
pub(crate) struct DiagnosticsCli {
    #[arg(long = "telemetry", help_heading = "Diagnostics")]
    pub(crate) telemetry: Option<TelemetryModeValue>,
    #[arg(long = "log_dir", help_heading = "Diagnostics")]
    pub(crate) log_dir: Option<String>,
    #[arg(long = "stage_timings_json", help_heading = "Diagnostics")]
    pub(crate) stage_timings_json: Option<String>,
    #[arg(long = "log_filter", help_heading = "Diagnostics")]
    pub(crate) log_filter: Option<String>,
    #[arg(long = "log_file", help_heading = "Diagnostics")]
    pub(crate) log_file: Option<String>,
    #[arg(long = "log_stderr", action = ArgAction::SetTrue, help_heading = "Diagnostics")]
    pub(crate) log_stderr: bool,
    #[arg(long = "no-log_stderr", hide = true, action = ArgAction::SetTrue, help_heading = "Diagnostics")]
    pub(crate) no_log_stderr: bool,
    #[arg(long = "progress_interval_seconds", help_heading = "Diagnostics")]
    pub(crate) progress_interval_seconds: Option<PositiveF32>,
    #[arg(long = "progress_interval_chunks", help_heading = "Diagnostics")]
    pub(crate) progress_interval_chunks: Option<NonZeroU32>,
    #[arg(long = "profile_summary_json", help_heading = "Diagnostics")]
    pub(crate) profile_summary_json: Option<String>,
    #[arg(long = "trace_file", help_heading = "Diagnostics")]
    pub(crate) trace_file: Option<String>,
    #[arg(long = "trace_filter", help_heading = "Diagnostics")]
    pub(crate) trace_filter: Option<String>,
    #[arg(long = "trace_event_cap", help_heading = "Diagnostics")]
    pub(crate) trace_event_cap: Option<u32>,
    #[arg(long = "log_queue_size", help_heading = "Diagnostics")]
    pub(crate) log_queue_size: Option<NonZeroU32>,
    #[arg(long = "log_lossy", action = ArgAction::SetTrue, help_heading = "Diagnostics")]
    pub(crate) log_lossy: bool,
    #[arg(long = "no-log_lossy", hide = true, action = ArgAction::SetTrue, help_heading = "Diagnostics")]
    pub(crate) no_log_lossy: bool,
    #[arg(long = "include_source_location", action = ArgAction::SetTrue, help_heading = "Diagnostics")]
    pub(crate) include_source_location: bool,
    #[arg(long = "no-include_source_location", hide = true, action = ArgAction::SetTrue, help_heading = "Diagnostics")]
    pub(crate) no_include_source_location: bool,
    #[arg(long = "include_span_events", action = ArgAction::SetTrue, help_heading = "Diagnostics")]
    pub(crate) include_span_events: bool,
    #[arg(long = "no-include_span_events", hide = true, action = ArgAction::SetTrue, help_heading = "Diagnostics")]
    pub(crate) no_include_span_events: bool,
}
