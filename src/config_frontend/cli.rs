use std::num::NonZeroU32;
use std::path::Path;

use clap::{ArgAction, CommandFactory, Parser};

use super::data::RegenieConfigData;
use super::domain::{
    ArrowCompressionValue, DeviceValue, FloatingPointDtypeValue, GpuGenotypeFormatValue, JaxMatmulPrecisionValue,
    MultiPhenotypeSampleModeValue, NameList, NullLogisticNonconvergencePolicyValue, OutputFormatValue,
    ParquetCompressionValue, PositiveF32, Probability, ProbabilityFloor, ResumeModeValue, SampleKeyModeValue,
    TelemetryModeValue, TrustedBgenValidationModeValue,
};
use super::resolve::{ConfigLayer, decode_toml_file_layer, resolve_config_layers};
use super::schema::{
    PartialBinaryConfig, PartialComputeConfig, PartialConfig, PartialDiagnosticsConfig, PartialInputConfig,
    PartialOutputConfig, PartialTraitConfig,
};
use super::{ConfigError, ConfigResult};

#[derive(Clone, Debug, PartialEq)]
pub struct CliOutcomeData {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub config: Option<RegenieConfigData>,
}

impl CliOutcomeData {
    fn output(exit_code: i32, stdout: impl Into<String>, stderr: impl Into<String>) -> Self {
        Self { exit_code, stdout: stdout.into(), stderr: stderr.into(), config: None }
    }

    fn config(config: RegenieConfigData) -> Self {
        Self { exit_code: 0, stdout: String::new(), stderr: String::new(), config: Some(config) }
    }
}

#[must_use]
pub fn dispatch_cli(args: &[String], direct_regenie: bool) -> CliOutcomeData {
    match dispatch_cli_result(args, direct_regenie) {
        Ok(outcome) => outcome,
        Err(error) => CliOutcomeData::output(1, String::new(), format!("Error: {}\n", error.message())),
    }
}

fn dispatch_cli_result(args: &[String], direct_regenie: bool) -> ConfigResult<CliOutcomeData> {
    if direct_regenie {
        return dispatch_regenie_command(args, "g-regenie");
    }
    if args.is_empty() {
        return Ok(CliOutcomeData::output(2, root_help("g"), String::new()));
    }
    match args[0].as_str() {
        "--help" | "-h" => Ok(CliOutcomeData::output(0, root_help("g"), String::new())),
        "regenie" => dispatch_regenie_command(&args[1..], "g regenie"),
        command_name => Ok(CliOutcomeData::output(2, String::new(), format!("No such command: {command_name}\n"))),
    }
}

fn dispatch_regenie_command(args: &[String], program_name: &'static str) -> ConfigResult<CliOutcomeData> {
    if args.iter().any(|argument| argument == "--help" || argument == "-h") {
        let mut command = RegenieCli::command();
        command = command.name(program_name);
        return Ok(CliOutcomeData::output(0, command.render_help().to_string(), String::new()));
    }
    let ParsedRegenieCli { config_path, cli_layer } = parse_regenie_cli(args, program_name)?;
    let toml_layer = decode_toml_file_layer(config_path.as_deref().map(Path::new))?;
    let config = resolve_config_layers([toml_layer, cli_layer])?;
    Ok(CliOutcomeData::config(config))
}

#[derive(Clone, Debug)]
struct ParsedRegenieCli {
    config_path: Option<String>,
    cli_layer: ConfigLayer,
}

fn parse_regenie_cli(args: &[String], program_name: &'static str) -> ConfigResult<ParsedRegenieCli> {
    let mut clap_arguments = Vec::with_capacity(args.len() + 1);
    clap_arguments.push(program_name.to_string());
    clap_arguments.extend(args.iter().cloned());
    let parsed_cli = RegenieCli::try_parse_from(clap_arguments).map_err(|error| ConfigError::new(error.to_string()))?;
    let config_path = parsed_cli.config.clone();
    let cli_layer = parsed_cli.into_config_layer()?;
    Ok(ParsedRegenieCli { config_path, cli_layer })
}

#[derive(Clone, Debug, Parser)]
#[command(
    about = "Run a REGENIE-compatible step 2 association scan.",
    disable_version_flag = true,
    rename_all = "verbatim"
)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "Typed Clap parser mirrors supported flags and hidden negative flags."
)]
struct RegenieCli {
    #[arg(long)]
    config: Option<String>,

    #[arg(long)]
    step: Option<u8>,
    #[arg(long, action = ArgAction::SetTrue)]
    qt: bool,
    #[arg(long = "no-qt", hide = true, action = ArgAction::SetTrue)]
    no_qt: bool,
    #[arg(long, action = ArgAction::SetTrue)]
    bt: bool,
    #[arg(long = "no-bt", hide = true, action = ArgAction::SetTrue)]
    no_bt: bool,
    #[arg(long)]
    bsize: Option<NonZeroU32>,
    #[arg(long)]
    threads: Option<NonZeroU32>,

    #[arg(long)]
    bgen: Option<String>,
    #[arg(long)]
    sample: Option<String>,
    #[arg(long = "phenoFile")]
    pheno_file: Option<String>,
    #[arg(long = "phenoCol", action = ArgAction::Append)]
    pheno_col: Vec<String>,
    #[arg(long = "phenoColList")]
    pheno_col_list: Option<NameList>,
    #[arg(long = "covarFile")]
    covar_file: Option<String>,
    #[arg(long = "covarCol", action = ArgAction::Append)]
    covar_col: Vec<String>,
    #[arg(long = "covarColList")]
    covar_col_list: Option<NameList>,
    #[arg(long)]
    pred: Option<String>,

    #[arg(long, action = ArgAction::SetTrue)]
    firth: bool,
    #[arg(long = "no-firth", hide = true, action = ArgAction::SetTrue)]
    no_firth: bool,
    #[arg(long, action = ArgAction::SetTrue)]
    approx: bool,
    #[arg(long = "no-approx", hide = true, action = ArgAction::SetTrue)]
    no_approx: bool,
    #[arg(long = "pThresh")]
    p_threshold: Option<Probability>,
    #[arg(long = "firth-se", action = ArgAction::SetTrue)]
    firth_se: bool,
    #[arg(long = "no-firth-se", hide = true, action = ArgAction::SetTrue)]
    no_firth_se: bool,

    #[arg(long)]
    out: Option<String>,
    #[arg(long)]
    format: Option<OutputFormatValue>,
    #[arg(long)]
    output_run_directory: Option<String>,
    #[arg(long)]
    writer_threads: Option<NonZeroU32>,
    #[arg(long)]
    writer_queue_depth: Option<NonZeroU32>,
    #[arg(long)]
    chunks_per_arrow_file: Option<NonZeroU32>,
    #[arg(long)]
    arrow_compression: Option<ArrowCompressionValue>,
    #[arg(long)]
    parquet_compression: Option<ParquetCompressionValue>,
    #[arg(long, action = ArgAction::SetTrue)]
    resume: bool,
    #[arg(long = "no-resume", hide = true, action = ArgAction::SetTrue)]
    no_resume: bool,
    #[arg(long)]
    resume_mode: Option<ResumeModeValue>,
    #[arg(long, action = ArgAction::SetTrue)]
    finalize_parquet: bool,
    #[arg(long = "no-finalize_parquet", hide = true, action = ArgAction::SetTrue)]
    no_finalize_parquet: bool,

    #[arg(long)]
    device: Option<DeviceValue>,
    #[arg(long)]
    staging_depth: Option<NonZeroU32>,
    #[arg(long)]
    variant_limit: Option<NonZeroU32>,
    #[arg(long, action = ArgAction::SetTrue)]
    trusted_no_missing_diploid: bool,
    #[arg(long = "no-trusted_no_missing_diploid", hide = true, action = ArgAction::SetTrue)]
    no_trusted_no_missing_diploid: bool,
    #[arg(long)]
    trusted_bgen_validation_mode: Option<TrustedBgenValidationModeValue>,
    #[arg(long)]
    sample_key_mode: Option<SampleKeyModeValue>,
    #[arg(long)]
    multi_phenotype_sample_mode: Option<MultiPhenotypeSampleModeValue>,
    #[arg(long)]
    firth_batch_size: Option<NonZeroU32>,
    #[arg(long)]
    firth_candidate_capacity: Option<NonZeroU32>,
    #[arg(long)]
    binary_null_maximum_iterations: Option<NonZeroU32>,
    #[arg(long)]
    binary_null_coefficient_tolerance: Option<PositiveF32>,
    #[arg(long)]
    null_logistic_nonconvergence_policy: Option<NullLogisticNonconvergencePolicyValue>,
    #[arg(long)]
    binary_minimum_probability: Option<ProbabilityFloor>,
    #[arg(long)]
    binary_minimum_variance: Option<PositiveF32>,
    #[arg(long)]
    binary_relative_variance_tolerance: Option<PositiveF32>,
    #[arg(long)]
    linear_minimum_variance: Option<PositiveF32>,
    #[arg(long)]
    linear_relative_variance_tolerance: Option<PositiveF32>,
    #[arg(long)]
    firth_maximum_iterations: Option<NonZeroU32>,
    #[arg(long)]
    firth_gradient_tolerance: Option<PositiveF32>,
    #[arg(long)]
    firth_coefficient_tolerance: Option<PositiveF32>,
    #[arg(long)]
    firth_likelihood_tolerance: Option<PositiveF32>,
    #[arg(long)]
    firth_maximum_step_size: Option<PositiveF32>,
    #[arg(long)]
    firth_pseudo_maximum_iterations: Option<NonZeroU32>,
    #[arg(long)]
    firth_pseudo_inner_maximum_iterations: Option<NonZeroU32>,
    #[arg(long)]
    firth_newton_raphson_zero_start_iterations: Option<NonZeroU32>,
    #[arg(long)]
    firth_line_search_maximum_attempts: Option<NonZeroU32>,
    #[arg(long)]
    firth_step_halving_maximum_attempts: Option<NonZeroU32>,
    #[arg(long)]
    firth_initial_response_scale: Option<PositiveF32>,
    #[arg(long)]
    firth_sparse_carrier_dosage_threshold: Option<PositiveF32>,
    #[arg(long)]
    firth_step_halving_scale: Option<PositiveF32>,
    #[arg(long)]
    null_firth_maximum_iterations: Option<NonZeroU32>,
    #[arg(long)]
    null_firth_gradient_tolerance: Option<PositiveF32>,
    #[arg(long)]
    null_firth_maximum_step_size: Option<PositiveF32>,
    #[arg(long)]
    null_firth_fallback_iteration_multiplier: Option<NonZeroU32>,
    #[arg(long)]
    null_firth_fallback_step_divisor: Option<PositiveF32>,
    #[arg(long)]
    null_firth_line_search_maximum_attempts: Option<NonZeroU32>,
    #[arg(long)]
    null_firth_step_halving_scale: Option<PositiveF32>,
    #[arg(long, action = ArgAction::SetTrue)]
    use_block_firth_math: bool,
    #[arg(long = "no-use_block_firth_math", hide = true, action = ArgAction::SetTrue)]
    no_use_block_firth_math: bool,
    #[arg(long)]
    bgen_decode_tile_variant_count: Option<NonZeroU32>,
    #[arg(long)]
    gpu_genotype_format: Option<GpuGenotypeFormatValue>,
    #[arg(long)]
    score_dtype: Option<FloatingPointDtypeValue>,
    #[arg(long)]
    firth_dtype: Option<FloatingPointDtypeValue>,
    #[arg(long)]
    jax_cache_dir: Option<String>,
    #[arg(long)]
    jax_matmul_precision: Option<JaxMatmulPrecisionValue>,
    #[arg(long, action = ArgAction::SetTrue)]
    jax_persistent_cache: bool,
    #[arg(long = "no-jax_persistent_cache", hide = true, action = ArgAction::SetTrue)]
    no_jax_persistent_cache: bool,
    #[arg(long, allow_hyphen_values = true)]
    jax_persistent_cache_min_entry_size_bytes: Option<i64>,
    #[arg(long)]
    jax_persistent_cache_min_compile_time_seconds: Option<u32>,
    #[arg(long, action = ArgAction::SetTrue)]
    jax_xla_autotune_cache: bool,
    #[arg(long = "no-jax_xla_autotune_cache", hide = true, action = ArgAction::SetTrue)]
    no_jax_xla_autotune_cache: bool,
    #[arg(long, action = ArgAction::SetTrue)]
    jax_transfer_guard: bool,
    #[arg(long = "no-jax_transfer_guard", hide = true, action = ArgAction::SetTrue)]
    no_jax_transfer_guard: bool,

    #[arg(long)]
    telemetry: Option<TelemetryModeValue>,
    #[arg(long)]
    log_dir: Option<String>,
    #[arg(long)]
    stage_timings_json: Option<String>,
    #[arg(long)]
    log_filter: Option<String>,
    #[arg(long)]
    log_file: Option<String>,
    #[arg(long, action = ArgAction::SetTrue)]
    log_stderr: bool,
    #[arg(long = "no-log_stderr", hide = true, action = ArgAction::SetTrue)]
    no_log_stderr: bool,
    #[arg(long)]
    progress_interval_seconds: Option<PositiveF32>,
    #[arg(long)]
    progress_interval_chunks: Option<NonZeroU32>,
    #[arg(long)]
    profile_summary_json: Option<String>,
    #[arg(long)]
    trace_file: Option<String>,
    #[arg(long)]
    trace_filter: Option<String>,
    #[arg(long)]
    trace_event_cap: Option<u32>,
    #[arg(long)]
    log_queue_size: Option<NonZeroU32>,
    #[arg(long, action = ArgAction::SetTrue)]
    log_lossy: bool,
    #[arg(long = "no-log_lossy", hide = true, action = ArgAction::SetTrue)]
    no_log_lossy: bool,
    #[arg(long, action = ArgAction::SetTrue)]
    include_source_location: bool,
    #[arg(long = "no-include_source_location", hide = true, action = ArgAction::SetTrue)]
    no_include_source_location: bool,
    #[arg(long, action = ArgAction::SetTrue)]
    include_span_events: bool,
    #[arg(long = "no-include_span_events", hide = true, action = ArgAction::SetTrue)]
    no_include_span_events: bool,
}

impl RegenieCli {
    fn into_config_layer(self) -> ConfigResult<ConfigLayer> {
        let partial_config = PartialConfig {
            input: self.input_config(),
            trait_config: self.trait_config()?,
            binary: self.binary_config()?,
            compute: self.compute_config()?,
            output: self.output_config()?,
            diagnostics: self.diagnostics_config()?,
            metadata: None,
        };
        Ok(ConfigLayer::from_partial_config(partial_config))
    }

    fn input_config(&self) -> PartialInputConfig {
        PartialInputConfig {
            bgen: self.bgen.clone(),
            sample: self.sample.clone(),
            pheno_file: self.pheno_file.clone(),
            pheno_columns: repeated_name_list(&self.pheno_col),
            pheno_col: None,
            pheno_col_list: self.pheno_col_list.clone(),
            covar_file: self.covar_file.clone(),
            covar_columns: repeated_name_list(&self.covar_col),
            covar_col: None,
            covar_col_list: self.covar_col_list.clone(),
            pred: self.pred.clone(),
        }
    }

    fn trait_config(&self) -> ConfigResult<PartialTraitConfig> {
        Ok(PartialTraitConfig {
            step: self.step,
            trait_type: None,
            qt: optional_flag("qt", self.qt, self.no_qt)?,
            bt: optional_flag("bt", self.bt, self.no_bt)?,
            bsize: self.bsize,
            threads: self.threads,
        })
    }

    fn binary_config(&self) -> ConfigResult<PartialBinaryConfig> {
        Ok(PartialBinaryConfig {
            firth: optional_flag("firth", self.firth, self.no_firth)?,
            approx: optional_flag("approx", self.approx, self.no_approx)?,
            p_threshold: self.p_threshold,
            firth_se: optional_flag("firth-se", self.firth_se, self.no_firth_se)?,
        })
    }

    fn output_config(&self) -> ConfigResult<PartialOutputConfig> {
        Ok(PartialOutputConfig {
            out: self.out.clone(),
            format: self.format,
            output_run_directory: self.output_run_directory.clone(),
            writer_threads: self.writer_threads,
            writer_queue_depth: self.writer_queue_depth,
            chunks_per_arrow_file: self.chunks_per_arrow_file,
            arrow_compression: self.arrow_compression,
            parquet_compression: self.parquet_compression,
            resume: optional_flag("resume", self.resume, self.no_resume)?,
            resume_mode: self.resume_mode,
            finalize_parquet: optional_flag("finalize_parquet", self.finalize_parquet, self.no_finalize_parquet)?,
        })
    }

    fn compute_config(&self) -> ConfigResult<PartialComputeConfig> {
        Ok(PartialComputeConfig {
            device: self.device,
            staging_depth: self.staging_depth,
            variant_limit: self.variant_limit,
            trusted_no_missing_diploid: optional_flag(
                "trusted_no_missing_diploid",
                self.trusted_no_missing_diploid,
                self.no_trusted_no_missing_diploid,
            )?,
            trusted_bgen_validation_mode: self.trusted_bgen_validation_mode,
            sample_key_mode: self.sample_key_mode,
            multi_phenotype_sample_mode: self.multi_phenotype_sample_mode,
            firth_batch_size: self.firth_batch_size,
            firth_candidate_capacity: self.firth_candidate_capacity,
            binary_null_maximum_iterations: self.binary_null_maximum_iterations,
            binary_null_coefficient_tolerance: self.binary_null_coefficient_tolerance,
            null_logistic_nonconvergence_policy: self.null_logistic_nonconvergence_policy,
            binary_minimum_probability: self.binary_minimum_probability,
            binary_minimum_variance: self.binary_minimum_variance,
            binary_relative_variance_tolerance: self.binary_relative_variance_tolerance,
            linear_minimum_variance: self.linear_minimum_variance,
            linear_relative_variance_tolerance: self.linear_relative_variance_tolerance,
            firth_maximum_iterations: self.firth_maximum_iterations,
            firth_gradient_tolerance: self.firth_gradient_tolerance,
            firth_coefficient_tolerance: self.firth_coefficient_tolerance,
            firth_likelihood_tolerance: self.firth_likelihood_tolerance,
            firth_maximum_step_size: self.firth_maximum_step_size,
            firth_pseudo_maximum_iterations: self.firth_pseudo_maximum_iterations,
            firth_pseudo_inner_maximum_iterations: self.firth_pseudo_inner_maximum_iterations,
            firth_newton_raphson_zero_start_iterations: self.firth_newton_raphson_zero_start_iterations,
            firth_line_search_maximum_attempts: self.firth_line_search_maximum_attempts,
            firth_step_halving_maximum_attempts: self.firth_step_halving_maximum_attempts,
            firth_initial_response_scale: self.firth_initial_response_scale,
            firth_sparse_carrier_dosage_threshold: self.firth_sparse_carrier_dosage_threshold,
            firth_step_halving_scale: self.firth_step_halving_scale,
            null_firth_maximum_iterations: self.null_firth_maximum_iterations,
            null_firth_gradient_tolerance: self.null_firth_gradient_tolerance,
            null_firth_maximum_step_size: self.null_firth_maximum_step_size,
            null_firth_fallback_iteration_multiplier: self.null_firth_fallback_iteration_multiplier,
            null_firth_fallback_step_divisor: self.null_firth_fallback_step_divisor,
            null_firth_line_search_maximum_attempts: self.null_firth_line_search_maximum_attempts,
            null_firth_step_halving_scale: self.null_firth_step_halving_scale,
            use_block_firth_math: optional_flag(
                "use_block_firth_math",
                self.use_block_firth_math,
                self.no_use_block_firth_math,
            )?,
            bgen_decode_tile_variant_count: self.bgen_decode_tile_variant_count,
            gpu_genotype_format: self.gpu_genotype_format,
            score_dtype: self.score_dtype,
            firth_dtype: self.firth_dtype,
            jax_cache_dir: self.jax_cache_dir.clone(),
            jax_matmul_precision: self.jax_matmul_precision,
            jax_persistent_cache: optional_flag(
                "jax_persistent_cache",
                self.jax_persistent_cache,
                self.no_jax_persistent_cache,
            )?,
            jax_persistent_cache_min_entry_size_bytes: self.jax_persistent_cache_min_entry_size_bytes,
            jax_persistent_cache_min_compile_time_seconds: self.jax_persistent_cache_min_compile_time_seconds,
            jax_xla_autotune_cache: optional_flag(
                "jax_xla_autotune_cache",
                self.jax_xla_autotune_cache,
                self.no_jax_xla_autotune_cache,
            )?,
            jax_transfer_guard: optional_flag(
                "jax_transfer_guard",
                self.jax_transfer_guard,
                self.no_jax_transfer_guard,
            )?,
        })
    }

    fn diagnostics_config(&self) -> ConfigResult<PartialDiagnosticsConfig> {
        Ok(PartialDiagnosticsConfig {
            telemetry: self.telemetry,
            log_dir: self.log_dir.clone(),
            stage_timings_json: self.stage_timings_json.clone(),
            log_filter: self.log_filter.clone(),
            log_file: self.log_file.clone(),
            log_stderr: optional_flag("log_stderr", self.log_stderr, self.no_log_stderr)?,
            progress_interval_seconds: self.progress_interval_seconds,
            progress_interval_chunks: self.progress_interval_chunks,
            profile_summary_json: self.profile_summary_json.clone(),
            trace_file: self.trace_file.clone(),
            trace_filter: self.trace_filter.clone(),
            trace_event_cap: self.trace_event_cap,
            log_queue_size: self.log_queue_size,
            log_lossy: optional_flag("log_lossy", self.log_lossy, self.no_log_lossy)?,
            include_source_location: optional_flag(
                "include_source_location",
                self.include_source_location,
                self.no_include_source_location,
            )?,
            include_span_events: optional_flag(
                "include_span_events",
                self.include_span_events,
                self.no_include_span_events,
            )?,
        })
    }
}

fn repeated_name_list(values: &[String]) -> Option<NameList> {
    (!values.is_empty()).then(|| NameList::from_values(values.to_vec()))
}

fn optional_flag(option_name: &str, positive_value: bool, negative_value: bool) -> ConfigResult<Option<bool>> {
    if positive_value && negative_value {
        return Err(ConfigError::new(format!("--{option_name} and --no-{option_name} cannot be used together.")));
    }
    Ok(if positive_value { Some(true) } else { negative_value.then_some(false) })
}

fn root_help(program_name: &str) -> String {
    format!(
        "Blazing fast REGENIE step 2 GWAS engine.\n\nUsage: {program_name} <COMMAND> [OPTIONS]\n\nCommands:\n  regenie  Run a REGENIE-compatible step 2 association scan.\n\nOptions:\n  -h, --help  Print help\n"
    )
}
