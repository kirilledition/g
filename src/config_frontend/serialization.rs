use std::fs;
use std::path::Path;

use serde::Serialize;

use super::data::RegenieConfigData;
use super::defaults::load_default_config_data;
use super::{ConfigError, ConfigResult, OPTION_SCHEMA_VERSION};

/// Write deterministic effective TOML for a resolved config.
///
/// # Errors
///
/// Returns an error when serialization fails or the file cannot be written.
pub fn write_toml(config: &RegenieConfigData, path: &Path) -> ConfigResult<()> {
    fs::write(path, dumps_toml(config)?)
        .map_err(|error| ConfigError::new(format!("Failed to write TOML config {}: {error}", path.display())))
}

/// Serialize a resolved config to TOML.
///
/// # Errors
///
/// Returns an error when serialization fails or default metadata cannot be loaded.
pub fn dumps_toml(config: &RegenieConfigData) -> ConfigResult<String> {
    toml::to_string(&EffectiveConfigToml::from_config(config)?)
        .map_err(|error| ConfigError::new(format!("Failed to serialize TOML config: {error}")))
}

#[derive(Serialize)]
struct EffectiveConfigToml {
    #[serde(skip_serializing_if = "InputToml::is_empty")]
    input: InputToml,
    #[serde(rename = "trait")]
    trait_config: TraitToml,
    #[serde(skip_serializing_if = "Option::is_none")]
    binary: Option<BinaryToml>,
    output: OutputToml,
    compute: ComputeToml,
    diagnostics: DiagnosticsToml,
    metadata: MetadataToml,
}

impl EffectiveConfigToml {
    fn from_config(config: &RegenieConfigData) -> ConfigResult<Self> {
        Ok(Self {
            input: InputToml::from_config(config),
            trait_config: TraitToml::from_config(config),
            binary: (config.trait_config.trait_type == "binary").then(|| BinaryToml::from_config(config)),
            output: OutputToml::from_config(config),
            compute: ComputeToml::from_config(config),
            diagnostics: DiagnosticsToml::from_config(config),
            metadata: MetadataToml::new()?,
        })
    }
}

#[derive(Serialize)]
struct InputToml {
    #[serde(skip_serializing_if = "Option::is_none")]
    bgen: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sample: Option<String>,
    #[serde(rename = "phenoFile", skip_serializing_if = "Option::is_none")]
    pheno_file: Option<String>,
    #[serde(rename = "phenoCol", skip_serializing_if = "Option::is_none")]
    pheno_col: Option<String>,
    #[serde(rename = "phenoColList", skip_serializing_if = "Option::is_none")]
    pheno_col_list: Option<String>,
    #[serde(rename = "covarFile", skip_serializing_if = "Option::is_none")]
    covar_file: Option<String>,
    #[serde(rename = "covarCol", skip_serializing_if = "Option::is_none")]
    covar_col: Option<String>,
    #[serde(rename = "covarColList", skip_serializing_if = "Option::is_none")]
    covar_col_list: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pred: Option<String>,
}

impl InputToml {
    fn from_config(config: &RegenieConfigData) -> Self {
        let (pheno_col, pheno_col_list) = serialize_column_options(&config.input.pheno_columns);
        let (covar_col, covar_col_list) = serialize_column_options(&config.input.covar_columns);
        Self {
            bgen: config.input.bgen.clone(),
            sample: config.input.sample.clone(),
            pheno_file: config.input.pheno_file.clone(),
            pheno_col,
            pheno_col_list,
            covar_file: config.input.covar_file.clone(),
            covar_col,
            covar_col_list,
            pred: config.input.pred.clone(),
        }
    }

    fn is_empty(&self) -> bool {
        self.bgen.is_none()
            && self.sample.is_none()
            && self.pheno_file.is_none()
            && self.pheno_col.is_none()
            && self.pheno_col_list.is_none()
            && self.covar_file.is_none()
            && self.covar_col.is_none()
            && self.covar_col_list.is_none()
            && self.pred.is_none()
    }
}

#[derive(Serialize)]
struct TraitToml {
    step: u8,
    qt: bool,
    bt: bool,
    bsize: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    threads: Option<u32>,
}

impl TraitToml {
    fn from_config(config: &RegenieConfigData) -> Self {
        Self {
            step: config.trait_config.step,
            qt: config.trait_config.trait_type == "quantitative",
            bt: config.trait_config.trait_type == "binary",
            bsize: config.trait_config.bsize,
            threads: config.trait_config.threads,
        }
    }
}

#[derive(Serialize)]
struct BinaryToml {
    firth: bool,
    approx: bool,
    #[serde(rename = "pThresh")]
    p_threshold: f32,
    #[serde(rename = "firth-se")]
    firth_se: bool,
}

impl BinaryToml {
    fn from_config(config: &RegenieConfigData) -> Self {
        Self {
            firth: config.binary.firth,
            approx: config.binary.approx,
            p_threshold: config.binary.p_threshold,
            firth_se: config.binary.firth_se,
        }
    }
}

#[derive(Serialize)]
struct OutputToml {
    #[serde(skip_serializing_if = "Option::is_none")]
    out: Option<String>,
    format: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_run_directory: Option<String>,
    writer_threads: u32,
    writer_queue_depth: u32,
    chunks_per_arrow_file: u32,
    arrow_compression: String,
    parquet_compression: String,
    resume: bool,
    resume_mode: String,
    finalize_parquet: bool,
}

impl OutputToml {
    fn from_config(config: &RegenieConfigData) -> Self {
        Self {
            out: config.g_output.out.clone(),
            format: config.g_output.format.clone(),
            output_run_directory: config.g_output.output_run_directory.clone(),
            writer_threads: config.g_output.writer_threads,
            writer_queue_depth: config.g_output.writer_queue_depth,
            chunks_per_arrow_file: config.g_output.chunks_per_arrow_file,
            arrow_compression: config.g_output.arrow_compression.clone(),
            parquet_compression: config.g_output.parquet_compression.clone(),
            resume: config.g_output.resume,
            resume_mode: config.g_output.resume_mode.clone(),
            finalize_parquet: config.g_output.finalize_parquet,
        }
    }
}

#[derive(Serialize)]
#[expect(clippy::struct_excessive_bools, reason = "TOML serialization mirrors the public compute schema.")]
struct ComputeToml {
    device: String,
    staging_depth: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    variant_limit: Option<u32>,
    trusted_no_missing_diploid: bool,
    trusted_bgen_validation_mode: String,
    sample_key_mode: String,
    multi_phenotype_sample_mode: String,
    firth_batch_size: u32,
    firth_candidate_capacity: u32,
    binary_null_maximum_iterations: u32,
    binary_null_coefficient_tolerance: f32,
    null_logistic_nonconvergence_policy: String,
    binary_minimum_probability: f32,
    binary_minimum_variance: f32,
    binary_relative_variance_tolerance: f32,
    linear_minimum_variance: f32,
    linear_relative_variance_tolerance: f32,
    firth_maximum_iterations: u32,
    firth_gradient_tolerance: f32,
    firth_coefficient_tolerance: f32,
    firth_likelihood_tolerance: f32,
    firth_maximum_step_size: f32,
    firth_pseudo_maximum_iterations: u32,
    firth_pseudo_inner_maximum_iterations: u32,
    firth_newton_raphson_zero_start_iterations: u32,
    firth_line_search_maximum_attempts: u32,
    firth_step_halving_maximum_attempts: u32,
    firth_initial_response_scale: f32,
    firth_sparse_carrier_dosage_threshold: f32,
    firth_step_halving_scale: f32,
    null_firth_maximum_iterations: u32,
    null_firth_gradient_tolerance: f32,
    null_firth_maximum_step_size: f32,
    null_firth_fallback_iteration_multiplier: u32,
    null_firth_fallback_step_divisor: f32,
    null_firth_line_search_maximum_attempts: u32,
    null_firth_step_halving_scale: f32,
    use_block_firth_math: bool,
    bgen_decode_tile_variant_count: u32,
    gpu_genotype_format: String,
    score_dtype: String,
    firth_dtype: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    jax_cache_dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    jax_matmul_precision: Option<String>,
    jax_persistent_cache: bool,
    jax_persistent_cache_min_entry_size_bytes: i64,
    jax_persistent_cache_min_compile_time_seconds: u32,
    jax_xla_autotune_cache: bool,
    jax_transfer_guard: bool,
}

impl ComputeToml {
    fn from_config(config: &RegenieConfigData) -> Self {
        Self {
            device: config.g_compute.device.clone(),
            staging_depth: config.g_compute.staging_depth,
            variant_limit: config.g_compute.variant_limit,
            trusted_no_missing_diploid: config.g_compute.trusted_no_missing_diploid,
            trusted_bgen_validation_mode: config.g_compute.trusted_bgen_validation_mode.clone(),
            sample_key_mode: config.g_compute.sample_key_mode.clone(),
            multi_phenotype_sample_mode: config.g_compute.multi_phenotype_sample_mode.clone(),
            firth_batch_size: config.g_compute.firth_batch_size,
            firth_candidate_capacity: config.g_compute.firth_candidate_capacity,
            binary_null_maximum_iterations: config.g_compute.binary_null_maximum_iterations,
            binary_null_coefficient_tolerance: config.g_compute.binary_null_coefficient_tolerance,
            null_logistic_nonconvergence_policy: config.g_compute.null_logistic_nonconvergence_policy.clone(),
            binary_minimum_probability: config.g_compute.binary_minimum_probability,
            binary_minimum_variance: config.g_compute.binary_minimum_variance,
            binary_relative_variance_tolerance: config.g_compute.binary_relative_variance_tolerance,
            linear_minimum_variance: config.g_compute.linear_minimum_variance,
            linear_relative_variance_tolerance: config.g_compute.linear_relative_variance_tolerance,
            firth_maximum_iterations: config.g_compute.firth_maximum_iterations,
            firth_gradient_tolerance: config.g_compute.firth_gradient_tolerance,
            firth_coefficient_tolerance: config.g_compute.firth_coefficient_tolerance,
            firth_likelihood_tolerance: config.g_compute.firth_likelihood_tolerance,
            firth_maximum_step_size: config.g_compute.firth_maximum_step_size,
            firth_pseudo_maximum_iterations: config.g_compute.firth_pseudo_maximum_iterations,
            firth_pseudo_inner_maximum_iterations: config.g_compute.firth_pseudo_inner_maximum_iterations,
            firth_newton_raphson_zero_start_iterations: config.g_compute.firth_newton_raphson_zero_start_iterations,
            firth_line_search_maximum_attempts: config.g_compute.firth_line_search_maximum_attempts,
            firth_step_halving_maximum_attempts: config.g_compute.firth_step_halving_maximum_attempts,
            firth_initial_response_scale: config.g_compute.firth_initial_response_scale,
            firth_sparse_carrier_dosage_threshold: config.g_compute.firth_sparse_carrier_dosage_threshold,
            firth_step_halving_scale: config.g_compute.firth_step_halving_scale,
            null_firth_maximum_iterations: config.g_compute.null_firth_maximum_iterations,
            null_firth_gradient_tolerance: config.g_compute.null_firth_gradient_tolerance,
            null_firth_maximum_step_size: config.g_compute.null_firth_maximum_step_size,
            null_firth_fallback_iteration_multiplier: config.g_compute.null_firth_fallback_iteration_multiplier,
            null_firth_fallback_step_divisor: config.g_compute.null_firth_fallback_step_divisor,
            null_firth_line_search_maximum_attempts: config.g_compute.null_firth_line_search_maximum_attempts,
            null_firth_step_halving_scale: config.g_compute.null_firth_step_halving_scale,
            use_block_firth_math: config.g_compute.use_block_firth_math,
            bgen_decode_tile_variant_count: config.g_compute.bgen_decode_tile_variant_count,
            gpu_genotype_format: config.g_compute.gpu_genotype_format.clone(),
            score_dtype: config.g_compute.score_dtype.clone(),
            firth_dtype: config.g_compute.firth_dtype.clone(),
            jax_cache_dir: config.g_compute.jax_cache_dir.clone(),
            jax_matmul_precision: config.g_compute.jax_matmul_precision.clone(),
            jax_persistent_cache: config.g_compute.jax_persistent_cache,
            jax_persistent_cache_min_entry_size_bytes: config.g_compute.jax_persistent_cache_min_entry_size_bytes,
            jax_persistent_cache_min_compile_time_seconds: config
                .g_compute
                .jax_persistent_cache_min_compile_time_seconds,
            jax_xla_autotune_cache: config.g_compute.jax_xla_autotune_cache,
            jax_transfer_guard: config.g_compute.jax_transfer_guard,
        }
    }
}

#[derive(Serialize)]
#[expect(clippy::struct_excessive_bools, reason = "TOML serialization mirrors the public diagnostics schema.")]
struct DiagnosticsToml {
    telemetry: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    log_dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stage_timings_json: Option<String>,
    log_filter: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    log_file: Option<String>,
    log_stderr: bool,
    progress_interval_seconds: f32,
    progress_interval_chunks: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    profile_summary_json: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    trace_file: Option<String>,
    trace_filter: String,
    trace_event_cap: u32,
    log_queue_size: u32,
    log_lossy: bool,
    include_source_location: bool,
    include_span_events: bool,
}

impl DiagnosticsToml {
    fn from_config(config: &RegenieConfigData) -> Self {
        Self {
            telemetry: config.g_diagnostics.telemetry.clone(),
            log_dir: config.g_diagnostics.log_dir.clone(),
            stage_timings_json: config.g_diagnostics.stage_timings_json.clone(),
            log_filter: config.g_diagnostics.log_filter.clone(),
            log_file: config.g_diagnostics.log_file.clone(),
            log_stderr: config.g_diagnostics.log_stderr,
            progress_interval_seconds: config.g_diagnostics.progress_interval_seconds,
            progress_interval_chunks: config.g_diagnostics.progress_interval_chunks,
            profile_summary_json: config.g_diagnostics.profile_summary_json.clone(),
            trace_file: config.g_diagnostics.trace_file.clone(),
            trace_filter: config.g_diagnostics.trace_filter.clone(),
            trace_event_cap: config.g_diagnostics.trace_event_cap,
            log_queue_size: config.g_diagnostics.log_queue_size,
            log_lossy: config.g_diagnostics.log_lossy,
            include_source_location: config.g_diagnostics.include_source_location,
            include_span_events: config.g_diagnostics.include_span_events,
        }
    }
}

#[derive(Serialize)]
struct MetadataToml {
    #[serde(rename = "default-config-hash")]
    default_config_hash: String,
    #[serde(rename = "option-schema-version")]
    option_schema_version: i64,
}

impl MetadataToml {
    fn new() -> ConfigResult<Self> {
        Ok(Self {
            default_config_hash: load_default_config_data()?.default_config_hash.clone(),
            option_schema_version: OPTION_SCHEMA_VERSION,
        })
    }
}

fn serialize_column_options(column_names: &[String]) -> (Option<String>, Option<String>) {
    match column_names {
        [] => (None, None),
        [column_name] => (Some(column_name.clone()), None),
        _ => (None, Some(column_names.join(","))),
    }
}
