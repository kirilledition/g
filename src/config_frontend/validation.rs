use std::collections::BTreeSet;
use std::path::Path;

use super::{
    ConfigError, ConfigResult, DefaultPolicy, OptionTable, QUANTITATIVE_BINARY_ONLY_OPTION_NAMES, RegenieConfigData,
    load_packaged_config_data, option_registry,
};

pub(super) fn validate_unknown_options(normalized_options: &OptionTable) -> ConfigResult<()> {
    let registry = option_registry();
    let known_options = registry.supported_option_names();
    for option_name in normalized_options.keys() {
        if !known_options.contains(option_name) {
            return Err(ConfigError::new(format!("Unknown g regenie option: {option_name}")));
        }
    }
    Ok(())
}

pub(super) fn reject_missing_resolved_default_options(normalized_options: &OptionTable) -> ConfigResult<()> {
    if let Some(missing_option_name) = option_registry().specs.iter().find_map(|option_spec| {
        (option_spec.default_policy == DefaultPolicy::Value && !normalized_options.contains_key(option_spec.cli_name))
            .then_some(option_spec.cli_name.to_string())
    }) {
        return Err(ConfigError::new(format!(
            "Default config is missing required default option {missing_option_name:?}."
        )));
    }
    Ok(())
}

pub(super) fn reject_quantitative_binary_only_options(
    explicit_options: &BTreeSet<String>,
    trait_type: &str,
) -> ConfigResult<()> {
    if trait_type != "quantitative" {
        return Ok(());
    }
    let binary_only_option_names = QUANTITATIVE_BINARY_ONLY_OPTION_NAMES
        .iter()
        .filter(|option_name| explicit_options.contains(**option_name))
        .copied()
        .collect::<Vec<_>>();
    raise_for_quantitative_binary_only_options(&binary_only_option_names)
}

/// Validate a resolved runtime config before execution.
///
/// # Errors
///
/// Returns an error when required inputs are missing, options are inconsistent, or unsupported modes are requested.
pub fn validate_config(config: &RegenieConfigData) -> ConfigResult<()> {
    validate_trait_config(config)?;
    validate_required_input_config(config)?;
    validate_compute_config(config)?;
    validate_output_config(config)?;
    validate_binary_config(config)?;
    Ok(())
}

pub(super) fn validate_existing_input_paths(config: &RegenieConfigData) -> ConfigResult<()> {
    validate_existing_path("--bgen", config.input.bgen.as_ref())?;
    validate_existing_path("--sample", config.input.sample.as_ref())?;
    validate_existing_path("--phenoFile", config.input.pheno_file.as_ref())?;
    validate_existing_path("--covarFile", config.input.covar_file.as_ref())?;
    validate_existing_path("--pred", config.input.pred.as_ref())?;
    Ok(())
}

fn validate_existing_path(option_name: &str, path: Option<&String>) -> ConfigResult<()> {
    let Some(path) = path else {
        return Ok(());
    };
    if !Path::new(path).exists() {
        return Err(ConfigError::new(format!("{option_name} path does not exist: {path}.")));
    }
    Ok(())
}

fn validate_trait_config(config: &RegenieConfigData) -> ConfigResult<()> {
    if config.trait_config.step == 1 {
        return Err(ConfigError::new("--step 1 is recognized, but g currently supports REGENIE Step 2 only."));
    }
    if config.trait_config.step != 2 {
        return Err(ConfigError::new("g regenie requires --step 2."));
    }
    if config.trait_config.bsize <= 0 {
        return Err(ConfigError::new("--bsize must be positive."));
    }
    if config.trait_config.threads.is_some_and(|threads| threads <= 0) {
        return Err(ConfigError::new("--threads must be positive when provided."));
    }
    Ok(())
}

fn validate_required_input_config(config: &RegenieConfigData) -> ConfigResult<()> {
    if config.input.bgen.is_none() {
        return Err(ConfigError::new("Exactly one genotype source is required; currently only --bgen is supported."));
    }
    if config.input.pheno_file.is_none() {
        return Err(ConfigError::new("--phenoFile is required."));
    }
    if config.input.pheno_columns.is_empty() {
        return Err(ConfigError::new("At least one --phenoCol or --phenoColList entry is required."));
    }
    validate_unique_phenotype_names(&config.input.pheno_columns)?;
    if config.input.pred.is_none() {
        return Err(ConfigError::new("--pred is required for REGENIE Step 2."));
    }
    if config.g_output.out.is_none() {
        return Err(ConfigError::new("--out is required."));
    }
    Ok(())
}

fn validate_compute_config(config: &RegenieConfigData) -> ConfigResult<()> {
    validate_positive_integer("--g-staging-depth", config.g_compute.staging_depth)?;
    if config.g_compute.variant_limit.is_some_and(|variant_limit| variant_limit <= 0) {
        return Err(ConfigError::new("--g-variant-limit must be positive when provided."));
    }
    validate_positive_integer("--g-firth-batch-size", config.g_compute.firth_batch_size)?;
    validate_positive_integer("--g-firth-candidate-capacity", config.g_compute.firth_candidate_capacity)?;
    validate_positive_integer("--g-binary-null-maximum-iterations", config.g_compute.binary_null_maximum_iterations)?;
    validate_positive_float(
        "--g-binary-null-coefficient-tolerance",
        config.g_compute.binary_null_coefficient_tolerance,
    )?;
    validate_probability_floor("--g-binary-minimum-probability", config.g_compute.binary_minimum_probability)?;
    validate_positive_float("--g-binary-minimum-variance", config.g_compute.binary_minimum_variance)?;
    validate_positive_float(
        "--g-binary-relative-variance-tolerance",
        config.g_compute.binary_relative_variance_tolerance,
    )?;
    validate_positive_float("--g-linear-minimum-variance", config.g_compute.linear_minimum_variance)?;
    validate_positive_float(
        "--g-linear-relative-variance-tolerance",
        config.g_compute.linear_relative_variance_tolerance,
    )?;
    validate_positive_integer("--g-firth-maximum-iterations", config.g_compute.firth_maximum_iterations)?;
    validate_positive_float("--g-firth-gradient-tolerance", config.g_compute.firth_gradient_tolerance)?;
    validate_positive_float("--g-firth-coefficient-tolerance", config.g_compute.firth_coefficient_tolerance)?;
    validate_positive_float("--g-firth-likelihood-tolerance", config.g_compute.firth_likelihood_tolerance)?;
    validate_positive_float("--g-firth-maximum-step-size", config.g_compute.firth_maximum_step_size)?;
    validate_positive_integer("--g-firth-pseudo-maximum-iterations", config.g_compute.firth_pseudo_maximum_iterations)?;
    validate_positive_integer(
        "--g-firth-pseudo-inner-maximum-iterations",
        config.g_compute.firth_pseudo_inner_maximum_iterations,
    )?;
    validate_positive_integer(
        "--g-firth-newton-raphson-zero-start-iterations",
        config.g_compute.firth_newton_raphson_zero_start_iterations,
    )?;
    validate_positive_integer(
        "--g-firth-line-search-maximum-attempts",
        config.g_compute.firth_line_search_maximum_attempts,
    )?;
    validate_positive_integer(
        "--g-firth-step-halving-maximum-attempts",
        config.g_compute.firth_step_halving_maximum_attempts,
    )?;
    validate_positive_float("--g-firth-initial-response-scale", config.g_compute.firth_initial_response_scale)?;
    validate_positive_float(
        "--g-firth-sparse-carrier-dosage-threshold",
        config.g_compute.firth_sparse_carrier_dosage_threshold,
    )?;
    validate_positive_float("--g-firth-step-halving-scale", config.g_compute.firth_step_halving_scale)?;
    validate_positive_integer("--g-null-firth-maximum-iterations", config.g_compute.null_firth_maximum_iterations)?;
    validate_positive_float("--g-null-firth-gradient-tolerance", config.g_compute.null_firth_gradient_tolerance)?;
    validate_positive_float("--g-null-firth-maximum-step-size", config.g_compute.null_firth_maximum_step_size)?;
    validate_positive_integer(
        "--g-null-firth-fallback-iteration-multiplier",
        config.g_compute.null_firth_fallback_iteration_multiplier,
    )?;
    validate_positive_float("--g-null-firth-fallback-step-divisor", config.g_compute.null_firth_fallback_step_divisor)?;
    validate_positive_integer(
        "--g-null-firth-line-search-maximum-attempts",
        config.g_compute.null_firth_line_search_maximum_attempts,
    )?;
    validate_positive_float("--g-null-firth-step-halving-scale", config.g_compute.null_firth_step_halving_scale)?;
    validate_positive_integer("--g-bgen-decode-tile-variant-count", config.g_compute.bgen_decode_tile_variant_count)?;
    if config.g_compute.gpu_genotype_format == "packed8" && config.g_compute.device != "gpu" {
        return Err(ConfigError::new("--g-gpu-genotype-format=packed8 requires --g-device=gpu."));
    }
    if config.g_compute.firth_dtype != "float64" {
        return Err(ConfigError::new("--g-firth-dtype currently supports float64 only."));
    }
    validate_quantitative_binary_config(config)?;
    Ok(())
}

fn validate_output_config(config: &RegenieConfigData) -> ConfigResult<()> {
    validate_positive_integer("--g-writer-threads", config.g_output.writer_threads)?;
    validate_positive_integer("--g-writer-queue-depth", config.g_output.writer_queue_depth)?;
    validate_positive_integer("--g-output-chunks-per-arrow-file", config.g_output.chunks_per_arrow_file)?;
    validate_positive_float("--g-progress-interval-seconds", config.g_diagnostics.progress_interval_seconds)?;
    validate_positive_integer("--g-progress-interval-chunks", config.g_diagnostics.progress_interval_chunks)?;
    validate_non_negative_integer("--g-trace-event-cap", config.g_diagnostics.trace_event_cap)?;
    validate_positive_integer("--g-log-queue-size", config.g_diagnostics.log_queue_size)?;
    Ok(())
}

fn validate_binary_config(config: &RegenieConfigData) -> ConfigResult<()> {
    if !(0.0..1.0).contains(&config.binary.p_threshold) || config.binary.p_threshold == 0.0 {
        return Err(ConfigError::new("--pThresh must be in (0, 1)."));
    }
    if config.binary.firth && !config.binary.approx {
        return Err(ConfigError::new("Exact --firth is not implemented yet. Use --firth --approx."));
    }
    if config.binary.approx && !config.binary.firth {
        return Err(ConfigError::new("--approx requires --firth."));
    }
    Ok(())
}

fn validate_unique_phenotype_names(phenotype_names: &[String]) -> ConfigResult<()> {
    let mut seen_phenotype_names = BTreeSet::new();
    let mut duplicate_phenotype_names = BTreeSet::new();
    for phenotype_name in phenotype_names {
        if !seen_phenotype_names.insert(phenotype_name.clone()) {
            duplicate_phenotype_names.insert(phenotype_name.clone());
        }
    }
    if duplicate_phenotype_names.is_empty() {
        return Ok(());
    }
    let duplicate_summary = duplicate_phenotype_names.into_iter().collect::<Vec<_>>().join(", ");
    Err(ConfigError::new(format!("Duplicate phenotype names are not allowed: {duplicate_summary}.")))
}

fn validate_quantitative_binary_config(config: &RegenieConfigData) -> ConfigResult<()> {
    if config.trait_config.trait_type != "quantitative" {
        return Ok(());
    }
    let mut binary_only_option_names = Vec::new();
    if config.binary.firth || config.explicit_options.contains("firth") {
        binary_only_option_names.push("firth");
    }
    if config.binary.approx || config.explicit_options.contains("approx") {
        binary_only_option_names.push("approx");
    }
    if config.binary.firth_se || config.explicit_options.contains("firth-se") {
        binary_only_option_names.push("firth-se");
    }
    if config.binary.p_threshold.to_bits() != load_packaged_config_data()?.binary.p_threshold.to_bits()
        || config.explicit_options.contains("pThresh")
    {
        binary_only_option_names.push("pThresh");
    }
    raise_for_quantitative_binary_only_options(&binary_only_option_names)
}

fn raise_for_quantitative_binary_only_options(option_names: &[&str]) -> ConfigResult<()> {
    if option_names.is_empty() {
        return Ok(());
    }
    let formatted_option_names =
        option_names.iter().map(|option_name| format!("--{option_name}")).collect::<Vec<_>>().join(", ");
    Err(ConfigError::new(format!(
        "{formatted_option_names} can only be used with --bt; omit binary-only options when using --qt."
    )))
}

/// Validate that an integer option is positive.
///
/// # Errors
///
/// Returns an error when `value` is less than or equal to zero.
pub fn validate_positive_integer(option_name: &str, value: i64) -> ConfigResult<()> {
    if value <= 0 {
        return Err(ConfigError::new(format!("{option_name} must be positive.")));
    }
    Ok(())
}

/// Validate that an integer option is non-negative.
///
/// # Errors
///
/// Returns an error when `value` is negative.
pub fn validate_non_negative_integer(option_name: &str, value: i64) -> ConfigResult<()> {
    if value < 0 {
        return Err(ConfigError::new(format!("{option_name} must be non-negative.")));
    }
    Ok(())
}

/// Validate that a floating-point option is positive.
///
/// # Errors
///
/// Returns an error when `value` is less than or equal to zero.
pub fn validate_positive_float(option_name: &str, value: f64) -> ConfigResult<()> {
    if value <= 0.0 {
        return Err(ConfigError::new(format!("{option_name} must be positive.")));
    }
    Ok(())
}

/// Validate that a probability floor is positive and below the symmetric midpoint.
///
/// # Errors
///
/// Returns an error when `value` is not positive or is greater than or equal to 0.5.
pub fn validate_probability_floor(option_name: &str, value: f64) -> ConfigResult<()> {
    validate_positive_float(option_name, value)?;
    if value >= 0.5 {
        return Err(ConfigError::new(format!("{option_name} must be less than 0.5.")));
    }
    Ok(())
}
