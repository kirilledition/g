use std::fs;
use std::path::Path;

use super::{ConfigError, ConfigResult, OPTION_SCHEMA_VERSION, RegenieConfigData, load_default_option_catalog_data};

/// Write a deterministic effective TOML file for a resolved config.
///
/// # Errors
///
/// Returns an error when serialization fails or the target path cannot be written.
pub fn write_toml(config: &RegenieConfigData, path: &Path) -> ConfigResult<()> {
    fs::write(path, dumps_toml(config)?)
        .map_err(|error| ConfigError::new(format!("Failed to write TOML config {}: {error}", path.display())))
}

/// Serialize a resolved config to deterministic TOML.
///
/// # Errors
///
/// Returns an error when default metadata cannot be loaded.
pub fn dumps_toml(config: &RegenieConfigData) -> ConfigResult<String> {
    let sections = build_toml_sections(config)?;
    let mut lines = Vec::new();
    for (section_name, section_values) in sections {
        if section_values.is_empty() {
            continue;
        }
        lines.push(format!("[{section_name}]"));
        for (key, value) in section_values {
            lines.push(format!("{} = {}", format_toml_key(&key), format_toml_value(&value)));
        }
        lines.push(String::new());
    }
    Ok(format!("{}\n", lines.join("\n").trim_end()))
}

#[derive(Clone, Debug, PartialEq)]
enum TomlOutputValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
}

type TomlSectionValues = Vec<(String, TomlOutputValue)>;
type TomlSections = Vec<(String, TomlSectionValues)>;

fn build_toml_sections(config: &RegenieConfigData) -> ConfigResult<TomlSections> {
    Ok(vec![
        ("input".to_string(), build_input_section(config)),
        ("trait".to_string(), build_trait_section(config)),
        ("binary".to_string(), build_binary_section(config)),
        ("output".to_string(), build_output_section(config)),
        ("compute".to_string(), build_compute_section(config)),
        ("diagnostics".to_string(), build_diagnostics_section(config)),
        ("metadata".to_string(), build_metadata_section()?),
    ])
}

fn build_input_section(config: &RegenieConfigData) -> TomlSectionValues {
    let mut input_section = Vec::new();
    push_optional_string(&mut input_section, "bgen", config.input.bgen.as_ref());
    push_optional_string(&mut input_section, "sample", config.input.sample.as_ref());
    push_optional_string(&mut input_section, "phenoFile", config.input.pheno_file.as_ref());
    if config.input.pheno_columns.len() == 1 {
        input_section.push(("phenoCol".to_string(), TomlOutputValue::String(config.input.pheno_columns[0].clone())));
    } else if !config.input.pheno_columns.is_empty() {
        input_section.push(("phenoColList".to_string(), TomlOutputValue::String(config.input.pheno_columns.join(","))));
    }
    push_optional_string(&mut input_section, "covarFile", config.input.covar_file.as_ref());
    if config.input.covar_columns.len() == 1 {
        input_section.push(("covarCol".to_string(), TomlOutputValue::String(config.input.covar_columns[0].clone())));
    } else if !config.input.covar_columns.is_empty() {
        input_section.push(("covarColList".to_string(), TomlOutputValue::String(config.input.covar_columns.join(","))));
    }
    push_optional_string(&mut input_section, "pred", config.input.pred.as_ref());
    input_section
}

fn build_trait_section(config: &RegenieConfigData) -> TomlSectionValues {
    vec![
        ("step".to_string(), TomlOutputValue::Integer(config.trait_config.step)),
        ("qt".to_string(), TomlOutputValue::Boolean(config.trait_config.trait_type == "quantitative")),
        ("bt".to_string(), TomlOutputValue::Boolean(config.trait_config.trait_type == "binary")),
        ("bsize".to_string(), TomlOutputValue::Integer(config.trait_config.bsize)),
    ]
    .into_iter()
    .chain(config.trait_config.threads.map(|threads| ("threads".to_string(), TomlOutputValue::Integer(threads))))
    .collect()
}

fn build_binary_section(config: &RegenieConfigData) -> TomlSectionValues {
    if config.trait_config.trait_type == "binary" {
        vec![
            ("firth".to_string(), TomlOutputValue::Boolean(config.binary.firth)),
            ("approx".to_string(), TomlOutputValue::Boolean(config.binary.approx)),
            ("pThresh".to_string(), TomlOutputValue::Float(config.binary.p_threshold)),
            ("firth-se".to_string(), TomlOutputValue::Boolean(config.binary.firth_se)),
        ]
    } else {
        Vec::new()
    }
}

fn build_output_section(config: &RegenieConfigData) -> TomlSectionValues {
    let mut output_section = Vec::new();
    push_optional_string(&mut output_section, "out", config.g_output.out.as_ref());
    output_section.extend(build_output_runtime_values(config));
    output_section
}

fn build_compute_section(config: &RegenieConfigData) -> TomlSectionValues {
    let mut compute_section = vec![
        ("device".to_string(), TomlOutputValue::String(config.g_compute.device.clone())),
        ("staging_depth".to_string(), TomlOutputValue::Integer(config.g_compute.staging_depth)),
    ];
    push_optional_integer(&mut compute_section, "variant_limit", config.g_compute.variant_limit);
    compute_section.extend(build_compute_core_values(config));
    compute_section.extend(build_firth_compute_values(config));
    compute_section.extend(build_null_firth_compute_values(config));
    compute_section.extend(build_genotype_compute_values(config));
    push_optional_string(&mut compute_section, "jax_cache_dir", config.g_compute.jax_cache_dir.as_ref());
    push_optional_string(&mut compute_section, "jax_matmul_precision", config.g_compute.jax_matmul_precision.as_ref());
    compute_section.extend(build_jax_compute_values(config));
    compute_section
}

fn build_compute_core_values(config: &RegenieConfigData) -> TomlSectionValues {
    vec![
        (
            "trusted_no_missing_diploid".to_string(),
            TomlOutputValue::Boolean(config.g_compute.trusted_no_missing_diploid),
        ),
        (
            "trusted_bgen_validation_mode".to_string(),
            TomlOutputValue::String(config.g_compute.trusted_bgen_validation_mode.clone()),
        ),
        ("sample_key_mode".to_string(), TomlOutputValue::String(config.g_compute.sample_key_mode.clone())),
        (
            "multi_phenotype_sample_mode".to_string(),
            TomlOutputValue::String(config.g_compute.multi_phenotype_sample_mode.clone()),
        ),
        ("firth_batch_size".to_string(), TomlOutputValue::Integer(config.g_compute.firth_batch_size)),
        ("firth_candidate_capacity".to_string(), TomlOutputValue::Integer(config.g_compute.firth_candidate_capacity)),
        (
            "binary_null_maximum_iterations".to_string(),
            TomlOutputValue::Integer(config.g_compute.binary_null_maximum_iterations),
        ),
        (
            "binary_null_coefficient_tolerance".to_string(),
            TomlOutputValue::Float(config.g_compute.binary_null_coefficient_tolerance),
        ),
        (
            "null_logistic_nonconvergence_policy".to_string(),
            TomlOutputValue::String(config.g_compute.null_logistic_nonconvergence_policy.clone()),
        ),
        ("binary_minimum_probability".to_string(), TomlOutputValue::Float(config.g_compute.binary_minimum_probability)),
        ("binary_minimum_variance".to_string(), TomlOutputValue::Float(config.g_compute.binary_minimum_variance)),
        (
            "binary_relative_variance_tolerance".to_string(),
            TomlOutputValue::Float(config.g_compute.binary_relative_variance_tolerance),
        ),
        ("linear_minimum_variance".to_string(), TomlOutputValue::Float(config.g_compute.linear_minimum_variance)),
        (
            "linear_relative_variance_tolerance".to_string(),
            TomlOutputValue::Float(config.g_compute.linear_relative_variance_tolerance),
        ),
    ]
}

fn build_firth_compute_values(config: &RegenieConfigData) -> TomlSectionValues {
    vec![
        ("firth_maximum_iterations".to_string(), TomlOutputValue::Integer(config.g_compute.firth_maximum_iterations)),
        ("firth_gradient_tolerance".to_string(), TomlOutputValue::Float(config.g_compute.firth_gradient_tolerance)),
        (
            "firth_coefficient_tolerance".to_string(),
            TomlOutputValue::Float(config.g_compute.firth_coefficient_tolerance),
        ),
        ("firth_likelihood_tolerance".to_string(), TomlOutputValue::Float(config.g_compute.firth_likelihood_tolerance)),
        ("firth_maximum_step_size".to_string(), TomlOutputValue::Float(config.g_compute.firth_maximum_step_size)),
        (
            "firth_pseudo_maximum_iterations".to_string(),
            TomlOutputValue::Integer(config.g_compute.firth_pseudo_maximum_iterations),
        ),
        (
            "firth_pseudo_inner_maximum_iterations".to_string(),
            TomlOutputValue::Integer(config.g_compute.firth_pseudo_inner_maximum_iterations),
        ),
        (
            "firth_newton_raphson_zero_start_iterations".to_string(),
            TomlOutputValue::Integer(config.g_compute.firth_newton_raphson_zero_start_iterations),
        ),
        (
            "firth_line_search_maximum_attempts".to_string(),
            TomlOutputValue::Integer(config.g_compute.firth_line_search_maximum_attempts),
        ),
        (
            "firth_step_halving_maximum_attempts".to_string(),
            TomlOutputValue::Integer(config.g_compute.firth_step_halving_maximum_attempts),
        ),
        (
            "firth_initial_response_scale".to_string(),
            TomlOutputValue::Float(config.g_compute.firth_initial_response_scale),
        ),
        (
            "firth_sparse_carrier_dosage_threshold".to_string(),
            TomlOutputValue::Float(config.g_compute.firth_sparse_carrier_dosage_threshold),
        ),
        ("firth_step_halving_scale".to_string(), TomlOutputValue::Float(config.g_compute.firth_step_halving_scale)),
    ]
}

fn build_null_firth_compute_values(config: &RegenieConfigData) -> TomlSectionValues {
    vec![
        (
            "null_firth_maximum_iterations".to_string(),
            TomlOutputValue::Integer(config.g_compute.null_firth_maximum_iterations),
        ),
        (
            "null_firth_gradient_tolerance".to_string(),
            TomlOutputValue::Float(config.g_compute.null_firth_gradient_tolerance),
        ),
        (
            "null_firth_maximum_step_size".to_string(),
            TomlOutputValue::Float(config.g_compute.null_firth_maximum_step_size),
        ),
        (
            "null_firth_fallback_iteration_multiplier".to_string(),
            TomlOutputValue::Integer(config.g_compute.null_firth_fallback_iteration_multiplier),
        ),
        (
            "null_firth_fallback_step_divisor".to_string(),
            TomlOutputValue::Float(config.g_compute.null_firth_fallback_step_divisor),
        ),
        (
            "null_firth_line_search_maximum_attempts".to_string(),
            TomlOutputValue::Integer(config.g_compute.null_firth_line_search_maximum_attempts),
        ),
        (
            "null_firth_step_halving_scale".to_string(),
            TomlOutputValue::Float(config.g_compute.null_firth_step_halving_scale),
        ),
    ]
}

fn build_genotype_compute_values(config: &RegenieConfigData) -> TomlSectionValues {
    vec![
        ("use_block_firth_math".to_string(), TomlOutputValue::Boolean(config.g_compute.use_block_firth_math)),
        (
            "bgen_decode_tile_variant_count".to_string(),
            TomlOutputValue::Integer(config.g_compute.bgen_decode_tile_variant_count),
        ),
        ("gpu_genotype_format".to_string(), TomlOutputValue::String(config.g_compute.gpu_genotype_format.clone())),
        ("score_dtype".to_string(), TomlOutputValue::String(config.g_compute.score_dtype.clone())),
        ("firth_dtype".to_string(), TomlOutputValue::String(config.g_compute.firth_dtype.clone())),
    ]
}

fn build_jax_compute_values(config: &RegenieConfigData) -> TomlSectionValues {
    vec![
        ("jax_persistent_cache".to_string(), TomlOutputValue::Boolean(config.g_compute.jax_persistent_cache)),
        (
            "jax_persistent_cache_min_entry_size_bytes".to_string(),
            TomlOutputValue::Integer(config.g_compute.jax_persistent_cache_min_entry_size_bytes),
        ),
        (
            "jax_persistent_cache_min_compile_time_seconds".to_string(),
            TomlOutputValue::Integer(config.g_compute.jax_persistent_cache_min_compile_time_seconds),
        ),
        ("jax_xla_autotune_cache".to_string(), TomlOutputValue::Boolean(config.g_compute.jax_xla_autotune_cache)),
        ("jax_transfer_guard".to_string(), TomlOutputValue::Boolean(config.g_compute.jax_transfer_guard)),
    ]
}

fn build_output_runtime_values(config: &RegenieConfigData) -> TomlSectionValues {
    let mut output_values = vec![("format".to_string(), TomlOutputValue::String(config.g_output.format.clone()))];
    push_optional_string(&mut output_values, "output_run_directory", config.g_output.output_run_directory.as_ref());
    output_values.extend([
        ("writer_threads".to_string(), TomlOutputValue::Integer(config.g_output.writer_threads)),
        ("writer_queue_depth".to_string(), TomlOutputValue::Integer(config.g_output.writer_queue_depth)),
        ("chunks_per_arrow_file".to_string(), TomlOutputValue::Integer(config.g_output.chunks_per_arrow_file)),
        ("arrow_compression".to_string(), TomlOutputValue::String(config.g_output.arrow_compression.clone())),
        ("parquet_compression".to_string(), TomlOutputValue::String(config.g_output.parquet_compression.clone())),
        ("resume".to_string(), TomlOutputValue::Boolean(config.g_output.resume)),
        ("resume_mode".to_string(), TomlOutputValue::String(config.g_output.resume_mode.clone())),
        ("finalize_parquet".to_string(), TomlOutputValue::Boolean(config.g_output.finalize_parquet)),
    ]);
    output_values
}

fn build_diagnostics_section(config: &RegenieConfigData) -> TomlSectionValues {
    let mut diagnostics_section =
        vec![("telemetry".to_string(), TomlOutputValue::String(config.g_diagnostics.telemetry.clone()))];
    push_optional_string(&mut diagnostics_section, "log_dir", config.g_diagnostics.log_dir.as_ref());
    push_optional_string(
        &mut diagnostics_section,
        "stage_timings_json",
        config.g_diagnostics.stage_timings_json.as_ref(),
    );
    diagnostics_section
        .push(("log_filter".to_string(), TomlOutputValue::String(config.g_diagnostics.log_filter.clone())));
    push_optional_string(&mut diagnostics_section, "log_file", config.g_diagnostics.log_file.as_ref());
    diagnostics_section.extend([
        ("log_stderr".to_string(), TomlOutputValue::Boolean(config.g_diagnostics.log_stderr)),
        (
            "progress_interval_seconds".to_string(),
            TomlOutputValue::Float(config.g_diagnostics.progress_interval_seconds),
        ),
        (
            "progress_interval_chunks".to_string(),
            TomlOutputValue::Integer(config.g_diagnostics.progress_interval_chunks),
        ),
    ]);
    push_optional_string(
        &mut diagnostics_section,
        "profile_summary_json",
        config.g_diagnostics.profile_summary_json.as_ref(),
    );
    push_optional_string(&mut diagnostics_section, "trace_file", config.g_diagnostics.trace_file.as_ref());
    diagnostics_section.extend([
        ("trace_filter".to_string(), TomlOutputValue::String(config.g_diagnostics.trace_filter.clone())),
        ("trace_event_cap".to_string(), TomlOutputValue::Integer(config.g_diagnostics.trace_event_cap)),
        ("log_queue_size".to_string(), TomlOutputValue::Integer(config.g_diagnostics.log_queue_size)),
        ("log_lossy".to_string(), TomlOutputValue::Boolean(config.g_diagnostics.log_lossy)),
        ("include_source_location".to_string(), TomlOutputValue::Boolean(config.g_diagnostics.include_source_location)),
        ("include_span_events".to_string(), TomlOutputValue::Boolean(config.g_diagnostics.include_span_events)),
    ]);
    diagnostics_section
}

fn build_metadata_section() -> ConfigResult<TomlSectionValues> {
    Ok(vec![
        (
            "default-config-hash".to_string(),
            TomlOutputValue::String(load_default_option_catalog_data()?.default_config_hash.clone()),
        ),
        ("option-schema-version".to_string(), TomlOutputValue::Integer(OPTION_SCHEMA_VERSION)),
    ])
}

fn push_optional_string(section_values: &mut Vec<(String, TomlOutputValue)>, key: &str, value: Option<&String>) {
    if let Some(value) = value {
        section_values.push((key.to_string(), TomlOutputValue::String(value.clone())));
    }
}

fn push_optional_integer(section_values: &mut Vec<(String, TomlOutputValue)>, key: &str, value: Option<i64>) {
    if let Some(value) = value {
        section_values.push((key.to_string(), TomlOutputValue::Integer(value)));
    }
}

fn format_toml_key(key: &str) -> String {
    if key.contains('-') { format!("\"{key}\"") } else { key.to_string() }
}

fn format_toml_value(value: &TomlOutputValue) -> String {
    match value {
        TomlOutputValue::String(value) => format_toml_string(value),
        TomlOutputValue::Integer(value) => value.to_string(),
        TomlOutputValue::Float(value) => value.to_string(),
        TomlOutputValue::Boolean(value) => value.to_string(),
    }
}

#[must_use]
pub fn format_toml_string(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}
