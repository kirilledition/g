use std::fs;
use std::path::Path;

use super::{
    ConfigError, ConfigResult, DefaultPolicy, OPTION_SCHEMA_VERSION, OptionTable, OptionValue, RegenieConfigData,
    load_default_option_catalog_data, option_registry,
};

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
        ("g.compute".to_string(), build_g_compute_section(config)),
        ("g.output".to_string(), build_g_output_section(config)),
        ("g.diagnostics".to_string(), build_diagnostics_section(config)),
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
    output_section
}

fn build_g_compute_section(config: &RegenieConfigData) -> TomlSectionValues {
    let mut g_compute_section = vec![
        ("device".to_string(), TomlOutputValue::String(config.g_compute.device.clone())),
        ("staging-depth".to_string(), TomlOutputValue::Integer(config.g_compute.staging_depth)),
    ];
    push_optional_integer(&mut g_compute_section, "variant-limit", config.g_compute.variant_limit);
    g_compute_section.extend(build_g_compute_core_values(config));
    g_compute_section.extend(build_firth_compute_values(config));
    g_compute_section.extend(build_null_firth_compute_values(config));
    g_compute_section.extend(build_genotype_compute_values(config));
    push_optional_string(&mut g_compute_section, "jax-cache-dir", config.g_compute.jax_cache_dir.as_ref());
    push_optional_string(
        &mut g_compute_section,
        "jax-matmul-precision",
        config.g_compute.jax_matmul_precision.as_ref(),
    );
    g_compute_section.extend(build_jax_compute_values(config));
    g_compute_section
}

fn build_g_compute_core_values(config: &RegenieConfigData) -> TomlSectionValues {
    vec![
        (
            "trusted-no-missing-diploid".to_string(),
            TomlOutputValue::Boolean(config.g_compute.trusted_no_missing_diploid),
        ),
        (
            "trusted-bgen-validation-mode".to_string(),
            TomlOutputValue::String(config.g_compute.trusted_bgen_validation_mode.clone()),
        ),
        ("sample-key-mode".to_string(), TomlOutputValue::String(config.g_compute.sample_key_mode.clone())),
        (
            "multi-phenotype-sample-mode".to_string(),
            TomlOutputValue::String(config.g_compute.multi_phenotype_sample_mode.clone()),
        ),
        ("firth-batch-size".to_string(), TomlOutputValue::Integer(config.g_compute.firth_batch_size)),
        ("firth-candidate-capacity".to_string(), TomlOutputValue::Integer(config.g_compute.firth_candidate_capacity)),
        (
            "binary-null-maximum-iterations".to_string(),
            TomlOutputValue::Integer(config.g_compute.binary_null_maximum_iterations),
        ),
        (
            "binary-null-coefficient-tolerance".to_string(),
            TomlOutputValue::Float(config.g_compute.binary_null_coefficient_tolerance),
        ),
        (
            "null-logistic-nonconvergence".to_string(),
            TomlOutputValue::String(config.g_compute.null_logistic_nonconvergence_policy.clone()),
        ),
        ("binary-minimum-probability".to_string(), TomlOutputValue::Float(config.g_compute.binary_minimum_probability)),
        ("binary-minimum-variance".to_string(), TomlOutputValue::Float(config.g_compute.binary_minimum_variance)),
        (
            "binary-relative-variance-tolerance".to_string(),
            TomlOutputValue::Float(config.g_compute.binary_relative_variance_tolerance),
        ),
        ("linear-minimum-variance".to_string(), TomlOutputValue::Float(config.g_compute.linear_minimum_variance)),
        (
            "linear-relative-variance-tolerance".to_string(),
            TomlOutputValue::Float(config.g_compute.linear_relative_variance_tolerance),
        ),
    ]
}

fn build_firth_compute_values(config: &RegenieConfigData) -> TomlSectionValues {
    vec![
        ("firth-maximum-iterations".to_string(), TomlOutputValue::Integer(config.g_compute.firth_maximum_iterations)),
        ("firth-gradient-tolerance".to_string(), TomlOutputValue::Float(config.g_compute.firth_gradient_tolerance)),
        (
            "firth-coefficient-tolerance".to_string(),
            TomlOutputValue::Float(config.g_compute.firth_coefficient_tolerance),
        ),
        ("firth-likelihood-tolerance".to_string(), TomlOutputValue::Float(config.g_compute.firth_likelihood_tolerance)),
        ("firth-maximum-step-size".to_string(), TomlOutputValue::Float(config.g_compute.firth_maximum_step_size)),
        (
            "firth-pseudo-maximum-iterations".to_string(),
            TomlOutputValue::Integer(config.g_compute.firth_pseudo_maximum_iterations),
        ),
        (
            "firth-pseudo-inner-maximum-iterations".to_string(),
            TomlOutputValue::Integer(config.g_compute.firth_pseudo_inner_maximum_iterations),
        ),
        (
            "firth-newton-raphson-zero-start-iterations".to_string(),
            TomlOutputValue::Integer(config.g_compute.firth_newton_raphson_zero_start_iterations),
        ),
        (
            "firth-line-search-maximum-attempts".to_string(),
            TomlOutputValue::Integer(config.g_compute.firth_line_search_maximum_attempts),
        ),
        (
            "firth-step-halving-maximum-attempts".to_string(),
            TomlOutputValue::Integer(config.g_compute.firth_step_halving_maximum_attempts),
        ),
        (
            "firth-initial-response-scale".to_string(),
            TomlOutputValue::Float(config.g_compute.firth_initial_response_scale),
        ),
        (
            "firth-sparse-carrier-dosage-threshold".to_string(),
            TomlOutputValue::Float(config.g_compute.firth_sparse_carrier_dosage_threshold),
        ),
        ("firth-step-halving-scale".to_string(), TomlOutputValue::Float(config.g_compute.firth_step_halving_scale)),
    ]
}

fn build_null_firth_compute_values(config: &RegenieConfigData) -> TomlSectionValues {
    vec![
        (
            "null-firth-maximum-iterations".to_string(),
            TomlOutputValue::Integer(config.g_compute.null_firth_maximum_iterations),
        ),
        (
            "null-firth-gradient-tolerance".to_string(),
            TomlOutputValue::Float(config.g_compute.null_firth_gradient_tolerance),
        ),
        (
            "null-firth-maximum-step-size".to_string(),
            TomlOutputValue::Float(config.g_compute.null_firth_maximum_step_size),
        ),
        (
            "null-firth-fallback-iteration-multiplier".to_string(),
            TomlOutputValue::Integer(config.g_compute.null_firth_fallback_iteration_multiplier),
        ),
        (
            "null-firth-fallback-step-divisor".to_string(),
            TomlOutputValue::Float(config.g_compute.null_firth_fallback_step_divisor),
        ),
        (
            "null-firth-line-search-maximum-attempts".to_string(),
            TomlOutputValue::Integer(config.g_compute.null_firth_line_search_maximum_attempts),
        ),
        (
            "null-firth-step-halving-scale".to_string(),
            TomlOutputValue::Float(config.g_compute.null_firth_step_halving_scale),
        ),
    ]
}

fn build_genotype_compute_values(config: &RegenieConfigData) -> TomlSectionValues {
    vec![
        ("use-block-firth-math".to_string(), TomlOutputValue::Boolean(config.g_compute.use_block_firth_math)),
        (
            "bgen-decode-tile-variant-count".to_string(),
            TomlOutputValue::Integer(config.g_compute.bgen_decode_tile_variant_count),
        ),
        ("gpu-genotype-format".to_string(), TomlOutputValue::String(config.g_compute.gpu_genotype_format.clone())),
        ("score-dtype".to_string(), TomlOutputValue::String(config.g_compute.score_dtype.clone())),
        ("firth-dtype".to_string(), TomlOutputValue::String(config.g_compute.firth_dtype.clone())),
    ]
}

fn build_jax_compute_values(config: &RegenieConfigData) -> TomlSectionValues {
    vec![
        ("jax-persistent-cache".to_string(), TomlOutputValue::Boolean(config.g_compute.jax_persistent_cache)),
        (
            "jax-persistent-cache-min-entry-size-bytes".to_string(),
            TomlOutputValue::Integer(config.g_compute.jax_persistent_cache_min_entry_size_bytes),
        ),
        (
            "jax-persistent-cache-min-compile-time-seconds".to_string(),
            TomlOutputValue::Integer(config.g_compute.jax_persistent_cache_min_compile_time_seconds),
        ),
        ("jax-xla-autotune-cache".to_string(), TomlOutputValue::Boolean(config.g_compute.jax_xla_autotune_cache)),
        ("jax-transfer-guard".to_string(), TomlOutputValue::Boolean(config.g_compute.jax_transfer_guard)),
    ]
}

fn build_g_output_section(config: &RegenieConfigData) -> TomlSectionValues {
    let mut g_output_section = vec![("format".to_string(), TomlOutputValue::String(config.g_output.format.clone()))];
    push_optional_string(&mut g_output_section, "output-run-directory", config.g_output.output_run_directory.as_ref());
    g_output_section.extend([
        ("writer-threads".to_string(), TomlOutputValue::Integer(config.g_output.writer_threads)),
        ("writer-queue-depth".to_string(), TomlOutputValue::Integer(config.g_output.writer_queue_depth)),
        ("chunks-per-arrow-file".to_string(), TomlOutputValue::Integer(config.g_output.chunks_per_arrow_file)),
        ("arrow-compression".to_string(), TomlOutputValue::String(config.g_output.arrow_compression.clone())),
        ("parquet-compression".to_string(), TomlOutputValue::String(config.g_output.parquet_compression.clone())),
        ("resume".to_string(), TomlOutputValue::Boolean(config.g_output.resume)),
        ("resume-mode".to_string(), TomlOutputValue::String(config.g_output.resume_mode.clone())),
        ("finalize-parquet".to_string(), TomlOutputValue::Boolean(config.g_output.finalize_parquet)),
    ]);
    g_output_section
}

fn build_diagnostics_section(config: &RegenieConfigData) -> TomlSectionValues {
    let mut diagnostics_section =
        vec![("telemetry".to_string(), TomlOutputValue::String(config.g_diagnostics.telemetry.clone()))];
    push_optional_string(&mut diagnostics_section, "log-dir", config.g_diagnostics.log_dir.as_ref());
    push_optional_string(
        &mut diagnostics_section,
        "stage-timings-json",
        config.g_diagnostics.stage_timings_json.as_ref(),
    );
    diagnostics_section
        .push(("log-filter".to_string(), TomlOutputValue::String(config.g_diagnostics.log_filter.clone())));
    push_optional_string(&mut diagnostics_section, "log-file", config.g_diagnostics.log_file.as_ref());
    diagnostics_section.extend([
        ("log-stderr".to_string(), TomlOutputValue::Boolean(config.g_diagnostics.log_stderr)),
        (
            "progress-interval-seconds".to_string(),
            TomlOutputValue::Float(config.g_diagnostics.progress_interval_seconds),
        ),
        (
            "progress-interval-chunks".to_string(),
            TomlOutputValue::Integer(config.g_diagnostics.progress_interval_chunks),
        ),
    ]);
    push_optional_string(
        &mut diagnostics_section,
        "profile-summary-json",
        config.g_diagnostics.profile_summary_json.as_ref(),
    );
    push_optional_string(&mut diagnostics_section, "trace-file", config.g_diagnostics.trace_file.as_ref());
    diagnostics_section.extend([
        ("trace-filter".to_string(), TomlOutputValue::String(config.g_diagnostics.trace_filter.clone())),
        ("trace-event-cap".to_string(), TomlOutputValue::Integer(config.g_diagnostics.trace_event_cap)),
        ("log-queue-size".to_string(), TomlOutputValue::Integer(config.g_diagnostics.log_queue_size)),
        ("log-lossy".to_string(), TomlOutputValue::Boolean(config.g_diagnostics.log_lossy)),
        ("include-source-location".to_string(), TomlOutputValue::Boolean(config.g_diagnostics.include_source_location)),
        ("include-span-events".to_string(), TomlOutputValue::Boolean(config.g_diagnostics.include_span_events)),
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

/// Build a starter TOML template from defaults and required placeholders.
///
/// # Errors
///
/// Returns an error when packaged defaults or option metadata cannot be loaded.
pub fn build_template() -> ConfigResult<String> {
    let default_catalog = load_default_option_catalog_data()?;
    let mut lines = Vec::new();
    lines.extend(build_commented_section("input"));
    lines.push(String::new());
    lines.extend(build_commented_section("output"));
    lines.push(String::new());
    append_raw_template_sections(&mut lines, &default_catalog.raw_toml);
    Ok(format!("{}\n", lines.join("\n").trim_end()))
}

fn build_commented_section(section_name: &str) -> Vec<String> {
    let mut placeholder_lines = vec![format!("[{section_name}]")];
    for option_spec in option_registry().specs {
        if option_spec.section != section_name {
            continue;
        }
        if !matches!(option_spec.default_policy, DefaultPolicy::RequiredAtRuntime | DefaultPolicy::AbsentIsNone) {
            continue;
        }
        if let Some(placeholder_value) = template_placeholder_for_option(option_spec.cli_name) {
            placeholder_lines.push(format!("# {} = {placeholder_value}", format_toml_key(option_spec.config_key())));
        }
    }
    placeholder_lines
}

fn template_placeholder_for_option(option_name: &str) -> Option<String> {
    match option_name {
        "bgen" => Some(format_toml_string("data/chr22.bgen")),
        "sample" => Some(format_toml_string("data/chr22.sample")),
        "phenoFile" => Some(format_toml_string("data/pheno.tsv")),
        "phenoCol" => Some(format_toml_string("BMI")),
        "covarFile" => Some(format_toml_string("data/covar.tsv")),
        "covarColList" => Some(format_toml_string("age,sex,PC1,PC2")),
        "pred" => Some(format_toml_string("data/step1_pred.list")),
        "out" => Some(format_toml_string("results/bmi")),
        _ => None,
    }
}

fn append_raw_template_sections(lines: &mut Vec<String>, raw_default_toml: &OptionTable) {
    for (section_name, section_value) in raw_default_toml {
        let Some(section_table) = section_value.as_table() else {
            continue;
        };
        if section_name == "g" {
            for (g_section_name, g_section_value) in section_table {
                if let Some(g_section_table) = g_section_value.as_table() {
                    append_raw_template_section(lines, &format!("g.{g_section_name}"), g_section_table);
                }
            }
            continue;
        }
        if section_name == "binary" {
            append_commented_raw_template_section(lines, section_name, section_table);
            continue;
        }
        append_raw_template_section(lines, section_name, section_table);
    }
}

fn append_raw_template_section(lines: &mut Vec<String>, section_name: &str, section_values: &OptionTable) {
    lines.push(format!("[{section_name}]"));
    for (key, value) in section_values {
        if let Ok(output_value) = toml_output_value_from_option_value(value) {
            lines.push(format!("{} = {}", format_toml_key(key), format_toml_value(&output_value)));
        }
    }
    lines.push(String::new());
}

fn append_commented_raw_template_section(lines: &mut Vec<String>, section_name: &str, section_values: &OptionTable) {
    lines.push(format!("# [{section_name}]"));
    for (key, value) in section_values {
        if let Ok(output_value) = toml_output_value_from_option_value(value) {
            lines.push(format!("# {} = {}", format_toml_key(key), format_toml_value(&output_value)));
        }
    }
    lines.push(String::new());
}

fn toml_output_value_from_option_value(value: &OptionValue) -> ConfigResult<TomlOutputValue> {
    match value {
        OptionValue::String(value) => Ok(TomlOutputValue::String(value.clone())),
        OptionValue::Integer(value) => Ok(TomlOutputValue::Integer(*value)),
        OptionValue::Float(value) => Ok(TomlOutputValue::Float(*value)),
        OptionValue::Boolean(value) => Ok(TomlOutputValue::Boolean(*value)),
        OptionValue::List(values) => Ok(TomlOutputValue::String(values.join(","))),
        OptionValue::None | OptionValue::Table(_) => Err(ConfigError::new("Cannot format nested TOML value.")),
    }
}
