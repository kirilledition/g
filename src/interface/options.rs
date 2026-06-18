#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConfigOptionValueKind {
    Boolean,
    Float,
    Integer,
    NameList,
    Path,
    String,
    StringEnum,
}

impl ConfigOptionValueKind {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Boolean => "boolean",
            Self::Float => "float",
            Self::Integer => "integer",
            Self::NameList => "name-list",
            Self::Path => "path",
            Self::String => "string",
            Self::StringEnum => "string-enum",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConfigOptionMetadata {
    pub section: &'static str,
    pub toml_name: &'static str,
    pub accepted_toml_names: &'static [&'static str],
    pub cli_long_name: Option<&'static str>,
    pub negative_cli_long_name: Option<&'static str>,
    pub flat_python_names: &'static [&'static str],
    pub value_kind: ConfigOptionValueKind,
}

macro_rules! option_metadata {
    (
        section: $section:literal,
        toml: $toml_name:literal,
        cli: $cli_long_name:expr,
        python: [$($python_name:literal),* $(,)?],
        kind: $value_kind:ident
        $(, accepted_toml: [$($accepted_toml_name:literal),* $(,)?])?
        $(, negative_cli: $negative_cli_long_name:literal)?
        $(,)?
    ) => {
        ConfigOptionMetadata {
            section: $section,
            toml_name: $toml_name,
            accepted_toml_names: &[$($($accepted_toml_name),*)?],
            cli_long_name: $cli_long_name,
            negative_cli_long_name: option_metadata!(@negative $($negative_cli_long_name)?),
            flat_python_names: &[$($python_name),*],
            value_kind: ConfigOptionValueKind::$value_kind,
        }
    };
    (@negative $negative_cli_long_name:literal) => {
        Some($negative_cli_long_name)
    };
    (@negative) => {
        None
    };
}

const CONFIG_OPTION_METADATA: &[ConfigOptionMetadata] = &[
    option_metadata!(section: "config", toml: "config", cli: Some("config"), python: [], kind: Path),
    option_metadata!(section: "input", toml: "bgen", cli: Some("bgen"), python: ["bgen"], kind: Path),
    option_metadata!(section: "input", toml: "sample", cli: Some("sample"), python: ["sample"], kind: Path),
    option_metadata!(
        section: "input",
        toml: "pheno_file",
        cli: Some("phenoFile"),
        python: ["phenoFile"],
        kind: Path,
        accepted_toml: ["phenoFile"],
    ),
    option_metadata!(
        section: "input",
        toml: "pheno_col",
        cli: Some("phenoCol"),
        python: ["phenoCol"],
        kind: NameList,
        accepted_toml: ["phenoCol"],
    ),
    option_metadata!(
        section: "input",
        toml: "pheno_col_list",
        cli: Some("phenoColList"),
        python: ["phenoColList"],
        kind: NameList,
        accepted_toml: ["phenoColList"],
    ),
    option_metadata!(
        section: "input",
        toml: "covar_file",
        cli: Some("covarFile"),
        python: ["covarFile"],
        kind: Path,
        accepted_toml: ["covarFile"],
    ),
    option_metadata!(
        section: "input",
        toml: "covar_col",
        cli: Some("covarCol"),
        python: ["covarCol"],
        kind: NameList,
        accepted_toml: ["covarCol"],
    ),
    option_metadata!(
        section: "input",
        toml: "covar_col_list",
        cli: Some("covarColList"),
        python: ["covarColList"],
        kind: NameList,
        accepted_toml: ["covarColList"],
    ),
    option_metadata!(section: "input", toml: "pred", cli: Some("pred"), python: ["pred"], kind: Path),
    option_metadata!(section: "trait", toml: "step", cli: Some("step"), python: ["step"], kind: Integer),
    option_metadata!(section: "trait", toml: "trait_type", cli: None, python: ["trait_type"], kind: StringEnum),
    option_metadata!(
        section: "trait",
        toml: "qt",
        cli: Some("qt"),
        python: ["qt"],
        kind: Boolean,
        negative_cli: "no-qt",
    ),
    option_metadata!(
        section: "trait",
        toml: "bt",
        cli: Some("bt"),
        python: ["bt"],
        kind: Boolean,
        negative_cli: "no-bt",
    ),
    option_metadata!(section: "trait", toml: "bsize", cli: Some("bsize"), python: ["bsize"], kind: Integer),
    option_metadata!(section: "trait", toml: "threads", cli: Some("threads"), python: ["threads"], kind: Integer),
    option_metadata!(
        section: "binary",
        toml: "firth",
        cli: Some("firth"),
        python: ["firth"],
        kind: Boolean,
        negative_cli: "no-firth",
    ),
    option_metadata!(
        section: "binary",
        toml: "approx",
        cli: Some("approx"),
        python: ["approx"],
        kind: Boolean,
        negative_cli: "no-approx",
    ),
    option_metadata!(
        section: "binary",
        toml: "p_threshold",
        cli: Some("pThresh"),
        python: ["pThresh"],
        kind: Float,
        accepted_toml: ["pThresh"],
    ),
    option_metadata!(
        section: "binary",
        toml: "firth_se",
        cli: Some("firth-se"),
        python: ["firth-se", "firth_se"],
        kind: Boolean,
        accepted_toml: ["firth-se"],
        negative_cli: "no-firth-se",
    ),
    option_metadata!(section: "output", toml: "out", cli: Some("out"), python: ["out"], kind: Path),
    option_metadata!(section: "output", toml: "format", cli: Some("format"), python: ["format"], kind: StringEnum),
    option_metadata!(
        section: "output",
        toml: "output_run_directory",
        cli: Some("output_run_directory"),
        python: ["output_run_directory"],
        kind: Path,
    ),
    option_metadata!(
        section: "output",
        toml: "writer_threads",
        cli: Some("writer_threads"),
        python: ["writer_threads"],
        kind: Integer,
    ),
    option_metadata!(
        section: "output",
        toml: "writer_queue_depth",
        cli: Some("writer_queue_depth"),
        python: ["writer_queue_depth"],
        kind: Integer,
    ),
    option_metadata!(
        section: "output",
        toml: "chunks_per_arrow_file",
        cli: Some("chunks_per_arrow_file"),
        python: ["chunks_per_arrow_file"],
        kind: Integer,
    ),
    option_metadata!(
        section: "output",
        toml: "arrow_compression",
        cli: Some("arrow_compression"),
        python: ["arrow_compression"],
        kind: StringEnum,
    ),
    option_metadata!(
        section: "output",
        toml: "parquet_compression",
        cli: Some("parquet_compression"),
        python: ["parquet_compression"],
        kind: StringEnum,
    ),
    option_metadata!(
        section: "output",
        toml: "output_statistic_dtype",
        cli: Some("output_statistic_dtype"),
        python: ["output_statistic_dtype"],
        kind: StringEnum,
    ),
    option_metadata!(
        section: "output",
        toml: "resume",
        cli: Some("resume"),
        python: ["resume"],
        kind: Boolean,
        negative_cli: "no-resume",
    ),
    option_metadata!(
        section: "output",
        toml: "resume_mode",
        cli: Some("resume_mode"),
        python: ["resume_mode"],
        kind: StringEnum,
    ),
    option_metadata!(
        section: "output",
        toml: "finalize_parquet",
        cli: Some("finalize_parquet"),
        python: ["finalize_parquet"],
        kind: Boolean,
        negative_cli: "no-finalize_parquet",
    ),
    option_metadata!(section: "compute", toml: "device", cli: Some("device"), python: ["device"], kind: StringEnum),
    option_metadata!(
        section: "compute",
        toml: "staging_depth",
        cli: Some("staging_depth"),
        python: ["staging_depth"],
        kind: Integer,
    ),
    option_metadata!(
        section: "compute",
        toml: "native_callback_batch_size",
        cli: Some("native_callback_batch_size"),
        python: ["native_callback_batch_size"],
        kind: Integer,
    ),
    option_metadata!(
        section: "compute",
        toml: "result_in_flight_limit",
        cli: Some("result_in_flight_limit"),
        python: ["result_in_flight_limit"],
        kind: Integer,
    ),
    option_metadata!(
        section: "compute",
        toml: "dosage_buffer_limit",
        cli: Some("dosage_buffer_limit"),
        python: ["dosage_buffer_limit"],
        kind: Integer,
    ),
    option_metadata!(
        section: "compute",
        toml: "variant_limit",
        cli: Some("variant_limit"),
        python: ["variant_limit"],
        kind: Integer,
    ),
    option_metadata!(
        section: "compute",
        toml: "trusted_no_missing_diploid",
        cli: Some("trusted_no_missing_diploid"),
        python: ["trusted_no_missing_diploid"],
        kind: Boolean,
        negative_cli: "no-trusted_no_missing_diploid",
    ),
    option_metadata!(
        section: "compute",
        toml: "trusted_bgen_validation_mode",
        cli: Some("trusted_bgen_validation_mode"),
        python: ["trusted_bgen_validation_mode"],
        kind: StringEnum,
    ),
    option_metadata!(
        section: "compute",
        toml: "sample_key_mode",
        cli: Some("sample_key_mode"),
        python: ["sample_key_mode"],
        kind: StringEnum,
    ),
    option_metadata!(
        section: "compute",
        toml: "multi_phenotype_sample_mode",
        cli: Some("multi_phenotype_sample_mode"),
        python: ["multi_phenotype_sample_mode"],
        kind: StringEnum,
    ),
    option_metadata!(
        section: "compute",
        toml: "firth_batch_size",
        cli: Some("firth_batch_size"),
        python: ["firth_batch_size"],
        kind: Integer,
    ),
    option_metadata!(
        section: "compute",
        toml: "firth_candidate_capacity",
        cli: Some("firth_candidate_capacity"),
        python: ["firth_candidate_capacity"],
        kind: Integer,
    ),
    option_metadata!(
        section: "compute",
        toml: "binary_null_maximum_iterations",
        cli: Some("binary_null_maximum_iterations"),
        python: ["binary_null_maximum_iterations"],
        kind: Integer,
    ),
    option_metadata!(
        section: "compute",
        toml: "binary_null_coefficient_tolerance",
        cli: Some("binary_null_coefficient_tolerance"),
        python: ["binary_null_coefficient_tolerance"],
        kind: Float,
    ),
    option_metadata!(
        section: "compute",
        toml: "null_logistic_nonconvergence_policy",
        cli: Some("null_logistic_nonconvergence_policy"),
        python: ["null_logistic_nonconvergence_policy"],
        kind: StringEnum,
    ),
    option_metadata!(
        section: "compute",
        toml: "binary_minimum_probability",
        cli: Some("binary_minimum_probability"),
        python: ["binary_minimum_probability"],
        kind: Float,
    ),
    option_metadata!(
        section: "compute",
        toml: "binary_minimum_variance",
        cli: Some("binary_minimum_variance"),
        python: ["binary_minimum_variance"],
        kind: Float,
    ),
    option_metadata!(
        section: "compute",
        toml: "binary_relative_variance_tolerance",
        cli: Some("binary_relative_variance_tolerance"),
        python: ["binary_relative_variance_tolerance"],
        kind: Float,
    ),
    option_metadata!(
        section: "compute",
        toml: "linear_minimum_variance",
        cli: Some("linear_minimum_variance"),
        python: ["linear_minimum_variance"],
        kind: Float,
    ),
    option_metadata!(
        section: "compute",
        toml: "linear_relative_variance_tolerance",
        cli: Some("linear_relative_variance_tolerance"),
        python: ["linear_relative_variance_tolerance"],
        kind: Float,
    ),
    option_metadata!(
        section: "compute",
        toml: "firth_maximum_iterations",
        cli: Some("firth_maximum_iterations"),
        python: ["firth_maximum_iterations"],
        kind: Integer,
    ),
    option_metadata!(
        section: "compute",
        toml: "firth_gradient_tolerance",
        cli: Some("firth_gradient_tolerance"),
        python: ["firth_gradient_tolerance"],
        kind: Float,
    ),
    option_metadata!(
        section: "compute",
        toml: "firth_coefficient_tolerance",
        cli: Some("firth_coefficient_tolerance"),
        python: ["firth_coefficient_tolerance"],
        kind: Float,
    ),
    option_metadata!(
        section: "compute",
        toml: "firth_likelihood_tolerance",
        cli: Some("firth_likelihood_tolerance"),
        python: ["firth_likelihood_tolerance"],
        kind: Float,
    ),
    option_metadata!(
        section: "compute",
        toml: "firth_maximum_step_size",
        cli: Some("firth_maximum_step_size"),
        python: ["firth_maximum_step_size"],
        kind: Float,
    ),
    option_metadata!(
        section: "compute",
        toml: "firth_pseudo_maximum_iterations",
        cli: Some("firth_pseudo_maximum_iterations"),
        python: ["firth_pseudo_maximum_iterations"],
        kind: Integer,
    ),
    option_metadata!(
        section: "compute",
        toml: "firth_pseudo_inner_maximum_iterations",
        cli: Some("firth_pseudo_inner_maximum_iterations"),
        python: ["firth_pseudo_inner_maximum_iterations"],
        kind: Integer,
    ),
    option_metadata!(
        section: "compute",
        toml: "firth_newton_raphson_zero_start_iterations",
        cli: Some("firth_newton_raphson_zero_start_iterations"),
        python: ["firth_newton_raphson_zero_start_iterations"],
        kind: Integer,
    ),
    option_metadata!(
        section: "compute",
        toml: "firth_line_search_maximum_attempts",
        cli: Some("firth_line_search_maximum_attempts"),
        python: ["firth_line_search_maximum_attempts"],
        kind: Integer,
    ),
    option_metadata!(
        section: "compute",
        toml: "firth_step_halving_maximum_attempts",
        cli: Some("firth_step_halving_maximum_attempts"),
        python: ["firth_step_halving_maximum_attempts"],
        kind: Integer,
    ),
    option_metadata!(
        section: "compute",
        toml: "firth_initial_response_scale",
        cli: Some("firth_initial_response_scale"),
        python: ["firth_initial_response_scale"],
        kind: Float,
    ),
    option_metadata!(
        section: "compute",
        toml: "firth_sparse_carrier_dosage_threshold",
        cli: Some("firth_sparse_carrier_dosage_threshold"),
        python: ["firth_sparse_carrier_dosage_threshold"],
        kind: Float,
    ),
    option_metadata!(
        section: "compute",
        toml: "firth_step_halving_scale",
        cli: Some("firth_step_halving_scale"),
        python: ["firth_step_halving_scale"],
        kind: Float,
    ),
    option_metadata!(
        section: "compute",
        toml: "null_firth_maximum_iterations",
        cli: Some("null_firth_maximum_iterations"),
        python: ["null_firth_maximum_iterations"],
        kind: Integer,
    ),
    option_metadata!(
        section: "compute",
        toml: "null_firth_gradient_tolerance",
        cli: Some("null_firth_gradient_tolerance"),
        python: ["null_firth_gradient_tolerance"],
        kind: Float,
    ),
    option_metadata!(
        section: "compute",
        toml: "null_firth_maximum_step_size",
        cli: Some("null_firth_maximum_step_size"),
        python: ["null_firth_maximum_step_size"],
        kind: Float,
    ),
    option_metadata!(
        section: "compute",
        toml: "null_firth_fallback_iteration_multiplier",
        cli: Some("null_firth_fallback_iteration_multiplier"),
        python: ["null_firth_fallback_iteration_multiplier"],
        kind: Integer,
    ),
    option_metadata!(
        section: "compute",
        toml: "null_firth_fallback_step_divisor",
        cli: Some("null_firth_fallback_step_divisor"),
        python: ["null_firth_fallback_step_divisor"],
        kind: Float,
    ),
    option_metadata!(
        section: "compute",
        toml: "null_firth_line_search_maximum_attempts",
        cli: Some("null_firth_line_search_maximum_attempts"),
        python: ["null_firth_line_search_maximum_attempts"],
        kind: Integer,
    ),
    option_metadata!(
        section: "compute",
        toml: "null_firth_step_halving_scale",
        cli: Some("null_firth_step_halving_scale"),
        python: ["null_firth_step_halving_scale"],
        kind: Float,
    ),
    option_metadata!(
        section: "compute",
        toml: "use_block_firth_math",
        cli: Some("use_block_firth_math"),
        python: ["use_block_firth_math"],
        kind: Boolean,
        negative_cli: "no-use_block_firth_math",
    ),
    option_metadata!(
        section: "compute",
        toml: "bgen_decode_tile_variant_count",
        cli: Some("bgen_decode_tile_variant_count"),
        python: ["bgen_decode_tile_variant_count"],
        kind: Integer,
    ),
    option_metadata!(
        section: "compute",
        toml: "gpu_genotype_format",
        cli: Some("gpu_genotype_format"),
        python: ["gpu_genotype_format"],
        kind: StringEnum,
    ),
    option_metadata!(
        section: "compute",
        toml: "score_dtype",
        cli: Some("score_dtype"),
        python: ["score_dtype"],
        kind: StringEnum,
    ),
    option_metadata!(
        section: "compute",
        toml: "firth_dtype",
        cli: Some("firth_dtype"),
        python: ["firth_dtype"],
        kind: StringEnum,
    ),
    option_metadata!(
        section: "compute",
        toml: "jax_cache_dir",
        cli: Some("jax_cache_dir"),
        python: ["jax_cache_dir"],
        kind: Path,
    ),
    option_metadata!(
        section: "compute",
        toml: "jax_matmul_precision",
        cli: Some("jax_matmul_precision"),
        python: ["jax_matmul_precision"],
        kind: StringEnum,
    ),
    option_metadata!(
        section: "compute",
        toml: "jax_persistent_cache",
        cli: Some("jax_persistent_cache"),
        python: ["jax_persistent_cache"],
        kind: Boolean,
        negative_cli: "no-jax_persistent_cache",
    ),
    option_metadata!(
        section: "compute",
        toml: "jax_persistent_cache_min_entry_size_bytes",
        cli: Some("jax_persistent_cache_min_entry_size_bytes"),
        python: ["jax_persistent_cache_min_entry_size_bytes"],
        kind: Integer,
    ),
    option_metadata!(
        section: "compute",
        toml: "jax_persistent_cache_min_compile_time_seconds",
        cli: Some("jax_persistent_cache_min_compile_time_seconds"),
        python: ["jax_persistent_cache_min_compile_time_seconds"],
        kind: Integer,
    ),
    option_metadata!(
        section: "compute",
        toml: "jax_xla_autotune_cache",
        cli: Some("jax_xla_autotune_cache"),
        python: ["jax_xla_autotune_cache"],
        kind: Boolean,
        negative_cli: "no-jax_xla_autotune_cache",
    ),
    option_metadata!(
        section: "compute",
        toml: "jax_transfer_guard",
        cli: Some("jax_transfer_guard"),
        python: ["jax_transfer_guard"],
        kind: Boolean,
        negative_cli: "no-jax_transfer_guard",
    ),
    option_metadata!(
        section: "diagnostics",
        toml: "telemetry",
        cli: Some("telemetry"),
        python: ["telemetry"],
        kind: StringEnum,
    ),
    option_metadata!(section: "diagnostics", toml: "log_dir", cli: Some("log_dir"), python: ["log_dir"], kind: Path),
    option_metadata!(
        section: "diagnostics",
        toml: "stage_timings_json",
        cli: Some("stage_timings_json"),
        python: ["stage_timings_json"],
        kind: Path,
    ),
    option_metadata!(
        section: "diagnostics",
        toml: "log_filter",
        cli: Some("log_filter"),
        python: ["log_filter"],
        kind: String,
    ),
    option_metadata!(
        section: "diagnostics",
        toml: "log_file",
        cli: Some("log_file"),
        python: ["log_file"],
        kind: Path,
    ),
    option_metadata!(
        section: "diagnostics",
        toml: "log_stderr",
        cli: Some("log_stderr"),
        python: ["log_stderr"],
        kind: Boolean,
        negative_cli: "no-log_stderr",
    ),
    option_metadata!(
        section: "diagnostics",
        toml: "progress_interval_seconds",
        cli: Some("progress_interval_seconds"),
        python: ["progress_interval_seconds"],
        kind: Float,
    ),
    option_metadata!(
        section: "diagnostics",
        toml: "progress_interval_chunks",
        cli: Some("progress_interval_chunks"),
        python: ["progress_interval_chunks"],
        kind: Integer,
    ),
    option_metadata!(
        section: "diagnostics",
        toml: "profile_summary_json",
        cli: Some("profile_summary_json"),
        python: ["profile_summary_json"],
        kind: Path,
    ),
    option_metadata!(
        section: "diagnostics",
        toml: "trace_file",
        cli: Some("trace_file"),
        python: ["trace_file"],
        kind: Path,
    ),
    option_metadata!(
        section: "diagnostics",
        toml: "trace_filter",
        cli: Some("trace_filter"),
        python: ["trace_filter"],
        kind: String,
    ),
    option_metadata!(
        section: "diagnostics",
        toml: "trace_event_cap",
        cli: Some("trace_event_cap"),
        python: ["trace_event_cap"],
        kind: Integer,
    ),
    option_metadata!(
        section: "diagnostics",
        toml: "log_queue_size",
        cli: Some("log_queue_size"),
        python: ["log_queue_size"],
        kind: Integer,
    ),
    option_metadata!(
        section: "diagnostics",
        toml: "log_lossy",
        cli: Some("log_lossy"),
        python: ["log_lossy"],
        kind: Boolean,
        negative_cli: "no-log_lossy",
    ),
    option_metadata!(
        section: "diagnostics",
        toml: "include_source_location",
        cli: Some("include_source_location"),
        python: ["include_source_location"],
        kind: Boolean,
        negative_cli: "no-include_source_location",
    ),
    option_metadata!(
        section: "diagnostics",
        toml: "include_span_events",
        cli: Some("include_span_events"),
        python: ["include_span_events"],
        kind: Boolean,
        negative_cli: "no-include_span_events",
    ),
];

#[must_use]
pub fn config_option_metadata() -> &'static [ConfigOptionMetadata] {
    CONFIG_OPTION_METADATA
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::config_option_metadata;
    use crate::interface;

    #[test]
    fn option_metadata_covers_visible_regenie_cli_flags() {
        let help = interface::dispatch_cli(&["regenie".to_string(), "--help".to_string()]).stdout;
        let mut help_flags = extract_long_flags(&help);
        help_flags.remove("help");
        let metadata_flags =
            config_option_metadata().iter().filter_map(|metadata| metadata.cli_long_name).collect::<BTreeSet<_>>();

        assert_eq!(help_flags, metadata_flags);
    }

    #[test]
    fn flat_python_option_names_are_unique_and_do_not_use_legacy_g_prefixes() {
        let mut seen_names = BTreeSet::new();
        for metadata in config_option_metadata() {
            for python_name in metadata.flat_python_names {
                assert!(
                    !python_name.starts_with("g-") && !python_name.starts_with("g_"),
                    "legacy Python option alias should not be exposed: {python_name}",
                );
                assert!(seen_names.insert(*python_name), "duplicate Python option alias: {python_name}");
            }
        }
    }

    fn extract_long_flags(help: &str) -> BTreeSet<&str> {
        let mut flags = BTreeSet::new();
        for token in help.split_whitespace() {
            if let Some(stripped_token) = token.strip_prefix("--") {
                let flag_name =
                    stripped_token.trim_end_matches(',').split_once(' ').map_or(stripped_token, |(name, _)| name);
                flags.insert(flag_name.trim_end_matches(','));
            }
        }
        flags
    }
}
