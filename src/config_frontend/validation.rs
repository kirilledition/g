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
    validate_existing_input_paths(config)?;
    validate_compute_config(config)?;
    validate_binary_config(config)?;
    Ok(())
}

fn validate_existing_input_paths(config: &RegenieConfigData) -> ConfigResult<()> {
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
    if config.g_compute.gpu_genotype_format == "packed8" && config.g_compute.device != "gpu" {
        return Err(ConfigError::new("--gpu_genotype_format=packed8 requires --device=gpu."));
    }
    if config.g_compute.firth_dtype != "float64" {
        return Err(ConfigError::new("--firth_dtype currently supports float64 only."));
    }
    validate_quantitative_binary_config(config)?;
    Ok(())
}

fn validate_binary_config(config: &RegenieConfigData) -> ConfigResult<()> {
    if !(0.0..1.0).contains(&config.binary.p_threshold) {
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
