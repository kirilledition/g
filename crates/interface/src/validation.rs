use std::collections::BTreeSet;

use g_plan as plan;

use super::resolved::RegenieConfigData;
use super::{ConfigError, ConfigResult};

/// Validate a resolved runtime config before execution.
///
/// # Errors
///
/// Returns an error when required inputs are missing or options conflict semantically.
pub fn validate_config(config: &RegenieConfigData) -> ConfigResult<()> {
    validate_required_input_config(config)?;
    validate_compute_config(config)?;
    validate_binary_config(config)?;
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
    if config.g_compute.gpu_genotype_format == plan::GpuGenotypeFormat::Packed8
        && config.g_compute.device != plan::Device::Gpu
    {
        return Err(ConfigError::new("--gpu_genotype_format=packed8 requires --device=gpu."));
    }
    if config.g_compute.score_dtype == plan::FloatingPointDtype::Float32
        && config.g_output.output_statistic_dtype == plan::FloatingPointDtype::Float64
    {
        return Err(ConfigError::new(
            "output_statistic_dtype cannot be wider than score_dtype; use float32 output or float64 score computation.",
        ));
    }
    validate_quantitative_binary_config(config)?;
    Ok(())
}

fn validate_binary_config(config: &RegenieConfigData) -> ConfigResult<()> {
    if config.binary.firth_se && config.binary.fallback_method != plan::BinaryFallbackMethod::FirthApproximate {
        return Err(ConfigError::new("--firth-se requires --binary-fallback=firth_approximate."));
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
    if config.trait_config.trait_type != plan::RegenieTraitType::Quantitative {
        return Ok(());
    }
    let mut binary_only_option_names = Vec::new();
    if config.provenance.binary.fallback_method {
        binary_only_option_names.push("binary-fallback");
    }
    if config.provenance.binary.firth_se {
        binary_only_option_names.push("firth-se");
    }
    if config.provenance.binary.p_threshold {
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
