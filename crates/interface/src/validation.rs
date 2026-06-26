use std::collections::BTreeSet;

use super::domain::{DeviceValue, FloatingPointDtypeValue, GpuGenotypeFormatValue, RegenieTraitTypeValue};
use super::resolved::RegenieConfigData;
use super::{ConfigError, ConfigResult};

/// Validate a resolved runtime config before execution.
///
/// # Errors
///
/// Returns an error when required inputs are missing or options conflict semantically.
pub fn validate_config(config: &RegenieConfigData) -> ConfigResult<()> {
    validate_trait_config(config)?;
    validate_required_input_config(config)?;
    validate_compute_config(config)?;
    validate_binary_config(config)?;
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
    if config.g_compute.gpu_genotype_format == GpuGenotypeFormatValue::Packed8
        && config.g_compute.device != DeviceValue::Gpu
    {
        return Err(ConfigError::new("--gpu_genotype_format=packed8 requires --device=gpu."));
    }
    if config.g_compute.firth_dtype != FloatingPointDtypeValue::Float64 {
        return Err(ConfigError::new("--firth_dtype currently supports float64 only."));
    }
    validate_quantitative_binary_config(config)?;
    Ok(())
}

fn validate_binary_config(config: &RegenieConfigData) -> ConfigResult<()> {
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
    if config.trait_config.trait_type != RegenieTraitTypeValue::Quantitative {
        return Ok(());
    }
    let mut binary_only_option_names = Vec::new();
    if config.provenance.binary.firth {
        binary_only_option_names.push("firth");
    }
    if config.provenance.binary.approx {
        binary_only_option_names.push("approx");
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
