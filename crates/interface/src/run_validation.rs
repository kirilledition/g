use std::path::Path;

use super::resolved::RegenieConfigData;
use super::validation::validate_config;
use super::{ConfigError, ConfigResult};

/// Validate a resolved runtime config at the execution boundary.
///
/// # Errors
///
/// Returns an error when semantic validation fails or an input path does not exist.
pub(crate) fn validate_config_for_run(config: &RegenieConfigData) -> ConfigResult<()> {
    validate_config(config)?;
    validate_existing_input_paths(config)
}

fn validate_existing_input_paths(config: &RegenieConfigData) -> ConfigResult<()> {
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
