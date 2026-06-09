use std::fs;
use std::path::Path;

use toml::Table;

use super::data::RegenieConfigData;
use super::defaults::load_default_config_data;
use super::schema::PartialConfig;
use super::validation::validate_config;
use super::{ConfigError, ConfigResult};

#[derive(Clone, Debug, Default)]
pub(crate) struct ConfigLayer {
    partial_config: PartialConfig,
}

impl ConfigLayer {
    pub(crate) fn from_partial_config(partial_config: PartialConfig) -> Self {
        Self { partial_config }
    }

    pub(crate) fn from_toml_table(toml_table: &Table, source: &str) -> ConfigResult<Self> {
        let partial_config = partial_config_from_toml_table(toml_table.clone(), source)?;
        Ok(Self { partial_config })
    }
}

pub(crate) fn partial_config_from_toml_text(toml_text: &str, source: &str) -> ConfigResult<PartialConfig> {
    toml::from_str::<PartialConfig>(toml_text)
        .map_err(|error| ConfigError::new(format!("Invalid TOML config {source}: {error}")))
}

fn partial_config_from_toml_table(toml_table: Table, source: &str) -> ConfigResult<PartialConfig> {
    toml_table
        .try_into::<PartialConfig>()
        .map_err(|error| ConfigError::new(format!("Invalid TOML config {source}: {error}")))
}

pub(crate) fn decode_toml_file_layer(path: Option<&Path>) -> ConfigResult<ConfigLayer> {
    let Some(config_path) = path else {
        return Ok(ConfigLayer::default());
    };
    let toml_text = fs::read_to_string(config_path)
        .map_err(|error| ConfigError::new(format!("Failed to read TOML config {}: {error}", config_path.display())))?;
    let partial_config = partial_config_from_toml_text(&toml_text, &config_path.display().to_string())?;
    Ok(ConfigLayer { partial_config })
}

/// Resolve a config from a TOML path.
///
/// # Errors
///
/// Returns an error when the file cannot be read, decoded, or validated.
pub fn from_toml_path(path: &Path) -> ConfigResult<RegenieConfigData> {
    resolve_config_layers([decode_toml_file_layer(Some(path))?])
}

/// Resolve a config from Python-provided options.
///
/// # Errors
///
/// Returns an error when option names, values, or the resolved runtime config are invalid.
pub fn from_options(raw_options: &Table) -> ConfigResult<RegenieConfigData> {
    resolve_config_layers([ConfigLayer::from_toml_table(raw_options, "Python options")?])
}

pub(crate) fn resolve_config_layers(
    explicit_layers: impl IntoIterator<Item = ConfigLayer>,
) -> ConfigResult<RegenieConfigData> {
    let mut merged_config = load_default_config_data()?.partial_config.clone();
    for explicit_layer in explicit_layers {
        merged_config.overlay(explicit_layer.partial_config)?;
    }
    resolve_partial_config(merged_config, true)
}

pub(crate) fn resolve_partial_config(partial_config: PartialConfig, validate: bool) -> ConfigResult<RegenieConfigData> {
    let mut config = partial_config.resolve()?;
    if validate {
        validate_config(&config)?;
        config.is_validated = true;
    }
    Ok(config)
}
