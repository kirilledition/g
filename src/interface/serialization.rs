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
    toml::to_string(&EffectiveConfigToml::new(config)?)
        .map_err(|error| ConfigError::new(format!("Failed to serialize TOML config: {error}")))
}

#[derive(Serialize)]
struct EffectiveConfigToml<'a> {
    #[serde(flatten)]
    config: &'a RegenieConfigData,
    metadata: MetadataToml<'a>,
}

impl<'a> EffectiveConfigToml<'a> {
    fn new(config: &'a RegenieConfigData) -> ConfigResult<Self> {
        Ok(Self { config, metadata: MetadataToml::new()? })
    }
}

#[derive(Serialize)]
struct MetadataToml<'a> {
    #[serde(rename = "default-config-hash")]
    default_config_hash: &'a str,
    #[serde(rename = "option-schema-version")]
    option_schema_version: i64,
}

impl MetadataToml<'_> {
    fn new() -> ConfigResult<Self> {
        Ok(Self {
            default_config_hash: &load_default_config_data()?.default_config_hash,
            option_schema_version: OPTION_SCHEMA_VERSION,
        })
    }
}
