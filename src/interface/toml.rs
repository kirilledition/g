use std::fs;
use std::path::Path;

use ::toml::{Table, Value};
use serde::Serialize;

use super::defaults::load_default_config_data;
use super::overlay::{ConfigLayer, resolve_config_layers};
use super::partial::PartialConfig;
use super::resolved::{ConfigProvenance, RegenieConfigData};
use super::{ConfigError, ConfigResult, OPTION_SCHEMA_VERSION};

impl ConfigLayer {
    pub(crate) fn from_toml_table(toml_table: &Table, source: &str) -> ConfigResult<Self> {
        let partial_config = partial_config_from_toml_table(toml_table.clone(), source)?;
        config_layer_from_toml_partial_config(partial_config)
    }
}

pub(crate) fn partial_config_from_toml_text(toml_text: &str, source: &str) -> ConfigResult<PartialConfig> {
    ::toml::from_str::<PartialConfig>(toml_text)
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
    config_layer_from_toml_partial_config(partial_config)
}

fn config_layer_from_toml_partial_config(partial_config: PartialConfig) -> ConfigResult<ConfigLayer> {
    let mut provenance = ConfigProvenance::from_partial_config(&partial_config);
    if is_current_effective_toml_metadata(partial_config.metadata.as_ref())? {
        clear_effective_default_binary_provenance(&partial_config, &mut provenance)?;
    }
    Ok(ConfigLayer::from_partial_config_with_provenance(partial_config, provenance))
}

fn is_current_effective_toml_metadata(metadata: Option<&Table>) -> ConfigResult<bool> {
    let Some(metadata_table) = metadata else {
        return Ok(false);
    };
    let Some(default_config_hash) = metadata_table.get("default-config-hash").and_then(Value::as_str) else {
        return Ok(false);
    };
    let Some(option_schema_version) = metadata_table.get("option-schema-version").and_then(Value::as_integer) else {
        return Ok(false);
    };
    Ok(default_config_hash == load_default_config_data()?.default_config_hash
        && option_schema_version == OPTION_SCHEMA_VERSION)
}

fn clear_effective_default_binary_provenance(
    partial_config: &PartialConfig,
    provenance: &mut ConfigProvenance,
) -> ConfigResult<()> {
    let default_binary_config = load_default_config_data()?.partial_config.binary;
    if partial_config.binary.firth == default_binary_config.firth {
        provenance.binary.firth = false;
    }
    if partial_config.binary.approx == default_binary_config.approx {
        provenance.binary.approx = false;
    }
    if partial_config.binary.p_threshold == default_binary_config.p_threshold {
        provenance.binary.p_threshold = false;
    }
    if partial_config.binary.firth_se == default_binary_config.firth_se {
        provenance.binary.firth_se = false;
    }
    Ok(())
}

/// Resolve a config from a TOML path.
///
/// # Errors
///
/// Returns an error when the file cannot be read, decoded, or semantically validated.
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
    ::toml::to_string(&EffectiveConfigToml::new(config)?)
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
