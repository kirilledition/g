use std::fs;
use std::path::Path;

use serde::Serialize;
use toml::{Table, Value};

use super::defaults::load_default_config_data;
use super::options::{ConfigOptionMetadata, ConfigOptionValueKind, config_option_metadata};
use super::overlay::{ConfigLayer, resolve_config_layers};
use super::partial::PartialConfig;
use super::resolved::{ConfigProvenance, RegenieConfigData};
use super::{ConfigError, ConfigResult, OPTION_SCHEMA_VERSION};

const NATIVE_CONFIG_SECTION_NAMES: &[&str] =
    &["input", "trait", "binary", "compute", "output", "diagnostics", "metadata"];

pub(crate) fn partial_config_from_toml_text(toml_text: &str, source: &str) -> ConfigResult<PartialConfig> {
    ::toml::from_str::<PartialConfig>(toml_text)
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
    let option_table = normalize_python_option_table(raw_options)?;
    let partial_config = option_table
        .clone()
        .try_into::<PartialConfig>()
        .map_err(|error| ConfigError::new(format!("Invalid TOML config Python options: {error}")))?;
    resolve_config_layers([config_layer_from_toml_partial_config(partial_config)?])
}

fn normalize_python_option_table(raw_options: &Table) -> ConfigResult<Table> {
    let mut option_table = Table::new();
    for (option_name, option_value) in raw_options {
        normalize_python_option(&mut option_table, option_name, option_value)?;
    }
    Ok(option_table)
}

fn normalize_python_option(option_table: &mut Table, option_name: &str, option_value: &Value) -> ConfigResult<()> {
    let Some(option_metadata) = metadata_for_flat_python_name(option_name) else {
        normalize_native_or_unknown_option(option_table, option_name, option_value)?;
        return Ok(());
    };
    let normalized_value = normalize_python_option_value(option_metadata.value_kind, option_value)?;
    let section_value =
        option_table.entry(option_metadata.section.to_string()).or_insert_with(|| Value::Table(Table::new()));
    let Value::Table(section_table) = section_value else {
        option_table.insert(option_name.to_string(), normalized_value);
        return Ok(());
    };
    section_table.insert(option_metadata.toml_name.to_string(), normalized_value);
    Ok(())
}

fn normalize_native_or_unknown_option(
    option_table: &mut Table,
    option_name: &str,
    option_value: &Value,
) -> ConfigResult<()> {
    if NATIVE_CONFIG_SECTION_NAMES.contains(&option_name) {
        if let Value::Table(section_updates) = option_value {
            match option_table.get_mut(option_name) {
                Some(Value::Table(section_table)) => {
                    section_table.extend(section_updates.clone());
                }
                Some(section_value) => {
                    *section_value = Value::Table(section_updates.clone());
                }
                None => {
                    option_table.insert(option_name.to_string(), Value::Table(section_updates.clone()));
                }
            }
        } else {
            option_table.insert(option_name.to_string(), option_value.clone());
        }
        return Ok(());
    }
    if matches!(option_value, Value::Table(_)) {
        return Err(ConfigError::new(format!(
            "Unknown g regenie option: {}",
            flatten_unknown_option_name(option_name, option_value)
        )));
    }
    Err(ConfigError::new(format!("Unknown g regenie option: {option_name}")))
}

fn metadata_for_flat_python_name(option_name: &str) -> Option<&'static ConfigOptionMetadata> {
    config_option_metadata().iter().find(|metadata| metadata.flat_python_names.contains(&option_name))
}

fn normalize_python_option_value(value_kind: ConfigOptionValueKind, option_value: &Value) -> ConfigResult<Value> {
    if value_kind != ConfigOptionValueKind::Boolean {
        return Ok(option_value.clone());
    }
    if let Value::Boolean(boolean_value) = option_value {
        return Ok(Value::Boolean(*boolean_value));
    }
    if let Value::String(option_text) = option_value {
        let normalized_value = option_text.trim().to_lowercase();
        if matches!(normalized_value.as_str(), "1" | "true" | "yes" | "on") {
            return Ok(Value::Boolean(true));
        }
        if matches!(normalized_value.as_str(), "0" | "false" | "no" | "off") {
            return Ok(Value::Boolean(false));
        }
    }
    Err(ConfigError::new("Boolean option value must be a bool or one of true/false/on/off/yes/no/1/0."))
}

fn flatten_unknown_option_name(option_name: &str, option_value: &Value) -> String {
    let Value::Table(option_table) = option_value else {
        return option_name.to_string();
    };
    let Some((nested_key, nested_value)) = option_table.iter().next() else {
        return option_name.to_string();
    };
    if matches!(nested_value, Value::Table(_)) {
        return format!("{option_name}.{}", flatten_unknown_option_name(nested_key, nested_value));
    }
    format!("{option_name}.{nested_key}")
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
