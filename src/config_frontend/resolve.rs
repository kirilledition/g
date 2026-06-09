use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use toml::{Table, Value};

use super::data::RegenieConfigData;
use super::defaults::load_default_config_data;
use super::metadata::option_registry;
use super::schema::PartialConfig;
use super::validation::validate_config;
use super::{ConfigError, ConfigResult};

#[derive(Clone, Debug, Default)]
pub(crate) struct ConfigLayer {
    partial_config: PartialConfig,
    explicit_options: BTreeSet<String>,
}

impl ConfigLayer {
    pub(crate) fn from_toml_table(toml_table: &Table, source: &str) -> ConfigResult<Self> {
        let partial_config = partial_config_from_toml_table(toml_table.clone(), source)?;
        let explicit_options = collect_explicit_options_from_toml_table(toml_table, source)?;
        Ok(Self { partial_config, explicit_options })
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
    let toml_table = toml::from_str::<Table>(&toml_text)
        .map_err(|error| ConfigError::new(format!("Invalid TOML config {}: {error}", config_path.display())))?;
    let explicit_options = collect_explicit_options_from_toml_table(&toml_table, &config_path.display().to_string())?;
    Ok(ConfigLayer { partial_config, explicit_options })
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
    resolve_config_layers([python_options_to_config_layer(raw_options, "Python options")?])
}

pub(crate) fn resolve_config_layers(
    explicit_layers: impl IntoIterator<Item = ConfigLayer>,
) -> ConfigResult<RegenieConfigData> {
    let mut merged_config = load_default_config_data()?.partial_config.clone();
    let mut explicit_options = BTreeSet::new();
    for explicit_layer in explicit_layers {
        merged_config.overlay(explicit_layer.partial_config)?;
        explicit_options.extend(explicit_layer.explicit_options);
    }
    resolve_partial_config(merged_config, explicit_options, true)
}

pub(crate) fn resolve_partial_config(
    partial_config: PartialConfig,
    explicit_options: BTreeSet<String>,
    validate: bool,
) -> ConfigResult<RegenieConfigData> {
    let mut config = partial_config.resolve(explicit_options)?;
    if validate {
        validate_config(&config)?;
        config.is_validated = true;
    }
    Ok(config)
}

pub(crate) fn set_cli_option_value(toml_table: &mut Table, option_name: &str, option_value: Value) -> ConfigResult<()> {
    let Some(option_spec) = option_registry().get_by_cli_name(option_name) else {
        return Err(ConfigError::new(format!("Unknown g regenie option: {option_name}")));
    };
    set_toml_option_value(toml_table, option_spec.section, option_spec.cli_name, option_value)
}

fn python_options_to_config_layer(raw_options: &Table, source: &str) -> ConfigResult<ConfigLayer> {
    let mut toml_table = Table::new();
    for (option_name, option_value) in raw_options {
        if is_section_name(option_name) {
            toml_table.insert(option_name.clone(), option_value.clone());
            continue;
        }
        set_cli_option_value(&mut toml_table, option_name, option_value.clone())?;
    }
    ConfigLayer::from_toml_table(&toml_table, source)
}

fn set_toml_option_value(
    toml_table: &mut Table,
    section_name: &str,
    option_name: &str,
    option_value: Value,
) -> ConfigResult<()> {
    let section_value = toml_table.entry(section_name.to_string()).or_insert_with(|| Value::Table(Table::new()));
    let Value::Table(section_table) = section_value else {
        return Err(ConfigError::new(format!("TOML section {section_name:?} must be a table.")));
    };
    section_table.insert(option_name.to_string(), option_value);
    Ok(())
}

fn collect_explicit_options_from_toml_table(toml_table: &Table, source: &str) -> ConfigResult<BTreeSet<String>> {
    let mut explicit_options = BTreeSet::new();
    for (section_name, section_value) in toml_table {
        if section_name == "metadata" {
            continue;
        }
        let Value::Table(section_table) = section_value else {
            return Err(ConfigError::new(format!(
                "Invalid TOML config {source}: expected [{section_name}] to be a table."
            )));
        };
        for option_name in section_table.keys() {
            if let Some(option_spec) = option_registry().get_by_toml_path(section_name, option_name) {
                explicit_options.insert(option_spec.cli_name.to_string());
            } else {
                explicit_options.insert(format!("{section_name}.{option_name}"));
            }
        }
    }
    Ok(explicit_options)
}

fn is_section_name(option_name: &str) -> bool {
    matches!(option_name, "input" | "trait" | "binary" | "compute" | "output" | "diagnostics" | "metadata")
}
