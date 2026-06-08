#![allow(
    clippy::missing_errors_doc,
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::similar_names,
    clippy::struct_excessive_bools,
    clippy::too_many_lines
)]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Write as _};
use std::fs;
use std::mem::MaybeUninit;
use std::path::Path;
use std::sync::OnceLock;

use sha2::{Digest, Sha256};
use toml_spanner::{Arena, Array, DateTime, Item, Table, Value};

mod cli;
mod metadata;
mod render;
mod validation;

pub use cli::{CliOutcomeData, dispatch_cli, explain_option, iter_explanations};
pub use metadata::{DefaultPolicy, OptionSpec, SupportLevel};
pub use render::{build_template, dumps_toml, format_toml_string, write_toml};
pub use validation::{
    validate_config, validate_non_negative_integer, validate_positive_float, validate_positive_integer,
    validate_probability_floor,
};

use metadata::{OptionValueType, option_registry};
use validation::{
    reject_missing_resolved_default_options, reject_quantitative_binary_only_options, reject_unsupported_options,
    validate_unknown_options,
};

const DEFAULT_CONFIG_TOML: &str = include_str!("../g/config.default.toml");
const OPTION_METADATA_JSON: &str = include_str!("../config_options.json");
const OPTION_SCHEMA_VERSION: i64 = 1;
const QUANTITATIVE_BINARY_ONLY_OPTION_NAMES: &[&str] = &["firth", "approx", "firth-se", "spa", "pThresh"];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigError {
    message: String,
}

impl ConfigError {
    fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ConfigError {}

type ConfigResult<T> = Result<T, ConfigError>;

#[derive(Clone, Debug, PartialEq)]
pub enum OptionValue {
    None,
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    List(Vec<String>),
    Table(BTreeMap<String, OptionValue>),
}

impl OptionValue {
    fn is_explicit_some(&self) -> bool {
        !matches!(self, Self::None)
    }

    fn as_table(&self) -> Option<&BTreeMap<String, OptionValue>> {
        match self {
            Self::Table(table) => Some(table),
            _ => None,
        }
    }
}

pub type OptionTable = BTreeMap<String, OptionValue>;

#[derive(Clone, Debug, PartialEq)]
pub struct TomlConfigLayer {
    toml_config: OptionTable,
    explicit_options: BTreeSet<String>,
}

impl TomlConfigLayer {
    pub fn toml_config(&self) -> &OptionTable {
        &self.toml_config
    }

    pub fn explicit_options(&self) -> &BTreeSet<String> {
        &self.explicit_options
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DefaultOptionCatalogData {
    pub raw_toml: OptionTable,
    pub normalized_options: OptionTable,
    pub default_config_hash: String,
}

static DEFAULT_CATALOG: OnceLock<Result<DefaultOptionCatalogData, ConfigError>> = OnceLock::new();

pub fn load_default_option_catalog_data() -> ConfigResult<&'static DefaultOptionCatalogData> {
    DEFAULT_CATALOG
        .get_or_init(|| {
            let toml_config = parse_toml_document(DEFAULT_CONFIG_TOML, "config.default.toml")?;
            validate_toml_schema(&toml_config, "config.default.toml")?;
            let normalized_options = flatten_toml_mapping(&toml_config)?;
            validate_default_catalog(&normalized_options)?;
            let default_config_hash = build_default_config_hash(&toml_config)?;
            Ok(DefaultOptionCatalogData { raw_toml: toml_config, normalized_options, default_config_hash })
        })
        .as_ref()
        .map_err(Clone::clone)
}

fn validate_default_catalog(normalized_options: &OptionTable) -> ConfigResult<()> {
    let registry = option_registry()?;
    let unknown_option_names = normalized_options
        .keys()
        .filter(|option_name| registry.get_by_name(option_name).is_none())
        .cloned()
        .collect::<Vec<_>>();
    if !unknown_option_names.is_empty() {
        return Err(ConfigError::new(format!(
            "Default config contains unknown option(s): {}.",
            unknown_option_names.join(", ")
        )));
    }

    let missing_default_names = registry
        .specs
        .iter()
        .filter(|option_spec| {
            option_spec.default_policy == DefaultPolicy::Value && !normalized_options.contains_key(&option_spec.name)
        })
        .map(|option_spec| option_spec.name.clone())
        .collect::<Vec<_>>();
    if !missing_default_names.is_empty() {
        return Err(ConfigError::new(format!(
            "Default config is missing required default option(s): {}.",
            missing_default_names.join(", ")
        )));
    }

    let invalid_default_names = registry
        .specs
        .iter()
        .filter(|option_spec| {
            matches!(
                option_spec.default_policy,
                DefaultPolicy::RequiredAtRuntime | DefaultPolicy::Unsupported | DefaultPolicy::Derived
            ) && normalized_options.contains_key(&option_spec.name)
        })
        .map(|option_spec| option_spec.name.clone())
        .collect::<Vec<_>>();
    if !invalid_default_names.is_empty() {
        return Err(ConfigError::new(format!(
            "Default config contains non-defaultable option(s): {}.",
            invalid_default_names.join(", ")
        )));
    }

    Ok(())
}

fn build_default_config_hash(raw_toml: &OptionTable) -> ConfigResult<String> {
    let normalized_payload = json_value_from_option_table(raw_toml);
    let encoded_payload = serde_json::to_string(&normalized_payload)
        .map_err(|error| ConfigError::new(format!("Failed to encode default config hash payload: {error}")))?;
    let digest = Sha256::digest(encoded_payload.as_bytes());
    let mut encoded_digest = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(&mut encoded_digest, "{byte:02x}").expect("writing SHA-256 digest to a string");
    }
    Ok(encoded_digest)
}

fn json_value_from_option_value(value: &OptionValue) -> serde_json::Value {
    match value {
        OptionValue::None => serde_json::Value::Null,
        OptionValue::String(value) => serde_json::Value::String(value.clone()),
        OptionValue::Integer(value) => serde_json::Value::Number((*value).into()),
        OptionValue::Float(value) => {
            serde_json::Number::from_f64(*value).map_or(serde_json::Value::Null, serde_json::Value::Number)
        }
        OptionValue::Boolean(value) => serde_json::Value::Bool(*value),
        OptionValue::List(values) => {
            serde_json::Value::Array(values.iter().map(|value| serde_json::Value::String(value.clone())).collect())
        }
        OptionValue::Table(values) => json_value_from_option_table(values),
    }
}

fn json_value_from_option_table(table: &OptionTable) -> serde_json::Value {
    serde_json::Value::Object(
        table.iter().map(|(key, value)| (key.clone(), json_value_from_option_value(value))).collect(),
    )
}

fn parse_toml_document(toml_text: &str, source: &str) -> ConfigResult<OptionTable> {
    let arena = Arena::new();
    let parsed_document = toml_spanner::parse(toml_text, &arena)
        .map_err(|error| ConfigError::new(format!("Invalid TOML config {source}: {error}")))?;
    Ok(table_from_toml_spanner_table(parsed_document.table()))
}

fn option_value_from_toml_item(item: &Item<'_>) -> OptionValue {
    match item.value() {
        Value::String(value) => OptionValue::String(value.to_string()),
        Value::Integer(value) => match value.as_i64() {
            Some(value) => OptionValue::Integer(value),
            None => OptionValue::String(value.as_i128().to_string()),
        },
        Value::Float(value) => OptionValue::Float(*value),
        Value::Boolean(value) => OptionValue::Boolean(*value),
        Value::Array(array) => option_value_from_toml_array(array),
        Value::Table(table) => OptionValue::Table(table_from_toml_spanner_table(table)),
        Value::DateTime(value) => OptionValue::String(date_time_string(value)),
    }
}

fn option_value_from_toml_array(array: &Array<'_>) -> OptionValue {
    if array.iter().all(|item| matches!(item.value(), Value::Table(_))) {
        return OptionValue::List(
            array.iter().map(|item| format_option_value(&option_value_from_toml_item(item))).collect(),
        );
    }
    OptionValue::List(array.iter().map(string_from_toml_item).collect())
}

fn string_from_toml_item(item: &Item<'_>) -> String {
    match item.value() {
        Value::String(value) => value.to_string(),
        Value::Integer(value) => value.as_i128().to_string(),
        Value::Float(value) => value.to_string(),
        Value::Boolean(value) => value.to_string(),
        Value::Array(_) | Value::Table(_) => format_option_value(&option_value_from_toml_item(item)),
        Value::DateTime(value) => date_time_string(value),
    }
}

fn date_time_string(value: &DateTime) -> String {
    let mut buffer = MaybeUninit::uninit();
    value.format(&mut buffer).to_string()
}

fn table_from_toml_spanner_table(table: &Table<'_>) -> OptionTable {
    table.iter().map(|(key, item)| (key.name.to_string(), option_value_from_toml_item(item))).collect()
}

fn validate_toml_schema(table: &OptionTable, source: &str) -> ConfigResult<()> {
    let registry = option_registry()?;
    for (section_name, section_value) in table {
        if section_name == "metadata" {
            continue;
        }
        if section_name == "g" {
            validate_g_toml_schema(section_value, source)?;
            continue;
        }
        let Some(section_table) = section_value.as_table() else {
            return Err(ConfigError::new(format!(
                "Invalid TOML config {source}: expected [{section_name}] to be a table."
            )));
        };
        for (toml_key, option_value) in section_table {
            let Some(option_spec) = registry.get_by_toml_path(section_name, toml_key) else {
                return Err(ConfigError::new(format!("Invalid TOML config {source}: unknown field `{toml_key}`.")));
            };
            validate_toml_option_value(option_spec, option_value, source)?;
        }
    }
    Ok(())
}

fn validate_g_toml_schema(g_value: &OptionValue, source: &str) -> ConfigResult<()> {
    let registry = option_registry()?;
    let Some(g_table) = g_value.as_table() else {
        return Err(ConfigError::new(format!("Invalid TOML config {source}: expected [g] to be a table.")));
    };
    for (g_section_name, g_section_value) in g_table {
        let full_section_name = format!("g.{g_section_name}");
        let Some(g_section_table) = g_section_value.as_table() else {
            return Err(ConfigError::new(format!(
                "Invalid TOML config {source}: expected [{full_section_name}] to be a table."
            )));
        };
        for (toml_key, option_value) in g_section_table {
            let Some(option_spec) = registry.get_by_toml_path(&full_section_name, toml_key) else {
                return Err(ConfigError::new(format!("Invalid TOML config {source}: unknown field `{toml_key}`.")));
            };
            validate_toml_option_value(option_spec, option_value, source)?;
        }
    }
    Ok(())
}

fn validate_toml_option_value(option_spec: &OptionSpec, option_value: &OptionValue, source: &str) -> ConfigResult<()> {
    let value_is_valid = match option_spec.value_type {
        OptionValueType::String | OptionValueType::Path => {
            matches!(option_value, OptionValue::String(_))
                || (option_spec.multiple && matches!(option_value, OptionValue::List(_)))
        }
        OptionValueType::Integer => matches!(option_value, OptionValue::Integer(_)),
        OptionValueType::Float => matches!(option_value, OptionValue::Integer(_) | OptionValue::Float(_)),
        OptionValueType::Boolean => matches!(option_value, OptionValue::Boolean(_)),
    };
    if !value_is_valid {
        let expected_type = match option_spec.value_type {
            OptionValueType::String | OptionValueType::Path => "string",
            OptionValueType::Integer => "int",
            OptionValueType::Float => "float",
            OptionValueType::Boolean => "bool",
        };
        return Err(ConfigError::new(format!(
            "Invalid TOML config {source}: Expected `{expected_type}` for {}.",
            option_spec.name
        )));
    }
    if !option_spec.accepted_values.is_empty() {
        let value = string_value(option_value, &option_spec.name)?;
        if !option_spec.accepted_values.iter().any(|accepted_value| accepted_value == &value) {
            return Err(ConfigError::new(format!(
                "Invalid TOML config {source}: invalid value {value:?} for --{}.",
                option_spec.name
            )));
        }
    }
    Ok(())
}

pub fn decode_toml_file_layer(path: Option<&Path>) -> ConfigResult<TomlConfigLayer> {
    let Some(config_path) = path else {
        return Ok(TomlConfigLayer { toml_config: OptionTable::new(), explicit_options: BTreeSet::new() });
    };
    let toml_text = fs::read_to_string(config_path)
        .map_err(|error| ConfigError::new(format!("Failed to read TOML config {}: {error}", config_path.display())))?;
    let toml_config = decode_toml_text(&toml_text, &config_path.display().to_string())?;
    let explicit_options = flatten_toml_mapping(&toml_config)?.into_keys().collect();
    Ok(TomlConfigLayer { toml_config, explicit_options })
}

pub fn decode_toml_text(toml_text: &str, source: &str) -> ConfigResult<OptionTable> {
    let toml_config = parse_toml_document(toml_text, source)?;
    validate_toml_schema(&toml_config, source)?;
    Ok(toml_config)
}

pub fn flatten_toml_mapping(raw_options: &OptionTable) -> ConfigResult<OptionTable> {
    let mut flattened_options = OptionTable::new();
    for (section_name, section_value) in raw_options {
        if section_name == "metadata" {
            continue;
        }
        if let Some(section_table) = section_value.as_table() {
            if section_name == "g" {
                flattened_options.extend(flatten_g_toml_section(section_table)?);
            } else {
                flattened_options.extend(flatten_toml_section(section_name, section_table)?);
            }
        } else {
            flattened_options.insert(section_name.clone(), section_value.clone());
        }
    }
    Ok(flattened_options)
}

fn flatten_g_toml_section(raw_g_options: &OptionTable) -> ConfigResult<OptionTable> {
    let mut flattened_options = OptionTable::new();
    for (section_name, section_value) in raw_g_options {
        if let Some(section_table) = section_value.as_table() {
            flattened_options.extend(flatten_toml_section(&format!("g.{section_name}"), section_table)?);
        } else {
            flattened_options.insert(format!("g.{section_name}"), section_value.clone());
        }
    }
    Ok(flattened_options)
}

fn flatten_toml_section(section_name: &str, section_options: &OptionTable) -> ConfigResult<OptionTable> {
    let registry = option_registry()?;
    let mut flattened_options = OptionTable::new();
    for (toml_key, option_value) in section_options {
        if let Some(option_spec) = registry.get_by_toml_path(section_name, toml_key) {
            flattened_options.insert(option_spec.name.clone(), option_value.clone());
        } else {
            flattened_options.insert(format!("{section_name}.{toml_key}"), option_value.clone());
        }
    }
    Ok(flattened_options)
}

pub fn normalize_option_name(option_name: &str) -> ConfigResult<String> {
    let registry = option_registry()?;
    if option_name == "trait_type" || registry.get_by_name(option_name).is_some() {
        return Ok(option_name.to_string());
    }
    if let Some(option_spec) = registry.get_by_destination(option_name) {
        return Ok(option_spec.name.clone());
    }
    if let Some(option_spec) = registry.get_by_python_alias(option_name) {
        return Ok(option_spec.name.clone());
    }
    if option_name.starts_with("g_") {
        return Ok(option_name.replace('_', "-"));
    }
    Ok(option_name.to_string())
}

pub fn option_dictionary_to_toml_config_layer(
    raw_options: &OptionTable,
    source: &str,
) -> ConfigResult<TomlConfigLayer> {
    let normalized_options = normalize_option_dictionary(raw_options)?;
    let registry = option_registry()?;
    let mut toml_mapping = OptionTable::new();

    for (option_name, option_value) in &normalized_options {
        if !option_value.is_explicit_some() || option_name == "trait_type" {
            continue;
        }
        let Some(option_spec) = registry.get_by_name(option_name) else {
            return Err(ConfigError::new(format!("Unknown g regenie option: {option_name}")));
        };
        let coerced_value = coerce_option_value(option_value, option_spec)?;
        set_toml_option_value(&mut toml_mapping, option_spec, coerced_value);
    }

    apply_trait_type_alias(&mut toml_mapping, normalized_options.get("trait_type"))?;
    validate_toml_schema(&toml_mapping, source)?;
    let explicit_options = normalized_options
        .iter()
        .filter(|(_, option_value)| option_value.is_explicit_some())
        .map(|(option_name, _)| option_name.clone())
        .collect();
    Ok(TomlConfigLayer { toml_config: toml_mapping, explicit_options })
}

pub fn normalize_option_dictionary(raw_options: &OptionTable) -> ConfigResult<OptionTable> {
    flatten_toml_mapping(raw_options)?
        .into_iter()
        .map(|(option_name, option_value)| normalize_option_name(&option_name).map(|name| (name, option_value)))
        .collect()
}

fn coerce_option_value(option_value: &OptionValue, option_spec: &OptionSpec) -> ConfigResult<OptionValue> {
    if option_spec.multiple {
        return Ok(coerce_string_list_value(option_value));
    }
    if !option_spec.accepted_values.is_empty() {
        let value = string_value(option_value, &option_spec.name)?;
        if !option_spec.accepted_values.iter().any(|accepted_value| accepted_value == &value) {
            return Err(ConfigError::new(format!("Invalid value for --{}: {value:?}.", option_spec.name)));
        }
        return Ok(OptionValue::String(value));
    }
    match option_spec.value_type {
        OptionValueType::String | OptionValueType::Path => {
            Ok(OptionValue::String(string_value(option_value, &option_spec.name)?))
        }
        OptionValueType::Integer => Ok(OptionValue::Integer(integer_value(option_value, &option_spec.name)?)),
        OptionValueType::Float => Ok(OptionValue::Float(float_value(option_value, &option_spec.name)?)),
        OptionValueType::Boolean => match boolean_value(option_value, &option_spec.name) {
            Ok(value) => Ok(OptionValue::Boolean(value)),
            Err(error) if option_spec.support_level == SupportLevel::RecognizedUnsupported => {
                let _ = error;
                Ok(OptionValue::Boolean(true))
            }
            Err(error) => Err(error),
        },
    }
}

fn coerce_string_list_value(option_value: &OptionValue) -> OptionValue {
    match option_value {
        OptionValue::List(values) => OptionValue::List(values.clone()),
        OptionValue::String(value) => OptionValue::String(value.clone()),
        OptionValue::Integer(value) => OptionValue::String(value.to_string()),
        OptionValue::Float(value) => OptionValue::String(value.to_string()),
        OptionValue::Boolean(value) => OptionValue::String(value.to_string()),
        OptionValue::None => OptionValue::None,
        OptionValue::Table(_) => OptionValue::String(format_option_value(option_value)),
    }
}

fn set_toml_option_value(toml_mapping: &mut OptionTable, option_spec: &OptionSpec, option_value: OptionValue) {
    if let Some((namespace_name, section_name)) = option_spec.section.split_once('.') {
        let namespace_entry =
            toml_mapping.entry(namespace_name.to_string()).or_insert_with(|| OptionValue::Table(OptionTable::new()));
        let OptionValue::Table(namespace_table) = namespace_entry else {
            return;
        };
        let section_entry =
            namespace_table.entry(section_name.to_string()).or_insert_with(|| OptionValue::Table(OptionTable::new()));
        if let OptionValue::Table(section_table) = section_entry {
            section_table.insert(option_spec.toml_key.clone(), option_value);
        }
        return;
    }

    let section_entry =
        toml_mapping.entry(option_spec.section.clone()).or_insert_with(|| OptionValue::Table(OptionTable::new()));
    if let OptionValue::Table(section_table) = section_entry {
        section_table.insert(option_spec.toml_key.clone(), option_value);
    }
}

fn apply_trait_type_alias(toml_mapping: &mut OptionTable, raw_trait_type: Option<&OptionValue>) -> ConfigResult<()> {
    let Some(raw_value) = raw_trait_type else {
        return Ok(());
    };
    if !raw_value.is_explicit_some() {
        return Ok(());
    }
    let trait_type = string_value(raw_value, "trait_type")?;
    let trait_entry = toml_mapping.entry("trait".to_string()).or_insert_with(|| OptionValue::Table(OptionTable::new()));
    let OptionValue::Table(trait_table) = trait_entry else {
        return Ok(());
    };
    match trait_type.as_str() {
        "quantitative" => {
            trait_table.insert("qt".to_string(), OptionValue::Boolean(true));
            trait_table.insert("bt".to_string(), OptionValue::Boolean(false));
        }
        "binary" => {
            trait_table.insert("qt".to_string(), OptionValue::Boolean(false));
            trait_table.insert("bt".to_string(), OptionValue::Boolean(true));
        }
        _ => {
            return Err(ConfigError::new(format!(
                "Invalid trait_type value {trait_type:?}; expected 'quantitative' or 'binary'."
            )));
        }
    }
    Ok(())
}

fn overlay_toml_configs(base_config: &OptionTable, override_config: &OptionTable) -> OptionTable {
    let mut merged_config = base_config.clone();
    for (key, override_value) in override_config {
        match (merged_config.get_mut(key), override_value) {
            (Some(OptionValue::Table(base_table)), OptionValue::Table(override_table)) => {
                *base_table = overlay_toml_configs(base_table, override_table);
            }
            (_, OptionValue::None) => {}
            _ => {
                merged_config.insert(key.clone(), override_value.clone());
            }
        }
    }
    merged_config
}

fn from_toml_config_layers(
    base_config: &OptionTable,
    explicit_layers: impl IntoIterator<Item = TomlConfigLayer>,
) -> ConfigResult<RegenieConfigData> {
    let mut merged_toml_config = base_config.clone();
    let mut explicit_option_names = BTreeSet::new();
    for explicit_layer in explicit_layers {
        reject_layer_trait_flag_conflict(&explicit_layer.toml_config)?;
        merged_toml_config = overlay_toml_configs(&merged_toml_config, &explicit_layer.toml_config);
        apply_trait_flag_layer_precedence(&mut merged_toml_config, &explicit_layer.toml_config);
        explicit_option_names.extend(explicit_layer.explicit_options);
    }
    from_toml_config(&merged_toml_config, &explicit_option_names)
}

fn reject_layer_trait_flag_conflict(toml_config: &OptionTable) -> ConfigResult<()> {
    if get_bool(toml_config, "trait", "qt")? == Some(true) && get_bool(toml_config, "trait", "bt")? == Some(true) {
        return Err(ConfigError::new("--qt and --bt are mutually exclusive."));
    }
    Ok(())
}

fn apply_trait_flag_layer_precedence(merged_config: &mut OptionTable, override_config: &OptionTable) {
    let override_qt = get_bool(override_config, "trait", "qt").ok().flatten();
    let override_bt = get_bool(override_config, "trait", "bt").ok().flatten();
    if override_qt == Some(true) {
        set_nested_value(merged_config, "trait", "bt", OptionValue::Boolean(false));
    }
    if override_bt == Some(true) {
        set_nested_value(merged_config, "trait", "qt", OptionValue::Boolean(false));
    }
}

fn set_nested_value(toml_config: &mut OptionTable, section_name: &str, key: &str, value: OptionValue) {
    let section_entry =
        toml_config.entry(section_name.to_string()).or_insert_with(|| OptionValue::Table(OptionTable::new()));
    if let OptionValue::Table(section_table) = section_entry {
        section_table.insert(key.to_string(), value);
    }
}

pub fn load_packaged_config_data() -> ConfigResult<RegenieConfigData> {
    let explicit_options = BTreeSet::new();
    build_runtime_config_from_toml_config(&load_default_option_catalog_data()?.raw_toml, &explicit_options)
}

pub fn from_options(raw_options: &OptionTable) -> ConfigResult<RegenieConfigData> {
    let explicit_layer = option_dictionary_to_toml_config_layer(raw_options, "Python options")?;
    from_toml_config_layers(&load_default_option_catalog_data()?.raw_toml, [explicit_layer])
}

pub fn from_toml_path(path: &Path) -> ConfigResult<RegenieConfigData> {
    let toml_layer = decode_toml_file_layer(Some(path))?;
    from_toml_config_layers(&load_default_option_catalog_data()?.raw_toml, [toml_layer])
}

fn from_toml_config(toml_config: &OptionTable, explicit_options: &BTreeSet<String>) -> ConfigResult<RegenieConfigData> {
    let config = build_runtime_config_from_toml_config(toml_config, explicit_options)?;
    let normalized_options = flatten_toml_mapping(toml_config)?;
    reject_quantitative_binary_only_options(explicit_options, &config.trait_config.trait_type)?;
    reject_unsupported_options(&normalized_options)?;
    validate_unknown_options(&normalized_options)?;
    reject_missing_resolved_default_options(&normalized_options)?;
    validate_config(&config)?;
    Ok(config)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputConfigData {
    pub bgen: Option<String>,
    pub sample: Option<String>,
    pub pheno_file: Option<String>,
    pub pheno_columns: Vec<String>,
    pub covar_file: Option<String>,
    pub covar_columns: Vec<String>,
    pub pred: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraitConfigData {
    pub step: i64,
    pub trait_type: String,
    pub bsize: i64,
    pub threads: Option<i64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BinaryConfigData {
    pub firth: bool,
    pub approx: bool,
    pub spa: bool,
    pub p_threshold: f64,
    pub firth_se: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GComputeConfigData {
    pub device: String,
    pub staging_depth: i64,
    pub variant_limit: Option<i64>,
    pub trusted_no_missing_diploid: bool,
    pub trusted_bgen_validation_mode: String,
    pub sample_key_mode: String,
    pub multi_phenotype_sample_mode: String,
    pub firth_batch_size: i64,
    pub firth_candidate_capacity: i64,
    pub binary_null_maximum_iterations: i64,
    pub binary_null_coefficient_tolerance: f64,
    pub null_logistic_nonconvergence_policy: String,
    pub binary_minimum_probability: f64,
    pub binary_minimum_variance: f64,
    pub binary_relative_variance_tolerance: f64,
    pub linear_minimum_variance: f64,
    pub linear_relative_variance_tolerance: f64,
    pub firth_maximum_iterations: i64,
    pub firth_gradient_tolerance: f64,
    pub firth_coefficient_tolerance: f64,
    pub firth_likelihood_tolerance: f64,
    pub firth_maximum_step_size: f64,
    pub firth_pseudo_maximum_iterations: i64,
    pub firth_pseudo_inner_maximum_iterations: i64,
    pub firth_newton_raphson_zero_start_iterations: i64,
    pub firth_line_search_maximum_attempts: i64,
    pub firth_step_halving_maximum_attempts: i64,
    pub firth_initial_response_scale: f64,
    pub firth_sparse_carrier_dosage_threshold: f64,
    pub firth_step_halving_scale: f64,
    pub null_firth_maximum_iterations: i64,
    pub null_firth_gradient_tolerance: f64,
    pub null_firth_maximum_step_size: f64,
    pub null_firth_fallback_iteration_multiplier: i64,
    pub null_firth_fallback_step_divisor: f64,
    pub null_firth_line_search_maximum_attempts: i64,
    pub null_firth_step_halving_scale: f64,
    pub use_block_firth_math: bool,
    pub bgen_decode_tile_variant_count: i64,
    pub gpu_genotype_format: String,
    pub score_dtype: String,
    pub firth_dtype: String,
    pub jax_cache_dir: Option<String>,
    pub jax_matmul_precision: Option<String>,
    pub jax_persistent_cache: bool,
    pub jax_persistent_cache_min_entry_size_bytes: i64,
    pub jax_persistent_cache_min_compile_time_seconds: i64,
    pub jax_xla_autotune_cache: bool,
    pub jax_transfer_guard: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GOutputConfigData {
    pub out: Option<String>,
    pub format: String,
    pub output_run_directory: Option<String>,
    pub writer_threads: i64,
    pub writer_queue_depth: i64,
    pub chunks_per_arrow_file: i64,
    pub arrow_compression: String,
    pub parquet_compression: String,
    pub resume: bool,
    pub resume_mode: String,
    pub finalize_parquet: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GDiagnosticsConfigData {
    pub telemetry: String,
    pub log_dir: Option<String>,
    pub stage_timings_json: Option<String>,
    pub log_filter: String,
    pub log_file: Option<String>,
    pub log_stderr: bool,
    pub progress_interval_seconds: f64,
    pub progress_interval_chunks: i64,
    pub profile_summary_json: Option<String>,
    pub trace_file: Option<String>,
    pub trace_filter: String,
    pub trace_event_cap: i64,
    pub log_queue_size: i64,
    pub log_lossy: bool,
    pub include_source_location: bool,
    pub include_span_events: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RegenieConfigData {
    pub input: InputConfigData,
    pub trait_config: TraitConfigData,
    pub binary: BinaryConfigData,
    pub g_compute: GComputeConfigData,
    pub g_output: GOutputConfigData,
    pub g_diagnostics: GDiagnosticsConfigData,
    pub explicit_options: BTreeSet<String>,
}

fn build_runtime_config_from_toml_config(
    toml_config: &OptionTable,
    explicit_options: &BTreeSet<String>,
) -> ConfigResult<RegenieConfigData> {
    let trait_type = resolve_configured_toml_trait_type(toml_config)?;
    let pheno_columns = resolve_exclusive_column_values(toml_config, "input", "phenoCol", "phenoColList")?;
    let covar_columns = resolve_exclusive_column_values(toml_config, "input", "covarCol", "covarColList")?;

    Ok(RegenieConfigData {
        input: InputConfigData {
            bgen: get_string(toml_config, "input", "bgen")?,
            sample: get_string(toml_config, "input", "sample")?,
            pheno_file: get_string(toml_config, "input", "phenoFile")?,
            pheno_columns,
            covar_file: get_string(toml_config, "input", "covarFile")?,
            covar_columns,
            pred: get_string(toml_config, "input", "pred")?,
        },
        trait_config: TraitConfigData {
            step: required_i64(toml_config, "trait", "step", "step")?,
            trait_type,
            bsize: required_i64(toml_config, "trait", "bsize", "bsize")?,
            threads: get_i64(toml_config, "trait", "threads")?,
        },
        binary: BinaryConfigData {
            firth: required_bool(toml_config, "binary", "firth", "firth")?,
            approx: required_bool(toml_config, "binary", "approx", "approx")?,
            spa: get_bool(toml_config, "binary", "spa")?.unwrap_or(false),
            p_threshold: required_f64(toml_config, "binary", "pThresh", "pThresh")?,
            firth_se: required_bool(toml_config, "binary", "firth-se", "firth-se")?,
        },
        g_compute: GComputeConfigData {
            device: required_string(toml_config, "g.compute", "device", "g-device")?,
            staging_depth: required_i64(toml_config, "g.compute", "staging-depth", "g-staging-depth")?,
            variant_limit: get_i64(toml_config, "g.compute", "variant-limit")?,
            trusted_no_missing_diploid: required_bool(
                toml_config,
                "g.compute",
                "trusted-no-missing-diploid",
                "g-trusted-no-missing-diploid",
            )?,
            trusted_bgen_validation_mode: required_string(
                toml_config,
                "g.compute",
                "trusted-bgen-validation-mode",
                "g-trusted-bgen-validation-mode",
            )?,
            sample_key_mode: required_string(toml_config, "g.compute", "sample-key-mode", "g-sample-key-mode")?,
            multi_phenotype_sample_mode: required_string(
                toml_config,
                "g.compute",
                "multi-phenotype-sample-mode",
                "g-multi-phenotype-sample-mode",
            )?,
            firth_batch_size: required_i64(toml_config, "g.compute", "firth-batch-size", "g-firth-batch-size")?,
            firth_candidate_capacity: required_i64(
                toml_config,
                "g.compute",
                "firth-candidate-capacity",
                "g-firth-candidate-capacity",
            )?,
            binary_null_maximum_iterations: required_i64(
                toml_config,
                "g.compute",
                "binary-null-maximum-iterations",
                "g-binary-null-maximum-iterations",
            )?,
            binary_null_coefficient_tolerance: required_f64(
                toml_config,
                "g.compute",
                "binary-null-coefficient-tolerance",
                "g-binary-null-coefficient-tolerance",
            )?,
            null_logistic_nonconvergence_policy: required_string(
                toml_config,
                "g.compute",
                "null-logistic-nonconvergence",
                "g-null-logistic-nonconvergence",
            )?,
            binary_minimum_probability: required_f64(
                toml_config,
                "g.compute",
                "binary-minimum-probability",
                "g-binary-minimum-probability",
            )?,
            binary_minimum_variance: required_f64(
                toml_config,
                "g.compute",
                "binary-minimum-variance",
                "g-binary-minimum-variance",
            )?,
            binary_relative_variance_tolerance: required_f64(
                toml_config,
                "g.compute",
                "binary-relative-variance-tolerance",
                "g-binary-relative-variance-tolerance",
            )?,
            linear_minimum_variance: required_f64(
                toml_config,
                "g.compute",
                "linear-minimum-variance",
                "g-linear-minimum-variance",
            )?,
            linear_relative_variance_tolerance: required_f64(
                toml_config,
                "g.compute",
                "linear-relative-variance-tolerance",
                "g-linear-relative-variance-tolerance",
            )?,
            firth_maximum_iterations: required_i64(
                toml_config,
                "g.compute",
                "firth-maximum-iterations",
                "g-firth-maximum-iterations",
            )?,
            firth_gradient_tolerance: required_f64(
                toml_config,
                "g.compute",
                "firth-gradient-tolerance",
                "g-firth-gradient-tolerance",
            )?,
            firth_coefficient_tolerance: required_f64(
                toml_config,
                "g.compute",
                "firth-coefficient-tolerance",
                "g-firth-coefficient-tolerance",
            )?,
            firth_likelihood_tolerance: required_f64(
                toml_config,
                "g.compute",
                "firth-likelihood-tolerance",
                "g-firth-likelihood-tolerance",
            )?,
            firth_maximum_step_size: required_f64(
                toml_config,
                "g.compute",
                "firth-maximum-step-size",
                "g-firth-maximum-step-size",
            )?,
            firth_pseudo_maximum_iterations: required_i64(
                toml_config,
                "g.compute",
                "firth-pseudo-maximum-iterations",
                "g-firth-pseudo-maximum-iterations",
            )?,
            firth_pseudo_inner_maximum_iterations: required_i64(
                toml_config,
                "g.compute",
                "firth-pseudo-inner-maximum-iterations",
                "g-firth-pseudo-inner-maximum-iterations",
            )?,
            firth_newton_raphson_zero_start_iterations: required_i64(
                toml_config,
                "g.compute",
                "firth-newton-raphson-zero-start-iterations",
                "g-firth-newton-raphson-zero-start-iterations",
            )?,
            firth_line_search_maximum_attempts: required_i64(
                toml_config,
                "g.compute",
                "firth-line-search-maximum-attempts",
                "g-firth-line-search-maximum-attempts",
            )?,
            firth_step_halving_maximum_attempts: required_i64(
                toml_config,
                "g.compute",
                "firth-step-halving-maximum-attempts",
                "g-firth-step-halving-maximum-attempts",
            )?,
            firth_initial_response_scale: required_f64(
                toml_config,
                "g.compute",
                "firth-initial-response-scale",
                "g-firth-initial-response-scale",
            )?,
            firth_sparse_carrier_dosage_threshold: required_f64(
                toml_config,
                "g.compute",
                "firth-sparse-carrier-dosage-threshold",
                "g-firth-sparse-carrier-dosage-threshold",
            )?,
            firth_step_halving_scale: required_f64(
                toml_config,
                "g.compute",
                "firth-step-halving-scale",
                "g-firth-step-halving-scale",
            )?,
            null_firth_maximum_iterations: required_i64(
                toml_config,
                "g.compute",
                "null-firth-maximum-iterations",
                "g-null-firth-maximum-iterations",
            )?,
            null_firth_gradient_tolerance: required_f64(
                toml_config,
                "g.compute",
                "null-firth-gradient-tolerance",
                "g-null-firth-gradient-tolerance",
            )?,
            null_firth_maximum_step_size: required_f64(
                toml_config,
                "g.compute",
                "null-firth-maximum-step-size",
                "g-null-firth-maximum-step-size",
            )?,
            null_firth_fallback_iteration_multiplier: required_i64(
                toml_config,
                "g.compute",
                "null-firth-fallback-iteration-multiplier",
                "g-null-firth-fallback-iteration-multiplier",
            )?,
            null_firth_fallback_step_divisor: required_f64(
                toml_config,
                "g.compute",
                "null-firth-fallback-step-divisor",
                "g-null-firth-fallback-step-divisor",
            )?,
            null_firth_line_search_maximum_attempts: required_i64(
                toml_config,
                "g.compute",
                "null-firth-line-search-maximum-attempts",
                "g-null-firth-line-search-maximum-attempts",
            )?,
            null_firth_step_halving_scale: required_f64(
                toml_config,
                "g.compute",
                "null-firth-step-halving-scale",
                "g-null-firth-step-halving-scale",
            )?,
            use_block_firth_math: required_bool(
                toml_config,
                "g.compute",
                "use-block-firth-math",
                "g-use-block-firth-math",
            )?,
            bgen_decode_tile_variant_count: required_i64(
                toml_config,
                "g.compute",
                "bgen-decode-tile-variant-count",
                "g-bgen-decode-tile-variant-count",
            )?,
            gpu_genotype_format: required_string(
                toml_config,
                "g.compute",
                "gpu-genotype-format",
                "g-gpu-genotype-format",
            )?,
            score_dtype: required_string(toml_config, "g.compute", "score-dtype", "g-score-dtype")?,
            firth_dtype: required_string(toml_config, "g.compute", "firth-dtype", "g-firth-dtype")?,
            jax_cache_dir: get_string(toml_config, "g.compute", "jax-cache-dir")?,
            jax_matmul_precision: get_string(toml_config, "g.compute", "jax-matmul-precision")?,
            jax_persistent_cache: required_bool(
                toml_config,
                "g.compute",
                "jax-persistent-cache",
                "g-jax-persistent-cache",
            )?,
            jax_persistent_cache_min_entry_size_bytes: required_i64(
                toml_config,
                "g.compute",
                "jax-persistent-cache-min-entry-size-bytes",
                "g-jax-persistent-cache-min-entry-size-bytes",
            )?,
            jax_persistent_cache_min_compile_time_seconds: required_i64(
                toml_config,
                "g.compute",
                "jax-persistent-cache-min-compile-time-seconds",
                "g-jax-persistent-cache-min-compile-time-seconds",
            )?,
            jax_xla_autotune_cache: required_bool(
                toml_config,
                "g.compute",
                "jax-xla-autotune-cache",
                "g-jax-xla-autotune-cache",
            )?,
            jax_transfer_guard: required_bool(toml_config, "g.compute", "jax-transfer-guard", "g-jax-transfer-guard")?,
        },
        g_output: GOutputConfigData {
            out: get_string(toml_config, "output", "out")?,
            format: required_string(toml_config, "g.output", "format", "g-output-format")?,
            output_run_directory: get_string(toml_config, "g.output", "output-run-directory")?,
            writer_threads: required_i64(toml_config, "g.output", "writer-threads", "g-writer-threads")?,
            writer_queue_depth: required_i64(toml_config, "g.output", "writer-queue-depth", "g-writer-queue-depth")?,
            chunks_per_arrow_file: required_i64(
                toml_config,
                "g.output",
                "chunks-per-arrow-file",
                "g-output-chunks-per-arrow-file",
            )?,
            arrow_compression: required_string(
                toml_config,
                "g.output",
                "arrow-compression",
                "g-output-arrow-compression",
            )?,
            parquet_compression: required_string(
                toml_config,
                "g.output",
                "parquet-compression",
                "g-output-parquet-compression",
            )?,
            resume: required_bool(toml_config, "g.output", "resume", "g-resume")?,
            resume_mode: required_string(toml_config, "g.output", "resume-mode", "g-resume-mode")?,
            finalize_parquet: required_bool(toml_config, "g.output", "finalize-parquet", "g-finalize-parquet")?,
        },
        g_diagnostics: GDiagnosticsConfigData {
            telemetry: required_string(toml_config, "g.diagnostics", "telemetry", "g-telemetry")?,
            log_dir: get_string(toml_config, "g.diagnostics", "log-dir")?,
            stage_timings_json: get_string(toml_config, "g.diagnostics", "stage-timings-json")?,
            log_filter: required_string(toml_config, "g.diagnostics", "log-filter", "g-log-filter")?,
            log_file: get_string(toml_config, "g.diagnostics", "log-file")?,
            log_stderr: required_bool(toml_config, "g.diagnostics", "log-stderr", "g-log-stderr")?,
            progress_interval_seconds: required_f64(
                toml_config,
                "g.diagnostics",
                "progress-interval-seconds",
                "g-progress-interval-seconds",
            )?,
            progress_interval_chunks: required_i64(
                toml_config,
                "g.diagnostics",
                "progress-interval-chunks",
                "g-progress-interval-chunks",
            )?,
            profile_summary_json: get_string(toml_config, "g.diagnostics", "profile-summary-json")?,
            trace_file: get_string(toml_config, "g.diagnostics", "trace-file")?,
            trace_filter: required_string(toml_config, "g.diagnostics", "trace-filter", "g-trace-filter")?,
            trace_event_cap: required_i64(toml_config, "g.diagnostics", "trace-event-cap", "g-trace-event-cap")?,
            log_queue_size: required_i64(toml_config, "g.diagnostics", "log-queue-size", "g-log-queue-size")?,
            log_lossy: required_bool(toml_config, "g.diagnostics", "log-lossy", "g-log-lossy")?,
            include_source_location: required_bool(
                toml_config,
                "g.diagnostics",
                "include-source-location",
                "g-include-source-location",
            )?,
            include_span_events: required_bool(
                toml_config,
                "g.diagnostics",
                "include-span-events",
                "g-include-span-events",
            )?,
        },
        explicit_options: explicit_options.clone(),
    })
}

fn resolve_configured_toml_trait_type(toml_config: &OptionTable) -> ConfigResult<String> {
    normalize_trait_type(get_bool(toml_config, "trait", "qt")?, get_bool(toml_config, "trait", "bt")?)
}

pub fn normalize_trait_type(qt: Option<bool>, bt: Option<bool>) -> ConfigResult<String> {
    if qt == Some(true) && bt == Some(true) {
        return Err(ConfigError::new("--qt and --bt are mutually exclusive."));
    }
    if bt == Some(true) {
        return Ok("binary".to_string());
    }
    Ok("quantitative".to_string())
}

fn resolve_exclusive_column_values(
    toml_config: &OptionTable,
    section_name: &str,
    repeated_key: &str,
    list_key: &str,
) -> ConfigResult<Vec<String>> {
    let repeated_columns = split_name_list(get_option_value(toml_config, section_name, repeated_key));
    let list_columns = split_name_list(get_option_value(toml_config, section_name, list_key));
    if !repeated_columns.is_empty() && !list_columns.is_empty() {
        return Err(ConfigError::new(format!("Use either --{repeated_key} or --{list_key}, not both.")));
    }
    if repeated_columns.is_empty() { Ok(list_columns) } else { Ok(repeated_columns) }
}

pub fn split_name_list(raw_value: Option<&OptionValue>) -> Vec<String> {
    match raw_value {
        None | Some(OptionValue::None) => Vec::new(),
        Some(OptionValue::String(value)) => {
            value.split(',').map(str::trim).filter(|name| !name.is_empty()).map(ToOwned::to_owned).collect()
        }
        Some(OptionValue::List(values)) => {
            values.iter().map(|name| name.trim()).filter(|name| !name.is_empty()).map(ToOwned::to_owned).collect()
        }
        Some(other_value) => {
            let value = format_option_value(other_value);
            value.split(',').map(str::trim).filter(|name| !name.is_empty()).map(ToOwned::to_owned).collect()
        }
    }
}

fn get_section_table<'a>(toml_config: &'a OptionTable, section_name: &str) -> Option<&'a OptionTable> {
    if let Some((namespace_name, nested_section_name)) = section_name.split_once('.') {
        return toml_config
            .get(namespace_name)
            .and_then(OptionValue::as_table)
            .and_then(|namespace_table| namespace_table.get(nested_section_name))
            .and_then(OptionValue::as_table);
    }
    toml_config.get(section_name).and_then(OptionValue::as_table)
}

fn get_option_value<'a>(toml_config: &'a OptionTable, section_name: &str, key: &str) -> Option<&'a OptionValue> {
    get_section_table(toml_config, section_name).and_then(|section_table| section_table.get(key))
}

fn get_string(toml_config: &OptionTable, section_name: &str, key: &str) -> ConfigResult<Option<String>> {
    get_option_value(toml_config, section_name, key).map(|value| string_value(value, key)).transpose()
}

fn required_string(
    toml_config: &OptionTable,
    section_name: &str,
    key: &str,
    option_name: &str,
) -> ConfigResult<String> {
    get_string(toml_config, section_name, key)?
        .ok_or_else(|| ConfigError::new(format!("Default config is missing required default option {option_name:?}.")))
}

fn get_i64(toml_config: &OptionTable, section_name: &str, key: &str) -> ConfigResult<Option<i64>> {
    get_option_value(toml_config, section_name, key).map(|value| integer_value(value, key)).transpose()
}

fn required_i64(toml_config: &OptionTable, section_name: &str, key: &str, option_name: &str) -> ConfigResult<i64> {
    get_i64(toml_config, section_name, key)?
        .ok_or_else(|| ConfigError::new(format!("Default config is missing required default option {option_name:?}.")))
}

fn get_f64(toml_config: &OptionTable, section_name: &str, key: &str) -> ConfigResult<Option<f64>> {
    get_option_value(toml_config, section_name, key).map(|value| float_value(value, key)).transpose()
}

fn required_f64(toml_config: &OptionTable, section_name: &str, key: &str, option_name: &str) -> ConfigResult<f64> {
    get_f64(toml_config, section_name, key)?
        .ok_or_else(|| ConfigError::new(format!("Default config is missing required default option {option_name:?}.")))
}

fn get_bool(toml_config: &OptionTable, section_name: &str, key: &str) -> ConfigResult<Option<bool>> {
    get_option_value(toml_config, section_name, key).map(|value| boolean_value(value, key)).transpose()
}

fn required_bool(toml_config: &OptionTable, section_name: &str, key: &str, option_name: &str) -> ConfigResult<bool> {
    get_bool(toml_config, section_name, key)?
        .ok_or_else(|| ConfigError::new(format!("Default config is missing required default option {option_name:?}.")))
}

fn string_value(option_value: &OptionValue, option_name: &str) -> ConfigResult<String> {
    match option_value {
        OptionValue::String(value) => Ok(value.clone()),
        OptionValue::Integer(value) => Ok(value.to_string()),
        OptionValue::Float(value) => Ok(value.to_string()),
        OptionValue::Boolean(value) => Ok(value.to_string()),
        OptionValue::List(values) => Ok(values.join(",")),
        OptionValue::None => Err(ConfigError::new(format!("Option {option_name} is not set."))),
        OptionValue::Table(_) => Err(ConfigError::new(format!("Option {option_name} must be a scalar value."))),
    }
}

fn integer_value(option_value: &OptionValue, option_name: &str) -> ConfigResult<i64> {
    match option_value {
        OptionValue::Integer(value) => Ok(*value),
        OptionValue::Float(value) if value.fract() == 0.0 => value
            .to_string()
            .parse::<i64>()
            .map_err(|_| ConfigError::new(format!("Integer option value for {option_name} must be an integer."))),
        OptionValue::String(value) => value
            .parse::<i64>()
            .map_err(|_| ConfigError::new(format!("Integer option value for {option_name} must be an integer."))),
        _ => Err(ConfigError::new(format!("Integer option value for {option_name} must be an integer."))),
    }
}

fn float_value(option_value: &OptionValue, option_name: &str) -> ConfigResult<f64> {
    match option_value {
        OptionValue::Integer(value) => value
            .to_string()
            .parse::<f64>()
            .map_err(|_| ConfigError::new(format!("Float option value for {option_name} must be a number."))),
        OptionValue::Float(value) => Ok(*value),
        OptionValue::String(value) => value
            .parse::<f64>()
            .map_err(|_| ConfigError::new(format!("Float option value for {option_name} must be a number."))),
        _ => Err(ConfigError::new(format!("Float option value for {option_name} must be a number."))),
    }
}

fn boolean_value(option_value: &OptionValue, _option_name: &str) -> ConfigResult<bool> {
    match option_value {
        OptionValue::Boolean(value) => Ok(*value),
        OptionValue::String(value) => match value.trim().to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" | "on" => Ok(true),
            "false" | "0" | "no" | "off" => Ok(false),
            _ => Err(ConfigError::new(format!(
                "Boolean option value must be a bool or explicit boolean string, got {value:?}."
            ))),
        },
        _ => Err(ConfigError::new(format!(
            "Boolean option value must be a bool or explicit boolean string, got {}.",
            format_option_value(option_value)
        ))),
    }
}

fn format_option_value(option_value: &OptionValue) -> String {
    match option_value {
        OptionValue::None => "None".to_string(),
        OptionValue::String(value) => value.clone(),
        OptionValue::Integer(value) => value.to_string(),
        OptionValue::Float(value) => value.to_string(),
        OptionValue::Boolean(value) => value.to_string(),
        OptionValue::List(values) => values.join(","),
        OptionValue::Table(_) => "<table>".to_string(),
    }
}
