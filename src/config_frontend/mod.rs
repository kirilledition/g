use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Write as _};
use std::fs;
use std::path::Path;
use std::sync::OnceLock;

use sha2::{Digest, Sha256};
use toml::Value;

mod cli;
mod metadata;
mod options;
mod render;
mod validation;

pub use cli::{CliOutcomeData, dispatch_cli, explain_option, iter_explanations};
pub use metadata::{DefaultPolicy, OptionSpec, SupportLevel};
pub use render::{dumps_toml, format_toml_string, write_toml};
pub use validation::{
    validate_config, validate_non_negative_integer, validate_positive_float, validate_positive_integer,
    validate_probability_floor,
};

use metadata::{OptionValueType, option_registry};
use validation::{
    reject_missing_resolved_default_options, reject_quantitative_binary_only_options, validate_unknown_options,
};

const DEFAULT_CONFIG_TOML: &str = include_str!("../g/config.default.toml");
const OPTION_SCHEMA_VERSION: i64 = 1;
const QUANTITATIVE_BINARY_ONLY_OPTION_NAMES: &[&str] = &["firth", "approx", "firth-se", "pThresh"];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigError {
    message: String,
}

impl ConfigError {
    fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }

    #[must_use]
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
    #[must_use]
    pub fn toml_config(&self) -> &OptionTable {
        &self.toml_config
    }

    #[must_use]
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

/// Load and validate the packaged default option catalog.
///
/// # Errors
///
/// Returns an error when the packaged TOML defaults or JSON option metadata are invalid.
pub fn load_default_option_catalog_data() -> ConfigResult<&'static DefaultOptionCatalogData> {
    DEFAULT_CATALOG
        .get_or_init(|| {
            let toml_config = parse_toml_document(DEFAULT_CONFIG_TOML, "config.default.toml")?;
            validate_toml_schema(&toml_config, "config.default.toml")?;
            let normalized_options = flatten_toml_mapping(&toml_config);
            validate_default_catalog(&normalized_options)?;
            let default_config_hash = build_default_config_hash(&toml_config)?;
            Ok(DefaultOptionCatalogData { raw_toml: toml_config, normalized_options, default_config_hash })
        })
        .as_ref()
        .map_err(Clone::clone)
}

fn validate_default_catalog(normalized_options: &OptionTable) -> ConfigResult<()> {
    let registry = option_registry();
    let unknown_option_names = normalized_options
        .keys()
        .filter(|option_name| registry.get_by_cli_name(option_name).is_none())
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
            option_spec.default_policy == DefaultPolicy::Value && !normalized_options.contains_key(option_spec.cli_name)
        })
        .map(|option_spec| option_spec.cli_name.to_string())
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
            option_spec.default_policy == DefaultPolicy::RequiredAtRuntime
                && normalized_options.contains_key(option_spec.cli_name)
        })
        .map(|option_spec| option_spec.cli_name.to_string())
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
    let parsed_table = toml::from_str::<toml::Table>(toml_text)
        .map_err(|error| ConfigError::new(format!("Invalid TOML config {source}: {error}")))?;
    Ok(table_from_toml_table(parsed_table))
}

fn option_value_from_toml_value(value: Value) -> OptionValue {
    match value {
        Value::String(value) => OptionValue::String(value),
        Value::Integer(value) => OptionValue::Integer(value),
        Value::Float(value) => OptionValue::Float(value),
        Value::Boolean(value) => OptionValue::Boolean(value),
        Value::Array(array) => option_value_from_toml_array(array),
        Value::Table(table) => OptionValue::Table(table_from_toml_table(table)),
        Value::Datetime(value) => OptionValue::String(value.to_string()),
    }
}

fn option_value_from_toml_array(array: Vec<Value>) -> OptionValue {
    OptionValue::List(array.into_iter().map(option_value_string_from_toml_value).collect())
}

fn option_value_string_from_toml_value(value: Value) -> String {
    match value {
        Value::String(value) => value,
        Value::Integer(value) => value.to_string(),
        Value::Float(value) => value.to_string(),
        Value::Boolean(value) => value.to_string(),
        Value::Array(_) | Value::Table(_) => format_option_value(&option_value_from_toml_value(value)),
        Value::Datetime(value) => value.to_string(),
    }
}

fn table_from_toml_table(table: toml::Table) -> OptionTable {
    table.into_iter().map(|(key, value)| (key, option_value_from_toml_value(value))).collect()
}

fn validate_toml_schema(table: &OptionTable, source: &str) -> ConfigResult<()> {
    let registry = option_registry();
    for (section_name, section_value) in table {
        if section_name == "metadata" {
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
            option_spec.cli_name
        )));
    }
    if !option_spec.accepted_values.is_empty() {
        let value = string_value(option_value, option_spec.cli_name)?;
        if !option_spec.accepted_values.contains(&value.as_str()) {
            return Err(ConfigError::new(format!(
                "Invalid TOML config {source}: invalid value {value:?} for --{}.",
                option_spec.cli_name
            )));
        }
    }
    Ok(())
}

/// Decode an optional TOML config path into an explicit config layer.
///
/// # Errors
///
/// Returns an error when the file cannot be read or the TOML content is invalid.
pub fn decode_toml_file_layer(path: Option<&Path>) -> ConfigResult<TomlConfigLayer> {
    let Some(config_path) = path else {
        return Ok(TomlConfigLayer { toml_config: OptionTable::new(), explicit_options: BTreeSet::new() });
    };
    let toml_text = fs::read_to_string(config_path)
        .map_err(|error| ConfigError::new(format!("Failed to read TOML config {}: {error}", config_path.display())))?;
    let toml_config = decode_toml_text(&toml_text, &config_path.display().to_string())?;
    let explicit_options = flatten_toml_mapping(&toml_config).into_keys().collect();
    Ok(TomlConfigLayer { toml_config, explicit_options })
}

/// Decode and validate a TOML document.
///
/// # Errors
///
/// Returns an error when parsing fails or the document does not match the supported schema.
pub fn decode_toml_text(toml_text: &str, source: &str) -> ConfigResult<OptionTable> {
    let toml_config = parse_toml_document(toml_text, source)?;
    validate_toml_schema(&toml_config, source)?;
    Ok(toml_config)
}

/// Flatten TOML-shaped sections into canonical option names.
#[must_use]
pub fn flatten_toml_mapping(raw_options: &OptionTable) -> OptionTable {
    let mut flattened_options = OptionTable::new();
    for (section_name, section_value) in raw_options {
        if section_name == "metadata" {
            continue;
        }
        if let Some(section_table) = section_value.as_table() {
            flattened_options.extend(flatten_toml_section(section_name, section_table));
        } else {
            flattened_options.insert(section_name.clone(), section_value.clone());
        }
    }
    flattened_options
}

fn flatten_toml_section(section_name: &str, section_options: &OptionTable) -> OptionTable {
    let registry = option_registry();
    let mut flattened_options = OptionTable::new();
    for (toml_key, option_value) in section_options {
        if let Some(option_spec) = registry.get_by_toml_path(section_name, toml_key) {
            flattened_options.insert(option_spec.cli_name.to_string(), option_value.clone());
        } else {
            flattened_options.insert(format!("{section_name}.{toml_key}"), option_value.clone());
        }
    }
    flattened_options
}

/// Normalize a CLI or TOML option name into its canonical CLI name.
#[must_use]
pub fn normalize_option_name(option_name: &str) -> String {
    option_name.to_string()
}

/// Convert a flat or TOML-shaped option dictionary into a TOML config layer.
///
/// # Errors
///
/// Returns an error when an option is unknown or has an invalid value for its schema.
pub fn option_dictionary_to_toml_config_layer(
    raw_options: &OptionTable,
    source: &str,
) -> ConfigResult<TomlConfigLayer> {
    let normalized_options = normalize_option_dictionary(raw_options);
    let registry = option_registry();
    let mut toml_mapping = OptionTable::new();

    for (option_name, option_value) in &normalized_options {
        if !option_value.is_explicit_some() {
            continue;
        }
        let Some(option_spec) = registry.get_by_cli_name(option_name) else {
            return Err(ConfigError::new(format!("Unknown g regenie option: {option_name}")));
        };
        let coerced_value = coerce_option_value(option_value, option_spec)?;
        set_toml_option_value(&mut toml_mapping, option_spec, coerced_value);
    }

    validate_toml_schema(&toml_mapping, source)?;
    let explicit_options = normalized_options
        .iter()
        .filter(|(_, option_value)| option_value.is_explicit_some())
        .map(|(option_name, _)| option_name.clone())
        .collect();
    Ok(TomlConfigLayer { toml_config: toml_mapping, explicit_options })
}

/// Normalize a full option dictionary into canonical option names.
#[must_use]
pub fn normalize_option_dictionary(raw_options: &OptionTable) -> OptionTable {
    flatten_toml_mapping(raw_options)
        .into_iter()
        .map(|(option_name, option_value)| (normalize_option_name(&option_name), option_value))
        .collect()
}

fn coerce_option_value(option_value: &OptionValue, option_spec: &OptionSpec) -> ConfigResult<OptionValue> {
    if option_spec.multiple {
        return Ok(coerce_string_list_value(option_value));
    }
    if !option_spec.accepted_values.is_empty() {
        let value = string_value(option_value, option_spec.cli_name)?;
        if !option_spec.accepted_values.contains(&value.as_str()) {
            return Err(ConfigError::new(format!("Invalid value for --{}: {value:?}.", option_spec.cli_name)));
        }
        return Ok(OptionValue::String(value));
    }
    match option_spec.value_type {
        OptionValueType::String | OptionValueType::Path => {
            Ok(OptionValue::String(string_value(option_value, option_spec.cli_name)?))
        }
        OptionValueType::Integer => Ok(OptionValue::Integer(integer_value(option_value, option_spec.cli_name)?)),
        OptionValueType::Float => Ok(OptionValue::Float(float_value(option_value, option_spec.cli_name)?)),
        OptionValueType::Boolean => Ok(OptionValue::Boolean(boolean_value(option_value, option_spec.cli_name)?)),
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
    let section_entry =
        toml_mapping.entry(option_spec.section.to_string()).or_insert_with(|| OptionValue::Table(OptionTable::new()));
    if let OptionValue::Table(section_table) = section_entry {
        section_table.insert(option_spec.cli_name.to_string(), option_value);
    }
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
    let override_quantitative_trait = get_bool(override_config, "trait", "qt").ok().flatten();
    let override_binary_trait = get_bool(override_config, "trait", "bt").ok().flatten();
    if override_quantitative_trait == Some(true) {
        set_nested_value(merged_config, "trait", "bt", OptionValue::Boolean(false));
    }
    if override_binary_trait == Some(true) {
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

/// Load packaged defaults as an unvalidated runtime config object.
///
/// # Errors
///
/// Returns an error when packaged defaults are malformed.
pub fn load_packaged_config_data() -> ConfigResult<RegenieConfigData> {
    let explicit_options = BTreeSet::new();
    build_runtime_config_from_toml_config(&load_default_option_catalog_data()?.raw_toml, &explicit_options)
}

/// Resolve a config from explicit Python option values.
///
/// # Errors
///
/// Returns an error when options are invalid or the resolved config fails validation.
pub fn from_options(raw_options: &OptionTable) -> ConfigResult<RegenieConfigData> {
    let explicit_layer = option_dictionary_to_toml_config_layer(raw_options, "Python options")?;
    from_toml_config_layers(&load_default_option_catalog_data()?.raw_toml, [explicit_layer])
}

/// Resolve a config from a TOML file path.
///
/// # Errors
///
/// Returns an error when the TOML file is invalid or the resolved config fails validation.
pub fn from_toml_path(path: &Path) -> ConfigResult<RegenieConfigData> {
    let toml_layer = decode_toml_file_layer(Some(path))?;
    from_toml_config_layers(&load_default_option_catalog_data()?.raw_toml, [toml_layer])
}

fn from_toml_config(toml_config: &OptionTable, explicit_options: &BTreeSet<String>) -> ConfigResult<RegenieConfigData> {
    let mut config = build_runtime_config_from_toml_config(toml_config, explicit_options)?;
    let normalized_options = flatten_toml_mapping(toml_config);
    reject_quantitative_binary_only_options(explicit_options, &config.trait_config.trait_type)?;
    validate_unknown_options(&normalized_options)?;
    reject_missing_resolved_default_options(&normalized_options)?;
    validate_config(&config)?;
    config.is_validated = true;
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
#[expect(clippy::struct_excessive_bools, reason = "Runtime config mirrors public REGENIE boolean flags.")]
pub struct BinaryConfigData {
    pub firth: bool,
    pub approx: bool,
    pub spa: bool,
    pub p_threshold: f64,
    pub firth_se: bool,
}

#[derive(Clone, Debug, PartialEq)]
#[expect(clippy::struct_excessive_bools, reason = "Runtime config mirrors public g-specific boolean options.")]
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
#[expect(clippy::struct_excessive_bools, reason = "Diagnostics config mirrors public g-specific boolean options.")]
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
    pub is_validated: bool,
}

fn build_runtime_config_from_toml_config(
    toml_config: &OptionTable,
    explicit_options: &BTreeSet<String>,
) -> ConfigResult<RegenieConfigData> {
    Ok(RegenieConfigData {
        input: build_input_config(toml_config)?,
        trait_config: build_trait_config(toml_config)?,
        binary: build_binary_config(toml_config)?,
        g_compute: build_g_compute_config(toml_config)?,
        g_output: build_g_output_config(toml_config)?,
        g_diagnostics: build_g_diagnostics_config(toml_config)?,
        explicit_options: explicit_options.clone(),
        is_validated: false,
    })
}

fn build_input_config(toml_config: &OptionTable) -> ConfigResult<InputConfigData> {
    Ok(InputConfigData {
        bgen: get_string(toml_config, "input", "bgen")?,
        sample: get_string(toml_config, "input", "sample")?,
        pheno_file: get_string(toml_config, "input", "phenoFile")?,
        pheno_columns: resolve_exclusive_column_values(toml_config, "input", "phenoCol", "phenoColList")?,
        covar_file: get_string(toml_config, "input", "covarFile")?,
        covar_columns: resolve_exclusive_column_values(toml_config, "input", "covarCol", "covarColList")?,
        pred: get_string(toml_config, "input", "pred")?,
    })
}

fn build_trait_config(toml_config: &OptionTable) -> ConfigResult<TraitConfigData> {
    let trait_type = resolve_configured_toml_trait_type(toml_config)?;
    Ok(TraitConfigData {
        step: required_i64(toml_config, "trait", "step", "step")?,
        trait_type,
        bsize: required_i64(toml_config, "trait", "bsize", "bsize")?,
        threads: get_i64(toml_config, "trait", "threads")?,
    })
}

fn build_binary_config(toml_config: &OptionTable) -> ConfigResult<BinaryConfigData> {
    Ok(BinaryConfigData {
        firth: required_bool(toml_config, "binary", "firth", "firth")?,
        approx: required_bool(toml_config, "binary", "approx", "approx")?,
        spa: false,
        p_threshold: required_f64(toml_config, "binary", "pThresh", "pThresh")?,
        firth_se: required_bool(toml_config, "binary", "firth-se", "firth-se")?,
    })
}

struct ComputeCoreFields {
    device: String,
    staging_depth: i64,
    variant_limit: Option<i64>,
    trusted_no_missing_diploid: bool,
    trusted_bgen_validation_mode: String,
    sample_key_mode: String,
    multi_phenotype_sample_mode: String,
    firth_batch_size: i64,
    firth_candidate_capacity: i64,
    binary_null_maximum_iterations: i64,
    binary_null_coefficient_tolerance: f64,
    null_logistic_nonconvergence_policy: String,
    binary_minimum_probability: f64,
    binary_minimum_variance: f64,
    binary_relative_variance_tolerance: f64,
    linear_minimum_variance: f64,
    linear_relative_variance_tolerance: f64,
}

struct FirthComputeFields {
    maximum_iterations: i64,
    gradient_tolerance: f64,
    coefficient_tolerance: f64,
    likelihood_tolerance: f64,
    maximum_step_size: f64,
    pseudo_maximum_iterations: i64,
    pseudo_inner_maximum_iterations: i64,
    newton_raphson_zero_start_iterations: i64,
    line_search_maximum_attempts: i64,
    step_halving_maximum_attempts: i64,
    initial_response_scale: f64,
    sparse_carrier_dosage_threshold: f64,
    step_halving_scale: f64,
}

struct NullFirthComputeFields {
    maximum_iterations: i64,
    gradient_tolerance: f64,
    maximum_step_size: f64,
    fallback_iteration_multiplier: i64,
    fallback_step_divisor: f64,
    line_search_maximum_attempts: i64,
    step_halving_scale: f64,
}

struct GenotypeComputeFields {
    use_block_firth_math: bool,
    bgen_decode_tile_variant_count: i64,
    gpu_genotype_format: String,
    score_dtype: String,
    firth_dtype: String,
}

struct JaxComputeFields {
    cache_dir: Option<String>,
    matmul_precision: Option<String>,
    persistent_cache: bool,
    persistent_cache_min_entry_size_bytes: i64,
    persistent_cache_min_compile_time_seconds: i64,
    xla_autotune_cache: bool,
    transfer_guard: bool,
}

fn build_g_compute_config(toml_config: &OptionTable) -> ConfigResult<GComputeConfigData> {
    let core = build_compute_core_fields(toml_config)?;
    let firth = build_firth_compute_fields(toml_config)?;
    let null_firth = build_null_firth_compute_fields(toml_config)?;
    let genotype = build_genotype_compute_fields(toml_config)?;
    let jax = build_jax_compute_fields(toml_config)?;
    Ok(GComputeConfigData {
        device: core.device,
        staging_depth: core.staging_depth,
        variant_limit: core.variant_limit,
        trusted_no_missing_diploid: core.trusted_no_missing_diploid,
        trusted_bgen_validation_mode: core.trusted_bgen_validation_mode,
        sample_key_mode: core.sample_key_mode,
        multi_phenotype_sample_mode: core.multi_phenotype_sample_mode,
        firth_batch_size: core.firth_batch_size,
        firth_candidate_capacity: core.firth_candidate_capacity,
        binary_null_maximum_iterations: core.binary_null_maximum_iterations,
        binary_null_coefficient_tolerance: core.binary_null_coefficient_tolerance,
        null_logistic_nonconvergence_policy: core.null_logistic_nonconvergence_policy,
        binary_minimum_probability: core.binary_minimum_probability,
        binary_minimum_variance: core.binary_minimum_variance,
        binary_relative_variance_tolerance: core.binary_relative_variance_tolerance,
        linear_minimum_variance: core.linear_minimum_variance,
        linear_relative_variance_tolerance: core.linear_relative_variance_tolerance,
        firth_maximum_iterations: firth.maximum_iterations,
        firth_gradient_tolerance: firth.gradient_tolerance,
        firth_coefficient_tolerance: firth.coefficient_tolerance,
        firth_likelihood_tolerance: firth.likelihood_tolerance,
        firth_maximum_step_size: firth.maximum_step_size,
        firth_pseudo_maximum_iterations: firth.pseudo_maximum_iterations,
        firth_pseudo_inner_maximum_iterations: firth.pseudo_inner_maximum_iterations,
        firth_newton_raphson_zero_start_iterations: firth.newton_raphson_zero_start_iterations,
        firth_line_search_maximum_attempts: firth.line_search_maximum_attempts,
        firth_step_halving_maximum_attempts: firth.step_halving_maximum_attempts,
        firth_initial_response_scale: firth.initial_response_scale,
        firth_sparse_carrier_dosage_threshold: firth.sparse_carrier_dosage_threshold,
        firth_step_halving_scale: firth.step_halving_scale,
        null_firth_maximum_iterations: null_firth.maximum_iterations,
        null_firth_gradient_tolerance: null_firth.gradient_tolerance,
        null_firth_maximum_step_size: null_firth.maximum_step_size,
        null_firth_fallback_iteration_multiplier: null_firth.fallback_iteration_multiplier,
        null_firth_fallback_step_divisor: null_firth.fallback_step_divisor,
        null_firth_line_search_maximum_attempts: null_firth.line_search_maximum_attempts,
        null_firth_step_halving_scale: null_firth.step_halving_scale,
        use_block_firth_math: genotype.use_block_firth_math,
        bgen_decode_tile_variant_count: genotype.bgen_decode_tile_variant_count,
        gpu_genotype_format: genotype.gpu_genotype_format,
        score_dtype: genotype.score_dtype,
        firth_dtype: genotype.firth_dtype,
        jax_cache_dir: jax.cache_dir,
        jax_matmul_precision: jax.matmul_precision,
        jax_persistent_cache: jax.persistent_cache,
        jax_persistent_cache_min_entry_size_bytes: jax.persistent_cache_min_entry_size_bytes,
        jax_persistent_cache_min_compile_time_seconds: jax.persistent_cache_min_compile_time_seconds,
        jax_xla_autotune_cache: jax.xla_autotune_cache,
        jax_transfer_guard: jax.transfer_guard,
    })
}

fn build_compute_core_fields(toml_config: &OptionTable) -> ConfigResult<ComputeCoreFields> {
    Ok(ComputeCoreFields {
        device: required_string(toml_config, "compute", "device", "device")?,
        staging_depth: required_i64(toml_config, "compute", "staging_depth", "staging_depth")?,
        variant_limit: get_i64(toml_config, "compute", "variant_limit")?,
        trusted_no_missing_diploid: required_bool(
            toml_config,
            "compute",
            "trusted_no_missing_diploid",
            "trusted_no_missing_diploid",
        )?,
        trusted_bgen_validation_mode: required_string(
            toml_config,
            "compute",
            "trusted_bgen_validation_mode",
            "trusted_bgen_validation_mode",
        )?,
        sample_key_mode: required_string(toml_config, "compute", "sample_key_mode", "sample_key_mode")?,
        multi_phenotype_sample_mode: required_string(
            toml_config,
            "compute",
            "multi_phenotype_sample_mode",
            "multi_phenotype_sample_mode",
        )?,
        firth_batch_size: required_i64(toml_config, "compute", "firth_batch_size", "firth_batch_size")?,
        firth_candidate_capacity: required_i64(
            toml_config,
            "compute",
            "firth_candidate_capacity",
            "firth_candidate_capacity",
        )?,
        binary_null_maximum_iterations: required_i64(
            toml_config,
            "compute",
            "binary_null_maximum_iterations",
            "binary_null_maximum_iterations",
        )?,
        binary_null_coefficient_tolerance: required_f64(
            toml_config,
            "compute",
            "binary_null_coefficient_tolerance",
            "binary_null_coefficient_tolerance",
        )?,
        null_logistic_nonconvergence_policy: required_string(
            toml_config,
            "compute",
            "null_logistic_nonconvergence_policy",
            "null_logistic_nonconvergence_policy",
        )?,
        binary_minimum_probability: required_f64(
            toml_config,
            "compute",
            "binary_minimum_probability",
            "binary_minimum_probability",
        )?,
        binary_minimum_variance: required_f64(
            toml_config,
            "compute",
            "binary_minimum_variance",
            "binary_minimum_variance",
        )?,
        binary_relative_variance_tolerance: required_f64(
            toml_config,
            "compute",
            "binary_relative_variance_tolerance",
            "binary_relative_variance_tolerance",
        )?,
        linear_minimum_variance: required_f64(
            toml_config,
            "compute",
            "linear_minimum_variance",
            "linear_minimum_variance",
        )?,
        linear_relative_variance_tolerance: required_f64(
            toml_config,
            "compute",
            "linear_relative_variance_tolerance",
            "linear_relative_variance_tolerance",
        )?,
    })
}

fn build_firth_compute_fields(toml_config: &OptionTable) -> ConfigResult<FirthComputeFields> {
    Ok(FirthComputeFields {
        maximum_iterations: required_i64(
            toml_config,
            "compute",
            "firth_maximum_iterations",
            "firth_maximum_iterations",
        )?,
        gradient_tolerance: required_f64(
            toml_config,
            "compute",
            "firth_gradient_tolerance",
            "firth_gradient_tolerance",
        )?,
        coefficient_tolerance: required_f64(
            toml_config,
            "compute",
            "firth_coefficient_tolerance",
            "firth_coefficient_tolerance",
        )?,
        likelihood_tolerance: required_f64(
            toml_config,
            "compute",
            "firth_likelihood_tolerance",
            "firth_likelihood_tolerance",
        )?,
        maximum_step_size: required_f64(toml_config, "compute", "firth_maximum_step_size", "firth_maximum_step_size")?,
        pseudo_maximum_iterations: required_i64(
            toml_config,
            "compute",
            "firth_pseudo_maximum_iterations",
            "firth_pseudo_maximum_iterations",
        )?,
        pseudo_inner_maximum_iterations: required_i64(
            toml_config,
            "compute",
            "firth_pseudo_inner_maximum_iterations",
            "firth_pseudo_inner_maximum_iterations",
        )?,
        newton_raphson_zero_start_iterations: required_i64(
            toml_config,
            "compute",
            "firth_newton_raphson_zero_start_iterations",
            "firth_newton_raphson_zero_start_iterations",
        )?,
        line_search_maximum_attempts: required_i64(
            toml_config,
            "compute",
            "firth_line_search_maximum_attempts",
            "firth_line_search_maximum_attempts",
        )?,
        step_halving_maximum_attempts: required_i64(
            toml_config,
            "compute",
            "firth_step_halving_maximum_attempts",
            "firth_step_halving_maximum_attempts",
        )?,
        initial_response_scale: required_f64(
            toml_config,
            "compute",
            "firth_initial_response_scale",
            "firth_initial_response_scale",
        )?,
        sparse_carrier_dosage_threshold: required_f64(
            toml_config,
            "compute",
            "firth_sparse_carrier_dosage_threshold",
            "firth_sparse_carrier_dosage_threshold",
        )?,
        step_halving_scale: required_f64(
            toml_config,
            "compute",
            "firth_step_halving_scale",
            "firth_step_halving_scale",
        )?,
    })
}

fn build_null_firth_compute_fields(toml_config: &OptionTable) -> ConfigResult<NullFirthComputeFields> {
    Ok(NullFirthComputeFields {
        maximum_iterations: required_i64(
            toml_config,
            "compute",
            "null_firth_maximum_iterations",
            "null_firth_maximum_iterations",
        )?,
        gradient_tolerance: required_f64(
            toml_config,
            "compute",
            "null_firth_gradient_tolerance",
            "null_firth_gradient_tolerance",
        )?,
        maximum_step_size: required_f64(
            toml_config,
            "compute",
            "null_firth_maximum_step_size",
            "null_firth_maximum_step_size",
        )?,
        fallback_iteration_multiplier: required_i64(
            toml_config,
            "compute",
            "null_firth_fallback_iteration_multiplier",
            "null_firth_fallback_iteration_multiplier",
        )?,
        fallback_step_divisor: required_f64(
            toml_config,
            "compute",
            "null_firth_fallback_step_divisor",
            "null_firth_fallback_step_divisor",
        )?,
        line_search_maximum_attempts: required_i64(
            toml_config,
            "compute",
            "null_firth_line_search_maximum_attempts",
            "null_firth_line_search_maximum_attempts",
        )?,
        step_halving_scale: required_f64(
            toml_config,
            "compute",
            "null_firth_step_halving_scale",
            "null_firth_step_halving_scale",
        )?,
    })
}

fn build_genotype_compute_fields(toml_config: &OptionTable) -> ConfigResult<GenotypeComputeFields> {
    Ok(GenotypeComputeFields {
        use_block_firth_math: required_bool(toml_config, "compute", "use_block_firth_math", "use_block_firth_math")?,
        bgen_decode_tile_variant_count: required_i64(
            toml_config,
            "compute",
            "bgen_decode_tile_variant_count",
            "bgen_decode_tile_variant_count",
        )?,
        gpu_genotype_format: required_string(toml_config, "compute", "gpu_genotype_format", "gpu_genotype_format")?,
        score_dtype: required_string(toml_config, "compute", "score_dtype", "score_dtype")?,
        firth_dtype: required_string(toml_config, "compute", "firth_dtype", "firth_dtype")?,
    })
}

fn build_jax_compute_fields(toml_config: &OptionTable) -> ConfigResult<JaxComputeFields> {
    Ok(JaxComputeFields {
        cache_dir: get_string(toml_config, "compute", "jax_cache_dir")?,
        matmul_precision: get_string(toml_config, "compute", "jax_matmul_precision")?,
        persistent_cache: required_bool(toml_config, "compute", "jax_persistent_cache", "jax_persistent_cache")?,
        persistent_cache_min_entry_size_bytes: required_i64(
            toml_config,
            "compute",
            "jax_persistent_cache_min_entry_size_bytes",
            "jax_persistent_cache_min_entry_size_bytes",
        )?,
        persistent_cache_min_compile_time_seconds: required_i64(
            toml_config,
            "compute",
            "jax_persistent_cache_min_compile_time_seconds",
            "jax_persistent_cache_min_compile_time_seconds",
        )?,
        xla_autotune_cache: required_bool(toml_config, "compute", "jax_xla_autotune_cache", "jax_xla_autotune_cache")?,
        transfer_guard: required_bool(toml_config, "compute", "jax_transfer_guard", "jax_transfer_guard")?,
    })
}

fn build_g_output_config(toml_config: &OptionTable) -> ConfigResult<GOutputConfigData> {
    Ok(GOutputConfigData {
        out: get_string(toml_config, "output", "out")?,
        format: required_string(toml_config, "output", "format", "format")?,
        output_run_directory: get_string(toml_config, "output", "output_run_directory")?,
        writer_threads: required_i64(toml_config, "output", "writer_threads", "writer_threads")?,
        writer_queue_depth: required_i64(toml_config, "output", "writer_queue_depth", "writer_queue_depth")?,
        chunks_per_arrow_file: required_i64(toml_config, "output", "chunks_per_arrow_file", "chunks_per_arrow_file")?,
        arrow_compression: required_string(toml_config, "output", "arrow_compression", "arrow_compression")?,
        parquet_compression: required_string(toml_config, "output", "parquet_compression", "parquet_compression")?,
        resume: required_bool(toml_config, "output", "resume", "resume")?,
        resume_mode: required_string(toml_config, "output", "resume_mode", "resume_mode")?,
        finalize_parquet: required_bool(toml_config, "output", "finalize_parquet", "finalize_parquet")?,
    })
}

fn build_g_diagnostics_config(toml_config: &OptionTable) -> ConfigResult<GDiagnosticsConfigData> {
    Ok(GDiagnosticsConfigData {
        telemetry: required_string(toml_config, "diagnostics", "telemetry", "telemetry")?,
        log_dir: get_string(toml_config, "diagnostics", "log_dir")?,
        stage_timings_json: get_string(toml_config, "diagnostics", "stage_timings_json")?,
        log_filter: required_string(toml_config, "diagnostics", "log_filter", "log_filter")?,
        log_file: get_string(toml_config, "diagnostics", "log_file")?,
        log_stderr: required_bool(toml_config, "diagnostics", "log_stderr", "log_stderr")?,
        progress_interval_seconds: required_f64(
            toml_config,
            "diagnostics",
            "progress_interval_seconds",
            "progress_interval_seconds",
        )?,
        progress_interval_chunks: required_i64(
            toml_config,
            "diagnostics",
            "progress_interval_chunks",
            "progress_interval_chunks",
        )?,
        profile_summary_json: get_string(toml_config, "diagnostics", "profile_summary_json")?,
        trace_file: get_string(toml_config, "diagnostics", "trace_file")?,
        trace_filter: required_string(toml_config, "diagnostics", "trace_filter", "trace_filter")?,
        trace_event_cap: required_i64(toml_config, "diagnostics", "trace_event_cap", "trace_event_cap")?,
        log_queue_size: required_i64(toml_config, "diagnostics", "log_queue_size", "log_queue_size")?,
        log_lossy: required_bool(toml_config, "diagnostics", "log_lossy", "log_lossy")?,
        include_source_location: required_bool(
            toml_config,
            "diagnostics",
            "include_source_location",
            "include_source_location",
        )?,
        include_span_events: required_bool(toml_config, "diagnostics", "include_span_events", "include_span_events")?,
    })
}

fn resolve_configured_toml_trait_type(toml_config: &OptionTable) -> ConfigResult<String> {
    normalize_trait_type(get_bool(toml_config, "trait", "qt")?, get_bool(toml_config, "trait", "bt")?)
}

/// Resolve REGENIE trait flags into the normalized runtime trait type.
///
/// # Errors
///
/// Returns an error when both quantitative and binary trait flags are enabled.
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
