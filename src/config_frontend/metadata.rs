#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SupportLevel {
    Supported,
    RecognizedUnsupported,
    GExtension,
    DeprecatedAlias,
}

impl SupportLevel {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Supported => "supported",
            Self::RecognizedUnsupported => "recognized_unsupported",
            Self::GExtension => "g_extension",
            Self::DeprecatedAlias => "deprecated_alias",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(super) enum OptionValueType {
    String,
    Integer,
    Float,
    Boolean,
    Path,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum DefaultPolicy {
    Value,
    AbsentIsNone,
    RequiredAtRuntime,
    Derived,
    Unsupported,
}

#[derive(Clone, Debug, Deserialize)]
pub struct OptionSpec {
    pub name: String,
    pub(super) destination: String,
    pub(super) support_level: SupportLevel,
    pub(super) section: String,
    pub(super) help_text: String,
    pub(super) value_type: OptionValueType,
    pub(super) multiple: bool,
    pub(super) is_flag: bool,
    pub(super) accepted_values: Vec<String>,
    pub(super) toml_key: String,
    pub(super) default_policy: DefaultPolicy,
    pub(super) python_aliases: Vec<String>,
}

#[derive(Clone, Debug)]
pub(super) struct OptionRegistry {
    pub(super) specs: Vec<OptionSpec>,
    by_name: BTreeMap<String, usize>,
    by_destination: BTreeMap<String, usize>,
    by_python_alias: BTreeMap<String, usize>,
    by_toml_path: BTreeMap<(String, String), usize>,
}

impl OptionRegistry {
    fn new(specs: Vec<OptionSpec>) -> Self {
        let mut by_name = BTreeMap::new();
        let mut by_destination = BTreeMap::new();
        let mut by_python_alias = BTreeMap::new();
        let mut by_toml_path = BTreeMap::new();

        for (spec_index, option_spec) in specs.iter().enumerate() {
            by_name.insert(option_spec.name.clone(), spec_index);
            by_destination.insert(option_spec.destination.clone(), spec_index);
            by_toml_path.insert((option_spec.section.clone(), option_spec.toml_key.clone()), spec_index);
            for python_alias in &option_spec.python_aliases {
                by_python_alias.insert(python_alias.clone(), spec_index);
            }
        }

        Self { specs, by_name, by_destination, by_python_alias, by_toml_path }
    }

    pub(super) fn get_by_name(&self, option_name: &str) -> Option<&OptionSpec> {
        self.by_name.get(option_name).and_then(|option_index| self.specs.get(*option_index))
    }

    pub(super) fn get_by_destination(&self, option_name: &str) -> Option<&OptionSpec> {
        self.by_destination.get(option_name).and_then(|option_index| self.specs.get(*option_index))
    }

    pub(super) fn get_by_python_alias(&self, option_name: &str) -> Option<&OptionSpec> {
        self.by_python_alias.get(option_name).and_then(|option_index| self.specs.get(*option_index))
    }

    pub(super) fn get_by_toml_path(&self, section_name: &str, toml_key: &str) -> Option<&OptionSpec> {
        self.by_toml_path
            .get(&(section_name.to_string(), toml_key.to_string()))
            .and_then(|option_index| self.specs.get(*option_index))
    }

    pub(super) fn supported_option_names(&self) -> BTreeSet<String> {
        self.specs
            .iter()
            .filter(|option_spec| {
                matches!(option_spec.support_level, SupportLevel::Supported | SupportLevel::GExtension)
            })
            .map(|option_spec| option_spec.name.clone())
            .collect()
    }

    pub(super) fn unsupported_option_names(&self) -> BTreeSet<String> {
        self.specs
            .iter()
            .filter(|option_spec| option_spec.support_level == SupportLevel::RecognizedUnsupported)
            .map(|option_spec| option_spec.name.clone())
            .collect()
    }
}

static OPTION_REGISTRY: OnceLock<Result<OptionRegistry, ConfigError>> = OnceLock::new();

pub(super) fn option_registry() -> ConfigResult<&'static OptionRegistry> {
    OPTION_REGISTRY
        .get_or_init(|| {
            let specs = serde_json::from_str::<Vec<OptionSpec>>(OPTION_METADATA_JSON)
                .map_err(|error| ConfigError::new(format!("Invalid packaged option metadata: {error}")))?;
            Ok(OptionRegistry::new(specs))
        })
        .as_ref()
        .map_err(Clone::clone)
}
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use serde::Deserialize;

use super::{ConfigError, ConfigResult, OPTION_METADATA_JSON};
