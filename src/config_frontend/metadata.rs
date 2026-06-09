use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use super::options;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SupportLevel {
    Supported,
    GExtension,
}

impl SupportLevel {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Supported => "supported",
            Self::GExtension => "g_extension",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum OptionValueType {
    String,
    Integer,
    Float,
    Boolean,
    Path,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DefaultPolicy {
    Value,
    AbsentIsNone,
    RequiredAtRuntime,
}

#[derive(Clone, Copy, Debug)]
pub struct OptionSpec {
    pub cli_name: &'static str,
    pub(super) support_level: SupportLevel,
    pub(super) section: &'static str,
    pub(super) help_text: &'static str,
    pub(super) value_type: OptionValueType,
    pub(super) multiple: bool,
    pub(super) is_flag: bool,
    pub(super) accepted_values: &'static [&'static str],
    pub(super) default_policy: DefaultPolicy,
}

impl OptionSpec {
    pub(super) fn config_key(&self) -> &'static str {
        if self.section == "g.output"
            && let Some(config_key) = self.cli_name.strip_prefix("g-output-")
        {
            return config_key;
        }
        if self.section.starts_with("g.") {
            return self.cli_name.strip_prefix("g-").unwrap_or(self.cli_name);
        }
        self.cli_name
    }
}

#[derive(Clone, Debug)]
pub(super) struct OptionRegistry {
    pub(super) specs: &'static [OptionSpec],
    by_cli_name: BTreeMap<&'static str, usize>,
    by_toml_path: BTreeMap<&'static str, BTreeMap<&'static str, usize>>,
}

impl OptionRegistry {
    fn new(specs: &'static [OptionSpec]) -> Self {
        let mut by_cli_name = BTreeMap::new();
        let mut by_toml_path = BTreeMap::new();

        for (spec_index, option_spec) in specs.iter().enumerate() {
            by_cli_name.insert(option_spec.cli_name, spec_index);
            by_toml_path
                .entry(option_spec.section)
                .or_insert_with(BTreeMap::new)
                .insert(option_spec.config_key(), spec_index);
        }

        Self { specs, by_cli_name, by_toml_path }
    }

    pub(super) fn get_by_cli_name(&self, option_name: &str) -> Option<&OptionSpec> {
        self.by_cli_name.get(option_name).and_then(|option_index| self.specs.get(*option_index))
    }

    pub(super) fn get_by_toml_path(&self, section_name: &str, toml_key: &str) -> Option<&OptionSpec> {
        self.by_toml_path
            .get(section_name)
            .and_then(|section_options| section_options.get(toml_key))
            .and_then(|option_index| self.specs.get(*option_index))
    }

    pub(super) fn supported_option_names(&self) -> BTreeSet<String> {
        self.specs
            .iter()
            .filter(|option_spec| {
                matches!(option_spec.support_level, SupportLevel::Supported | SupportLevel::GExtension)
            })
            .map(|option_spec| option_spec.cli_name.to_string())
            .collect()
    }
}

static OPTION_REGISTRY: OnceLock<OptionRegistry> = OnceLock::new();

pub(super) fn option_registry() -> &'static OptionRegistry {
    OPTION_REGISTRY.get_or_init(|| OptionRegistry::new(options::OPTION_SPECS))
}
