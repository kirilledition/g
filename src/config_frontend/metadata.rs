use std::collections::BTreeMap;
use std::sync::OnceLock;

use super::options;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OptionValueKind {
    String,
    Integer,
    Float,
    Boolean,
    Path,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct OptionSpec {
    pub(crate) cli_name: &'static str,
    pub(crate) section: &'static str,
    pub(crate) help_text: &'static str,
    pub(crate) value_kind: OptionValueKind,
    pub(crate) multiple: bool,
    pub(crate) is_flag: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct OptionRegistry {
    pub(crate) specs: &'static [OptionSpec],
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
                .insert(option_spec.cli_name, spec_index);
        }

        Self { specs, by_cli_name, by_toml_path }
    }

    pub(crate) fn get_by_cli_name(&self, option_name: &str) -> Option<&OptionSpec> {
        self.by_cli_name.get(option_name).and_then(|option_index| self.specs.get(*option_index))
    }

    pub(crate) fn get_by_toml_path(&self, section_name: &str, toml_key: &str) -> Option<&OptionSpec> {
        self.by_toml_path
            .get(section_name)
            .and_then(|section_options| section_options.get(toml_key))
            .and_then(|option_index| self.specs.get(*option_index))
    }
}

static OPTION_REGISTRY: OnceLock<OptionRegistry> = OnceLock::new();

pub(crate) fn option_registry() -> &'static OptionRegistry {
    OPTION_REGISTRY.get_or_init(|| OptionRegistry::new(options::OPTION_SPECS))
}
