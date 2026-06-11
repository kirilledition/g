use std::fmt;

mod cli;
mod defaults;
mod domain;
mod options;
mod overlay;
mod partial;
mod resolved;
mod run_validation;
mod toml;
mod validation;

pub use cli::{CliOutcomeData, dispatch_cli};
pub use defaults::load_packaged_config_data;
pub use options::{ConfigOptionMetadata, ConfigOptionValueKind, config_option_metadata};
pub use resolved::{
    BinaryConfigData, GComputeConfigData, GDiagnosticsConfigData, GOutputConfigData, InputConfigData,
    RegenieConfigData, TraitConfigData,
};
pub use run_validation::validate_config_for_run;
pub use toml::{dumps_toml, from_options, from_toml_path, write_toml};
pub use validation::validate_config;

const DEFAULT_CONFIG_TOML: &str = include_str!("config.default.toml");
const OPTION_SCHEMA_VERSION: i64 = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigError {
    message: String,
}

impl ConfigError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
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

pub(crate) type ConfigResult<T> = Result<T, ConfigError>;
