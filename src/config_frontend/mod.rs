use std::fmt;

mod cli;
mod data;
mod defaults;
mod domain;
mod metadata;
mod options;
mod resolve;
mod schema;
mod serialization;
mod validation;

pub use cli::{CliOutcomeData, dispatch_cli};
pub use data::{
    BinaryConfigData, GComputeConfigData, GDiagnosticsConfigData, GOutputConfigData, InputConfigData,
    RegenieConfigData, TraitConfigData,
};
pub use defaults::load_packaged_config_data;
pub use resolve::{from_options, from_toml_path};
pub use serialization::{dumps_toml, write_toml};
pub use validation::validate_config;

const DEFAULT_CONFIG_TOML: &str = include_str!("../g/config.default.toml");
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
