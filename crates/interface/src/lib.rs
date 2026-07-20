#[cfg(not(target_pointer_width = "64"))]
compile_error!("g requires a 64-bit target.");

mod api;
mod cli;
mod defaults;
mod domain;
mod error;
mod overlay;
mod partial;
mod plan_request;
mod resolved;
mod run_validation;
#[cfg(test)]
mod test_support;
#[cfg(test)]
mod tests;
mod toml;
mod validation;

pub use api::*;
pub(crate) use error::{ConfigError, ConfigResult};

const DEFAULT_CONFIG_TOML: &str = include_str!("config.default.toml");
const OPTION_SCHEMA_VERSION: i64 = 0;
