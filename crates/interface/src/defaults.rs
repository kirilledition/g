use std::sync::OnceLock;

use sha2::{Digest, Sha256};

use super::partial::PartialConfig;
use super::toml::partial_config_from_toml_text;
use super::{ConfigError, ConfigResult, DEFAULT_CONFIG_TOML};

#[derive(Clone, Debug)]
pub(crate) struct DefaultConfigData {
    pub(crate) partial_config: PartialConfig,
    pub(crate) default_config_hash: String,
}

static DEFAULT_CONFIG: OnceLock<Result<DefaultConfigData, ConfigError>> = OnceLock::new();

pub(crate) fn load_default_config_data() -> ConfigResult<&'static DefaultConfigData> {
    DEFAULT_CONFIG
        .get_or_init(|| {
            let partial_config = partial_config_from_toml_text(DEFAULT_CONFIG_TOML, "config.default.toml")?;
            let default_config_hash = hex::encode(Sha256::digest(DEFAULT_CONFIG_TOML.as_bytes()));
            Ok(DefaultConfigData { partial_config, default_config_hash })
        })
        .as_ref()
        .map_err(Clone::clone)
}
