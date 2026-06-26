use std::fmt::Write as _;
use std::sync::OnceLock;

use sha2::{Digest, Sha256};

use super::overlay::resolve_partial_config;
use super::partial::PartialConfig;
use super::resolved::{ConfigProvenance, RegenieConfigData};
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
            let default_config_hash = build_default_config_hash(DEFAULT_CONFIG_TOML);
            Ok(DefaultConfigData { partial_config, default_config_hash })
        })
        .as_ref()
        .map_err(Clone::clone)
}

/// Load packaged defaults as an unvalidated runtime config object.
///
/// # Errors
///
/// Returns an error when the packaged default TOML cannot be decoded.
pub fn load_packaged_config_data() -> ConfigResult<RegenieConfigData> {
    resolve_partial_config(load_default_config_data()?.partial_config.clone(), ConfigProvenance::default(), false)
}

fn build_default_config_hash(default_toml_text: &str) -> String {
    let digest = Sha256::digest(default_toml_text.as_bytes());
    let mut encoded_digest = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(&mut encoded_digest, "{byte:02x}").expect("writing SHA-256 digest to a string");
    }
    encoded_digest
}
