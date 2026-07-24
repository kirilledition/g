use super::super::domain::NameList;
use super::super::overlay::ConfigLayer;
use super::super::partial::{
    PartialBinaryConfig, PartialConfig, PartialInputConfig, PartialOutputConfig, PartialTraitConfig,
};
use super::super::{ConfigError, ConfigResult};
use super::parser::RegenieCli;

impl RegenieCli {
    pub(crate) fn into_config_layer(self) -> ConfigResult<ConfigLayer> {
        let RegenieCli { trait_options, input, binary, out, recover_output_attempt, fenced_output_owner_claim, .. } =
            self;
        let partial_config = PartialConfig {
            input: PartialInputConfig {
                bgen: input.bgen,
                bgen_content_sha256: None,
                sample: input.sample,
                pheno_file: input.pheno_file,
                pheno_columns: canonical_columns("phenoCol", input.pheno_col)?,
                covar_file: input.covar_file,
                covar_columns: canonical_columns("covarCol", input.covar_col)?,
                pred: input.pred,
            },
            trait_config: PartialTraitConfig {
                trait_type: None,
                qt: trait_options.qt.then_some(true),
                bt: trait_options.bt.then_some(true),
                bsize: trait_options.bsize,
            },
            binary: PartialBinaryConfig {
                fallback_method: binary.fallback_method,
                p_threshold: binary.p_threshold,
                firth_se: binary.firth_se.then_some(true),
            },
            output: PartialOutputConfig {
                out,
                recover_attempt: recover_output_attempt,
                fenced_owner_claim_id: fenced_output_owner_claim,
                ..PartialOutputConfig::default()
            },
            ..PartialConfig::default()
        };
        Ok(ConfigLayer::from_partial_config(partial_config))
    }
}

fn canonical_columns(repeated_option_name: &str, repeated_values: Vec<String>) -> ConfigResult<Option<NameList>> {
    if repeated_values.is_empty() {
        return Ok(None);
    }
    NameList::from_values(repeated_values)
        .map(Some)
        .map_err(|error| ConfigError::new(format!("--{repeated_option_name}: {error}")))
}
