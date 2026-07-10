use super::super::domain::NameList;
use super::super::overlay::ConfigLayer;
use super::super::partial::{
    PartialBinaryConfig, PartialConfig, PartialInputConfig, PartialOutputConfig, PartialTraitConfig,
};
use super::super::{ConfigError, ConfigResult};
use super::parser::RegenieCli;

impl RegenieCli {
    pub(crate) fn into_config_layer(self) -> ConfigResult<ConfigLayer> {
        let partial_config = PartialConfig {
            input: self.input_config()?,
            trait_config: self.trait_config(),
            binary: self.binary_config(),
            output: PartialOutputConfig { out: self.out.clone(), ..PartialOutputConfig::default() },
            ..PartialConfig::default()
        };
        Ok(ConfigLayer::from_partial_config(partial_config))
    }

    fn input_config(&self) -> ConfigResult<PartialInputConfig> {
        Ok(PartialInputConfig {
            bgen: self.input.bgen.clone(),
            sample: self.input.sample.clone(),
            pheno_file: self.input.pheno_file.clone(),
            pheno_columns: canonical_columns(
                "phenoCol",
                &self.input.pheno_col,
                "phenoColList",
                self.input.pheno_col_list.clone(),
            )?,
            covar_file: self.input.covar_file.clone(),
            covar_columns: canonical_columns(
                "covarCol",
                &self.input.covar_col,
                "covarColList",
                self.input.covar_col_list.clone(),
            )?,
            pred: self.input.pred.clone(),
        })
    }

    fn trait_config(&self) -> PartialTraitConfig {
        PartialTraitConfig {
            step: self.trait_options.step,
            trait_type: None,
            qt: self.trait_options.qt.then_some(true),
            bt: self.trait_options.bt.then_some(true),
            bsize: self.trait_options.bsize,
            threads: self.trait_options.threads,
        }
    }

    fn binary_config(&self) -> PartialBinaryConfig {
        PartialBinaryConfig {
            firth: self.binary.firth.then_some(true),
            approx: self.binary.approx.then_some(true),
            p_threshold: self.binary.p_threshold,
            firth_se: self.binary.firth_se.then_some(true),
        }
    }
}

fn canonical_columns(
    repeated_option_name: &str,
    repeated_values: &[String],
    list_option_name: &str,
    list_values: Option<NameList>,
) -> ConfigResult<Option<NameList>> {
    if !repeated_values.is_empty() && list_values.is_some() {
        return Err(ConfigError::new(format!(
            "--{repeated_option_name} and --{list_option_name} cannot be used together."
        )));
    }
    if repeated_values.is_empty() {
        return Ok(list_values);
    }
    NameList::from_values(repeated_values.to_vec())
        .map(Some)
        .map_err(|error| ConfigError::new(format!("--{repeated_option_name}: {error}")))
}
