use std::num::NonZeroU32;

use clap::{ArgAction, Args, Parser};
use g_plan as plan;

use super::super::overlay::ConfigLayer;
use super::super::{ConfigError, ConfigResult};

pub(crate) struct ParsedRegenieCli {
    pub(crate) config_path: Option<String>,
    pub(crate) cli_layer: ConfigLayer,
}

pub(crate) fn parse_regenie_cli(args: &[String], program_name: &'static str) -> ConfigResult<ParsedRegenieCli> {
    let mut parsed_cli =
        RegenieCli::try_parse_from(std::iter::once(program_name).chain(args.iter().map(String::as_str)))
            .map_err(|error| ConfigError::new(error.to_string()))?;
    let config_path = parsed_cli.config.take();
    let cli_layer = parsed_cli.into_config_layer()?;
    Ok(ParsedRegenieCli { config_path, cli_layer })
}

#[derive(Parser)]
#[command(about = "Run a REGENIE-compatible step 2 association scan.", disable_version_flag = true)]
pub(crate) struct RegenieCli {
    #[arg(long = "config", help_heading = "Config")]
    pub(crate) config: Option<String>,
    #[command(flatten)]
    pub(crate) trait_options: TraitCli,
    #[command(flatten)]
    pub(crate) input: InputCli,
    #[command(flatten)]
    pub(crate) binary: BinaryCli,
    #[arg(long = "out", help_heading = "Output")]
    pub(crate) out: Option<String>,
}

#[derive(Args)]
pub(crate) struct TraitCli {
    #[arg(long = "qt", action = ArgAction::SetTrue, help_heading = "Trait")]
    pub(crate) qt: bool,
    #[arg(long = "bt", action = ArgAction::SetTrue, help_heading = "Trait")]
    pub(crate) bt: bool,
    #[arg(long = "bsize", help_heading = "Trait")]
    pub(crate) bsize: Option<NonZeroU32>,
}

#[derive(Args)]
pub(crate) struct InputCli {
    #[arg(long = "bgen", help_heading = "Input")]
    pub(crate) bgen: Option<String>,
    #[arg(long = "sample", help_heading = "Input")]
    pub(crate) sample: Option<String>,
    #[arg(long = "phenoFile", help_heading = "Input")]
    pub(crate) pheno_file: Option<String>,
    #[arg(long = "phenoCol", action = ArgAction::Append, help_heading = "Input")]
    pub(crate) pheno_col: Vec<String>,
    #[arg(long = "covarFile", help_heading = "Input")]
    pub(crate) covar_file: Option<String>,
    #[arg(long = "covarCol", action = ArgAction::Append, help_heading = "Input")]
    pub(crate) covar_col: Vec<String>,
    #[arg(long = "pred", help_heading = "Input")]
    pub(crate) pred: Option<String>,
}

#[derive(Args)]
pub(crate) struct BinaryCli {
    #[arg(long = "binary-fallback", help_heading = "Binary")]
    pub(crate) fallback_method: Option<plan::BinaryFallbackMethod>,
    #[arg(long = "pThresh", help_heading = "Binary")]
    pub(crate) p_threshold: Option<plan::Probability>,
    #[arg(long = "firth-se", action = ArgAction::SetTrue, help_heading = "Binary")]
    pub(crate) firth_se: bool,
}
