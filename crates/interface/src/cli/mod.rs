use std::path::Path;

use clap::CommandFactory;

use super::ConfigResult;
use super::overlay::resolve_config_layers;
use super::resolved::RegenieConfigData;
use super::run_validation::validate_config_for_run;
use super::toml::decode_toml_file_layer;

mod layer;
mod parser;

use parser::{ParsedRegenieCli, RegenieCli, parse_regenie_cli};

#[derive(Clone, Debug, PartialEq)]
pub struct CliOutcomeData {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub config: Option<RegenieConfigData>,
}

impl CliOutcomeData {
    fn output(exit_code: i32, stdout: impl Into<String>, stderr: impl Into<String>) -> Self {
        Self { exit_code, stdout: stdout.into(), stderr: stderr.into(), config: None }
    }

    fn config(config: RegenieConfigData) -> Self {
        Self { exit_code: 0, stdout: String::new(), stderr: String::new(), config: Some(config) }
    }
}

#[must_use]
pub fn dispatch_cli(args: &[String]) -> CliOutcomeData {
    match dispatch_cli_result(args) {
        Ok(outcome) => outcome,
        Err(error) => CliOutcomeData::output(1, String::new(), format!("Error: {}\n", error.message())),
    }
}

fn dispatch_cli_result(args: &[String]) -> ConfigResult<CliOutcomeData> {
    if args.is_empty() {
        return Ok(CliOutcomeData::output(2, root_help("g"), String::new()));
    }
    match args[0].as_str() {
        "--help" | "-h" => Ok(CliOutcomeData::output(0, root_help("g"), String::new())),
        "regenie" => dispatch_regenie_command(&args[1..], "g regenie"),
        command_name => Ok(CliOutcomeData::output(2, String::new(), format!("No such command: {command_name}\n"))),
    }
}

fn dispatch_regenie_command(args: &[String], program_name: &'static str) -> ConfigResult<CliOutcomeData> {
    if args.iter().any(|argument| argument == "--help" || argument == "-h") {
        let mut command = RegenieCli::command();
        command = command.name(program_name);
        return Ok(CliOutcomeData::output(0, command.render_help().to_string(), String::new()));
    }
    let ParsedRegenieCli { config_path, cli_layer } = parse_regenie_cli(args, program_name)?;
    let toml_layer = decode_toml_file_layer(config_path.as_deref().map(Path::new))?;
    let config = resolve_config_layers([toml_layer, cli_layer])?;
    validate_config_for_run(&config)?;
    Ok(CliOutcomeData::config(config))
}

fn root_help(program_name: &str) -> String {
    format!(
        "Blazing fast REGENIE step 2 GWAS engine.\n\nUsage: {program_name} <COMMAND> [OPTIONS]\n\nCommands:\n  regenie  Run a REGENIE-compatible step 2 association scan.\n\nOptions:\n  -h, --help  Print help\n"
    )
}
