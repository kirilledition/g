use std::path::Path;

use clap::CommandFactory;

use super::ConfigResult;
use super::overlay::resolve_config_layers;
use super::plan_request::compile_run_plan;
use super::run_validation::validate_config_for_run;
use super::toml::{decode_toml_file_layer, dumps_toml};

mod layer;
mod parser;

use parser::{ParsedRegenieCli, RegenieCli, parse_regenie_cli};

#[derive(Debug, PartialEq)]
pub enum CliDispatch {
    Exit { exit_code: i32, stdout: String, stderr: String },
    Run(Box<CompiledCliRun>),
}

#[derive(Debug, PartialEq)]
pub struct CompiledCliRun {
    pub run_plan: g_plan::RunPlan,
    pub effective_config_toml: String,
}

#[must_use]
pub fn dispatch_cli(args: &[String]) -> CliDispatch {
    match dispatch_cli_result(args) {
        Ok(outcome) => outcome,
        Err(error) => CliDispatch::Exit { exit_code: 1, stdout: String::new(), stderr: format!("Error: {error}\n") },
    }
}

fn dispatch_cli_result(args: &[String]) -> ConfigResult<CliDispatch> {
    if args.is_empty() {
        return Ok(CliDispatch::Exit { exit_code: 2, stdout: root_help("g"), stderr: String::new() });
    }
    match args[0].as_str() {
        "--help" | "-h" => Ok(CliDispatch::Exit { exit_code: 0, stdout: root_help("g"), stderr: String::new() }),
        "regenie" => dispatch_regenie_command(&args[1..], "g regenie"),
        command_name => Ok(CliDispatch::Exit {
            exit_code: 2,
            stdout: String::new(),
            stderr: format!("No such command: {command_name}\n"),
        }),
    }
}

fn dispatch_regenie_command(args: &[String], program_name: &'static str) -> ConfigResult<CliDispatch> {
    if args.iter().any(|argument| argument == "--help" || argument == "-h") {
        let mut command = RegenieCli::command();
        command = command.name(program_name);
        return Ok(CliDispatch::Exit {
            exit_code: 0,
            stdout: command.render_help().to_string(),
            stderr: String::new(),
        });
    }
    let ParsedRegenieCli { config_path, cli_layer } = parse_regenie_cli(args, program_name)?;
    let toml_layer = decode_toml_file_layer(config_path.as_deref().map(Path::new))?;
    let config = resolve_config_layers([toml_layer, cli_layer])?;
    validate_config_for_run(&config)?;
    let run_plan = compile_run_plan(&config)?;
    let effective_config_toml = dumps_toml(&config)?;
    Ok(CliDispatch::Run(Box::new(CompiledCliRun { run_plan, effective_config_toml })))
}

fn root_help(program_name: &str) -> String {
    format!(
        "Blazing fast REGENIE step 2 GWAS engine.\n\nUsage: {program_name} <COMMAND> [OPTIONS]\n\nCommands:\n  regenie  Run a REGENIE-compatible step 2 association scan.\n\nOptions:\n  -h, --help  Print help\n"
    )
}
