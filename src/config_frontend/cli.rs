use std::path::Path;

use clap::{Arg, ArgAction, Command, builder::ValueParser};
use toml::{Table, Value};

use super::data::RegenieConfigData;
use super::domain::parse_cli_option_value;
use super::metadata::{OptionSpec, OptionValueKind, option_registry};
use super::resolve::{ConfigLayer, decode_toml_file_layer, resolve_config_layers, set_cli_option_value};
use super::{ConfigError, ConfigResult};

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
pub fn dispatch_cli(args: &[String], direct_regenie: bool) -> CliOutcomeData {
    match dispatch_cli_result(args, direct_regenie) {
        Ok(outcome) => outcome,
        Err(error) => CliOutcomeData::output(1, String::new(), format!("Error: {}\n", error.message())),
    }
}

fn dispatch_cli_result(args: &[String], direct_regenie: bool) -> ConfigResult<CliOutcomeData> {
    if direct_regenie {
        return dispatch_regenie_command(args, "g-regenie");
    }
    if args.is_empty() {
        return Ok(CliOutcomeData::output(2, root_help("g"), String::new()));
    }
    match args[0].as_str() {
        "--help" | "-h" => Ok(CliOutcomeData::output(0, root_help("g"), String::new())),
        "regenie" => dispatch_regenie_command(&args[1..], "g regenie"),
        command_name => Ok(CliOutcomeData::output(2, String::new(), format!("No such command: {command_name}\n"))),
    }
}

fn dispatch_regenie_command(args: &[String], program_name: &str) -> ConfigResult<CliOutcomeData> {
    if args.iter().any(|argument| argument == "--help" || argument == "-h") {
        let mut command = build_regenie_clap_command(program_name);
        return Ok(CliOutcomeData::output(0, command.render_help().to_string(), String::new()));
    }
    let ParsedRegenieCli { config_path, cli_layer } = parse_regenie_cli(args, program_name)?;
    let toml_layer = decode_toml_file_layer(config_path.as_deref().map(Path::new))?;
    let config = resolve_config_layers([toml_layer, cli_layer])?;
    Ok(CliOutcomeData::config(config))
}

#[derive(Clone, Debug)]
struct ParsedRegenieCli {
    config_path: Option<String>,
    cli_layer: ConfigLayer,
}

fn parse_regenie_cli(args: &[String], program_name: &str) -> ConfigResult<ParsedRegenieCli> {
    let registry = option_registry();
    let mut clap_arguments = Vec::with_capacity(args.len() + 1);
    clap_arguments.push(program_name.to_string());
    clap_arguments.extend(args.iter().cloned());
    let matches = build_regenie_clap_command(program_name)
        .try_get_matches_from(clap_arguments)
        .map_err(|error| ConfigError::new(error.to_string()))?;

    let config_path = matches.get_one::<String>("config").cloned();
    let mut toml_table = Table::new();
    for option_spec in registry.specs {
        if option_spec.is_flag {
            if matches.get_flag(option_spec.cli_name) {
                set_cli_option_value(&mut toml_table, option_spec.cli_name, Value::Boolean(true))?;
            }
            let negative_name = format!("no-{}", option_spec.cli_name);
            if matches.get_flag(&negative_name) {
                set_cli_option_value(&mut toml_table, option_spec.cli_name, Value::Boolean(false))?;
            }
            continue;
        }
        if option_spec.multiple {
            if let Some(values) = matches.get_many::<String>(option_spec.cli_name) {
                let toml_values = values.cloned().map(Value::String).collect::<Vec<_>>();
                set_cli_option_value(&mut toml_table, option_spec.cli_name, Value::Array(toml_values))?;
            }
            continue;
        }
        if let Some(value) = matches.get_one::<String>(option_spec.cli_name) {
            set_cli_option_value(&mut toml_table, option_spec.cli_name, cli_toml_value(option_spec, value)?)?;
        }
    }

    Ok(ParsedRegenieCli { config_path, cli_layer: ConfigLayer::from_toml_table(&toml_table, "CLI options")? })
}

fn build_regenie_clap_command(program_name: &str) -> Command {
    let mut command = Command::new(leak_string(program_name.to_string()))
        .about("Run a REGENIE-compatible step 2 association scan.")
        .disable_version_flag(true)
        .arg(Arg::new("config").long("config").help("TOML config file.").num_args(1).action(ArgAction::Set));

    for option_spec in option_registry().specs {
        if option_spec.is_flag {
            command = command.arg(
                Arg::new(option_spec.cli_name)
                    .long(option_spec.cli_name)
                    .help(option_spec.help_text)
                    .action(ArgAction::SetTrue),
            );
            let negative_name = format!("no-{}", option_spec.cli_name);
            command = command.arg(
                Arg::new(leak_string(negative_name.clone()))
                    .long(leak_string(negative_name))
                    .hide(true)
                    .action(ArgAction::SetTrue),
            );
            continue;
        }
        let action = if option_spec.multiple { ArgAction::Append } else { ArgAction::Set };
        let mut argument = Arg::new(option_spec.cli_name)
            .long(option_spec.cli_name)
            .help(option_spec.help_text)
            .num_args(1)
            .action(action)
            .value_parser(cli_value_parser(option_spec));
        if matches!(option_spec.value_kind, OptionValueKind::Integer | OptionValueKind::Float) {
            argument = argument.allow_negative_numbers(true);
        }
        command = command.arg(argument);
    }
    command
}

fn cli_value_parser(option_spec: &'static OptionSpec) -> ValueParser {
    ValueParser::new(move |raw_value: &str| -> Result<String, String> {
        parse_cli_option_value(option_spec.cli_name, raw_value)
            .map(|()| raw_value.to_string())
            .map_err(|error| error.message().to_string())
    })
}

fn cli_toml_value(option_spec: &OptionSpec, raw_value: &str) -> ConfigResult<Value> {
    match option_spec.value_kind {
        OptionValueKind::String | OptionValueKind::Path => Ok(Value::String(raw_value.to_string())),
        OptionValueKind::Integer => raw_value
            .parse::<i64>()
            .map(Value::Integer)
            .map_err(|_| ConfigError::new(format!("Invalid value for --{}: {raw_value:?}.", option_spec.cli_name))),
        OptionValueKind::Float => raw_value
            .parse::<f64>()
            .map(Value::Float)
            .map_err(|_| ConfigError::new(format!("Invalid value for --{}: {raw_value:?}.", option_spec.cli_name))),
        OptionValueKind::Boolean => {
            Err(ConfigError::new(format!("Boolean option --{} must be passed as a flag.", option_spec.cli_name)))
        }
    }
}

fn leak_string(value: String) -> &'static str {
    Box::leak(value.into_boxed_str())
}

fn root_help(program_name: &str) -> String {
    format!(
        "Blazing fast REGENIE step 2 GWAS engine.\n\nUsage: {program_name} <COMMAND> [OPTIONS]\n\nCommands:\n  regenie  Run a REGENIE-compatible step 2 association scan.\n\nOptions:\n  -h, --help  Print help\n"
    )
}
