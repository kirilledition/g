use std::path::Path;

use clap::{Arg, ArgAction, Command, error::ErrorKind};

use super::validation::validate_existing_input_paths;
use super::{
    ConfigError, ConfigResult, OptionTable, OptionValue, RegenieConfigData, decode_toml_file_layer,
    from_toml_config_layers, load_default_option_catalog_data, option_dictionary_to_toml_config_layer, option_registry,
};

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
    let ParsedRegenieCli { config_path, cli_options } = parse_regenie_cli(args)?;
    let toml_layer = decode_toml_file_layer(config_path.as_deref().map(Path::new))?;
    let cli_layer = option_dictionary_to_toml_config_layer(&cli_options, "CLI options")?;
    let config = from_toml_config_layers(&load_default_option_catalog_data()?.raw_toml, [toml_layer, cli_layer])?;
    validate_existing_input_paths(&config)?;
    Ok(CliOutcomeData::config(config))
}

#[derive(Clone, Debug, PartialEq)]
struct ParsedRegenieCli {
    config_path: Option<String>,
    cli_options: OptionTable,
}

fn parse_regenie_cli(args: &[String]) -> ConfigResult<ParsedRegenieCli> {
    let registry = option_registry();
    let mut clap_arguments = Vec::with_capacity(args.len() + 1);
    clap_arguments.push("g-regenie".to_string());
    clap_arguments.extend(args.iter().cloned());
    let matches = match build_regenie_clap_command("g-regenie").try_get_matches_from(clap_arguments) {
        Ok(matches) => matches,
        Err(error) if error.kind() == ErrorKind::DisplayHelp => {
            return Err(ConfigError::new(error.to_string()));
        }
        Err(error) => {
            return Err(ConfigError::new(error.to_string()));
        }
    };

    let config_path = matches.get_one::<String>("config").cloned();
    let mut cli_options = OptionTable::new();
    for option_spec in registry.specs {
        if option_spec.is_flag {
            if matches.get_flag(option_spec.cli_name) {
                cli_options.insert(option_spec.cli_name.to_string(), OptionValue::Boolean(true));
            }
            let negative_name = format!("no-{}", option_spec.cli_name);
            if matches.get_flag(&negative_name) {
                cli_options.insert(option_spec.cli_name.to_string(), OptionValue::Boolean(false));
            }
            continue;
        }
        if option_spec.multiple {
            if let Some(values) = matches.get_many::<String>(option_spec.cli_name) {
                cli_options.insert(option_spec.cli_name.to_string(), OptionValue::List(values.cloned().collect()));
            }
            continue;
        }
        if let Some(value) = matches.get_one::<String>(option_spec.cli_name) {
            cli_options.insert(option_spec.cli_name.to_string(), OptionValue::String(value.clone()));
        }
    }

    Ok(ParsedRegenieCli { config_path, cli_options })
}

fn build_regenie_clap_command(program_name: &str) -> Command {
    let mut command = Command::new(leak_string(program_name.to_string()))
        .about("Run a REGENIE-compatible step 2 association scan.")
        .disable_version_flag(true)
        .arg(Arg::new("config").long("config").help("TOML config file.").num_args(1).action(ArgAction::Set));

    let registry = option_registry();
    for option_spec in registry.specs {
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
        command = command.arg(
            Arg::new(option_spec.cli_name)
                .long(option_spec.cli_name)
                .help(option_spec.help_text)
                .num_args(1)
                .action(action),
        );
    }
    command
}

fn leak_string(value: String) -> &'static str {
    Box::leak(value.into_boxed_str())
}

/// Return a user-facing explanation for one option.
///
/// # Errors
///
/// Returns an error when the option name is unknown.
pub fn explain_option(cli_name: &str) -> ConfigResult<String> {
    let Some(option_spec) = option_registry().get_by_cli_name(cli_name) else {
        return Err(ConfigError::new(format!("Unknown option: {cli_name}")));
    };
    Ok(format!("{}: {}. {}", option_spec.cli_name, option_spec.support_level.as_str(), option_spec.help_text))
}

#[must_use]
pub fn iter_explanations() -> Vec<String> {
    option_registry()
        .specs
        .iter()
        .map(|option_spec| {
            format!("{}: {}. {}", option_spec.cli_name, option_spec.support_level.as_str(), option_spec.help_text)
        })
        .collect()
}

fn root_help(program_name: &str) -> String {
    format!(
        "Blazing fast REGENIE step 2 GWAS engine.\n\nUsage: {program_name} <COMMAND> [OPTIONS]\n\nCommands:\n  regenie  Run a REGENIE-compatible step 2 association scan.\n\nOptions:\n  -h, --help  Print help\n"
    )
}
