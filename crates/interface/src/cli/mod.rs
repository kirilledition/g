use std::io::ErrorKind;
use std::path::{Component, Path, PathBuf};

use clap::CommandFactory;

use super::ConfigResult;
use super::overlay::{ConfigLayer, resolve_config_layers};
use super::plan_request::compile_run_plan;
use super::run_validation::validate_config_for_run;
use super::toml::{decode_toml_file_layer, dumps_toml};

mod layer;
mod parser;

use parser::{BatchCli, ParsedRegenieCli, RegenieCli, parse_batch_cli, parse_regenie_cli};

#[derive(Debug, PartialEq)]
pub enum CliDispatch {
    Exit { exit_code: i32, stdout: String, stderr: String },
    Runs(Vec<CompiledCliRun>),
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
        "batch" => dispatch_batch_command(&args[1..], "g batch"),
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
    compile_run(config_path.as_deref(), cli_layer).map(|compiled_run| CliDispatch::Runs(vec![compiled_run]))
}

fn dispatch_batch_command(args: &[String], program_name: &'static str) -> ConfigResult<CliDispatch> {
    if args.iter().any(|argument| argument == "--help" || argument == "-h") {
        let mut command = BatchCli::command();
        command = command.name(program_name);
        return Ok(CliDispatch::Exit {
            exit_code: 0,
            stdout: command.render_help().to_string(),
            stderr: String::new(),
        });
    }
    let config_paths = parse_batch_cli(args, program_name)?;
    let mut compiled_runs = Vec::with_capacity(config_paths.len());
    for (config_index, config_path) in config_paths.iter().enumerate() {
        let compiled_run = compile_run(Some(config_path), ConfigLayer::default()).map_err(|error| {
            super::ConfigError::new(format!(
                "Batch config {} ({}): {error}",
                config_index + 1,
                Path::new(config_path).display(),
            ))
        })?;
        compiled_runs.push(compiled_run);
    }
    validate_disjoint_output_roots(&compiled_runs)?;
    Ok(CliDispatch::Runs(compiled_runs))
}

fn compile_run(config_path: Option<&str>, cli_layer: ConfigLayer) -> ConfigResult<CompiledCliRun> {
    let toml_layer = decode_toml_file_layer(config_path.map(Path::new))?;
    let config = resolve_config_layers([toml_layer, cli_layer])?;
    validate_config_for_run(&config)?;
    let run_plan = compile_run_plan(&config)?;
    let effective_config_toml = dumps_toml(&config)?;
    Ok(CompiledCliRun { run_plan, effective_config_toml })
}

fn validate_disjoint_output_roots(compiled_runs: &[CompiledCliRun]) -> ConfigResult<()> {
    let mut indexed_output_roots: Vec<(usize, PathBuf)> = Vec::with_capacity(compiled_runs.len());
    for (config_index, compiled_run) in compiled_runs.iter().enumerate() {
        let output_root = resolve_output_root(Path::new(&compiled_run.run_plan.output.output_run_root))?;
        for (existing_config_index, existing_output_root) in &indexed_output_roots {
            if output_root.starts_with(existing_output_root) || existing_output_root.starts_with(&output_root) {
                return Err(super::ConfigError::new(format!(
                    "Batch configs {} ({}) and {} ({}) resolve to equal or nested output run roots.",
                    existing_config_index + 1,
                    existing_output_root.display(),
                    config_index + 1,
                    output_root.display(),
                )));
            }
        }
        indexed_output_roots.push((config_index, output_root));
    }
    Ok(())
}

fn resolve_output_root(path: &Path) -> ConfigResult<PathBuf> {
    let absolute_path = std::path::absolute(path).map_err(|error| {
        super::ConfigError::new(format!("Failed to resolve output run root {}: {error}", path.display()))
    })?;
    let mut normalized_path = PathBuf::new();
    for component in absolute_path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized_path.pop();
            }
            Component::Prefix(_) | Component::RootDir | Component::Normal(_) => {
                normalized_path.push(component.as_os_str());
            }
        }
    }
    for existing_ancestor in normalized_path.ancestors() {
        match existing_ancestor.canonicalize() {
            Ok(canonical_ancestor) => {
                let missing_suffix =
                    normalized_path.strip_prefix(existing_ancestor).expect("an ancestor is always a path prefix");
                return Ok(canonical_ancestor.join(missing_suffix));
            }
            Err(error) if matches!(error.kind(), ErrorKind::NotFound | ErrorKind::NotADirectory) => {}
            Err(error) => {
                return Err(super::ConfigError::new(format!(
                    "Failed to resolve output run root {} at existing ancestor {}: {error}",
                    path.display(),
                    existing_ancestor.display(),
                )));
            }
        }
    }
    Err(super::ConfigError::new(format!(
        "Failed to resolve an existing ancestor for output run root {}.",
        path.display(),
    )))
}

fn root_help(program_name: &str) -> String {
    format!(
        "Blazing fast REGENIE step 2 GWAS engine.\n\nUsage: {program_name} <COMMAND> [OPTIONS]\n\nCommands:\n  regenie  Run a REGENIE-compatible step 2 association scan.\n  batch    Run complete configurations sequentially in one process.\n\nOptions:\n  -h, --help  Print help\n"
    )
}
