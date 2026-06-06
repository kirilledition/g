"""Command-line interface for the GWAS engine."""

from __future__ import annotations

import sys
import typing
from pathlib import Path

import click

from g import api, types
from g.engine import shutdown
from g.interface import config, config_layers, defaults, options


class NaturalOrderGroup(click.Group):
    """Click group that keeps command insertion order in help output."""

    def list_commands(self, ctx: click.Context) -> list[str]:
        """List commands in registration order."""
        del ctx
        return list(self.commands)


@click.group(
    cls=NaturalOrderGroup,
    no_args_is_help=True,
    context_settings={"help_option_names": ["--help", "-h"]},
)
def app() -> None:
    """Blazing fast REGENIE step 2 GWAS engine."""


def print_success_message(artifacts: api.RunArtifacts) -> None:
    """Print a concise success message for a completed CLI run."""
    if artifacts.phenotype_artifacts:
        for phenotype_artifact in artifacts.phenotype_artifacts:
            print_success_message(phenotype_artifact)
        return
    if artifacts.output_run_directory is not None:
        click.echo(f"Success. Chunked run saved to {artifacts.output_run_directory}")
        if artifacts.final_parquet is not None:
            click.echo(f"Finalized Parquet saved to {artifacts.final_parquet}")
        return
    click.echo("Success. Run completed.")


def print_warm_cache_message(report: typing.Any) -> None:
    """Print a concise success message for cache warming."""
    warmed_shape_descriptions = ", ".join(
        f"({shape.sample_count}, {shape.variant_count})" for shape in report.warmed_shapes
    )
    click.echo(f"Success. Warmed JAX cache shapes: {warmed_shape_descriptions}")


def print_interrupted_message(shutdown_request: shutdown.GracefulShutdownRequested) -> None:
    """Print a concise interrupted-run message."""
    click.echo(
        f"Interrupted by {shutdown_request.signal_name}. "
        "Flushed queued chunks and saved committed output for --resume.",
        err=True,
    )


def resolve_trusted_bgen_validation_mode(
    *,
    validate_trusted_bgen: bool,
    assume_trusted_bgen_validated: bool,
) -> types.TrustedBgenValidationMode:
    """Resolve trusted BGEN validation mode from CLI flags."""
    if validate_trusted_bgen and assume_trusted_bgen_validated:
        message = "--validate-trusted-bgen and --assume-trusted-bgen-validated are mutually exclusive."
        raise click.BadParameter(message)
    if assume_trusted_bgen_validated:
        return types.TrustedBgenValidationMode.ASSUME_VALIDATED
    if validate_trusted_bgen:
        return types.TrustedBgenValidationMode.FORCE_VALIDATE
    return types.TrustedBgenValidationMode.CACHE_ON_MISS


def read_typed_toml_mapping(path: Path | None) -> dict[str, typing.Any]:
    """Read a TOML file through the typed schema into built-in containers."""
    if path is None:
        return {}
    return config_layers.toml_config_to_builtin_mapping(config_layers.decode_toml_file(path))


def explicit_cli_options(context: click.Context, parameters: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """Return only command-line provided options."""
    cli_options: dict[str, typing.Any] = {}
    for parameter_name, parameter_value in parameters.items():
        if parameter_name == "config":
            continue
        source = context.get_parameter_source(parameter_name)
        if source == click.core.ParameterSource.COMMANDLINE:
            cli_options[parameter_name] = parameter_value
    return cli_options


def build_regenie_config_from_cli(context: click.Context, parameters: dict[str, typing.Any]) -> config.RegenieConfig:
    """Apply built-in defaults, TOML config, and explicit CLI overrides."""
    try:
        toml_layer = config_layers.decode_toml_file_layer(parameters.get("config"))
        cli_layer = config_layers.option_dictionary_to_toml_config_layer(
            explicit_cli_options(context, parameters),
            source="CLI options",
        )
        return config.from_toml_config_layers(
            base_config=defaults.load_default_option_catalog().toml_config,
            explicit_layers=(toml_layer, cli_layer),
        )
    except ValueError as error:
        raise click.ClickException(str(error)) from error


def path_option(*parameter_declarations: str, **kwargs: typing.Any) -> typing.Callable[[typing.Any], typing.Any]:
    """Create a reusable optional path option decorator."""
    return click.option(*parameter_declarations, type=click.Path(path_type=Path), default=None, **kwargs)


def click_type_for_option(option_spec: options.OptionSpec) -> typing.Any:
    """Return the Click type declared by an option spec."""
    if option_spec.accepted_values:
        return click.Choice(option_spec.accepted_values)
    if option_spec.type == options.OptionValueType.PATH:
        return click.Path(path_type=Path)
    if option_spec.type == options.OptionValueType.INTEGER:
        return int
    if option_spec.type == options.OptionValueType.FLOAT:
        return float
    return None


def click_declarations_for_option(option_spec: options.OptionSpec) -> tuple[str, ...]:
    """Return Click parameter declarations for an option spec."""
    if option_spec.cli_flags:
        return option_spec.cli_flags
    return (f"--{option_spec.name}", option_spec.destination)


def click_keyword_arguments_for_option(option_spec: options.OptionSpec) -> dict[str, typing.Any]:
    """Return Click keyword arguments for an option spec."""
    keyword_arguments: dict[str, typing.Any] = {"help": option_spec.help_text}
    click_type = click_type_for_option(option_spec)
    if click_type is not None:
        keyword_arguments["type"] = click_type
    if not option_spec.multiple:
        keyword_arguments["default"] = None
    if option_spec.multiple:
        keyword_arguments["multiple"] = True
    if option_spec.is_flag and not any("/" in flag for flag in option_spec.cli_flags):
        keyword_arguments["is_flag"] = True
    return keyword_arguments


def regenie_options(function: typing.Callable[..., typing.Any]) -> typing.Callable[..., typing.Any]:
    """Apply g regenie options from the central option table."""
    decorated_function = function
    for option_spec in reversed(options.OPTION_SPECS):
        decorated_function = click.option(
            *click_declarations_for_option(option_spec),
            **click_keyword_arguments_for_option(option_spec),
        )(decorated_function)
    return decorated_function


@app.command("regenie", context_settings={"help_option_names": ["--help", "-h"]})
@click.pass_context
@path_option("--config", help="TOML config file.")
@regenie_options
def run_regenie_command(context: click.Context, **parameters: typing.Any) -> None:
    """Run a REGENIE-compatible step 2 association scan."""
    regenie_config = build_regenie_config_from_cli(context, parameters)
    try:
        with shutdown.install_graceful_shutdown_handlers():
            artifacts = api.regenie(regenie_config)
    except shutdown.GracefulShutdownRequested as shutdown_request:
        print_interrupted_message(shutdown_request)
        raise click.exceptions.Exit(shutdown_request.exit_code) from shutdown_request
    print_success_message(artifacts)


@app.group("config", cls=NaturalOrderGroup)
def config_group() -> None:
    """Inspect and manage g regenie TOML configs."""


@config_group.command("init")
@path_option("--out", help="Config file to write. Defaults to stdout.")
def config_init_command(out: Path | None) -> None:
    """Write a starter g regenie TOML config."""
    template = config.build_template()
    if out is None:
        click.echo(template, nl=False)
        return
    out.write_text(template, encoding="utf-8")
    click.echo(f"Wrote {out}")


@config_group.command("validate")
@click.argument("config_path", type=click.Path(path_type=Path))
def config_validate_command(config_path: Path) -> None:
    """Validate a g regenie TOML config."""
    try:
        config.RegenieConfig.from_toml(config_path)
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    click.echo("Config is valid.")


@config_group.command("explain")
@click.argument("option_name", required=False)
def config_explain_command(option_name: str | None) -> None:
    """Explain supported and recognized options."""
    if option_name is None:
        for explanation in options.iter_explanations():
            click.echo(explanation)
        return
    normalized_name = option_name.removeprefix("--")
    try:
        click.echo(options.explain_option(normalized_name))
    except KeyError as error:
        raise click.ClickException(f"Unknown option: {option_name}") from error


def regenie_main() -> None:
    """Run the direct g-regenie executable."""
    run_regenie_command.main(args=sys.argv[1:], prog_name="g-regenie", standalone_mode=True)


def main() -> None:
    """Run the GWAS CLI."""
    app()
