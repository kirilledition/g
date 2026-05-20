"""Command-line interface for the GWAS engine."""

from __future__ import annotations

import sys
import tomllib
import typing
from pathlib import Path

import click

from g import api, types
from g.interface import config, options


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


def read_raw_toml(path: Path | None) -> dict[str, typing.Any]:
    """Read a TOML file into a raw dictionary."""
    if path is None:
        return {}
    with path.open("rb") as config_file:
        return tomllib.load(config_file)


def explicit_cli_options(context: click.Context, parameters: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """Return only command-line provided options."""
    cli_options: dict[str, typing.Any] = {}
    for parameter_name, parameter_value in parameters.items():
        if parameter_name == "config":
            continue
        source = context.get_parameter_source(parameter_name)
        if source == click.core.ParameterSource.COMMANDLINE:
            cli_options[parameter_name] = parameter_value
    if cli_options.get("qt") is True:
        cli_options["bt"] = False
    if cli_options.get("bt") is True:
        cli_options["qt"] = False
    return cli_options


def build_regenie_config_from_cli(context: click.Context, parameters: dict[str, typing.Any]) -> config.RegenieConfig:
    """Apply built-in defaults, TOML config, and explicit CLI overrides."""
    raw_toml_options = read_raw_toml(parameters.get("config"))
    raw_cli_options = explicit_cli_options(context, parameters)
    try:
        merged_options = config.merge_option_dictionaries(raw_toml_options, raw_cli_options)
        return config.RegenieConfig.from_options(merged_options)
    except ValueError as error:
        raise click.ClickException(str(error)) from error


def path_option(*parameter_declarations: str, **kwargs: typing.Any) -> typing.Callable[[typing.Any], typing.Any]:
    """Create a reusable optional path option decorator."""
    return click.option(*parameter_declarations, type=click.Path(path_type=Path), default=None, **kwargs)


@app.command("regenie", context_settings={"help_option_names": ["--help", "-h"]})
@click.pass_context
@path_option("--config", help="TOML config file.")
@click.option("--step", type=int, default=None, help="REGENIE analysis step. Only step 2 is supported.")
@click.option("--qt/--no-qt", default=None, help="Analyze quantitative traits.")
@click.option("--bt/--no-bt", default=None, help="Analyze binary traits.")
@path_option("--bgen", help="BGEN genotype file.")
@path_option("--sample", help="BGEN sample file.")
@path_option("--phenoFile", "pheno_file", help="Phenotype table.")
@click.option("--phenoCol", "pheno_col", multiple=True, help="Phenotype column.")
@click.option("--phenoColList", "pheno_col_list", default=None, help="Comma-separated phenotype columns.")
@path_option("--covarFile", "covar_file", help="Covariate table.")
@click.option("--covarCol", "covar_col", multiple=True, help="Covariate column.")
@click.option("--covarColList", "covar_col_list", default=None, help="Comma-separated covariate columns.")
@path_option("--pred", help="REGENIE step 1 prediction list.")
@click.option("--bsize", type=int, default=None, help="Variants per processing block.")
@click.option("--threads", type=int, default=None, help="Requested CPU thread count.")
@path_option("--out", help="Output prefix.")
@click.option("--firth/--no-firth", default=None, help="Use Firth fallback.")
@click.option("--approx/--no-approx", default=None, help="Use approximate Firth fallback.")
@click.option("--spa/--no-spa", default=None, help="Use SPA fallback.")
@click.option("--pThresh", "p_threshold", type=float, default=None, help="Fallback p-value threshold.")
@click.option("--firth-se/--no-firth-se", "firth_se", default=None, help="Use Firth-derived standard errors.")
@path_option("--bed", help="Recognized REGENIE option; unsupported by g.")
@path_option("--pgen", help="Recognized REGENIE option; unsupported by g.")
@path_option("--keep", help="Recognized REGENIE option; unsupported by g.")
@path_option("--remove", help="Recognized REGENIE option; unsupported by g.")
@path_option("--extract", help="Recognized REGENIE option; unsupported by g.")
@path_option("--exclude", help="Recognized REGENIE option; unsupported by g.")
@click.option("--catCovarList", "cat_covar_list", default=None, help="Recognized REGENIE option; unsupported by g.")
@click.option("--test", default=None, help="Recognized REGENIE option; unsupported by g.")
@click.option("--t2e", is_flag=True, default=None, help="Recognized REGENIE option; unsupported by g.")
@click.option("--g-device", type=click.Choice([item.value for item in types.Device]), default=None, help="JAX device.")
@click.option("--g-staging-depth", type=int, default=None, help="Native callback staging depth.")
@click.option("--g-variant-limit", type=int, default=None, help="Debug variant cap.")
@click.option(
    "--g-trusted-no-missing-diploid/--no-g-trusted-no-missing-diploid",
    default=None,
    help="Trusted BGEN fast path.",
)
@click.option(
    "--g-trusted-bgen-validation-mode",
    type=click.Choice([item.value for item in types.TrustedBgenValidationMode]),
    default=None,
    help="Trusted BGEN validation mode.",
)
@click.option(
    "--g-sample-key-mode",
    type=click.Choice([item.value for item in types.SampleKeyMode]),
    default=None,
    help="Sample key mode.",
)
@click.option(
    "--g-output-format",
    type=click.Choice([item.value for item in types.OutputFormat]),
    default=None,
    help="Output materialization format.",
)
@path_option("--g-output-run-directory", help="Internal g run directory.")
@click.option("--g-writer-threads", type=int, default=None, help="Output writer thread count.")
@click.option("--g-writer-queue-depth", type=int, default=None, help="Output writer queue depth.")
@click.option("--g-output-chunks-per-arrow-file", type=int, default=None, help="Chunks grouped into one Arrow file.")
@click.option(
    "--g-output-arrow-compression",
    type=click.Choice([item.value for item in types.ArrowCompression]),
    default=None,
    help="Internal Arrow chunk compression.",
)
@click.option("--g-resume/--no-g-resume", default=None, help="Resume a previous run.")
@click.option(
    "--g-resume-mode",
    type=click.Choice([item.value for item in types.ResumeMode]),
    default=None,
    help="Resume validation mode.",
)
@click.option("--g-finalize-parquet/--no-g-finalize-parquet", default=None, help="Finalize Parquet.")
@click.option("--g-firth-batch-size", type=int, default=None, help="Firth batch size.")
@click.option("--g-firth-candidate-capacity", type=int, default=None, help="Firth candidate capacity.")
@click.option("--g-binary-null-maximum-iterations", type=int, default=None, help="Maximum null-logistic iterations.")
@click.option("--g-binary-null-coefficient-tolerance", type=float, default=None, help="Null-logistic tolerance.")
@click.option("--g-firth-maximum-iterations", type=int, default=None, help="Maximum Firth iterations.")
@click.option("--g-firth-gradient-tolerance", type=float, default=None, help="Firth gradient tolerance.")
@click.option("--g-firth-coefficient-tolerance", type=float, default=None, help="Firth coefficient tolerance.")
@click.option("--g-firth-likelihood-tolerance", type=float, default=None, help="Firth likelihood tolerance.")
@click.option("--g-firth-maximum-step-size", type=float, default=None, help="Firth maximum step size.")
@click.option("--g-use-block-firth-math/--no-g-use-block-firth-math", default=None, help="Use block Firth math.")
@click.option("--g-bgen-decode-tile-variant-count", type=int, default=None, help="BGEN decode tile variant count.")
@path_option("--g-jax-cache-dir", help="JAX compilation cache directory.")
@click.option(
    "--g-jax-matmul-precision",
    type=click.Choice([item.value for item in types.JaxMatmulPrecision]),
    default=None,
    help="JAX matmul precision.",
)
@click.option("--g-jax-persistent-cache/--no-g-jax-persistent-cache", default=None, help="Use JAX persistent cache.")
@click.option("--g-jax-persistent-cache-min-entry-size-bytes", type=int, default=None, help="Minimum cache entry size.")
@click.option(
    "--g-jax-persistent-cache-min-compile-time-seconds",
    type=int,
    default=None,
    help="Minimum compile time for persistent cache writes.",
)
@click.option("--g-jax-xla-autotune-cache/--no-g-jax-xla-autotune-cache", default=None, help="Use XLA autotune cache.")
@click.option("--g-jax-transfer-guard/--no-g-jax-transfer-guard", default=None, help="Enable JAX transfer guard.")
@path_option("--g-stage-timings-json", help="Stage timing diagnostics JSON path.")
def run_regenie_command(context: click.Context, **parameters: typing.Any) -> None:
    """Run a REGENIE-compatible step 2 association scan."""
    regenie_config = build_regenie_config_from_cli(context, parameters)
    artifacts = api.regenie(regenie_config)
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
