"""Command-line interface for the GWAS engine."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003

import typer

from g import api, types

run_linear_api = api.linear
run_logistic_api = api.logistic
run_regenie2_linear_api = api.regenie2_linear

app = typer.Typer(
    name="g",
    help="Blazing fast GWAS engine.",
    no_args_is_help=True,
    rich_markup_mode=None,
)


def resolve_chunk_size(requested_chunk_size: int | None, association_mode: types.AssociationMode) -> int:
    """Resolve the effective chunk size for an association mode."""
    if requested_chunk_size is not None:
        return requested_chunk_size
    if association_mode == types.AssociationMode.LOGISTIC:
        return api.DEFAULT_LOGISTIC_CHUNK_SIZE
    return api.DEFAULT_LINEAR_CHUNK_SIZE


def print_success_message(artifacts: api.RunArtifacts) -> None:
    """Print a concise success message for a completed CLI run."""
    if artifacts.sumstats_tsv is not None:
        typer.echo(f"Success. Results saved to {artifacts.sumstats_tsv}")
        return
    if artifacts.output_run_directory is not None:
        typer.echo(f"Success. Chunked run saved to {artifacts.output_run_directory}")
        if artifacts.final_parquet is not None:
            typer.echo(f"Finalized Parquet saved to {artifacts.final_parquet}")
        return
    typer.echo("Success. Run completed.")


@app.command("linear", no_args_is_help=True)
def run_linear_command(
    bfile: Path | None = typer.Option(None, help="PLINK dataset prefix."),
    bgen: Path | None = typer.Option(None, help="BGEN file path."),
    sample: Path | None = typer.Option(
        None,
        help="Optional BGEN sample-file path. Defaults to embedded samples or an adjacent .sample file.",
    ),
    pheno: Path = typer.Option(..., help="Phenotype table path."),
    pheno_name: str = typer.Option(..., "--pheno-name", help="Phenotype column name to analyze."),
    out: Path = typer.Option(..., help="Output prefix or TSV path."),
    covar: Path | None = typer.Option(None, help="Optional covariate table path."),
    covar_names: str | None = typer.Option(None, "--covar-names", help="Comma-separated covariate column names."),
    chunk_size: int | None = typer.Option(None, help="Variants per BED chunk."),
    variant_limit: int | None = typer.Option(None, help="Optional variant cap for debugging or tests."),
    device: types.Device = typer.Option(types.Device.CPU, help="JAX execution device."),
    output_mode: types.OutputMode = typer.Option(types.OutputMode.TSV, help="Output format mode."),
    output_run_directory: Path | None = typer.Option(None, help="Run directory for chunked output mode."),
    resume: bool = typer.Option(  # noqa: FBT001
        default=False,
        help="Resume a previous chunked run.",
    ),
    finalize_parquet: bool = typer.Option(  # noqa: FBT001
        default=False,
        help="Compact committed Arrow chunks into Parquet.",
    ),
) -> None:
    """Run a linear association scan."""
    compute_config = api.ComputeConfig(
        chunk_size=resolve_chunk_size(chunk_size, types.AssociationMode.LINEAR),
        device=device,
        variant_limit=variant_limit,
        output_mode=output_mode,
        output_run_directory=output_run_directory,
        resume=resume,
        finalize_parquet=finalize_parquet,
    )
    artifacts = run_linear_api(
        bfile=bfile,
        bgen=bgen,
        sample=sample,
        pheno=pheno,
        pheno_name=pheno_name,
        out=out,
        covar=covar,
        covar_names=api.parse_covariate_name_list(covar_names),
        compute=compute_config,
        solver=api.LinearConfig(),
    )
    print_success_message(artifacts)


@app.command("logistic", no_args_is_help=True)
def run_logistic_command(
    bfile: Path | None = typer.Option(None, help="PLINK dataset prefix."),
    bgen: Path | None = typer.Option(None, help="BGEN file path."),
    sample: Path | None = typer.Option(
        None,
        help="Optional BGEN sample-file path. Defaults to embedded samples or an adjacent .sample file.",
    ),
    pheno: Path = typer.Option(..., help="Phenotype table path."),
    pheno_name: str = typer.Option(..., "--pheno-name", help="Phenotype column name to analyze."),
    out: Path = typer.Option(..., help="Output prefix or TSV path."),
    covar: Path | None = typer.Option(None, help="Optional covariate table path."),
    covar_names: str | None = typer.Option(None, "--covar-names", help="Comma-separated covariate column names."),
    chunk_size: int | None = typer.Option(None, help="Variants per BED chunk."),
    variant_limit: int | None = typer.Option(None, help="Optional variant cap for debugging or tests."),
    device: types.Device = typer.Option(types.Device.CPU, help="JAX execution device."),
    max_iterations: int = typer.Option(50, help="Maximum logistic IRLS iterations."),
    tolerance: float = typer.Option(1.0e-8, help="Logistic convergence tolerance."),
    firth_fallback: bool = typer.Option(  # noqa: FBT001
        True,  # noqa: FBT003
        "--firth/--no-firth",
        help="Use Firth fallback when needed.",
    ),
    output_mode: types.OutputMode = typer.Option(types.OutputMode.TSV, help="Output format mode."),
    output_run_directory: Path | None = typer.Option(None, help="Run directory for chunked output mode."),
    resume: bool = typer.Option(  # noqa: FBT001
        default=False,
        help="Resume a previous chunked run.",
    ),
    finalize_parquet: bool = typer.Option(  # noqa: FBT001
        default=False,
        help="Compact committed Arrow chunks into Parquet.",
    ),
) -> None:
    """Run a logistic association scan."""
    compute_config = api.ComputeConfig(
        chunk_size=resolve_chunk_size(chunk_size, types.AssociationMode.LOGISTIC),
        device=device,
        variant_limit=variant_limit,
        output_mode=output_mode,
        output_run_directory=output_run_directory,
        resume=resume,
        finalize_parquet=finalize_parquet,
    )
    solver_config = api.LogisticConfig(
        max_iterations=max_iterations,
        tolerance=tolerance,
        firth_fallback=firth_fallback,
    )
    artifacts = run_logistic_api(
        bfile=bfile,
        bgen=bgen,
        sample=sample,
        pheno=pheno,
        pheno_name=pheno_name,
        out=out,
        covar=covar,
        covar_names=api.parse_covariate_name_list(covar_names),
        compute=compute_config,
        solver=solver_config,
    )
    print_success_message(artifacts)


@app.command("regenie2-linear", no_args_is_help=True)
def run_regenie2_linear_command(
    bgen: Path = typer.Option(..., help="BGEN file path."),
    sample: Path | None = typer.Option(
        None,
        help="Optional BGEN sample-file path. Defaults to embedded samples or an adjacent .sample file.",
    ),
    pheno: Path = typer.Option(..., help="Phenotype table path."),
    pheno_name: str = typer.Option(..., "--pheno-name", help="Phenotype column name to analyze."),
    out: Path = typer.Option(..., help="Output prefix or TSV path."),
    covar: Path | None = typer.Option(None, help="Optional covariate table path."),
    covar_names: str | None = typer.Option(None, "--covar-names", help="Comma-separated covariate column names."),
    pred: Path = typer.Option(..., help="REGENIE step 1 _pred.list file path."),
    chunk_size: int | None = typer.Option(None, help="Variants per chunk."),
    variant_limit: int | None = typer.Option(None, help="Optional variant cap for debugging or tests."),
    device: types.Device = typer.Option(types.Device.CPU, help="JAX execution device."),
    output_mode: types.OutputMode = typer.Option(types.OutputMode.TSV, help="Output format mode."),
    output_run_directory: Path | None = typer.Option(None, help="Run directory for chunked output mode."),
    resume: bool = typer.Option(  # noqa: FBT001
        default=False,
        help="Resume a previous chunked run.",
    ),
    finalize_parquet: bool = typer.Option(  # noqa: FBT001
        default=False,
        help="Compact committed Arrow chunks into Parquet.",
    ),
) -> None:
    """Run a REGENIE step 2 linear association scan."""
    compute_config = api.ComputeConfig(
        chunk_size=resolve_chunk_size(chunk_size, types.AssociationMode.REGENIE2_LINEAR),
        device=device,
        variant_limit=variant_limit,
        output_mode=output_mode,
        output_run_directory=output_run_directory,
        resume=resume,
        finalize_parquet=finalize_parquet,
    )
    artifacts = run_regenie2_linear_api(
        bgen=bgen,
        sample=sample,
        pheno=pheno,
        pheno_name=pheno_name,
        out=out,
        covar=covar,
        covar_names=api.parse_covariate_name_list(covar_names),
        pred=pred,
        compute=compute_config,
        solver=api.Regenie2LinearConfig(),
    )
    print_success_message(artifacts)


def main() -> None:
    """Run the GWAS CLI."""
    app()
