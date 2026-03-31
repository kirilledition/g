"""Command-line interface for the GWAS engine."""

from __future__ import annotations

import pathlib  # noqa: TC003

import typer

from g.api import (
    DEFAULT_LINEAR_CHUNK_SIZE,
    DEFAULT_LOGISTIC_CHUNK_SIZE,
    ComputeConfig,
    LinearConfig,
    LogisticConfig,
    RunArtifacts,
    parse_covariate_name_list,
)
from g.api import (
    linear as run_linear_api,
)
from g.api import (
    logistic as run_logistic_api,
)

app = typer.Typer(
    name="g",
    help="Blazing fast GWAS engine.",
    no_args_is_help=True,
    rich_markup_mode=None,
)


def resolve_chunk_size(requested_chunk_size: int | None, association_mode: str) -> int:
    """Resolve the effective chunk size for an association mode."""
    if requested_chunk_size is not None:
        return requested_chunk_size
    if association_mode == "linear":
        return DEFAULT_LINEAR_CHUNK_SIZE
    return DEFAULT_LOGISTIC_CHUNK_SIZE


def print_success_message(artifacts: RunArtifacts) -> None:
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
    bfile: pathlib.Path = typer.Option(..., help="PLINK dataset prefix."),
    pheno: pathlib.Path = typer.Option(..., help="Phenotype table path."),
    pheno_name: str = typer.Option(..., "--pheno-name", help="Phenotype column name to analyze."),
    out: pathlib.Path = typer.Option(..., help="Output prefix or TSV path."),
    covar: pathlib.Path | None = typer.Option(None, help="Optional covariate table path."),
    covar_names: str | None = typer.Option(None, "--covar-names", help="Comma-separated covariate column names."),
    chunk_size: int | None = typer.Option(None, help="Variants per BED chunk."),
    variant_limit: int | None = typer.Option(None, help="Optional variant cap for debugging or tests."),
    device: str = typer.Option("cpu", help="JAX execution device. Use 'gpu' to enable GPU acceleration."),
    output_mode: str = typer.Option("tsv", help="Output mode: 'tsv' or 'arrow_chunks'."),
    output_run_directory: pathlib.Path | None = typer.Option(None, help="Run directory for chunked output mode."),
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
    compute_config = ComputeConfig(
        chunk_size=resolve_chunk_size(chunk_size, "linear"),
        device=device,
        variant_limit=variant_limit,
        output_mode=output_mode,
        output_run_directory=output_run_directory,
        resume=resume,
        finalize_parquet=finalize_parquet,
    )
    artifacts = run_linear_api(
        bfile=bfile,
        pheno=pheno,
        pheno_name=pheno_name,
        out=out,
        covar=covar,
        covar_names=parse_covariate_name_list(covar_names),
        compute=compute_config,
        solver=LinearConfig(),
    )
    print_success_message(artifacts)


@app.command("logistic", no_args_is_help=True)
def run_logistic_command(
    bfile: pathlib.Path = typer.Option(..., help="PLINK dataset prefix."),
    pheno: pathlib.Path = typer.Option(..., help="Phenotype table path."),
    pheno_name: str = typer.Option(..., "--pheno-name", help="Phenotype column name to analyze."),
    out: pathlib.Path = typer.Option(..., help="Output prefix or TSV path."),
    covar: pathlib.Path | None = typer.Option(None, help="Optional covariate table path."),
    covar_names: str | None = typer.Option(None, "--covar-names", help="Comma-separated covariate column names."),
    chunk_size: int | None = typer.Option(None, help="Variants per BED chunk."),
    variant_limit: int | None = typer.Option(None, help="Optional variant cap for debugging or tests."),
    device: str = typer.Option("cpu", help="JAX execution device. Use 'gpu' to enable GPU acceleration."),
    max_iterations: int = typer.Option(50, help="Maximum logistic IRLS iterations."),
    tolerance: float = typer.Option(1.0e-8, help="Logistic convergence tolerance."),
    firth_fallback: bool = typer.Option(  # noqa: FBT001
        True,  # noqa: FBT003
        "--firth/--no-firth",
        help="Use Firth fallback when needed.",
    ),
    output_mode: str = typer.Option("tsv", help="Output mode: 'tsv' or 'arrow_chunks'."),
    output_run_directory: pathlib.Path | None = typer.Option(None, help="Run directory for chunked output mode."),
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
    compute_config = ComputeConfig(
        chunk_size=resolve_chunk_size(chunk_size, "logistic"),
        device=device,
        variant_limit=variant_limit,
        output_mode=output_mode,
        output_run_directory=output_run_directory,
        resume=resume,
        finalize_parquet=finalize_parquet,
    )
    solver_config = LogisticConfig(
        max_iterations=max_iterations,
        tolerance=tolerance,
        firth_fallback=firth_fallback,
    )
    artifacts = run_logistic_api(
        bfile=bfile,
        pheno=pheno,
        pheno_name=pheno_name,
        out=out,
        covar=covar,
        covar_names=parse_covariate_name_list(covar_names),
        compute=compute_config,
        solver=solver_config,
    )
    print_success_message(artifacts)


def main() -> None:
    """Run the GWAS CLI."""
    app()
