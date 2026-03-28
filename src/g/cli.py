"""Command-line interface for the GWAS engine."""

from __future__ import annotations

from pathlib import Path

import typer

from g.api import (
    DEFAULT_LINEAR_CHUNK_SIZE,
    DEFAULT_LOGISTIC_CHUNK_SIZE,
    ComputeConfig,
    LinearConfig,
    LogisticConfig,
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


def print_success_message(artifacts_path: Path) -> None:
    """Print a concise success message for a completed CLI run."""
    output_path = Path(artifacts_path)
    typer.echo(f"Success. Results saved to {output_path}")


@app.command("linear", no_args_is_help=True)
def run_linear_command(
    bfile: Path = typer.Option(..., help="PLINK dataset prefix."),
    pheno: Path = typer.Option(..., help="Phenotype table path."),
    pheno_name: str = typer.Option(..., "--pheno-name", help="Phenotype column name to analyze."),
    out: Path = typer.Option(..., help="Output prefix or TSV path."),
    covar: Path | None = typer.Option(None, help="Optional covariate table path."),
    covar_names: str | None = typer.Option(None, "--covar-names", help="Comma-separated covariate column names."),
    chunk_size: int | None = typer.Option(None, help="Variants per BED chunk."),
    variant_limit: int | None = typer.Option(None, help="Optional variant cap for debugging or tests."),
    device: str = typer.Option("cpu", help="JAX execution device. Use 'gpu' to enable GPU acceleration."),
) -> None:
    """Run a linear association scan."""
    compute_config = ComputeConfig(
        chunk_size=resolve_chunk_size(chunk_size, "linear"),
        device=device,
        variant_limit=variant_limit,
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
    print_success_message(artifacts.sumstats_tsv)


@app.command("logistic", no_args_is_help=True)
def run_logistic_command(
    bfile: Path = typer.Option(..., help="PLINK dataset prefix."),
    pheno: Path = typer.Option(..., help="Phenotype table path."),
    pheno_name: str = typer.Option(..., "--pheno-name", help="Phenotype column name to analyze."),
    out: Path = typer.Option(..., help="Output prefix or TSV path."),
    covar: Path | None = typer.Option(None, help="Optional covariate table path."),
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
) -> None:
    """Run a logistic association scan."""
    compute_config = ComputeConfig(
        chunk_size=resolve_chunk_size(chunk_size, "logistic"),
        device=device,
        variant_limit=variant_limit,
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
    print_success_message(artifacts.sumstats_tsv)


def main() -> None:
    """Run the GWAS CLI."""
    app()
