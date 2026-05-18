"""Command-line interface for the GWAS engine."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003

import typer

from g import api, types

run_regenie2_linear_api = api.regenie2_linear
run_regenie2_api = api.regenie2
run_regenie2_warm_cache_api = api.regenie2_warm_cache

app = typer.Typer(
    name="g",
    help="Blazing fast REGENIE step 2 GWAS engine.",
    no_args_is_help=True,
    rich_markup_mode=None,
)


@app.callback()
def root_callback() -> None:
    """Run the GWAS CLI."""


def resolve_chunk_size(requested_chunk_size: int | None) -> int:
    """Resolve the effective chunk size."""
    if requested_chunk_size is not None:
        return requested_chunk_size
    return api.DEFAULT_REGENIE2_LINEAR_CHUNK_SIZE


def print_success_message(artifacts: api.RunArtifacts) -> None:
    """Print a concise success message for a completed CLI run."""
    if artifacts.output_run_directory is not None:
        typer.echo(f"Success. Chunked run saved to {artifacts.output_run_directory}")
        if artifacts.final_parquet is not None:
            typer.echo(f"Finalized Parquet saved to {artifacts.final_parquet}")
        return
    typer.echo("Success. Run completed.")


def print_warm_cache_message(report: api.WarmCacheReport) -> None:
    """Print a concise success message for cache warming."""
    warmed_shape_descriptions = ", ".join(
        f"({shape.sample_count}, {shape.variant_count})" for shape in report.warmed_shapes
    )
    typer.echo(f"Success. Warmed JAX cache shapes: {warmed_shape_descriptions}")


@app.command("regenie2-linear", no_args_is_help=True)
def run_regenie2_linear_command(
    bgen: Path = typer.Option(..., help="BGEN file path."),
    sample: Path | None = typer.Option(
        None,
        help="Optional BGEN sample-file path. Defaults to embedded samples or an adjacent .sample file.",
    ),
    pheno: Path = typer.Option(..., help="Phenotype table path."),
    pheno_name: str = typer.Option(..., "--pheno-name", help="Phenotype column name to analyze."),
    out: Path = typer.Option(..., help="Output prefix or run directory."),
    covar: Path | None = typer.Option(None, help="Optional covariate table path."),
    covar_names: str | None = typer.Option(None, "--covar-names", help="Comma-separated covariate column names."),
    pred: Path = typer.Option(..., help="REGENIE step 1 _pred.list file path."),
    chunk_size: int | None = typer.Option(None, help="Variants per chunk."),
    variant_limit: int | None = typer.Option(None, help="Optional variant cap for debugging or tests."),
    prefetch_chunks: int = typer.Option(1, help="Number of genotype chunks to prefetch on the host."),
    device: types.Device = typer.Option(types.Device.CPU, help="JAX execution device."),
    output_run_directory: Path | None = typer.Option(None, help="Run directory for Arrow chunked output."),
    output_writer_thread_count: int = typer.Option(
        api.output.DEFAULT_WRITER_THREAD_COUNT,
        help="Background output writer thread count.",
    ),
    output_writer_queue_depth: int = typer.Option(
        api.DEFAULT_OUTPUT_WRITER_QUEUE_DEPTH,
        help="Maximum number of queued output write jobs.",
    ),
    trusted_no_missing_diploid: bool = typer.Option(  # noqa: FBT001
        default=False,
        help="Use the native fast path for validated BGENs with no missing diploid genotypes.",
    ),
    warm_cache_first: bool = typer.Option(  # noqa: FBT001
        default=False,
        help="Warm exact JAX cache shapes in this process before running.",
    ),
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
        chunk_size=resolve_chunk_size(chunk_size),
        device=device,
        variant_limit=variant_limit,
        prefetch_chunks=prefetch_chunks,
        output_run_directory=output_run_directory,
        resume=resume,
        finalize_parquet=finalize_parquet,
        output_writer_thread_count=output_writer_thread_count,
        output_writer_queue_depth=output_writer_queue_depth,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
        warm_cache_first=warm_cache_first,
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


@app.command("regenie2", no_args_is_help=True)
def run_regenie2_command(
    bgen: Path = typer.Option(..., help="BGEN file path."),
    sample: Path | None = typer.Option(
        None,
        help="Optional BGEN sample-file path. Defaults to embedded samples or an adjacent .sample file.",
    ),
    pheno: Path = typer.Option(..., help="Phenotype table path."),
    pheno_name: str = typer.Option(..., "--pheno-name", help="Phenotype column name to analyze."),
    out: Path = typer.Option(..., help="Output prefix or run directory."),
    trait_type: types.RegenieTraitType = typer.Option(
        types.RegenieTraitType.QUANTITATIVE,
        "--trait-type",
        help="Trait type to analyze.",
    ),
    covar: Path | None = typer.Option(None, help="Optional covariate table path."),
    covar_names: str | None = typer.Option(None, "--covar-names", help="Comma-separated covariate column names."),
    pred: Path = typer.Option(..., help="REGENIE step 1 _pred.list file path."),
    firth: bool = typer.Option(  # noqa: FBT001
        default=False,
        help="Use Firth fallback for binary score-test p-values below pThresh.",
    ),
    approx: bool = typer.Option(  # noqa: FBT001
        default=False,
        help="Use approximate Firth fallback when --firth is enabled.",
    ),
    spa: bool = typer.Option(  # noqa: FBT001
        default=False,
        help="Use SPA fallback for binary score-test p-values below pThresh.",
    ),
    p_threshold: float = typer.Option(
        0.05,
        "--pThresh",
        "--p-thresh",
        help="Score-test p-value threshold for binary fallback correction.",
    ),
    firth_se: bool = typer.Option(  # noqa: FBT001
        default=False,
        help="Use LRT-derived standard errors for successful Firth rows.",
    ),
    chunk_size: int | None = typer.Option(None, help="Variants per chunk."),
    variant_limit: int | None = typer.Option(None, help="Optional variant cap for debugging or tests."),
    prefetch_chunks: int = typer.Option(1, help="Number of genotype chunks to prefetch on the host."),
    device: types.Device = typer.Option(types.Device.CPU, help="JAX execution device."),
    output_run_directory: Path | None = typer.Option(None, help="Run directory for Arrow chunked output."),
    output_writer_thread_count: int = typer.Option(
        api.output.DEFAULT_WRITER_THREAD_COUNT,
        help="Background output writer thread count.",
    ),
    output_writer_queue_depth: int = typer.Option(
        api.DEFAULT_OUTPUT_WRITER_QUEUE_DEPTH,
        help="Maximum number of queued output write jobs.",
    ),
    trusted_no_missing_diploid: bool = typer.Option(  # noqa: FBT001
        default=False,
        help="Use the native fast path for validated BGENs with no missing diploid genotypes.",
    ),
    warm_cache_first: bool = typer.Option(  # noqa: FBT001
        default=False,
        help="Warm exact JAX cache shapes in this process before running.",
    ),
    resume: bool = typer.Option(  # noqa: FBT001
        default=False,
        help="Resume a previous chunked run.",
    ),
    finalize_parquet: bool = typer.Option(  # noqa: FBT001
        default=False,
        help="Compact committed Arrow chunks into Parquet.",
    ),
) -> None:
    """Run a REGENIE step 2 association scan."""
    compute_config = api.ComputeConfig(
        chunk_size=resolve_chunk_size(chunk_size),
        device=device,
        variant_limit=variant_limit,
        prefetch_chunks=prefetch_chunks,
        output_run_directory=output_run_directory,
        resume=resume,
        finalize_parquet=finalize_parquet,
        output_writer_thread_count=output_writer_thread_count,
        output_writer_queue_depth=output_writer_queue_depth,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
        warm_cache_first=warm_cache_first,
    )
    artifacts = run_regenie2_api(
        bgen=bgen,
        sample=sample,
        pheno=pheno,
        pheno_name=pheno_name,
        out=out,
        covar=covar,
        covar_names=api.parse_covariate_name_list(covar_names),
        pred=pred,
        trait_type=trait_type,
        compute=compute_config,
        binary=api.Regenie2BinaryConfig(
            firth=firth,
            approx=approx,
            spa=spa,
            p_threshold=p_threshold,
            firth_se=firth_se,
        ),
    )
    print_success_message(artifacts)


@app.command("regenie2-warm-cache", no_args_is_help=True)
def run_regenie2_warm_cache_command(
    bgen: Path = typer.Option(..., help="BGEN file path."),
    sample: Path | None = typer.Option(
        None,
        help="Optional BGEN sample-file path. Defaults to embedded samples or an adjacent .sample file.",
    ),
    pheno: Path = typer.Option(..., help="Phenotype table path."),
    pheno_name: str = typer.Option(..., "--pheno-name", help="Phenotype column name to analyze."),
    trait_type: types.RegenieTraitType = typer.Option(
        types.RegenieTraitType.QUANTITATIVE,
        "--trait-type",
        help="Trait type to warm.",
    ),
    covar: Path | None = typer.Option(None, help="Optional covariate table path."),
    covar_names: str | None = typer.Option(None, "--covar-names", help="Comma-separated covariate column names."),
    pred: Path = typer.Option(..., help="REGENIE step 1 _pred.list file path."),
    firth: bool = typer.Option(  # noqa: FBT001
        default=False,
        help="Use Firth fallback for binary score-test p-values below pThresh.",
    ),
    approx: bool = typer.Option(  # noqa: FBT001
        default=False,
        help="Use approximate Firth fallback when --firth is enabled.",
    ),
    spa: bool = typer.Option(  # noqa: FBT001
        default=False,
        help="Use SPA fallback for binary score-test p-values below pThresh.",
    ),
    p_threshold: float = typer.Option(
        0.05,
        "--pThresh",
        "--p-thresh",
        help="Score-test p-value threshold for binary fallback correction.",
    ),
    firth_se: bool = typer.Option(  # noqa: FBT001
        default=False,
        help="Use LRT-derived standard errors for successful Firth rows.",
    ),
    chunk_size: int | None = typer.Option(None, help="Variants per chunk."),
    variant_limit: int | None = typer.Option(None, help="Optional variant cap for debugging or tests."),
    device: types.Device = typer.Option(types.Device.CPU, help="JAX execution device."),
    trusted_no_missing_diploid: bool = typer.Option(  # noqa: FBT001
        default=False,
        help="Use the native fast path for validated BGENs with no missing diploid genotypes.",
    ),
) -> None:
    """Warm JAX compilation-cache entries for a REGENIE step 2 association scan."""
    compute_config = api.ComputeConfig(
        chunk_size=resolve_chunk_size(chunk_size),
        device=device,
        variant_limit=variant_limit,
        finalize_parquet=False,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
    )
    report = run_regenie2_warm_cache_api(
        bgen=bgen,
        sample=sample,
        pheno=pheno,
        pheno_name=pheno_name,
        covar=covar,
        covar_names=api.parse_covariate_name_list(covar_names),
        pred=pred,
        trait_type=trait_type,
        compute=compute_config,
        binary=api.Regenie2BinaryConfig(
            firth=firth,
            approx=approx,
            spa=spa,
            p_threshold=p_threshold,
            firth_se=firth_se,
        ),
    )
    print_warm_cache_message(report)


def main() -> None:
    """Run the GWAS CLI."""
    app()
