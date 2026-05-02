"""Command-line interface for the GWAS engine."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003

import typer

from g import api, types

run_regenie2_linear_api = api.regenie2_linear
run_regenie2_api = api.regenie2

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


def resolve_arrow_payload_batch_size(requested_arrow_payload_batch_size: int | None) -> int:
    """Resolve the effective Arrow payload batch size."""
    if requested_arrow_payload_batch_size is not None:
        return requested_arrow_payload_batch_size
    return api.DEFAULT_ARROW_PAYLOAD_BATCH_SIZE


def print_success_message(artifacts: api.RunArtifacts) -> None:
    """Print a concise success message for a completed CLI run."""
    if artifacts.output_run_directory is not None:
        typer.echo(f"Success. Chunked run saved to {artifacts.output_run_directory}")
        if artifacts.final_parquet is not None:
            typer.echo(f"Finalized Parquet saved to {artifacts.final_parquet}")
        return
    typer.echo("Success. Run completed.")


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
    device: types.Device = typer.Option(types.Device.CPU, help="JAX execution device."),
    output_run_directory: Path | None = typer.Option(None, help="Run directory for Arrow chunked output."),
    arrow_payload_batch_size: int | None = typer.Option(
        None,
        help="Number of REGENIE output chunks to batch per Arrow IPC write.",
    ),
    resume: bool = typer.Option(  # noqa: FBT001
        default=False,
        help="Resume a previous chunked run.",
    ),
    finalize_parquet: bool = typer.Option(  # noqa: FBT001
        default=False,
        help="Compact committed Arrow chunks into Parquet.",
    ),
    output_writer_backend: types.OutputWriterBackend = typer.Option(
        types.OutputWriterBackend.PYTHON,
        help="Backend used for chunk output writing.",
    ),
) -> None:
    """Run a REGENIE step 2 linear association scan."""
    compute_config = api.ComputeConfig(
        chunk_size=resolve_chunk_size(chunk_size),
        device=device,
        variant_limit=variant_limit,
        output_run_directory=output_run_directory,
        resume=resume,
        finalize_parquet=finalize_parquet,
        arrow_payload_batch_size=resolve_arrow_payload_batch_size(arrow_payload_batch_size),
        output_writer_backend=output_writer_backend,
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
    binary_correction: types.RegenieBinaryCorrection = typer.Option(
        types.RegenieBinaryCorrection.FIRTH_APPROXIMATE,
        "--binary-correction",
        help="Correction path for binary score-test candidates.",
    ),
    chunk_size: int | None = typer.Option(None, help="Variants per chunk."),
    variant_limit: int | None = typer.Option(None, help="Optional variant cap for debugging or tests."),
    device: types.Device = typer.Option(types.Device.CPU, help="JAX execution device."),
    output_run_directory: Path | None = typer.Option(None, help="Run directory for Arrow chunked output."),
    arrow_payload_batch_size: int | None = typer.Option(
        None,
        help="Number of REGENIE output chunks to batch per Arrow IPC write.",
    ),
    resume: bool = typer.Option(  # noqa: FBT001
        default=False,
        help="Resume a previous chunked run.",
    ),
    finalize_parquet: bool = typer.Option(  # noqa: FBT001
        default=False,
        help="Compact committed Arrow chunks into Parquet.",
    ),
    output_writer_backend: types.OutputWriterBackend = typer.Option(
        types.OutputWriterBackend.PYTHON,
        help="Backend used for chunk output writing.",
    ),
) -> None:
    """Run a REGENIE step 2 association scan."""
    compute_config = api.ComputeConfig(
        chunk_size=resolve_chunk_size(chunk_size),
        device=device,
        variant_limit=variant_limit,
        output_run_directory=output_run_directory,
        resume=resume,
        finalize_parquet=finalize_parquet,
        arrow_payload_batch_size=resolve_arrow_payload_batch_size(arrow_payload_batch_size),
        output_writer_backend=output_writer_backend,
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
        binary=api.Regenie2BinaryConfig(correction=binary_correction),
    )
    print_success_message(artifacts)


def main() -> None:
    """Run the GWAS CLI."""
    app()
