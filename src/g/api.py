"""Public Python API for GWAS execution."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from g import engine, jax_setup, types
from g.io import output, source

configure_jax_device = jax_setup.configure_jax_device
iter_regenie2_linear_output_frames = engine.iter_regenie2_linear_output_frames
write_frame_iterator_to_tsv = engine.write_frame_iterator_to_tsv
prepare_output_run = output.prepare_output_run
persist_chunked_results = output.persist_chunked_results
finalize_chunks_to_parquet = output.finalize_chunks_to_parquet

DEFAULT_REGENIE2_LINEAR_CHUNK_SIZE = 2048


@dataclasses.dataclass(frozen=True)
class ComputeConfig:
    """Hardware and batching settings for REGENIE step 2 execution."""

    chunk_size: int = DEFAULT_REGENIE2_LINEAR_CHUNK_SIZE
    device: types.Device = types.Device.CPU
    variant_limit: int | None = None
    prefetch_chunks: int = 1
    output_mode: types.OutputMode = types.OutputMode.TSV
    output_run_directory: Path | None = None
    resume: bool = False
    finalize_parquet: bool = False


@dataclasses.dataclass(frozen=True)
class Regenie2LinearConfig:
    """Configuration for REGENIE step 2 linear association."""


@dataclasses.dataclass(frozen=True)
class RunArtifacts:
    """Immutable pointers to generated output files."""

    sumstats_tsv: Path | None = None
    output_run_directory: Path | None = None
    final_parquet: Path | None = None


def parse_covariate_name_list(raw_covariate_names: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...] | None:
    """Normalize covariate names into a tuple."""
    if raw_covariate_names is None:
        return None
    if isinstance(raw_covariate_names, str):
        covariate_names = tuple(
            stripped_name for name in raw_covariate_names.split(",") if (stripped_name := name.strip())
        )
        return covariate_names or None
    covariate_names = tuple(name.strip() for name in raw_covariate_names if name.strip())
    return covariate_names or None


def resolve_output_path(output_path_or_prefix: Path | str, association_mode: types.AssociationMode) -> Path:
    """Resolve the final TSV path for an association run."""
    output_path = Path(output_path_or_prefix)
    if output_path.suffix == ".tsv":
        return output_path
    return output_path.with_suffix(f".{association_mode}.tsv")


def validate_compute_config(compute_config: ComputeConfig) -> None:
    """Validate a compute configuration."""
    if compute_config.chunk_size <= 0:
        message = "Chunk size must be positive."
        raise ValueError(message)
    if compute_config.variant_limit is not None and compute_config.variant_limit <= 0:
        message = "Variant limit must be positive when provided."
        raise ValueError(message)
    if compute_config.prefetch_chunks < 0:
        message = "Prefetch chunk count must be zero or positive."
        raise ValueError(message)


def regenie2_linear(
    *,
    bgen: Path | str,
    sample: Path | str | None = None,
    pheno: Path | str,
    pheno_name: str,
    out: Path | str,
    covar: Path | str | None = None,
    covar_names: str | list[str] | tuple[str, ...] | None = None,
    pred: Path | str,
    compute: ComputeConfig | None = None,
    solver: Regenie2LinearConfig | None = None,
) -> RunArtifacts:
    """Run a REGENIE step 2 linear association scan and write results to disk."""
    del solver
    compute_config = compute or ComputeConfig()
    validate_compute_config(compute_config)
    output_path = resolve_output_path(out, types.AssociationMode.REGENIE2_LINEAR)
    configure_jax_device(compute_config.device)
    covariate_name_list = parse_covariate_name_list(covar_names)
    genotype_source_config = source.build_bgen_source_config(bgen, sample)
    output_run_directory = compute_config.output_run_directory or Path(out)
    committed_chunk_identifiers: set[int] = set()
    output_run_paths = None
    if compute_config.output_mode == types.OutputMode.ARROW_CHUNKS:
        prepared_output_run = prepare_output_run(
            output_root=output_run_directory,
            association_mode=types.AssociationMode.REGENIE2_LINEAR,
            resume=compute_config.resume,
        )
        output_run_paths = prepared_output_run.output_run_paths
        committed_chunk_identifiers = set(prepared_output_run.committed_chunk_identifiers)

    frame_iterator = iter_regenie2_linear_output_frames(
        genotype_source_config=genotype_source_config,
        phenotype_path=Path(pheno),
        phenotype_name=pheno_name,
        prediction_list_path=Path(pred),
        covariate_path=Path(covar) if covar is not None else None,
        covariate_names=covariate_name_list,
        chunk_size=compute_config.chunk_size,
        variant_limit=compute_config.variant_limit,
        prefetch_chunks=compute_config.prefetch_chunks,
        committed_chunk_identifiers=committed_chunk_identifiers,
    )
    if compute_config.output_mode == types.OutputMode.TSV:
        write_frame_iterator_to_tsv(frame_iterator, output_path)
        return RunArtifacts(sumstats_tsv=output_path)
    assert output_run_paths is not None

    persist_chunked_results(
        frame_iterator=frame_iterator,
        output_run_paths=output_run_paths,
        association_mode=types.AssociationMode.REGENIE2_LINEAR,
    )
    final_parquet_path = (
        finalize_chunks_to_parquet(output_run_paths, types.AssociationMode.REGENIE2_LINEAR)
        if compute_config.finalize_parquet
        else None
    )
    return RunArtifacts(
        output_run_directory=output_run_paths.run_directory,
        final_parquet=final_parquet_path,
    )
