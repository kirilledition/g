"""Public Python API for GWAS execution."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from g import engine, jax_setup, types
from g.io import output, source

configure_jax_device = jax_setup.configure_jax_device
iter_regenie2_linear_output_frames = engine.iter_regenie2_linear_output_frames
iter_regenie2_binary_output_frames = engine.iter_regenie2_binary_output_frames
prepare_output_run = output.prepare_output_run
persist_chunked_results = output.persist_chunked_results
finalize_chunks_to_parquet = output.finalize_chunks_to_parquet

DEFAULT_REGENIE2_LINEAR_CHUNK_SIZE = 8192
DEFAULT_ARROW_PAYLOAD_BATCH_SIZE = 1


@dataclasses.dataclass(frozen=True)
class ComputeConfig:
    """Hardware and batching settings for REGENIE step 2 execution."""

    chunk_size: int = DEFAULT_REGENIE2_LINEAR_CHUNK_SIZE
    device: types.Device = types.Device.CPU
    variant_limit: int | None = None
    prefetch_chunks: int = 1
    output_run_directory: Path | None = None
    resume: bool = False
    finalize_parquet: bool = True
    arrow_payload_batch_size: int = DEFAULT_ARROW_PAYLOAD_BATCH_SIZE
    output_writer_thread_count: int = output.DEFAULT_WRITER_THREAD_COUNT


@dataclasses.dataclass(frozen=True)
class Regenie2LinearConfig:
    """Configuration for REGENIE step 2 linear association."""


@dataclasses.dataclass(frozen=True)
class Regenie2BinaryConfig:
    """Configuration for REGENIE step 2 binary association."""

    correction: types.RegenieBinaryCorrection = types.RegenieBinaryCorrection.FIRTH_APPROXIMATE


@dataclasses.dataclass(frozen=True)
class RunArtifacts:
    """Immutable pointers to generated output files."""

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
    if compute_config.arrow_payload_batch_size <= 0:
        message = "Arrow payload batch size must be positive."
        raise ValueError(message)
    if compute_config.output_writer_thread_count <= 0:
        message = "Output writer thread count must be positive."
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
    return regenie2(
        bgen=bgen,
        sample=sample,
        pheno=pheno,
        pheno_name=pheno_name,
        out=out,
        covar=covar,
        covar_names=covar_names,
        pred=pred,
        trait_type=types.RegenieTraitType.QUANTITATIVE,
        compute=compute,
    )


def regenie2(
    *,
    bgen: Path | str,
    sample: Path | str | None = None,
    pheno: Path | str,
    pheno_name: str,
    out: Path | str,
    covar: Path | str | None = None,
    covar_names: str | list[str] | tuple[str, ...] | None = None,
    pred: Path | str,
    trait_type: types.RegenieTraitType = types.RegenieTraitType.QUANTITATIVE,
    compute: ComputeConfig | None = None,
    binary: Regenie2BinaryConfig | None = None,
) -> RunArtifacts:
    """Run a REGENIE step 2 association scan and write results to disk."""
    compute_config = compute or ComputeConfig()
    validate_compute_config(compute_config)
    configure_jax_device(compute_config.device)
    covariate_name_list = parse_covariate_name_list(covar_names)
    genotype_source_config = source.build_bgen_source_config(bgen, sample)
    output_run_directory = compute_config.output_run_directory or Path(out)
    association_mode = (
        types.AssociationMode.REGENIE2_BINARY
        if trait_type == types.RegenieTraitType.BINARY
        else types.AssociationMode.REGENIE2_LINEAR
    )
    prepared_output_run = prepare_output_run(
        output_root=output_run_directory,
        association_mode=association_mode,
        resume=compute_config.resume,
    )
    output_run_paths = prepared_output_run.output_run_paths
    committed_chunk_identifiers = set(prepared_output_run.committed_chunk_identifiers)

    if trait_type == types.RegenieTraitType.BINARY:
        binary_config = binary or Regenie2BinaryConfig()
        frame_iterator = iter_regenie2_binary_output_frames(
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
            correction=binary_config.correction,
        )
    else:
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

    persist_chunked_results(
        frame_iterator=frame_iterator,
        output_run_paths=output_run_paths,
        association_mode=association_mode,
        writer_thread_count=compute_config.output_writer_thread_count,
        payload_batch_size=compute_config.arrow_payload_batch_size,
    )
    final_parquet_path = (
        finalize_chunks_to_parquet(output_run_paths, association_mode) if compute_config.finalize_parquet else None
    )
    return RunArtifacts(
        output_run_directory=output_run_paths.run_directory,
        final_parquet=final_parquet_path,
    )
