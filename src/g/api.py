"""Public Python API for GWAS execution."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from g.engine import iter_linear_output_frames, iter_logistic_output_frames, write_frame_iterator_to_tsv
from g.jax_setup import configure_jax_device

DEFAULT_LINEAR_CHUNK_SIZE = 2048
DEFAULT_LOGISTIC_CHUNK_SIZE = 1024


@dataclasses.dataclass(frozen=True)
class ComputeConfig:
    """Hardware and batching settings shared across association methods.

    This stays command-agnostic so future GRM and mixed-model entrypoints can
    reuse the same execution controls without introducing another public module.
    """

    chunk_size: int = DEFAULT_LINEAR_CHUNK_SIZE
    device: str = "cpu"
    variant_limit: int | None = None


@dataclasses.dataclass(frozen=True)
class LogisticConfig:
    """Mathematical settings for logistic regression."""

    max_iterations: int = 50
    tolerance: float = 1.0e-8
    firth_fallback: bool = True


@dataclasses.dataclass(frozen=True)
class LinearConfig:
    """Mathematical settings for linear regression."""


@dataclasses.dataclass(frozen=True)
class RunArtifacts:
    """Immutable pointers to generated output files."""

    sumstats_tsv: Path


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


def resolve_output_path(output_path_or_prefix: Path | str, association_mode: str) -> Path:
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
    if compute_config.device not in {"cpu", "gpu"}:
        message = f"Unsupported device '{compute_config.device}'. Expected 'cpu' or 'gpu'."
        raise ValueError(message)


def validate_logistic_config(logistic_config: LogisticConfig) -> None:
    """Validate a logistic solver configuration."""
    if logistic_config.max_iterations <= 0:
        message = "Maximum iterations must be positive."
        raise ValueError(message)
    if logistic_config.tolerance <= 0.0:
        message = "Tolerance must be positive."
        raise ValueError(message)


def linear(
    bfile: Path | str,
    pheno: Path | str,
    pheno_name: str,
    out: Path | str,
    covar: Path | str | None = None,
    covar_names: str | list[str] | tuple[str, ...] | None = None,
    compute: ComputeConfig | None = None,
    solver: LinearConfig | None = None,
) -> RunArtifacts:
    """Run a linear association scan and write results to disk."""
    del solver
    compute_config = compute or ComputeConfig()
    validate_compute_config(compute_config)
    output_path = resolve_output_path(out, "linear")
    configure_jax_device(compute_config.device)

    frame_iterator = iter_linear_output_frames(
        bed_prefix=Path(bfile),
        phenotype_path=Path(pheno),
        phenotype_name=pheno_name,
        covariate_path=Path(covar) if covar is not None else None,
        covariate_names=parse_covariate_name_list(covar_names),
        chunk_size=compute_config.chunk_size,
        variant_limit=compute_config.variant_limit,
    )
    write_frame_iterator_to_tsv(frame_iterator, output_path)
    return RunArtifacts(sumstats_tsv=output_path)


def logistic(
    bfile: Path | str,
    pheno: Path | str,
    pheno_name: str,
    out: Path | str,
    covar: Path | str | None = None,
    covar_names: str | list[str] | tuple[str, ...] | None = None,
    compute: ComputeConfig | None = None,
    solver: LogisticConfig | None = None,
) -> RunArtifacts:
    """Run a logistic association scan and write results to disk."""
    compute_config = compute or ComputeConfig(chunk_size=DEFAULT_LOGISTIC_CHUNK_SIZE)
    solver_config = solver or LogisticConfig()
    validate_compute_config(compute_config)
    validate_logistic_config(solver_config)
    output_path = resolve_output_path(out, "logistic")
    configure_jax_device(compute_config.device)

    frame_iterator = iter_logistic_output_frames(
        bed_prefix=Path(bfile),
        phenotype_path=Path(pheno),
        phenotype_name=pheno_name,
        covariate_path=Path(covar) if covar is not None else None,
        covariate_names=parse_covariate_name_list(covar_names),
        chunk_size=compute_config.chunk_size,
        variant_limit=compute_config.variant_limit,
        max_iterations=solver_config.max_iterations,
        tolerance=solver_config.tolerance,
    )
    write_frame_iterator_to_tsv(frame_iterator, output_path)
    return RunArtifacts(sumstats_tsv=output_path)
