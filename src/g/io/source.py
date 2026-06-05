"""BGEN genotype source configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GenotypeSourceConfig:
    """Configuration describing one BGEN input source."""

    source_path: Path
    sample_path: Path | None = None


def build_bgen_source_config(bgen_path: Path | str, sample_path: Path | str | None = None) -> GenotypeSourceConfig:
    """Build a genotype source config for a BGEN file."""
    return GenotypeSourceConfig(
        source_path=Path(bgen_path),
        sample_path=Path(sample_path) if sample_path is not None else None,
    )


def resolve_genotype_source_config(
    bgen: Path | str | None,
    sample: Path | str | None = None,
) -> GenotypeSourceConfig:
    """Resolve the requested BGEN source from public API arguments."""
    if bgen is None:
        message = "A BGEN source must be provided via bgen."
        raise ValueError(message)
    return build_bgen_source_config(bgen, sample_path=sample)


def validate_genotype_source_config(genotype_source_config: GenotypeSourceConfig) -> None:
    """Validate a BGEN source config."""
    if genotype_source_config.source_path.suffix != ".bgen":
        message = f"Expected a .bgen source path, found '{genotype_source_config.source_path}'."
        raise ValueError(message)


def resolve_bgen_sample_path(bgen_path: Path, sample_path: Path | None = None) -> Path | None:
    """Resolve an explicit or adjacent Oxford sample file for one BGEN file."""
    if sample_path is not None:
        return sample_path
    adjacent_sample_path = bgen_path.with_suffix(".sample")
    return adjacent_sample_path if adjacent_sample_path.exists() else None


def build_genotype_source_signature_paths(genotype_source_config: GenotypeSourceConfig) -> tuple[Path, ...]:
    """Return the input files that define reproducibility for one source."""
    validate_genotype_source_config(genotype_source_config)
    resolved_sample_path = resolve_bgen_sample_path(
        genotype_source_config.source_path,
        genotype_source_config.sample_path,
    )
    if resolved_sample_path is None:
        return (genotype_source_config.source_path,)
    return (genotype_source_config.source_path, resolved_sample_path)
