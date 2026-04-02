"""Format-agnostic genotype source orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from g.io.bgen import (
    iter_genotype_chunks as iter_bgen_genotype_chunks,
)
from g.io.bgen import (
    iter_linear_genotype_chunks as iter_bgen_linear_genotype_chunks,
)
from g.io.bgen import (
    load_bgen_sample_table,
)
from g.io.plink import (
    iter_genotype_chunks as iter_plink_genotype_chunks,
)
from g.io.plink import (
    iter_linear_genotype_chunks as iter_plink_linear_genotype_chunks,
)
from g.io.prefetch import prefetch_iterator_values
from g.io.tabular import load_aligned_sample_data, load_aligned_sample_data_from_individual_identifier_table

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np

    from g.models import AlignedSampleData, GenotypeChunk, LinearGenotypeChunk


@dataclass(frozen=True)
class GenotypeSourceConfig:
    """Configuration describing one genotype input source."""

    source_format: str
    source_path: Path


SUPPORTED_SOURCE_FORMATS = frozenset({"plink", "bgen"})


def build_plink_source_config(bed_prefix: Path | str) -> GenotypeSourceConfig:
    """Build a genotype source config for a PLINK dataset prefix."""
    return GenotypeSourceConfig(source_format="plink", source_path=Path(bed_prefix))


def build_bgen_source_config(bgen_path: Path | str) -> GenotypeSourceConfig:
    """Build a genotype source config for a BGEN file."""
    return GenotypeSourceConfig(source_format="bgen", source_path=Path(bgen_path))


def resolve_genotype_source_config(
    bfile: Path | str | None,
    bgen: Path | str | None,
) -> GenotypeSourceConfig:
    """Resolve the requested genotype source from public API arguments."""
    if (bfile is None) == (bgen is None):
        message = "Exactly one genotype source must be provided via bfile or bgen."
        raise ValueError(message)
    if bfile is not None:
        return build_plink_source_config(bfile)
    if bgen is not None:
        return build_bgen_source_config(bgen)
    raise ValueError("Impossible state.")


def validate_genotype_source_config(genotype_source_config: GenotypeSourceConfig) -> None:
    """Validate that a genotype source config uses a supported format."""
    if genotype_source_config.source_format not in SUPPORTED_SOURCE_FORMATS:
        message = (
            f"Unsupported genotype source format '{genotype_source_config.source_format}'. "
            f"Expected one of {sorted(SUPPORTED_SOURCE_FORMATS)}."
        )
        raise ValueError(message)


def build_genotype_source_signature_paths(genotype_source_config: GenotypeSourceConfig) -> tuple[Path, ...]:
    """Return the input files that define reproducibility for one source."""
    validate_genotype_source_config(genotype_source_config)
    if genotype_source_config.source_format == "plink":
        return (
            genotype_source_config.source_path.with_suffix(".bed"),
            genotype_source_config.source_path.with_suffix(".bim"),
            genotype_source_config.source_path.with_suffix(".fam"),
        )
    return (genotype_source_config.source_path,)


def load_aligned_sample_data_from_source(
    genotype_source_config: GenotypeSourceConfig,
    phenotype_path: Path,
    phenotype_name: str,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    *,
    is_binary_trait: bool,
) -> AlignedSampleData:
    """Load aligned sample data for any supported genotype source."""
    validate_genotype_source_config(genotype_source_config)
    if genotype_source_config.source_format == "plink":
        return load_aligned_sample_data(
            bed_prefix=genotype_source_config.source_path,
            phenotype_path=phenotype_path,
            phenotype_name=phenotype_name,
            covariate_path=covariate_path,
            covariate_names=covariate_names,
            is_binary_trait=is_binary_trait,
        )
    return load_aligned_sample_data_from_individual_identifier_table(
        sample_table=load_bgen_sample_table(genotype_source_config.source_path),
        phenotype_path=phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=is_binary_trait,
    )


def iter_genotype_chunks_from_source(
    genotype_source_config: GenotypeSourceConfig,
    sample_indices: np.ndarray,
    expected_individual_identifiers: np.ndarray,
    chunk_size: int,
    variant_limit: int | None = None,
    *,
    include_missing_value_flag: bool = True,
    prefetch_chunks: int = 0,
) -> Iterator[GenotypeChunk]:
    """Yield genotype chunks for any supported source format."""
    validate_genotype_source_config(genotype_source_config)
    if genotype_source_config.source_format == "plink":
        base_iterator = iter_plink_genotype_chunks(
            bed_prefix=genotype_source_config.source_path,
            sample_indices=sample_indices,
            expected_individual_identifiers=expected_individual_identifiers,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
            include_missing_value_flag=include_missing_value_flag,
        )
    else:
        base_iterator = iter_bgen_genotype_chunks(
            bgen_path=genotype_source_config.source_path,
            sample_indices=sample_indices,
            expected_individual_identifiers=expected_individual_identifiers,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
            include_missing_value_flag=include_missing_value_flag,
        )
    return prefetch_iterator_values(base_iterator, prefetch_chunks)


def iter_linear_genotype_chunks_from_source(
    genotype_source_config: GenotypeSourceConfig,
    sample_indices: np.ndarray,
    expected_individual_identifiers: np.ndarray,
    chunk_size: int,
    variant_limit: int | None = None,
    *,
    prefetch_chunks: int = 0,
) -> Iterator[LinearGenotypeChunk]:
    """Yield linear-regression genotype chunks for any supported source format."""
    validate_genotype_source_config(genotype_source_config)
    if genotype_source_config.source_format == "plink":
        base_iterator = iter_plink_linear_genotype_chunks(
            bed_prefix=genotype_source_config.source_path,
            sample_indices=sample_indices,
            expected_individual_identifiers=expected_individual_identifiers,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
        )
    else:
        base_iterator = iter_bgen_linear_genotype_chunks(
            bgen_path=genotype_source_config.source_path,
            sample_indices=sample_indices,
            expected_individual_identifiers=expected_individual_identifiers,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
        )
    return prefetch_iterator_values(base_iterator, prefetch_chunks)
