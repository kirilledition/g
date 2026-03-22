"""PLINK BED/BIM/FAM input helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import polars as pl
from bed_reader import open_bed

from g import jax_setup  # noqa: F401
from g.models import GenotypeChunk, VariantMetadata

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

VARIANT_TABLE_COLUMNS = (
    "chromosome",
    "variant_identifier",
    "genetic_distance",
    "position",
    "allele_one",
    "allele_two",
)


def load_variant_table(bed_prefix: Path) -> pl.DataFrame:
    """Load a PLINK BIM file.

    Args:
        bed_prefix: PLINK dataset prefix.

    Returns:
        Parsed BIM table.

    """
    return pl.read_csv(
        bed_prefix.with_suffix(".bim"),
        has_header=False,
        new_columns=list(VARIANT_TABLE_COLUMNS),
        separator="\t",
    )


def iter_genotype_chunks(
    bed_prefix: Path,
    sample_indices: np.ndarray,
    expected_individual_identifiers: np.ndarray,
    chunk_size: int,
    variant_limit: int | None = None,
) -> Iterator[GenotypeChunk]:
    """Yield mean-imputed genotype chunks from a PLINK BED file.

    Args:
        bed_prefix: PLINK dataset prefix.
        sample_indices: Row indices to read from the BED file.
        expected_individual_identifiers: Expected sample order after alignment.
        chunk_size: Number of variants per chunk.
        variant_limit: Optional cap on the number of variants.

    Yields:
        Mean-imputed genotype chunks with metadata.

    Raises:
        ValueError: If BED sample order does not match the aligned sample order.

    """
    variant_table = load_variant_table(bed_prefix)
    total_variant_count = variant_table.height if variant_limit is None else min(variant_limit, variant_table.height)
    bed_path = bed_prefix.with_suffix(".bed")

    with open_bed(str(bed_path)) as bed_handle:
        observed_individual_identifiers = np.asarray(bed_handle.iid)[sample_indices]
        if not np.array_equal(observed_individual_identifiers, expected_individual_identifiers):
            message = "BED sample order does not match the aligned phenotype/covariate order."
            raise ValueError(message)

        for variant_start in range(0, total_variant_count, chunk_size):
            variant_stop = min(total_variant_count, variant_start + chunk_size)
            genotype_matrix = np.asarray(
                bed_handle.read(index=np.s_[sample_indices, variant_start:variant_stop], dtype=np.float64),
                dtype=np.float64,
            )
            column_means = np.nanmean(genotype_matrix, axis=0)
            sanitized_column_means = np.where(np.isnan(column_means), 0.0, column_means)
            imputed_matrix = np.where(np.isnan(genotype_matrix), sanitized_column_means[None, :], genotype_matrix)
            allele_one_frequency = sanitized_column_means / 2.0
            observation_count = np.full((variant_stop - variant_start,), imputed_matrix.shape[0], dtype=np.int64)
            metadata_table = variant_table.slice(variant_start, variant_stop - variant_start)

            yield GenotypeChunk(
                genotypes=jnp.asarray(imputed_matrix),
                metadata=VariantMetadata(
                    chromosome=metadata_table.get_column("chromosome").cast(pl.String).to_numpy(),
                    variant_identifiers=metadata_table.get_column("variant_identifier").cast(pl.String).to_numpy(),
                    position=metadata_table.get_column("position").cast(pl.Int64).to_numpy(),
                    allele_one=metadata_table.get_column("allele_one").cast(pl.String).to_numpy(),
                    allele_two=metadata_table.get_column("allele_two").cast(pl.String).to_numpy(),
                ),
                allele_one_frequency=jnp.asarray(allele_one_frequency),
                observation_count=jnp.asarray(observation_count),
            )
