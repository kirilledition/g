"""PLINK BED/BIM/FAM input helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from bed_reader import open_bed, read_f64
from bed_reader._open_bed import get_num_threads

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


def read_bed_chunk(
    bed_handle: open_bed,
    bed_path: Path,
    sample_index_array: np.ndarray,
    variant_start: int,
    variant_stop: int,
    num_threads: int,
) -> jax.Array:
    """Read one BED chunk into a JAX array.

    Args:
        bed_handle: Open BED reader handle.
        bed_path: PLINK BED file path.
        sample_index_array: Contiguous sample indices.
        variant_start: Inclusive variant start index.
        variant_stop: Exclusive variant stop index.
        num_threads: Thread count for the BED reader.

    Returns:
        Genotype chunk as a JAX array with float64 precision.

    """
    variant_index_array = np.arange(variant_start, variant_stop, dtype=np.intp)
    genotype_matrix_host = np.zeros(
        (sample_index_array.shape[0], variant_index_array.shape[0]),
        dtype=np.float64,
        order="C",
    )
    read_f64(
        str(bed_path),
        bed_handle.cloud_options,
        iid_count=bed_handle.iid_count,
        sid_count=bed_handle.sid_count,
        is_a1_counted=bed_handle.count_A1,
        iid_index=sample_index_array,
        sid_index=variant_index_array,
        val=genotype_matrix_host,
        num_threads=num_threads,
    )
    return jax.device_put(genotype_matrix_host)


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
    sample_index_array = np.ascontiguousarray(sample_indices, dtype=np.intp)
    chromosome_values = variant_table.get_column("chromosome").cast(pl.String).to_numpy()
    variant_identifier_values = variant_table.get_column("variant_identifier").cast(pl.String).to_numpy()
    position_values = variant_table.get_column("position").cast(pl.Int64).to_numpy()
    allele_one_values = variant_table.get_column("allele_one").cast(pl.String).to_numpy()
    allele_two_values = variant_table.get_column("allele_two").cast(pl.String).to_numpy()

    with open_bed(str(bed_path)) as bed_handle:
        observed_individual_identifiers = np.asarray(bed_handle.iid)[sample_index_array]
        if not np.array_equal(observed_individual_identifiers, expected_individual_identifiers):
            message = "BED sample order does not match the aligned phenotype/covariate order."
            raise ValueError(message)

        num_threads = get_num_threads(getattr(bed_handle, "_num_threads", None))

        for variant_start in range(0, total_variant_count, chunk_size):
            variant_stop = min(total_variant_count, variant_start + chunk_size)
            genotype_matrix = read_bed_chunk(
                bed_handle=bed_handle,
                bed_path=bed_path,
                sample_index_array=sample_index_array,
                variant_start=variant_start,
                variant_stop=variant_stop,
                num_threads=num_threads,
            )
            missing_mask = jnp.isnan(genotype_matrix)
            observed_genotype_total = jnp.where(missing_mask, 0.0, genotype_matrix).sum(axis=0)
            observation_count = jnp.sum(~missing_mask, axis=0, dtype=jnp.int64)
            column_means = observed_genotype_total / jnp.maximum(observation_count, 1)
            sanitized_column_means = jnp.where(observation_count > 0, column_means, 0.0)
            imputed_matrix = jnp.where(missing_mask, sanitized_column_means[None, :], genotype_matrix)
            allele_one_frequency = sanitized_column_means / 2.0

            yield GenotypeChunk(
                genotypes=imputed_matrix,
                missing_mask=missing_mask,
                metadata=VariantMetadata(
                    chromosome=chromosome_values[variant_start:variant_stop],
                    variant_identifiers=variant_identifier_values[variant_start:variant_stop],
                    position=position_values[variant_start:variant_stop],
                    allele_one=allele_one_values[variant_start:variant_stop],
                    allele_two=allele_two_values[variant_start:variant_stop],
                ),
                allele_one_frequency=allele_one_frequency,
                observation_count=observation_count,
            )
