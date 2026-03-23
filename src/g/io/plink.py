"""PLINK BED/BIM/FAM input helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from bed_reader import open_bed, read_f64
from bed_reader._open_bed import get_num_threads

from g import (
    _core,
    jax_setup,  # noqa: F401
)
from g.io.tabular import load_family_table
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


def read_bed_chunk_native(
    bed_path: Path,
    sample_index_array: np.ndarray,
    variant_start: int,
    variant_stop: int,
) -> jax.Array:
    """Read one BED chunk through the native Rust decoder.

    Args:
        bed_path: PLINK BED file path.
        sample_index_array: Contiguous sample indices.
        variant_start: Inclusive variant start index.
        variant_stop: Exclusive variant stop index.

    Returns:
        Genotype chunk as a JAX array with float64 precision.

    """
    native_chunk = _core.read_bed_chunk_f64(
        bed_path=bed_path,
        sample_indices=sample_index_array.tolist(),
        variant_start=variant_start,
        variant_stop=variant_stop,
    )
    genotype_matrix_host = np.frombuffer(native_chunk.genotype_values_le, dtype=np.float64).reshape(
        (native_chunk.sample_count, native_chunk.variant_count),
        order="C",
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


def validate_bed_sample_order(
    bed_prefix: Path,
    sample_index_array: np.ndarray,
    expected_individual_identifiers: np.ndarray,
) -> None:
    """Validate that FAM sample order matches the aligned sample order.

    Args:
        bed_prefix: PLINK dataset prefix.
        sample_index_array: Contiguous sample indices.
        expected_individual_identifiers: Expected sample order after alignment.

    Raises:
        ValueError: If FAM sample order does not match the aligned sample order.

    """
    family_table = load_family_table(bed_prefix.with_suffix(".fam"))
    observed_individual_identifiers = family_table.get_column("individual_identifier").to_numpy()[sample_index_array]
    if not np.array_equal(observed_individual_identifiers, expected_individual_identifiers):
        message = "BED sample order does not match the aligned phenotype/covariate order."
        raise ValueError(message)


def build_genotype_chunk(
    genotype_matrix: jax.Array,
    chromosome_values: np.ndarray,
    variant_identifier_values: np.ndarray,
    position_values: np.ndarray,
    allele_one_values: np.ndarray,
    allele_two_values: np.ndarray,
    variant_start: int,
    variant_stop: int,
) -> GenotypeChunk:
    """Build one mean-imputed genotype chunk from a raw genotype matrix.

    Args:
        genotype_matrix: Raw genotype matrix with NaNs for missing values.
        chromosome_values: Full chromosome value array.
        variant_identifier_values: Full variant identifier array.
        position_values: Full position array.
        allele_one_values: Full allele-one array.
        allele_two_values: Full allele-two array.
        variant_start: Inclusive variant start index.
        variant_stop: Exclusive variant stop index.

    Returns:
        Mean-imputed genotype chunk with metadata and summary statistics.

    """
    missing_mask = jnp.isnan(genotype_matrix)
    observed_genotype_total = jnp.where(missing_mask, 0.0, genotype_matrix).sum(axis=0)
    observation_count = jnp.sum(~missing_mask, axis=0, dtype=jnp.int64)
    column_means = observed_genotype_total / jnp.maximum(observation_count, 1)
    sanitized_column_means = jnp.where(observation_count > 0, column_means, 0.0)
    imputed_matrix = jnp.where(missing_mask, sanitized_column_means[None, :], genotype_matrix)
    allele_one_frequency = sanitized_column_means / 2.0

    return GenotypeChunk(
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


def iter_genotype_chunks(
    bed_prefix: Path,
    sample_indices: np.ndarray,
    expected_individual_identifiers: np.ndarray,
    chunk_size: int,
    variant_limit: int | None = None,
    *,
    use_native_reader: bool = False,
) -> Iterator[GenotypeChunk]:
    """Yield mean-imputed genotype chunks from a PLINK BED file.

    Args:
        bed_prefix: PLINK dataset prefix.
        sample_indices: Row indices to read from the BED file.
        expected_individual_identifiers: Expected sample order after alignment.
        chunk_size: Number of variants per chunk.
        variant_limit: Optional cap on the number of variants.
        use_native_reader: Whether to decode BED chunks through the Rust extension.

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

    validate_bed_sample_order(
        bed_prefix=bed_prefix,
        sample_index_array=sample_index_array,
        expected_individual_identifiers=expected_individual_identifiers,
    )

    if use_native_reader:
        for variant_start in range(0, total_variant_count, chunk_size):
            variant_stop = min(total_variant_count, variant_start + chunk_size)
            genotype_matrix = read_bed_chunk_native(
                bed_path=bed_path,
                sample_index_array=sample_index_array,
                variant_start=variant_start,
                variant_stop=variant_stop,
            )
            yield build_genotype_chunk(
                genotype_matrix=genotype_matrix,
                chromosome_values=chromosome_values,
                variant_identifier_values=variant_identifier_values,
                position_values=position_values,
                allele_one_values=allele_one_values,
                allele_two_values=allele_two_values,
                variant_start=variant_start,
                variant_stop=variant_stop,
            )
        return

    with open_bed(str(bed_path)) as bed_handle:
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
            yield build_genotype_chunk(
                genotype_matrix=genotype_matrix,
                chromosome_values=chromosome_values,
                variant_identifier_values=variant_identifier_values,
                position_values=position_values,
                allele_one_values=allele_one_values,
                allele_two_values=allele_two_values,
                variant_start=variant_start,
                variant_stop=variant_stop,
            )
