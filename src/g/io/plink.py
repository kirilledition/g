"""PLINK BED/BIM/FAM input helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import polars as pl
from bed_reader import open_bed, read_f32
from bed_reader._open_bed import get_num_threads

from g.io.genotype_processing import (
    build_genotype_chunk,
    build_linear_genotype_chunk,
    preprocess_genotype_matrix,
    preprocess_genotype_matrix_arrays,
)
from g.io.tabular import load_family_table

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from g.models import GenotypeChunk, LinearGenotypeChunk

VARIANT_TABLE_COLUMNS = (
    "chromosome",
    "variant_identifier",
    "genetic_distance",
    "position",
    "allele_one",
    "allele_two",
)


def read_bed_chunk_host(
    bed_handle: open_bed,
    bed_path: Path,
    sample_index_array: np.ndarray,
    variant_start: int,
    variant_stop: int,
    num_threads: int,
) -> npt.NDArray[np.float32]:
    """Read one BED chunk into a host NumPy array.

    Args:
        bed_handle: Open BED reader handle.
        bed_path: PLINK BED file path.
        sample_index_array: Contiguous sample indices.
        variant_start: Inclusive variant start index.
        variant_stop: Exclusive variant stop index.
        num_threads: Thread count for the BED reader.

    Returns:
        Genotype chunk as a host NumPy array with float32 precision.

    """
    variant_index_array = np.arange(variant_start, variant_stop, dtype=np.intp)
    genotype_matrix_host = np.zeros(
        (sample_index_array.shape[0], variant_index_array.shape[0]),
        dtype=np.float32,
        order="C",
    )
    read_f32(
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
    return genotype_matrix_host


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
        Genotype chunk as a float32 JAX array.

    """
    genotype_matrix_host = read_bed_chunk_host(
        bed_handle=bed_handle,
        bed_path=bed_path,
        sample_index_array=sample_index_array,
        variant_start=variant_start,
        variant_stop=variant_stop,
        num_threads=num_threads,
    )
    return jnp.asarray(genotype_matrix_host, dtype=jnp.float32)


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


def iter_genotype_chunks(
    bed_prefix: Path,
    sample_indices: np.ndarray,
    expected_individual_identifiers: np.ndarray,
    chunk_size: int,
    variant_limit: int | None = None,
    *,
    include_missing_value_flag: bool = True,
) -> Iterator[GenotypeChunk]:
    """Yield mean-imputed genotype chunks from a PLINK BED file.

    Args:
        bed_prefix: PLINK dataset prefix.
        sample_indices: Row indices to read from the BED file.
        expected_individual_identifiers: Expected sample order after alignment.
        chunk_size: Number of variants per chunk.
        variant_limit: Optional cap on the number of variants.
        include_missing_value_flag: Whether to materialize a host missing-value
            flag for each chunk.

    Yields:
        Mean-imputed genotype chunks with metadata.

    Raises:
        ValueError: If BED sample order does not match the aligned sample order.

    """
    variant_table = load_variant_table(bed_prefix)
    total_variant_count = variant_table.height if variant_limit is None else min(variant_limit, variant_table.height)
    if total_variant_count == 0:
        return
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

    with open_bed(str(bed_path)) as bed_handle:
        num_threads = get_num_threads(getattr(bed_handle, "_num_threads", None))

        for variant_start in range(0, total_variant_count, chunk_size):
            variant_stop = min(total_variant_count, variant_start + chunk_size)
            genotype_matrix_host = read_bed_chunk_host(
                bed_handle=bed_handle,
                bed_path=bed_path,
                sample_index_array=sample_index_array,
                variant_start=variant_start,
                variant_stop=variant_stop,
                num_threads=num_threads,
            )
            preprocessed_chunk_data = preprocess_genotype_matrix(
                jax.device_put(genotype_matrix_host),
                include_missing_value_flag=include_missing_value_flag,
            )
            yield build_genotype_chunk(
                preprocessed_chunk_data=preprocessed_chunk_data,
                chromosome_values=chromosome_values,
                variant_identifier_values=variant_identifier_values,
                position_values=position_values,
                allele_one_values=allele_one_values,
                allele_two_values=allele_two_values,
                variant_start=variant_start,
                variant_stop=variant_stop,
            )


def iter_linear_genotype_chunks(
    bed_prefix: Path,
    sample_indices: np.ndarray,
    expected_individual_identifiers: np.ndarray,
    chunk_size: int,
    variant_limit: int | None = None,
) -> Iterator[LinearGenotypeChunk]:
    """Yield linear-regression genotype chunks without missingness bookkeeping."""
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

    with open_bed(str(bed_path)) as bed_handle:
        num_threads = get_num_threads(getattr(bed_handle, "_num_threads", None))

        for variant_start in range(0, total_variant_count, chunk_size):
            variant_stop = min(total_variant_count, variant_start + chunk_size)
            with jax.profiler.TraceAnnotation("linear.read_bed_chunk_host"):
                genotype_matrix_host = read_bed_chunk_host(
                    bed_handle=bed_handle,
                    bed_path=bed_path,
                    sample_index_array=sample_index_array,
                    variant_start=variant_start,
                    variant_stop=variant_stop,
                    num_threads=num_threads,
                )
            with jax.profiler.TraceAnnotation("linear.device_put_genotypes"):
                genotype_matrix_device = jax.device_put(genotype_matrix_host)
            with jax.profiler.TraceAnnotation("linear.preprocess_genotypes"):
                preprocessed_genotype_arrays = preprocess_genotype_matrix_arrays(genotype_matrix_device)
            with jax.profiler.TraceAnnotation("linear.build_chunk"):
                yield build_linear_genotype_chunk(
                    preprocessed_genotype_arrays=preprocessed_genotype_arrays,
                    chromosome_values=chromosome_values,
                    variant_identifier_values=variant_identifier_values,
                    position_values=position_values,
                    allele_one_values=allele_one_values,
                    allele_two_values=allele_two_values,
                    variant_start=variant_start,
                    variant_stop=variant_stop,
                )
