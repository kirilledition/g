"""BGEN input helpers using bgen-reader."""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import polars as pl
from bgen_reader import open_bgen

from g.io.plink import (
    build_genotype_chunk,
    build_linear_genotype_chunk,
    preprocess_genotype_matrix,
    preprocess_genotype_matrix_arrays,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from g.models import GenotypeChunk, LinearGenotypeChunk


def read_bgen_chunk_host(
    bgen_handle: open_bgen,
    sample_index_array: np.ndarray,
    variant_start: int,
    variant_stop: int,
) -> npt.NDArray[np.float32]:
    """Read one BGEN chunk into a host NumPy array of dosages.

    Args:
        bgen_handle: Open BGEN reader handle.
        sample_index_array: Contiguous sample indices.
        variant_start: Inclusive variant start index.
        variant_stop: Exclusive variant stop index.

    Returns:
        Genotype chunk as a host NumPy array with float32 precision.
        Values are expected dosages (0 to 2) with NaNs for missing.

    """
    variant_index_array = np.arange(variant_start, variant_stop, dtype=np.intp)
    raw = bgen_handle.read((sample_index_array, variant_index_array), dtype=np.float32)
    def compute(p: typing.Any) -> typing.Any:
        return p[:, :, 1] + 2.0 * p[:, :, 2]
    return compute(np.asarray(raw, dtype=np.float32))


def read_bgen_chunk(
    bgen_handle: open_bgen,
    sample_index_array: np.ndarray,
    variant_start: int,
    variant_stop: int,
) -> jax.Array:
    """Read one BGEN chunk into a JAX array.

    Args:
        bgen_handle: Open BGEN reader handle.
        sample_index_array: Contiguous sample indices.
        variant_start: Inclusive variant start index.
        variant_stop: Exclusive variant stop index.

    Returns:
        Genotype chunk as a float32 JAX array.

    """
    genotype_matrix_host = read_bgen_chunk_host(
        bgen_handle=bgen_handle,
        sample_index_array=sample_index_array,
        variant_start=variant_start,
        variant_stop=variant_stop,
    )
    return jnp.asarray(genotype_matrix_host, dtype=jnp.float32)


def load_bgen_variant_table(bgen_handle: open_bgen) -> pl.DataFrame:
    """Load variant metadata from BGEN file.

    Args:
        bgen_handle: Open BGEN reader handle.

    Returns:
        Parsed variant table matching the PLINK BIM format columns.

    """
    allele_ids = bgen_handle.allele_ids
    # bgen_reader allele_ids are comma separated strings, e.g. "A,G"
    alleles = [a.split(",") for a in allele_ids]
    allele_one = [a[0] if len(a) > 0 else "" for a in alleles]
    allele_two = [a[1] if len(a) > 1 else "" for a in alleles]

    return pl.DataFrame(
        {
            "chromosome": bgen_handle.chromosomes,
            "variant_identifier": bgen_handle.rsids,
            "genetic_distance": np.zeros(bgen_handle.nvariants, dtype=np.float32),  # BGEN usually lacks cM
            "position": bgen_handle.positions,
            "allele_one": allele_one,
            "allele_two": allele_two,
        }
    )


def validate_bgen_sample_order(
    bgen_handle: open_bgen,
    sample_index_array: np.ndarray,
    expected_individual_identifiers: np.ndarray,
    bgen_path: Path,
) -> None:
    """Validate that BGEN sample order matches the aligned sample order.

    Args:
        bgen_handle: Open BGEN reader handle.
        sample_index_array: Contiguous sample indices.
        expected_individual_identifiers: Expected sample order after alignment.
        bgen_path: Path to the BGEN file.

    Raises:
        ValueError: If BGEN sample order does not match the aligned sample order.

    """
    try:
        bgen_samples = bgen_handle.samples
    except RuntimeError as err:
        sample_path = bgen_path.with_suffix(".sample")
        if sample_path.exists():
            message = "BGEN file does not contain samples. Supplying external sample file is not yet fully supported."
            raise ValueError(message) from None
        message = "BGEN file does not contain samples and no .sample file found."
        raise ValueError(message) from err

    observed_individual_identifiers = bgen_samples[sample_index_array]
    if not np.array_equal(observed_individual_identifiers, expected_individual_identifiers):
        message = "BGEN sample order does not match the aligned phenotype/covariate order."
        raise ValueError(message)


def iter_genotype_chunks(
    bgen_path: Path,
    sample_indices: np.ndarray,
    expected_individual_identifiers: np.ndarray,
    chunk_size: int,
    variant_limit: int | None = None,
    *,
    include_missing_value_flag: bool = True,
) -> Iterator[GenotypeChunk]:
    """Yield mean-imputed genotype chunks from a BGEN file.

    Args:
        bgen_path: Path to BGEN file.
        sample_indices: Row indices to read from the BGEN file.
        expected_individual_identifiers: Expected sample order after alignment.
        chunk_size: Number of variants per chunk.
        variant_limit: Optional cap on the number of variants.
        include_missing_value_flag: Whether to materialize a host missing-value
            flag for each chunk.

    Yields:
        Mean-imputed genotype chunks with metadata.

    Raises:
        ValueError: If BGEN sample order does not match the aligned sample order.

    """
    with open_bgen(str(bgen_path), verbose=False) as bgen_handle:
        variant_table = load_bgen_variant_table(bgen_handle)
        total_variant_count = variant_table.height
        if variant_limit is not None:
            total_variant_count = min(variant_limit, variant_table.height)
        if total_variant_count == 0:
            return

        sample_index_array = np.ascontiguousarray(sample_indices, dtype=np.intp)
        chromosome_values = variant_table.get_column("chromosome").cast(pl.String).to_numpy()
        variant_identifier_values = variant_table.get_column("variant_identifier").cast(pl.String).to_numpy()
        position_values = variant_table.get_column("position").cast(pl.Int64).to_numpy()
        allele_one_values = variant_table.get_column("allele_one").cast(pl.String).to_numpy()
        allele_two_values = variant_table.get_column("allele_two").cast(pl.String).to_numpy()

        validate_bgen_sample_order(
            bgen_handle=bgen_handle,
            sample_index_array=sample_index_array,
            expected_individual_identifiers=expected_individual_identifiers,
            bgen_path=bgen_path,
        )

        for variant_start in range(0, total_variant_count, chunk_size):
            variant_stop = min(total_variant_count, variant_start + chunk_size)
            genotype_matrix_host = read_bgen_chunk_host(
                bgen_handle=bgen_handle,
                sample_index_array=sample_index_array,
                variant_start=variant_start,
                variant_stop=variant_stop,
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
    bgen_path: Path,
    sample_indices: np.ndarray,
    expected_individual_identifiers: np.ndarray,
    chunk_size: int,
    variant_limit: int | None = None,
) -> Iterator[LinearGenotypeChunk]:
    """Yield linear-regression genotype chunks without missingness bookkeeping."""
    with open_bgen(str(bgen_path), verbose=False) as bgen_handle:
        variant_table = load_bgen_variant_table(bgen_handle)
        total_variant_count = variant_table.height
        if variant_limit is not None:
            total_variant_count = min(variant_limit, variant_table.height)
        if total_variant_count == 0:
            return

        sample_index_array = np.ascontiguousarray(sample_indices, dtype=np.intp)
        chromosome_values = variant_table.get_column("chromosome").cast(pl.String).to_numpy()
        variant_identifier_values = variant_table.get_column("variant_identifier").cast(pl.String).to_numpy()
        position_values = variant_table.get_column("position").cast(pl.Int64).to_numpy()
        allele_one_values = variant_table.get_column("allele_one").cast(pl.String).to_numpy()
        allele_two_values = variant_table.get_column("allele_two").cast(pl.String).to_numpy()

        validate_bgen_sample_order(
            bgen_handle=bgen_handle,
            sample_index_array=sample_index_array,
            expected_individual_identifiers=expected_individual_identifiers,
            bgen_path=bgen_path,
        )

        for variant_start in range(0, total_variant_count, chunk_size):
            variant_stop = min(total_variant_count, variant_start + chunk_size)
            with jax.profiler.TraceAnnotation("linear.read_bgen_chunk_host"):
                genotype_matrix_host = read_bgen_chunk_host(
                    bgen_handle=bgen_handle,
                    sample_index_array=sample_index_array,
                    variant_start=variant_start,
                    variant_stop=variant_stop,
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
