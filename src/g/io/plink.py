"""PLINK BED/BIM/FAM input helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import polars as pl
from bed_reader import open_bed, read_f64
from bed_reader._open_bed import get_num_threads

from g import (
    _core,
    jax_setup,  # noqa: F401
)
from g.io.tabular import load_family_table
from g.models import GenotypeChunk, PreprocessedGenotypeChunkData, VariantMetadata

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


def read_bed_chunk_host(
    bed_handle: open_bed,
    bed_path: Path,
    sample_index_array: np.ndarray,
    variant_start: int,
    variant_stop: int,
    num_threads: int,
) -> npt.NDArray[np.float64]:
    """Read one BED chunk into a host NumPy array.

    Args:
        bed_handle: Open BED reader handle.
        bed_path: PLINK BED file path.
        sample_index_array: Contiguous sample indices.
        variant_start: Inclusive variant start index.
        variant_stop: Exclusive variant stop index.
        num_threads: Thread count for the BED reader.

    Returns:
        Genotype chunk as a host NumPy array with float64 precision.

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
        Genotype chunk as a JAX array with float64 precision.

    """
    return jax.device_put(
        read_bed_chunk_host(
            bed_handle=bed_handle,
            bed_path=bed_path,
            sample_index_array=sample_index_array,
            variant_start=variant_start,
            variant_stop=variant_stop,
            num_threads=num_threads,
        )
    )


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


def preprocess_genotype_matrix(genotype_matrix: jax.Array) -> PreprocessedGenotypeChunkData:
    """Preprocess a raw genotype matrix with Phase 1 semantics.

    Args:
        genotype_matrix: Raw genotype matrix with NaNs for missing values.

    Returns:
        Mean-imputed genotype arrays and summary values.

    """
    missing_mask = jnp.isnan(genotype_matrix)
    observed_genotype_total = jnp.where(missing_mask, 0.0, genotype_matrix).sum(axis=0)
    observation_count = jnp.sum(~missing_mask, axis=0, dtype=jnp.int64)
    column_means = observed_genotype_total / jnp.maximum(observation_count, 1)
    sanitized_column_means = jnp.where(observation_count > 0, column_means, 0.0)
    imputed_matrix = jnp.where(missing_mask, sanitized_column_means[None, :], genotype_matrix)
    allele_one_frequency = sanitized_column_means / 2.0
    return PreprocessedGenotypeChunkData(
        genotypes=imputed_matrix,
        missing_mask=missing_mask,
        has_missing_values=bool(jax.device_get(jnp.any(missing_mask))),
        allele_one_frequency=allele_one_frequency,
        observation_count=observation_count,
    )


def preprocess_genotype_matrix_native(
    genotype_matrix_host: npt.NDArray[np.float64],
) -> PreprocessedGenotypeChunkData:
    """Preprocess a raw genotype matrix through the Rust extension.

    Args:
        genotype_matrix_host: Raw genotype matrix with NaNs for missing values.

    Returns:
        Mean-imputed genotype arrays and summary values.

    """
    native_result = _core.preprocess_genotype_matrix_f64(genotype_matrix=genotype_matrix_host)
    imputed_genotype_matrix = np.frombuffer(
        memoryview(native_result.imputed_genotype_values), dtype=np.float64
    ).reshape(
        (native_result.sample_count, native_result.variant_count),
        order="C",
    )
    missing_mask = np.frombuffer(memoryview(native_result.missing_mask_values), dtype=np.uint8).reshape(
        (native_result.sample_count, native_result.variant_count),
        order="C",
    )
    allele_one_frequency = np.frombuffer(memoryview(native_result.allele_one_frequency_values), dtype=np.float64)
    observation_count = np.frombuffer(memoryview(native_result.observation_count_values), dtype=np.int64)
    return PreprocessedGenotypeChunkData(
        genotypes=jax.device_put(imputed_genotype_matrix),
        missing_mask=jax.device_put(missing_mask.view(np.bool_)),
        has_missing_values=bool(np.any(missing_mask)),
        allele_one_frequency=jax.device_put(allele_one_frequency),
        observation_count=jax.device_put(observation_count),
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
    preprocessed_chunk_data: PreprocessedGenotypeChunkData,
    chromosome_values: np.ndarray,
    variant_identifier_values: np.ndarray,
    position_values: np.ndarray,
    allele_one_values: np.ndarray,
    allele_two_values: np.ndarray,
    variant_start: int,
    variant_stop: int,
) -> GenotypeChunk:
    """Build one genotype chunk from preprocessed arrays and metadata.

    Args:
        preprocessed_chunk_data: Preprocessed genotype arrays and summary values.
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
    return GenotypeChunk(
        genotypes=preprocessed_chunk_data.genotypes,
        missing_mask=preprocessed_chunk_data.missing_mask,
        has_missing_values=preprocessed_chunk_data.has_missing_values,
        metadata=VariantMetadata(
            chromosome=chromosome_values[variant_start:variant_stop],
            variant_identifiers=variant_identifier_values[variant_start:variant_stop],
            position=position_values[variant_start:variant_stop],
            allele_one=allele_one_values[variant_start:variant_stop],
            allele_two=allele_two_values[variant_start:variant_stop],
        ),
        allele_one_frequency=preprocessed_chunk_data.allele_one_frequency,
        observation_count=preprocessed_chunk_data.observation_count,
    )


def iter_genotype_chunks(
    bed_prefix: Path,
    sample_indices: np.ndarray,
    expected_individual_identifiers: np.ndarray,
    chunk_size: int,
    variant_limit: int | None = None,
    *,
    use_native_reader: bool = False,
    use_native_preprocessing: bool = False,
) -> Iterator[GenotypeChunk]:
    """Yield mean-imputed genotype chunks from a PLINK BED file.

    Args:
        bed_prefix: PLINK dataset prefix.
        sample_indices: Row indices to read from the BED file.
        expected_individual_identifiers: Expected sample order after alignment.
        chunk_size: Number of variants per chunk.
        variant_limit: Optional cap on the number of variants.
        use_native_reader: Whether to decode BED chunks through the Rust extension.
        use_native_preprocessing: Whether to preprocess BED chunks through the Rust extension.

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
            preprocessed_chunk_data = (
                preprocess_genotype_matrix_native(jax.device_get(genotype_matrix))
                if use_native_preprocessing
                else preprocess_genotype_matrix(genotype_matrix)
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
        return

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
            preprocessed_chunk_data = (
                preprocess_genotype_matrix_native(genotype_matrix_host)
                if use_native_preprocessing
                else preprocess_genotype_matrix(jax.device_put(genotype_matrix_host))
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
