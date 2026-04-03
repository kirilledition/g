"""Format-agnostic genotype reader interfaces and shared iteration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import jax
import numpy as np
import numpy.typing as npt
import polars as pl

from g.io.genotype_processing import preprocess_genotype_matrix, preprocess_genotype_matrix_arrays
from g.types import ArrayMemoryOrder

if TYPE_CHECKING:
    from collections.abc import Iterator

    from g.models import GenotypeChunk, LinearGenotypeChunk, VariantMetadata

@dataclass(frozen=True)
class VariantTableArrays:
    """Numpy views of variant metadata columns used during chunk construction.

    Attributes:
        chromosome_values: Chromosome identifiers per variant.
        variant_identifier_values: Variant identifiers per variant.
        position_values: Genomic positions per variant.
        allele_one_values: First allele per variant.
        allele_two_values: Second allele per variant.

    """

    chromosome_values: npt.NDArray[np.str_]
    variant_identifier_values: npt.NDArray[np.str_]
    position_values: npt.NDArray[np.int64]
    allele_one_values: npt.NDArray[np.str_]
    allele_two_values: npt.NDArray[np.str_]


@runtime_checkable
class GenotypeReader(Protocol):
    """Protocol implemented by concrete genotype readers."""

    @property
    def sample_count(self) -> int:
        """Return the number of samples."""

    @property
    def variant_count(self) -> int:
        """Return the number of variants."""

    @property
    def samples(self) -> npt.NDArray[np.str_]:
        """Return sample identifiers in file order."""

    @property
    def variant_table(self) -> pl.DataFrame:
        """Return normalized variant metadata."""

    def get_variant_table_arrays(self, variant_start: int, variant_stop: int) -> VariantTableArrays:
        """Return normalized metadata arrays for one variant slice."""

    def read(
        self,
        index: object = None,
        dtype: type[np.float32] | type[np.float64] = np.float32,
        order: ArrayMemoryOrder = ArrayMemoryOrder.C_CONTIGUOUS,
    ) -> npt.NDArray[np.float32] | npt.NDArray[np.float64]:
        """Read a genotype matrix subset as dosages with NaN for missing values."""

    def close(self) -> None:
        """Release any reader resources."""

    def __enter__(self) -> GenotypeReader:
        """Enter a context manager and return the reader."""

    def __exit__(self, exception_type: object, exception_value: object, traceback: object) -> None:
        """Exit a context manager and release resources."""


def build_variant_table_arrays(variant_table: pl.DataFrame) -> VariantTableArrays:
    """Convert normalized variant metadata into numpy arrays."""
    return VariantTableArrays(
        chromosome_values=np.asarray(variant_table.get_column("chromosome").cast(pl.String).to_numpy(), dtype=np.str_),
        variant_identifier_values=np.asarray(
            variant_table.get_column("variant_identifier").cast(pl.String).to_numpy(),
            dtype=np.str_,
        ),
        position_values=np.asarray(variant_table.get_column("position").cast(pl.Int64).to_numpy(), dtype=np.int64),
        allele_one_values=np.asarray(variant_table.get_column("allele_one").cast(pl.String).to_numpy(), dtype=np.str_),
        allele_two_values=np.asarray(variant_table.get_column("allele_two").cast(pl.String).to_numpy(), dtype=np.str_),
    )


def resolve_total_variant_count(variant_count: int, variant_limit: int | None) -> int:
    """Resolve the effective number of variants to iterate."""
    if variant_limit is None:
        return variant_count
    return min(variant_count, variant_limit)


def validate_sample_order(
    observed_individual_identifiers: npt.NDArray[np.str_],
    sample_index_array: npt.NDArray[np.intp],
    expected_individual_identifiers: npt.NDArray[np.str_],
    source_name: str,
) -> None:
    """Validate that reader sample order matches aligned sample order."""
    selected_individual_identifiers = observed_individual_identifiers[sample_index_array]
    if not np.array_equal(selected_individual_identifiers, expected_individual_identifiers):
        message = f"{source_name} sample order does not match the aligned phenotype/covariate order."
        raise ValueError(message)


def build_variant_metadata(
    variant_table_arrays: VariantTableArrays,
    variant_start: int,
    variant_stop: int,
) -> VariantMetadata:
    """Build one variant-metadata object from already-sliced arrays."""
    from g.models import VariantMetadata

    return VariantMetadata(
        variant_start_index=variant_start,
        variant_stop_index=variant_stop,
        chromosome=variant_table_arrays.chromosome_values,
        variant_identifiers=variant_table_arrays.variant_identifier_values,
        position=variant_table_arrays.position_values,
        allele_one=variant_table_arrays.allele_one_values,
        allele_two=variant_table_arrays.allele_two_values,
    )


def iter_genotype_chunks_from_reader(
    genotype_reader: GenotypeReader,
    source_name: str,
    sample_indices: npt.NDArray[np.int64],
    expected_individual_identifiers: npt.NDArray[np.str_],
    chunk_size: int,
    variant_limit: int | None = None,
    *,
    include_missing_value_flag: bool = True,
    validate_sample_order_flag: bool = True,
) -> Iterator[GenotypeChunk]:
    """Yield mean-imputed genotype chunks from any compatible reader."""
    total_variant_count = resolve_total_variant_count(genotype_reader.variant_count, variant_limit)
    if total_variant_count == 0:
        return

    sample_index_array = np.ascontiguousarray(sample_indices, dtype=np.intp)
    if validate_sample_order_flag:
        validate_sample_order(
            observed_individual_identifiers=genotype_reader.samples,
            sample_index_array=sample_index_array,
            expected_individual_identifiers=expected_individual_identifiers,
            source_name=source_name,
        )

    for variant_start in range(0, total_variant_count, chunk_size):
        variant_stop = min(total_variant_count, variant_start + chunk_size)
        variant_table_arrays = genotype_reader.get_variant_table_arrays(variant_start, variant_stop)
        genotype_matrix_host = genotype_reader.read(
            index=(sample_index_array, slice(variant_start, variant_stop)),
            dtype=np.float32,
            order=ArrayMemoryOrder.C_CONTIGUOUS,
        )
        preprocessed_chunk_data = preprocess_genotype_matrix(
            jax.device_put(genotype_matrix_host),
            include_missing_value_flag=include_missing_value_flag,
        )
        from g.models import GenotypeChunk

        yield GenotypeChunk(
            genotypes=preprocessed_chunk_data.genotypes,
            missing_mask=preprocessed_chunk_data.missing_mask,
            has_missing_values=preprocessed_chunk_data.has_missing_values,
            metadata=build_variant_metadata(variant_table_arrays, variant_start, variant_stop),
            allele_one_frequency=preprocessed_chunk_data.allele_one_frequency,
            observation_count=preprocessed_chunk_data.observation_count,
        )


def iter_linear_genotype_chunks_from_reader(
    genotype_reader: GenotypeReader,
    source_name: str,
    sample_indices: npt.NDArray[np.int64],
    expected_individual_identifiers: npt.NDArray[np.str_],
    chunk_size: int,
    variant_limit: int | None = None,
    *,
    validate_sample_order_flag: bool = True,
) -> Iterator[LinearGenotypeChunk]:
    """Yield linear-regression genotype chunks from any compatible reader."""
    total_variant_count = resolve_total_variant_count(genotype_reader.variant_count, variant_limit)
    if total_variant_count == 0:
        return

    sample_index_array = np.ascontiguousarray(sample_indices, dtype=np.intp)
    if validate_sample_order_flag:
        validate_sample_order(
            observed_individual_identifiers=genotype_reader.samples,
            sample_index_array=sample_index_array,
            expected_individual_identifiers=expected_individual_identifiers,
            source_name=source_name,
        )

    trace_prefix = source_name.lower()
    for variant_start in range(0, total_variant_count, chunk_size):
        variant_stop = min(total_variant_count, variant_start + chunk_size)
        variant_table_arrays = genotype_reader.get_variant_table_arrays(variant_start, variant_stop)
        with jax.profiler.TraceAnnotation(f"linear.read_{trace_prefix}_chunk"):
            genotype_matrix_host = genotype_reader.read(
                index=(sample_index_array, slice(variant_start, variant_stop)),
                dtype=np.float32,
                order=ArrayMemoryOrder.C_CONTIGUOUS,
            )
        with jax.profiler.TraceAnnotation("linear.device_put_genotypes"):
            genotype_matrix_device = jax.device_put(genotype_matrix_host)
        with jax.profiler.TraceAnnotation("linear.preprocess_genotypes"):
            preprocessed_genotype_arrays = preprocess_genotype_matrix_arrays(genotype_matrix_device)
        with jax.profiler.TraceAnnotation("linear.build_chunk"):
            from g.models import LinearGenotypeChunk

            yield LinearGenotypeChunk(
                genotypes=preprocessed_genotype_arrays.genotypes,
                metadata=build_variant_metadata(variant_table_arrays, variant_start, variant_stop),
                allele_one_frequency=preprocessed_genotype_arrays.allele_one_frequency,
                observation_count=preprocessed_genotype_arrays.observation_count,
            )
