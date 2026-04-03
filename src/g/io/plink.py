"""PLINK BED/BIM/FAM input helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import polars as pl
from bed_reader import open_bed, read_f32
from bed_reader._open_bed import get_num_threads

from g.io.genotype_processing import (
    build_genotype_chunk,
    preprocess_genotype_matrix,
    preprocess_genotype_matrix_arrays,
)
from g.io.reader import (
    ArrayMemoryOrder,
    iter_genotype_chunks_from_reader,
    iter_linear_genotype_chunks_from_reader,
    validate_sample_order,
)
from g.io.tabular import load_family_table

if TYPE_CHECKING:
    from collections.abc import Iterator

    from g.models import GenotypeChunk, LinearGenotypeChunk

VARIANT_TABLE_COLUMNS = (
    "chromosome",
    "variant_identifier",
    "genetic_distance",
    "position",
    "allele_one",
    "allele_two",
)

__all__ = (
    "PlinkReader",
    "build_genotype_chunk",
    "iter_genotype_chunks",
    "iter_linear_genotype_chunks",
    "load_variant_table",
    "preprocess_genotype_matrix",
    "preprocess_genotype_matrix_arrays",
    "read_bed_chunk",
    "read_bed_chunk_host",
    "validate_bed_sample_order",
)


class VariantSliceBounds(NamedTuple):
    """Normalized start and stop indices for one variant slice."""

    start: int
    stop: int


def normalize_variant_slice_bounds(variant_slice: slice, variant_count: int) -> VariantSliceBounds:
    """Resolve concrete start and stop indices for one variant slice."""
    return VariantSliceBounds(
        start=0 if variant_slice.start is None else int(variant_slice.start),
        stop=variant_count if variant_slice.stop is None else int(variant_slice.stop),
    )


class PlinkReader:
    """PLINK reader exposing a bed-reader-like public API."""

    def __init__(self, bed_prefix: Path | str) -> None:
        """Open one PLINK BED dataset."""
        self.bed_prefix = Path(bed_prefix)
        self.bed_path = self.bed_prefix.with_suffix(".bed")
        self.bed_handle = open_bed(str(self.bed_path))
        self.variant_table = load_variant_table(self.bed_prefix)
        self.num_threads = get_num_threads(getattr(self.bed_handle, "_num_threads", None))

    @property
    def sample_count(self) -> int:
        """Return the number of samples."""
        if hasattr(self.bed_handle, "iid_count"):
            return int(self.bed_handle.iid_count)
        if hasattr(self.bed_handle, "iid"):
            return len(self.bed_handle.iid)
        return 0

    @property
    def variant_count(self) -> int:
        """Return the number of variants."""
        if hasattr(self.bed_handle, "sid_count"):
            return int(self.bed_handle.sid_count)
        return int(self.variant_table.height)

    @property
    def samples(self) -> npt.NDArray[np.str_]:
        """Return sample identifiers in file order."""
        if not hasattr(self.bed_handle, "iid"):
            message = "PLINK reader handle does not expose sample identifiers."
            raise AttributeError(message)
        return np.asarray(self.bed_handle.iid, dtype=np.str_)

    def read(
        self,
        index: object = None,
        dtype: type[np.float32] | type[np.float64] = np.float32,
        order: ArrayMemoryOrder = "C",
    ) -> npt.NDArray[np.float32] | npt.NDArray[np.float64]:
        """Read PLINK genotypes with the same calling convention as `bed_handle.read`."""
        if (
            dtype is np.float32
            and order == "C"
            and isinstance(index, tuple)
            and len(index) == 2
            and isinstance(index[0], np.ndarray)
            and isinstance(index[1], slice)
            and (index[1].step is None or index[1].step == 1)
        ):
            variant_slice_bounds = normalize_variant_slice_bounds(index[1], self.variant_count)
            return read_bed_chunk_host(
                bed_handle=self.bed_handle,
                bed_path=self.bed_path,
                sample_index_array=np.ascontiguousarray(index[0], dtype=np.intp),
                variant_start=variant_slice_bounds.start,
                variant_stop=variant_slice_bounds.stop,
                num_threads=self.num_threads,
            )
        genotype_matrix = self.bed_handle.read(index=index, dtype=dtype, order=order)
        return np.asarray(genotype_matrix, dtype=dtype, order=order)

    def close(self) -> None:
        """Release the underlying BED handle."""
        self.bed_handle.__exit__(None, None, None)

    def __enter__(self) -> PlinkReader:
        """Return the open reader in a context manager."""
        return self

    def __exit__(self, exception_type: object, exception_value: object, traceback: object) -> None:
        """Close the reader when leaving a context manager."""
        del exception_type, exception_value, traceback
        self.close()


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
    validate_sample_order(
        observed_individual_identifiers=np.asarray(
            family_table.get_column("individual_identifier").cast(pl.String).to_numpy(),
            dtype=np.str_,
        ),
        sample_index_array=np.asarray(sample_index_array, dtype=np.intp),
        expected_individual_identifiers=np.asarray(expected_individual_identifiers, dtype=np.str_),
        source_name="BED",
    )


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
    sample_index_array = np.ascontiguousarray(sample_indices, dtype=np.intp)
    validate_bed_sample_order(
        bed_prefix=bed_prefix,
        sample_index_array=sample_index_array,
        expected_individual_identifiers=expected_individual_identifiers,
    )
    with PlinkReader(bed_prefix) as plink_reader:
        yield from iter_genotype_chunks_from_reader(
            genotype_reader=plink_reader,
            source_name="BED",
            sample_indices=sample_indices,
            expected_individual_identifiers=expected_individual_identifiers,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
            include_missing_value_flag=include_missing_value_flag,
            validate_sample_order_flag=False,
        )


def iter_linear_genotype_chunks(
    bed_prefix: Path,
    sample_indices: np.ndarray,
    expected_individual_identifiers: np.ndarray,
    chunk_size: int,
    variant_limit: int | None = None,
) -> Iterator[LinearGenotypeChunk]:
    """Yield linear-regression genotype chunks without missingness bookkeeping."""
    sample_index_array = np.ascontiguousarray(sample_indices, dtype=np.intp)
    validate_bed_sample_order(
        bed_prefix=bed_prefix,
        sample_index_array=sample_index_array,
        expected_individual_identifiers=expected_individual_identifiers,
    )
    with PlinkReader(bed_prefix) as plink_reader:
        yield from iter_linear_genotype_chunks_from_reader(
            genotype_reader=plink_reader,
            source_name="BED",
            sample_indices=sample_indices,
            expected_individual_identifiers=expected_individual_identifiers,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
            validate_sample_order_flag=False,
        )
