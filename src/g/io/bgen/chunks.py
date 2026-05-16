"""Chunked BGEN reads and sample-order validation helpers."""

from __future__ import annotations

import typing

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from g import types
from g.io import models, reader
from g.io.bgen import sample
from g.io.bgen.reader import BgenReader, open_bgen

if typing.TYPE_CHECKING:
    import collections.abc
    from pathlib import Path

    import jax
    import polars as pl


def read_bgen_chunk_host(
    bgen_reader: BgenReader,
    sample_index_array: npt.NDArray[np.intp],
    variant_start: int,
    variant_stop: int,
) -> npt.NDArray[np.float32]:
    """Read one BGEN dosage chunk into a host NumPy array."""
    genotype_matrix_host = bgen_reader.read_float32(
        np.ascontiguousarray(sample_index_array, dtype=np.int64), variant_start, variant_stop
    )
    return np.asarray(genotype_matrix_host, dtype=np.float32, order=types.ArrayMemoryOrder.C_CONTIGUOUS.value)


def read_bgen_chunk(
    bgen_reader: BgenReader, sample_index_array: npt.NDArray[np.intp], variant_start: int, variant_stop: int
) -> jax.Array:
    """Read one BGEN dosage chunk into a JAX array."""
    return jnp.asarray(
        read_bgen_chunk_host(
            bgen_reader=bgen_reader,
            sample_index_array=sample_index_array,
            variant_start=variant_start,
            variant_stop=variant_stop,
        ),
        dtype=jnp.float32,
    )


def validate_bgen_sample_order(
    bgen_reader: BgenReader,
    sample_index_array: npt.NDArray[np.intp],
    expected_individual_identifiers: npt.NDArray[np.str_],
    bgen_path: Path,
) -> None:
    """Validate that BGEN sample order matches the aligned sample order."""
    del bgen_path
    if bgen_reader.sample_identifier_source == types.SampleIdentifierSource.GENERATED:
        raise ValueError("BGEN file does not contain samples and no .sample file was found.")
    reader.validate_sample_order(
        observed_individual_identifiers=bgen_reader.samples,
        sample_index_array=sample_index_array,
        expected_individual_identifiers=expected_individual_identifiers,
        source_name="BGEN",
    )


def iter_genotype_chunks(
    bgen_path: Path,
    sample_indices: npt.NDArray[np.int64],
    expected_individual_identifiers: npt.NDArray[np.str_],
    chunk_size: int,
    variant_limit: int | None = None,
    *,
    include_missing_value_flag: bool = True,
    sample_path: Path | None = None,
) -> collections.abc.Iterator[models.GenotypeChunk]:
    """Yield mean-imputed genotype chunks from a BGEN file."""
    with open_bgen(bgen_path, sample_path=sample_path) as bgen_reader:
        if bgen_reader.sample_identifier_source == types.SampleIdentifierSource.GENERATED:
            raise ValueError("BGEN file does not contain samples and no .sample file was found.")
        yield from reader.iter_genotype_chunks_from_reader(
            genotype_reader=bgen_reader,
            source_name="BGEN",
            sample_indices=sample_indices,
            expected_individual_identifiers=expected_individual_identifiers,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
            include_missing_value_flag=include_missing_value_flag,
        )


def iter_dosage_genotype_chunks(
    bgen_path: Path,
    sample_indices: npt.NDArray[np.int64],
    expected_individual_identifiers: npt.NDArray[np.str_],
    chunk_size: int,
    variant_limit: int | None = None,
    *,
    sample_path: Path | None = None,
) -> collections.abc.Iterator[models.DosageGenotypeChunk]:
    """Yield raw dosage genotype chunks from a BGEN file."""
    with open_bgen(bgen_path, sample_path=sample_path) as bgen_reader:
        if bgen_reader.sample_identifier_source == types.SampleIdentifierSource.GENERATED:
            raise ValueError("BGEN file does not contain samples and no .sample file was found.")
        yield from reader.iter_dosage_genotype_chunks_from_reader(
            genotype_reader=bgen_reader,
            source_name="BGEN",
            sample_indices=sample_indices,
            expected_individual_identifiers=expected_individual_identifiers,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
        )


def load_bgen_sample_table(bgen_path: Path, sample_path: Path | None = None) -> pl.DataFrame:
    """Load BGEN sample identifiers into a normalized identifier table."""
    resolved_sample_path = sample.resolve_bgen_sample_path(bgen_path, sample_path)
    if resolved_sample_path is not None:
        sample_table = sample.load_sample_identifier_table(resolved_sample_path)
        with open_bgen(bgen_path, sample_path=resolved_sample_path) as bgen_reader:
            if sample_table.height != bgen_reader.sample_count:
                message = (
                    f"Expect number of samples in file to match BGEN sample count. "
                    f"Sample file '{resolved_sample_path}' contains {sample_table.height} rows, "
                    f"but '{bgen_path}' contains {bgen_reader.sample_count} samples."
                )
                raise ValueError(message)
        return sample_table
    with open_bgen(bgen_path) as bgen_reader:
        if bgen_reader.sample_identifier_source == types.SampleIdentifierSource.GENERATED:
            raise ValueError("BGEN file does not contain samples and no .sample file was found.")
        return sample.build_sample_identifier_table(np.asarray(bgen_reader.samples, dtype=np.str_))
