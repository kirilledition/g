"""BGEN input helpers with a bed-reader-like API."""

from __future__ import annotations

import importlib
import typing
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import polars as pl

from g.io.reader import (
    ArrayMemoryOrder,
    iter_genotype_chunks_from_reader,
    iter_linear_genotype_chunks_from_reader,
    validate_sample_order,
)
from g.io.sample import build_sample_identifier_table, load_sample_identifier_table, resolve_bgen_sample_path

if TYPE_CHECKING:
    from collections.abc import Iterator

    from g.models import GenotypeChunk, LinearGenotypeChunk


OpenBgenHandle = typing.Any
SampleIdentifierSource = Literal["embedded", "external", "generated"]


def load_backend_open_bgen() -> typing.Any:
    """Import the optional BGEN backend lazily."""
    try:
        return importlib.import_module("bgen_reader").open_bgen
    except ModuleNotFoundError as error:
        message = (
            "BGEN support requires the `bgen-reader` stack. "
            "Run the command inside `nix develop` and sync dependencies again."
        )
        raise ModuleNotFoundError(message) from error


def resolve_variant_identifier_values(
    variant_identifier_values: npt.NDArray[np.str_],
    rsid_values: npt.NDArray[np.str_],
) -> npt.NDArray[np.str_]:
    """Prefer rsids and fall back to file-specific variant identifiers."""
    resolved_variant_identifier_values = np.where(rsid_values != "", rsid_values, variant_identifier_values)
    return np.asarray(resolved_variant_identifier_values, dtype=np.str_)


def convert_probability_tensor_to_dosage(
    probability_tensor: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    combination_count: int,
    *,
    is_phased: bool,
    dtype: type[np.float32] | type[np.float64],
    order: ArrayMemoryOrder,
) -> npt.NDArray[np.float32] | npt.NDArray[np.float64]:
    """Convert a BGEN probability tensor into additive dosages.

    For diploid unphased biallelic data the tensor stores three genotype
    probabilities ordered by alternative-allele count: 0, 1, 2.

    For diploid phased biallelic data the tensor stores two one-hot haplotype
    distributions: [haplotype_1_ref, haplotype_1_alt, haplotype_2_ref,
    haplotype_2_alt].
    """
    if combination_count == 3 and not is_phased:
        dosage_matrix = probability_tensor[:, :, 1] + (2.0 * probability_tensor[:, :, 2])
        return np.asarray(dosage_matrix, dtype=dtype, order=order)
    if combination_count == 4 and is_phased:
        dosage_matrix = probability_tensor[:, :, 1] + probability_tensor[:, :, 3]
        return np.asarray(dosage_matrix, dtype=dtype, order=order)
    message = "Unsupported BGEN probability layout. Only diploid biallelic phased or unphased variants are supported."
    raise ValueError(message)


def build_bgen_variant_table(bgen_handle: OpenBgenHandle) -> pl.DataFrame:
    """Build normalized variant metadata from an open BGEN handle."""
    allele_identifier_values = np.asarray(bgen_handle.allele_ids, dtype=np.str_)
    allele_pairs = [allele_identifier_value.split(",") for allele_identifier_value in allele_identifier_values]
    counted_allele_values = [allele_pair[-1] if allele_pair else "" for allele_pair in allele_pairs]
    reference_allele_values = [allele_pair[0] if allele_pair else "" for allele_pair in allele_pairs]
    variant_identifier_values = np.asarray(bgen_handle.ids, dtype=np.str_)
    rsid_values = np.asarray(bgen_handle.rsids, dtype=np.str_)
    return pl.DataFrame(
        {
            "chromosome": np.asarray(bgen_handle.chromosomes, dtype=np.str_),
            "variant_identifier": resolve_variant_identifier_values(variant_identifier_values, rsid_values),
            "genetic_distance": np.zeros(int(bgen_handle.nvariants), dtype=np.float32),
            "position": np.asarray(bgen_handle.positions, dtype=np.int64),
            "allele_one": np.asarray(counted_allele_values, dtype=np.str_),
            "allele_two": np.asarray(reference_allele_values, dtype=np.str_),
        }
    )


def resolve_sample_identifier_source(bgen_handle: OpenBgenHandle, sample_path: Path | None) -> SampleIdentifierSource:
    """Resolve where sample identifiers originated for one open BGEN handle."""
    if sample_path is not None:
        return "external"
    if bool(bgen_handle._cbgen.contain_samples):
        return "embedded"
    return "generated"


class BgenReader:
    """BGEN reader exposing a bed-reader-like public API."""

    def __init__(
        self,
        bgen_path: Path | str,
        sample_path: Path | str | None = None,
        *,
        allow_complex: bool = False,
    ) -> None:
        """Open one BGEN file.

        Args:
            bgen_path: Path to the `.bgen` file.
            sample_path: Optional explicit `.sample` file path.
            allow_complex: Forwarded to the backend for complex BGEN metadata.

        Raises:
            ValueError: The file uses an unsupported genotype layout.

        """
        self.bgen_path = Path(bgen_path)
        self.sample_path = resolve_bgen_sample_path(
            self.bgen_path,
            Path(sample_path) if sample_path is not None else None,
        )
        backend_open_bgen = load_backend_open_bgen()
        self.backend_handle = backend_open_bgen(
            self.bgen_path,
            samples_filepath=self.sample_path,
            allow_complex=allow_complex,
            verbose=False,
        )
        self.sample_identifier_source = resolve_sample_identifier_source(self.backend_handle, self.sample_path)
        self.combination_count = int(np.asarray(self.backend_handle.ncombinations, dtype=np.int32)[0])
        self.is_phased = bool(np.asarray(self.backend_handle.phased, dtype=np.bool_)[0])
        self.variant_table = build_bgen_variant_table(self.backend_handle)
        self.validate_supported_layout()

    @property
    def sample_count(self) -> int:
        """Return the number of samples."""
        return int(self.backend_handle.nsamples)

    @property
    def variant_count(self) -> int:
        """Return the number of variants."""
        return int(self.backend_handle.nvariants)

    @property
    def samples(self) -> npt.NDArray[np.str_]:
        """Return sample identifiers in file order."""
        return np.asarray(self.backend_handle.samples, dtype=np.str_)

    def validate_supported_layout(self) -> None:
        """Validate that the file layout can be converted into additive dosages."""
        allele_count_values = np.asarray(self.backend_handle.nalleles, dtype=np.int32)
        if not np.all(allele_count_values == 2):
            message = "Only diploid biallelic BGEN variants are supported. This matches the UK Biobank release format."
            raise ValueError(message)
        if self.combination_count == 3 and not self.is_phased:
            return
        if self.combination_count == 4 and self.is_phased:
            return
        message = (
            "Unsupported BGEN probability layout. Only diploid biallelic phased or unphased variants are supported."
        )
        raise ValueError(message)

    def read(
        self,
        index: object = None,
        dtype: type[np.float32] | type[np.float64] = np.float32,
        order: ArrayMemoryOrder = "C",
    ) -> npt.NDArray[np.float32] | npt.NDArray[np.float64]:
        """Read BGEN dosages with the same calling convention as `bed_handle.read`."""
        probability_tensor = self.backend_handle.read(index=index, dtype=dtype, order=order)
        return convert_probability_tensor_to_dosage(
            probability_tensor=np.asarray(probability_tensor, dtype=dtype, order=order),
            combination_count=self.combination_count,
            is_phased=self.is_phased,
            dtype=dtype,
            order=order,
        )

    def close(self) -> None:
        """Close the underlying BGEN handle."""
        self.backend_handle.close()

    def __enter__(self) -> BgenReader:
        """Return the open reader in a context manager."""
        return self

    def __exit__(self, exception_type: object, exception_value: object, traceback: object) -> None:
        """Close the reader when leaving a context manager."""
        del exception_type, exception_value, traceback
        self.close()


def open_bgen(
    bgen_path: Path | str,
    sample_path: Path | str | None = None,
    *,
    allow_complex: bool = False,
) -> BgenReader:
    """Open one BGEN file with a bed-reader-like wrapper."""
    return BgenReader(bgen_path=bgen_path, sample_path=sample_path, allow_complex=allow_complex)


def load_bgen_sample_table(bgen_path: Path, sample_path: Path | None = None) -> pl.DataFrame:
    """Load BGEN sample identifiers into a normalized identifier table."""
    resolved_sample_path = resolve_bgen_sample_path(bgen_path, sample_path)
    backend_open_bgen = load_backend_open_bgen()
    if resolved_sample_path is not None:
        sample_table = load_sample_identifier_table(resolved_sample_path)
        with backend_open_bgen(
            bgen_path,
            samples_filepath=resolved_sample_path,
            allow_complex=True,
            verbose=False,
        ) as bgen_handle:
            if sample_table.height != int(bgen_handle.nsamples):
                message = (
                    f"Sample file '{resolved_sample_path}' contains {sample_table.height} rows, "
                    f"but '{bgen_path}' contains {int(bgen_handle.nsamples)} samples."
                )
                raise ValueError(message)
        return sample_table

    with backend_open_bgen(bgen_path, allow_complex=True, verbose=False) as bgen_handle:
        if resolve_sample_identifier_source(bgen_handle, None) == "generated":
            message = "BGEN file does not contain samples and no .sample file was found."
            raise ValueError(message)
        return build_sample_identifier_table(np.asarray(bgen_handle.samples, dtype=np.str_))


def read_bgen_chunk_host(
    bgen_reader: BgenReader,
    sample_index_array: npt.NDArray[np.intp],
    variant_start: int,
    variant_stop: int,
) -> npt.NDArray[np.float32]:
    """Read one BGEN chunk into a host NumPy array of dosages."""
    genotype_matrix_host = bgen_reader.read(
        index=(sample_index_array, slice(variant_start, variant_stop)),
        dtype=np.float32,
        order="C",
    )
    return np.asarray(genotype_matrix_host, dtype=np.float32, order="C")


def read_bgen_chunk(
    bgen_reader: BgenReader,
    sample_index_array: npt.NDArray[np.intp],
    variant_start: int,
    variant_stop: int,
) -> jax.Array:
    """Read one BGEN chunk into a JAX array."""
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
    if bgen_reader.sample_identifier_source == "generated":
        message = "BGEN file does not contain samples and no .sample file was found."
        raise ValueError(message)
    validate_sample_order(
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
) -> Iterator[GenotypeChunk]:
    """Yield mean-imputed genotype chunks from a BGEN file."""
    with open_bgen(bgen_path, sample_path=sample_path) as bgen_reader:
        if bgen_reader.sample_identifier_source == "generated":
            message = "BGEN file does not contain samples and no .sample file was found."
            raise ValueError(message)
        yield from iter_genotype_chunks_from_reader(
            genotype_reader=bgen_reader,
            source_name="BGEN",
            sample_indices=sample_indices,
            expected_individual_identifiers=expected_individual_identifiers,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
            include_missing_value_flag=include_missing_value_flag,
        )


def iter_linear_genotype_chunks(
    bgen_path: Path,
    sample_indices: npt.NDArray[np.int64],
    expected_individual_identifiers: npt.NDArray[np.str_],
    chunk_size: int,
    variant_limit: int | None = None,
    *,
    sample_path: Path | None = None,
) -> Iterator[LinearGenotypeChunk]:
    """Yield linear-regression genotype chunks without missingness bookkeeping."""
    with open_bgen(bgen_path, sample_path=sample_path) as bgen_reader:
        if bgen_reader.sample_identifier_source == "generated":
            message = "BGEN file does not contain samples and no .sample file was found."
            raise ValueError(message)
        yield from iter_linear_genotype_chunks_from_reader(
            genotype_reader=bgen_reader,
            source_name="BGEN",
            sample_indices=sample_indices,
            expected_individual_identifiers=expected_individual_identifiers,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
        )
