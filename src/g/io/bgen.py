"""BGEN input and Oxford sample-file helpers."""

from __future__ import annotations

import importlib
import typing
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import polars as pl

from g import models, types
from g.io import reader

if typing.TYPE_CHECKING:
    import collections.abc


OpenBgenHandle = typing.Any
CoreBgenHandle = typing.Any


@dataclass(frozen=True)
class CoreVariantMetadata:
    """Raw normalized variant metadata returned by the Rust core reader."""

    chromosome_values: list[str]
    variant_identifier_values: list[str]
    position_values: list[int]
    allele_one_values: list[str]
    allele_two_values: list[str]


@dataclass(frozen=True)
class ContiguousVariantSlice:
    """One contiguous variant span."""

    variant_start: int
    variant_stop: int


@dataclass(frozen=True)
class ReadSelection:
    """Normalized read selectors for one compatibility read."""

    sample_index_array: npt.NDArray[np.int64]
    variant_index_array: npt.NDArray[np.int64]


@dataclass(frozen=True)
class VariantReadRun:
    """One contiguous run inside a possibly non-contiguous variant request."""

    variant_start: int
    variant_stop: int
    output_start: int
    output_stop: int


def split_sample_file_line(raw_line: str) -> list[str]:
    """Split one Oxford sample-file line on arbitrary whitespace."""
    return raw_line.strip().split()


def resolve_bgen_sample_path(bgen_path: Path, sample_path: Path | None = None) -> Path | None:
    """Resolve an explicit or adjacent Oxford sample file for one BGEN file."""
    if sample_path is not None:
        return sample_path
    adjacent_sample_path = bgen_path.with_suffix(".sample")
    return adjacent_sample_path if adjacent_sample_path.exists() else None


def build_sample_identifier_table(sample_identifiers: npt.NDArray[np.str_]) -> pl.DataFrame:
    """Build the normalized identifier table used by the alignment pipeline."""
    normalized_sample_identifiers = np.asarray(sample_identifiers, dtype=np.str_)
    return pl.DataFrame(
        {
            "family_identifier": normalized_sample_identifiers,
            "individual_identifier": normalized_sample_identifiers,
        }
    ).with_row_index("sample_index")


def load_sample_identifier_table(sample_path: Path) -> pl.DataFrame:
    """Load normalized identifiers from an Oxford sample file.

    Args:
        sample_path: Path to the sample file.

    Returns:
        A normalized table containing `sample_index`, `family_identifier`, and
        `individual_identifier` columns.

    Raises:
        ValueError: The file is malformed or missing mandatory identifier columns.

    """
    sample_lines = sample_path.read_text(encoding="utf-8").splitlines()
    non_empty_lines = [line for line in sample_lines if line.strip()]
    if len(non_empty_lines) < 2:
        message = f"Sample file '{sample_path}' must contain at least two header lines."
        raise ValueError(message)

    column_names = split_sample_file_line(non_empty_lines[0])
    column_types = split_sample_file_line(non_empty_lines[1])
    if len(column_names) != len(column_types):
        message = f"Sample file '{sample_path}' header and type lines have different column counts."
        raise ValueError(message)
    if not column_names:
        message = f"Sample file '{sample_path}' does not contain any columns."
        raise ValueError(message)
    if column_types[0] != "0":
        message = f"Sample file '{sample_path}' must mark the first identifier column with type '0'."
        raise ValueError(message)
    if "ID_2" in column_names and column_types[column_names.index("ID_2")] != "0":
        message = f"Sample file '{sample_path}' must mark 'ID_2' with type '0'."
        raise ValueError(message)

    data_rows: list[list[str]] = []
    for row_index, raw_line in enumerate(non_empty_lines[2:], start=3):
        row_values = split_sample_file_line(raw_line)
        if len(row_values) != len(column_names):
            message = (
                f"Sample file '{sample_path}' line {row_index} has {len(row_values)} values, "
                f"but the header declares {len(column_names)} columns."
            )
            raise ValueError(message)
        data_rows.append(row_values)

    if not data_rows:
        empty_identifiers = np.asarray([], dtype=np.str_)
        return build_sample_identifier_table(empty_identifiers)

    column_values = {
        column_name: [row_values[column_index] for row_values in data_rows]
        for column_index, column_name in enumerate(column_names)
    }
    sample_table = pl.DataFrame(column_values)
    family_identifier_column_name = column_names[0]
    individual_identifier_column_name = "ID_2" if "ID_2" in column_names else family_identifier_column_name
    return pl.DataFrame(
        {
            "family_identifier": sample_table.get_column(family_identifier_column_name).cast(pl.String),
            "individual_identifier": sample_table.get_column(individual_identifier_column_name).cast(pl.String),
        }
    ).with_row_index("sample_index")


def load_backend_core() -> typing.Any:
    """Import Rust core helpers lazily."""
    try:
        return importlib.import_module("g._core")
    except ModuleNotFoundError as error:
        message = "Rust core helpers are unavailable. Ensure the extension module is built."
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
    order: types.ArrayMemoryOrder,
) -> npt.NDArray[np.float32] | npt.NDArray[np.float64]:
    """Convert a BGEN probability tensor into additive dosages."""
    if dtype is np.float32:
        core_module = load_backend_core()
        dosage_matrix = core_module.convert_probability_tensor_to_dosage_f32(
            np.asarray(probability_tensor, dtype=np.float32, order="C"),
            int(combination_count),
            bool(is_phased),
        )
        return np.asarray(dosage_matrix, dtype=np.float32, order=order.value)
    if combination_count == 3 and not is_phased:
        dosage_matrix = probability_tensor[:, :, 1] + (2.0 * probability_tensor[:, :, 2])
        return np.asarray(dosage_matrix, dtype=dtype, order=order.value)
    if combination_count == 4 and is_phased:
        dosage_matrix = probability_tensor[:, :, 1] + probability_tensor[:, :, 3]
        return np.asarray(dosage_matrix, dtype=dtype, order=order.value)
    message = "Unsupported BGEN probability layout. Only diploid biallelic phased or unphased variants are supported."
    raise ValueError(message)


def convert_probability_matrix_to_dosage(
    probability_matrix: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    combination_count: int,
    *,
    is_phased: bool,
) -> npt.NDArray[np.float32] | npt.NDArray[np.float64]:
    """Convert one variant's probability matrix into additive dosages."""
    if probability_matrix.dtype == np.float32:
        core_module = load_backend_core()
        dosage_vector = core_module.convert_probability_matrix_to_dosage_f32(
            np.asarray(probability_matrix, dtype=np.float32, order="C"),
            int(combination_count),
            bool(is_phased),
        )
        return np.asarray(dosage_vector, dtype=np.float32, order="C")
    if combination_count == 3 and not is_phased:
        return probability_matrix[:, 1] + (2.0 * probability_matrix[:, 2])
    if combination_count == 4 and is_phased:
        return probability_matrix[:, 1] + probability_matrix[:, 3]
    message = "Unsupported BGEN probability layout. Only diploid biallelic phased or unphased variants are supported."
    raise ValueError(message)


def build_bgen_variant_table(bgen_handle: OpenBgenHandle) -> pl.DataFrame:
    """Build normalized variant metadata from an open BGEN-like handle."""
    variant_table_arrays = build_bgen_variant_table_arrays(bgen_handle)
    return pl.DataFrame(
        {
            "chromosome": variant_table_arrays.chromosome_values,
            "variant_identifier": variant_table_arrays.variant_identifier_values,
            "genetic_distance": np.zeros(len(variant_table_arrays.position_values), dtype=np.float32),
            "position": variant_table_arrays.position_values,
            "allele_one": variant_table_arrays.allele_one_values,
            "allele_two": variant_table_arrays.allele_two_values,
        }
    )


def build_bgen_variant_table_arrays(bgen_handle: OpenBgenHandle) -> reader.VariantTableArrays:
    """Build normalized variant metadata arrays from a BGEN-like handle."""
    allele_identifier_values = np.asarray(bgen_handle.allele_ids, dtype=np.str_)
    allele_pairs = [allele_identifier_value.split(",") for allele_identifier_value in allele_identifier_values]
    counted_allele_values = [allele_pair[-1] if allele_pair else "" for allele_pair in allele_pairs]
    reference_allele_values = [allele_pair[0] if allele_pair else "" for allele_pair in allele_pairs]
    variant_identifier_values = np.asarray(bgen_handle.ids, dtype=np.str_)
    rsid_values = np.asarray(bgen_handle.rsids, dtype=np.str_)
    return reader.VariantTableArrays(
        chromosome_values=np.asarray(bgen_handle.chromosomes, dtype=np.str_),
        variant_identifier_values=resolve_variant_identifier_values(variant_identifier_values, rsid_values),
        position_values=np.asarray(bgen_handle.positions, dtype=np.int64),
        allele_one_values=np.asarray(counted_allele_values, dtype=np.str_),
        allele_two_values=np.asarray(reference_allele_values, dtype=np.str_),
    )


def build_variant_table_arrays_from_core_metadata(
    variant_metadata: CoreVariantMetadata,
) -> reader.VariantTableArrays:
    """Build normalized metadata arrays from Rust core metadata lists."""
    return reader.VariantTableArrays(
        chromosome_values=np.asarray(variant_metadata.chromosome_values, dtype=np.str_),
        variant_identifier_values=np.asarray(variant_metadata.variant_identifier_values, dtype=np.str_),
        position_values=np.asarray(variant_metadata.position_values, dtype=np.int64),
        allele_one_values=np.asarray(variant_metadata.allele_one_values, dtype=np.str_),
        allele_two_values=np.asarray(variant_metadata.allele_two_values, dtype=np.str_),
    )


def build_variant_table_from_core_metadata(variant_metadata: CoreVariantMetadata) -> pl.DataFrame:
    """Build normalized metadata table from Rust core metadata lists."""
    variant_table_arrays = build_variant_table_arrays_from_core_metadata(variant_metadata)
    return pl.DataFrame(
        {
            "chromosome": variant_table_arrays.chromosome_values,
            "variant_identifier": variant_table_arrays.variant_identifier_values,
            "genetic_distance": np.zeros(len(variant_table_arrays.position_values), dtype=np.float32),
            "position": variant_table_arrays.position_values,
            "allele_one": variant_table_arrays.allele_one_values,
            "allele_two": variant_table_arrays.allele_two_values,
        }
    )


def build_core_variant_metadata(
    core_reader: CoreBgenHandle,
    variant_start: int,
    variant_stop: int,
) -> CoreVariantMetadata:
    """Read one normalized metadata slice from the Rust core reader."""
    (
        chromosome_values,
        variant_identifier_values,
        position_values,
        allele_one_values,
        allele_two_values,
    ) = core_reader.variant_metadata_slice(variant_start, variant_stop)
    return CoreVariantMetadata(
        chromosome_values=list(chromosome_values),
        variant_identifier_values=list(variant_identifier_values),
        position_values=list(position_values),
        allele_one_values=list(allele_one_values),
        allele_two_values=list(allele_two_values),
    )


def resolve_sample_identifier_source(
    core_reader: CoreBgenHandle,
    sample_path: Path | None,
) -> types.SampleIdentifierSource:
    """Resolve where sample identifiers originated for one open BGEN handle."""
    if sample_path is not None:
        return types.SampleIdentifierSource.EXTERNAL
    if bool(core_reader.contains_embedded_samples):
        return types.SampleIdentifierSource.EMBEDDED
    return types.SampleIdentifierSource.GENERATED


def build_generated_sample_identifier_array(sample_count: int) -> npt.NDArray[np.str_]:
    """Build deterministic fallback sample identifiers when the file stores none."""
    return np.asarray([f"sample_{sample_index}" for sample_index in range(sample_count)], dtype=np.str_)


def normalize_axis_index(axis_index: int, axis_size: int, axis_name: str) -> int:
    """Normalize one possibly-negative axis index."""
    normalized_axis_index = axis_index + axis_size if axis_index < 0 else axis_index
    if normalized_axis_index < 0 or normalized_axis_index >= axis_size:
        message = f"{axis_name} index {axis_index} is out of bounds for axis size {axis_size}."
        raise IndexError(message)
    return normalized_axis_index


def normalize_axis_selector(
    axis_selector: object,
    axis_size: int,
    axis_name: str,
) -> npt.NDArray[np.int64]:
    """Normalize one row or column selector into explicit int64 indices."""
    if axis_selector is None:
        return np.arange(axis_size, dtype=np.int64)
    if isinstance(axis_selector, slice):
        normalized_slice = axis_selector.indices(axis_size)
        return np.arange(*normalized_slice, dtype=np.int64)
    if isinstance(axis_selector, (int, np.integer)):
        return np.asarray([normalize_axis_index(int(axis_selector), axis_size, axis_name)], dtype=np.int64)

    selector_array = np.asarray(axis_selector)
    if selector_array.dtype == np.bool_:
        if selector_array.ndim != 1 or selector_array.shape[0] != axis_size:
            message = (
                f"{axis_name} boolean selector must be one-dimensional with length {axis_size}. "
                f"Observed shape {selector_array.shape}."
            )
            raise ValueError(message)
        return np.flatnonzero(selector_array).astype(np.int64, copy=False)

    if selector_array.ndim != 1:
        message = f"{axis_name} selector must be one-dimensional. Observed shape {selector_array.shape}."
        raise ValueError(message)

    normalized_values = [
        normalize_axis_index(int(raw_axis_index), axis_size, axis_name)
        for raw_axis_index in selector_array.astype(np.int64, copy=False)
    ]
    return np.asarray(normalized_values, dtype=np.int64)


def normalize_read_selection(index: object, sample_count: int, variant_count: int) -> ReadSelection:
    """Normalize a bed-reader-like read selector."""
    sample_selector: object
    variant_selector: object
    if index is None:
        sample_selector = slice(None)
        variant_selector = slice(None)
    elif isinstance(index, tuple):
        if len(index) != 2:
            message = "BGEN read index tuples must contain exactly two selectors: samples and variants."
            raise ValueError(message)
        sample_selector, variant_selector = index
    else:
        sample_selector = index
        variant_selector = slice(None)

    return ReadSelection(
        sample_index_array=normalize_axis_selector(sample_selector, sample_count, "Sample"),
        variant_index_array=normalize_axis_selector(variant_selector, variant_count, "Variant"),
    )


def resolve_contiguous_variant_slice(
    variant_index_array: npt.NDArray[np.int64],
) -> ContiguousVariantSlice | None:
    """Resolve a contiguous variant slice when possible."""
    if variant_index_array.size == 0:
        return ContiguousVariantSlice(variant_start=0, variant_stop=0)
    if variant_index_array.size == 1:
        variant_start = int(variant_index_array[0])
        return ContiguousVariantSlice(variant_start=variant_start, variant_stop=variant_start + 1)
    consecutive_differences = np.diff(variant_index_array)
    if np.all(consecutive_differences == 1):
        return ContiguousVariantSlice(
            variant_start=int(variant_index_array[0]),
            variant_stop=int(variant_index_array[-1]) + 1,
        )
    return None


def build_variant_read_runs(variant_index_array: npt.NDArray[np.int64]) -> list[VariantReadRun]:
    """Split a possibly non-contiguous variant request into contiguous runs."""
    if variant_index_array.size == 0:
        return []

    variant_read_runs: list[VariantReadRun] = []
    run_variant_start = int(variant_index_array[0])
    run_output_start = 0

    for output_index in range(1, int(variant_index_array.size)):
        previous_variant_index = int(variant_index_array[output_index - 1])
        current_variant_index = int(variant_index_array[output_index])
        if current_variant_index != previous_variant_index + 1:
            variant_read_runs.append(
                VariantReadRun(
                    variant_start=run_variant_start,
                    variant_stop=previous_variant_index + 1,
                    output_start=run_output_start,
                    output_stop=output_index,
                )
            )
            run_variant_start = current_variant_index
            run_output_start = output_index

    variant_read_runs.append(
        VariantReadRun(
            variant_start=run_variant_start,
            variant_stop=int(variant_index_array[-1]) + 1,
            output_start=run_output_start,
            output_stop=int(variant_index_array.size),
        )
    )
    return variant_read_runs


class BgenReader:
    """Native Rust BGEN reader with a bed-reader-like compatibility surface."""

    def __init__(
        self,
        bgen_path: Path | str,
        sample_path: Path | str | None = None,
        *,
        allow_complex: bool = False,
        trusted_no_missing_diploid: bool = False,
    ) -> None:
        """Open one BGEN file.

        Args:
            bgen_path: Path to the `.bgen` file.
            sample_path: Optional explicit `.sample` file path.
            allow_complex: Present for compatibility. Native Rust BGEN reads
                currently reject unsupported layouts regardless of this flag.
            trusted_no_missing_diploid: Whether to enable the faster native
                reader path that trusts unphased diploid records have no
                missing probabilities.

        Raises:
            ValueError: The file uses an unsupported genotype layout.

        """
        del allow_complex
        self.bgen_path = Path(bgen_path)
        self.sample_path = resolve_bgen_sample_path(
            self.bgen_path,
            Path(sample_path) if sample_path is not None else None,
        )
        core_module = load_backend_core()
        self.core_reader = core_module.PyBgenReader(
            str(self.bgen_path),
            bool(trusted_no_missing_diploid),
        )
        self.sample_identifier_source = resolve_sample_identifier_source(self.core_reader, self.sample_path)
        self.sample_identifier_array = self.resolve_sample_identifier_array()
        self._variant_table: pl.DataFrame | None = None
        self._variant_table_arrays: reader.VariantTableArrays | None = None
        self._chromosome_boundary_indices: npt.NDArray[np.int64] | None = None
        self._prepared_sample_index_array: npt.NDArray[np.intp] | None = None
        self.trusted_no_missing_diploid = bool(trusted_no_missing_diploid)

    @property
    def sample_count(self) -> int:
        """Return the number of samples."""
        return int(self.core_reader.sample_count)

    @property
    def variant_count(self) -> int:
        """Return the number of variants."""
        return int(self.core_reader.variant_count)

    @property
    def samples(self) -> npt.NDArray[np.str_]:
        """Return sample identifiers in file order."""
        return self.sample_identifier_array

    @property
    def variant_table(self) -> pl.DataFrame:
        """Return normalized BGEN variant metadata."""
        if self._variant_table is None:
            variant_metadata = build_core_variant_metadata(self.core_reader, 0, self.variant_count)
            self._variant_table = build_variant_table_from_core_metadata(variant_metadata)
        return self._variant_table

    def get_variant_table_arrays(self, variant_start: int, variant_stop: int) -> reader.VariantTableArrays:
        """Return normalized metadata arrays for one BGEN variant slice."""
        if variant_start < 0 or variant_stop < variant_start or variant_stop > self.variant_count:
            message = (
                f"Variant bounds must satisfy 0 <= start <= stop <= {self.variant_count}. "
                f"Received start={variant_start}, stop={variant_stop}."
            )
            raise ValueError(message)
        if self._variant_table_arrays is None:
            full_variant_metadata = build_core_variant_metadata(self.core_reader, 0, self.variant_count)
            self._variant_table_arrays = build_variant_table_arrays_from_core_metadata(full_variant_metadata)
        return reader.VariantTableArrays(
            chromosome_values=self._variant_table_arrays.chromosome_values[variant_start:variant_stop],
            variant_identifier_values=self._variant_table_arrays.variant_identifier_values[variant_start:variant_stop],
            position_values=self._variant_table_arrays.position_values[variant_start:variant_stop],
            allele_one_values=self._variant_table_arrays.allele_one_values[variant_start:variant_stop],
            allele_two_values=self._variant_table_arrays.allele_two_values[variant_start:variant_stop],
        )

    def split_variant_slice_by_chromosome(
        self,
        variant_start: int,
        variant_stop: int,
    ) -> tuple[tuple[int, int], ...]:
        """Return chromosome-homogeneous absolute variant slices within one contiguous request."""
        if variant_start < 0 or variant_stop < variant_start or variant_stop > self.variant_count:
            message = (
                f"Variant bounds must satisfy 0 <= start <= stop <= {self.variant_count}. "
                f"Received start={variant_start}, stop={variant_stop}."
            )
            raise ValueError(message)
        if variant_start == variant_stop:
            return ((variant_start, variant_stop),)
        chromosome_boundary_indices = self.resolve_chromosome_boundary_indices()
        if chromosome_boundary_indices.size <= 2:
            return ((variant_start, variant_stop),)
        boundary_start_index = int(np.searchsorted(chromosome_boundary_indices, variant_start, side="right") - 1)
        boundary_stop_index = int(np.searchsorted(chromosome_boundary_indices, variant_stop, side="left"))
        chromosome_slices: list[tuple[int, int]] = []
        for boundary_index in range(boundary_start_index, boundary_stop_index):
            chromosome_slices.append(
                (
                    max(variant_start, int(chromosome_boundary_indices[boundary_index])),
                    min(variant_stop, int(chromosome_boundary_indices[boundary_index + 1])),
                )
            )
        return tuple(chromosome_slices)

    def resolve_sample_identifier_array(self) -> npt.NDArray[np.str_]:
        """Resolve normalized individual identifiers for the open BGEN reader."""
        if self.sample_identifier_source == types.SampleIdentifierSource.EXTERNAL:
            assert self.sample_path is not None
            sample_table = load_sample_identifier_table(self.sample_path)
            return np.asarray(sample_table.get_column("individual_identifier").to_numpy(), dtype=np.str_)
        if self.sample_identifier_source == types.SampleIdentifierSource.EMBEDDED:
            return np.asarray(self.core_reader.sample_identifiers(), dtype=np.str_)
        return build_generated_sample_identifier_array(self.sample_count)

    def resolve_chromosome_boundary_indices(self) -> npt.NDArray[np.int64]:
        """Resolve absolute variant indices where chromosome runs start and stop."""
        if self._chromosome_boundary_indices is None:
            self._chromosome_boundary_indices = np.asarray(
                self.core_reader.chromosome_boundary_indices(),
                dtype=np.int64,
            )
        return self._chromosome_boundary_indices

    def prepare_sample_selection(
        self,
        sample_index_array: npt.NDArray[np.int64] | npt.NDArray[np.intp],
    ) -> None:
        """Bind one reusable aligned sample selection for hot-path reads."""
        normalized_sample_index_array = np.ascontiguousarray(sample_index_array, dtype=np.int64)
        self.core_reader.prepare_sample_selection(normalized_sample_index_array)
        self._prepared_sample_index_array = np.asarray(normalized_sample_index_array, dtype=np.intp)

    def clear_prepared_sample_selection(self) -> None:
        """Clear one previously bound reusable aligned sample selection."""
        self.core_reader.clear_prepared_sample_selection()
        self._prepared_sample_index_array = None

    def reset_profile(self) -> None:
        """Reset cumulative Rust BGEN profiling counters."""
        self.core_reader.reset_profile()

    def profile_snapshot(self) -> dict[str, int]:
        """Return cumulative Rust BGEN profiling counters."""
        return dict(self.core_reader.profile_snapshot())

    def validate_trusted_no_missing_diploid(self) -> None:
        """Validate that the open file satisfies the trusted fast-path assumptions."""
        self.core_reader.validate_trusted_no_missing_diploid()

    def read(
        self,
        index: object = None,
        dtype: type[np.float32] | type[np.float64] = np.float32,
        order: types.ArrayMemoryOrder = types.ArrayMemoryOrder.C_CONTIGUOUS,
    ) -> npt.NDArray[np.float32] | npt.NDArray[np.float64]:
        """Read BGEN dosages with the same calling convention as `bed_handle.read`."""
        read_selection = normalize_read_selection(index, self.sample_count, self.variant_count)
        sample_index_array = np.ascontiguousarray(read_selection.sample_index_array, dtype=np.int64)
        variant_index_array = np.ascontiguousarray(read_selection.variant_index_array, dtype=np.int64)
        contiguous_variant_slice = resolve_contiguous_variant_slice(variant_index_array)

        if contiguous_variant_slice is not None:
            dosage_matrix = self.read_float32(
                sample_index_array,
                contiguous_variant_slice.variant_start,
                contiguous_variant_slice.variant_stop,
            )
        else:
            dosage_matrix = np.empty((sample_index_array.size, variant_index_array.size), dtype=np.float32, order="C")
            for variant_read_run in build_variant_read_runs(variant_index_array):
                dosage_matrix[:, variant_read_run.output_start : variant_read_run.output_stop] = self.read_float32(
                    sample_index_array,
                    variant_read_run.variant_start,
                    variant_read_run.variant_stop,
                )

        return np.asarray(dosage_matrix, dtype=dtype, order=order.value)

    def read_float32(
        self,
        sample_index_array: npt.NDArray[np.int64] | npt.NDArray[np.intp],
        variant_start: int,
        variant_stop: int,
    ) -> npt.NDArray[np.float32]:
        """Read one strict float32 dosage block for the BGEN hot path."""
        if variant_start < 0 or variant_stop < variant_start or variant_stop > self.variant_count:
            message = (
                f"Variant bounds must satisfy 0 <= start <= stop <= {self.variant_count}. "
                f"Received start={variant_start}, stop={variant_stop}."
            )
            raise ValueError(message)
        normalized_sample_index_array = np.ascontiguousarray(sample_index_array, dtype=np.int64)
        if variant_stop == variant_start:
            return np.empty((len(normalized_sample_index_array), 0), dtype=np.float32, order="C")
        if self._prepared_sample_index_array is not None and np.array_equal(
            self._prepared_sample_index_array,
            np.asarray(normalized_sample_index_array, dtype=np.intp),
        ):
            dosage_matrix = self.core_reader.read_dosage_f32_prepared(
                int(variant_start),
                int(variant_stop),
            )
            return np.asarray(dosage_matrix, dtype=np.float32, order="C")
        dosage_matrix = self.core_reader.read_dosage_f32(
            normalized_sample_index_array,
            int(variant_start),
            int(variant_stop),
        )
        return np.asarray(dosage_matrix, dtype=np.float32, order="C")

    def read_float32_prepared(
        self,
        variant_start: int,
        variant_stop: int,
    ) -> npt.NDArray[np.float32]:
        """Read one strict float32 dosage block using the prepared sample selection."""
        if self._prepared_sample_index_array is None:
            message = "Prepared BGEN sample selection was requested before aligned samples were bound."
            raise ValueError(message)
        if variant_start < 0 or variant_stop < variant_start or variant_stop > self.variant_count:
            message = (
                f"Variant bounds must satisfy 0 <= start <= stop <= {self.variant_count}. "
                f"Received start={variant_start}, stop={variant_stop}."
            )
            raise ValueError(message)
        if variant_stop == variant_start:
            return np.empty((len(self._prepared_sample_index_array), 0), dtype=np.float32, order="C")
        dosage_matrix = self.core_reader.read_dosage_f32_prepared(int(variant_start), int(variant_stop))
        return np.asarray(dosage_matrix, dtype=np.float32, order="C")

    def read_float32_into(
        self,
        output_array: npt.NDArray[np.float32],
        sample_index_array: npt.NDArray[np.int64] | npt.NDArray[np.intp],
        variant_start: int,
        variant_stop: int,
    ) -> npt.NDArray[np.float32]:
        """Fill one strict float32 dosage block into a caller-provided output buffer."""
        if variant_start < 0 or variant_stop < variant_start or variant_stop > self.variant_count:
            message = (
                f"Variant bounds must satisfy 0 <= start <= stop <= {self.variant_count}. "
                f"Received start={variant_start}, stop={variant_stop}."
            )
            raise ValueError(message)
        selected_sample_count = len(sample_index_array)
        selected_variant_count = variant_stop - variant_start
        expected_shape = (selected_sample_count, selected_variant_count)
        if output_array.shape != expected_shape:
            message = f"Output array shape mismatch: expected {expected_shape}, observed {output_array.shape}."
            raise ValueError(message)
        if output_array.dtype != np.float32:
            message = "Output array for BGEN dosage reads must have dtype float32."
            raise ValueError(message)
        if not output_array.flags.c_contiguous:
            message = "Output array for BGEN dosage reads must be C-contiguous."
            raise ValueError(message)
        normalized_sample_index_array = np.ascontiguousarray(sample_index_array, dtype=np.int64)
        if self._prepared_sample_index_array is not None and np.array_equal(
            self._prepared_sample_index_array,
            np.asarray(normalized_sample_index_array, dtype=np.intp),
        ):
            self.core_reader.read_dosage_f32_into_prepared(
                int(variant_start),
                int(variant_stop),
                output_array,
            )
            return output_array
        self.core_reader.read_dosage_f32_into(
            normalized_sample_index_array,
            int(variant_start),
            int(variant_stop),
            output_array,
        )
        return output_array

    def read_float32_into_prepared(
        self,
        output_array: npt.NDArray[np.float32],
        variant_start: int,
        variant_stop: int,
    ) -> npt.NDArray[np.float32]:
        """Fill one output buffer using the prepared sample selection."""
        if self._prepared_sample_index_array is None:
            message = "Prepared BGEN sample selection was requested before aligned samples were bound."
            raise ValueError(message)
        expected_shape = (len(self._prepared_sample_index_array), variant_stop - variant_start)
        if output_array.shape != expected_shape:
            message = f"Output array shape mismatch: expected {expected_shape}, observed {output_array.shape}."
            raise ValueError(message)
        if output_array.dtype != np.float32:
            message = "Output array for BGEN dosage reads must have dtype float32."
            raise ValueError(message)
        if not output_array.flags.c_contiguous:
            message = "Output array for BGEN dosage reads must be C-contiguous."
            raise ValueError(message)
        self.core_reader.read_dosage_f32_into_prepared(
            int(variant_start),
            int(variant_stop),
            output_array,
        )
        return output_array

    def close(self) -> None:
        """Close the underlying BGEN handle."""
        self.core_reader.close()

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
    trusted_no_missing_diploid: bool = False,
) -> BgenReader:
    """Open one BGEN file with a bed-reader-like wrapper."""
    return BgenReader(
        bgen_path=bgen_path,
        sample_path=sample_path,
        allow_complex=allow_complex,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
    )


def load_bgen_sample_table(bgen_path: Path, sample_path: Path | None = None) -> pl.DataFrame:
    """Load BGEN sample identifiers into a normalized identifier table."""
    resolved_sample_path = resolve_bgen_sample_path(bgen_path, sample_path)
    if resolved_sample_path is not None:
        sample_table = load_sample_identifier_table(resolved_sample_path)
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
            message = "BGEN file does not contain samples and no .sample file was found."
            raise ValueError(message)
        return build_sample_identifier_table(np.asarray(bgen_reader.samples, dtype=np.str_))


def read_bgen_chunk_host(
    bgen_reader: BgenReader,
    sample_index_array: npt.NDArray[np.intp],
    variant_start: int,
    variant_stop: int,
) -> npt.NDArray[np.float32]:
    """Read one BGEN chunk into a host NumPy array of dosages."""
    genotype_matrix_host = bgen_reader.read_float32(
        np.ascontiguousarray(sample_index_array, dtype=np.int64),
        variant_start,
        variant_stop,
    )
    return np.asarray(genotype_matrix_host, dtype=np.float32, order=types.ArrayMemoryOrder.C_CONTIGUOUS.value)


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
    if bgen_reader.sample_identifier_source == types.SampleIdentifierSource.GENERATED:
        message = "BGEN file does not contain samples and no .sample file was found."
        raise ValueError(message)
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
            message = "BGEN file does not contain samples and no .sample file was found."
            raise ValueError(message)
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
    """Yield dosage genotype chunks without missingness bookkeeping."""
    with open_bgen(bgen_path, sample_path=sample_path) as bgen_reader:
        if bgen_reader.sample_identifier_source == types.SampleIdentifierSource.GENERATED:
            message = "BGEN file does not contain samples and no .sample file was found."
            raise ValueError(message)
        yield from reader.iter_dosage_genotype_chunks_from_reader(
            genotype_reader=bgen_reader,
            source_name="BGEN",
            sample_indices=sample_indices,
            expected_individual_identifiers=expected_individual_identifiers,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
        )
