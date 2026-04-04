"""BGEN input and Oxford sample-file helpers."""

from __future__ import annotations

import importlib
import threading
import typing
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


def load_backend_cbgen_file() -> typing.Any:
    """Import the low-level cbgen file wrapper lazily."""
    try:
        return importlib.import_module("cbgen").bgen_file
    except ModuleNotFoundError as error:
        message = (
            "BGEN support requires the `bgen-reader` stack. "
            "Run the command inside `nix develop` and sync dependencies again."
        )
        raise ModuleNotFoundError(message) from error


def load_backend_cbgen_ffi() -> typing.Any:
    """Import the low-level cbgen FFI module lazily."""
    try:
        return importlib.import_module("cbgen._ffi")
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
    order: types.ArrayMemoryOrder,
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
    if combination_count == 3 and not is_phased:
        return probability_matrix[:, 1] + (2.0 * probability_matrix[:, 2])
    if combination_count == 4 and is_phased:
        return probability_matrix[:, 1] + probability_matrix[:, 3]
    message = "Unsupported BGEN probability layout. Only diploid biallelic phased or unphased variants are supported."
    raise ValueError(message)


def read_bgen_direct_dosage_matrix(
    *,
    bgen_path: Path,
    variant_offsets: npt.NDArray[np.uint64],
    sample_index_array: npt.NDArray[np.intp],
    total_sample_count: int,
    combination_count: int,
    is_phased: bool,
    dtype: type[np.float32] | type[np.float64],
    order: types.ArrayMemoryOrder,
    num_threads: int,
) -> npt.NDArray[np.float32] | npt.NDArray[np.float64]:
    """Read a BGEN slice directly into a dosage matrix without a chunk probability tensor."""
    variant_count = len(variant_offsets)
    selected_sample_count = len(sample_index_array)
    numpy_order = "F" if order == types.ArrayMemoryOrder.FORTRAN_CONTIGUOUS else "C"
    if variant_count == 0:
        return np.empty((selected_sample_count, 0), dtype=dtype, order=numpy_order)

    cbgen_file = load_backend_cbgen_file()
    cbgen_ffi = load_backend_cbgen_ffi()
    direct_dosage_by_variant = np.empty((variant_count, selected_sample_count), dtype=dtype, order="C")
    full_sample_index_array = np.arange(total_sample_count, dtype=np.intp)
    use_full_sample_range = selected_sample_count == total_sample_count and np.array_equal(
        sample_index_array,
        full_sample_index_array,
    )
    worker_errors: list[BaseException] = []
    worker_error_lock = threading.Lock()

    def read_probability_matrix(
        file_handle: typing.Any,
        variant_offset: int,
        probability_matrix: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    ) -> None:
        genotype_handle = cbgen_ffi.lib.bgen_file_open_genotype(file_handle._bgen_file, int(variant_offset))
        if genotype_handle == cbgen_ffi.ffi.NULL:
            message = f"Could not open genotype (offset {variant_offset})."
            raise RuntimeError(message)
        try:
            if dtype == np.float32:
                error_code = cbgen_ffi.lib.bgen_genotype_read32(
                    genotype_handle,
                    cbgen_ffi.ffi.cast("float *", probability_matrix.ctypes.data),
                )
            else:
                error_code = cbgen_ffi.lib.bgen_genotype_read64(
                    genotype_handle,
                    cbgen_ffi.ffi.cast("double *", probability_matrix.ctypes.data),
                )
        finally:
            cbgen_ffi.lib.bgen_genotype_close(genotype_handle)
        if error_code != 0:
            message = f"Could not read genotype probabilities (offset {variant_offset})."
            raise RuntimeError(message)

    def worker(variant_start_index: int, variant_stop_index: int) -> None:
        probability_matrix = np.empty((total_sample_count, combination_count), dtype=dtype, order="C")
        try:
            with cbgen_file(bgen_path) as file_handle:
                for variant_index in range(variant_start_index, variant_stop_index):
                    read_probability_matrix(file_handle, int(variant_offsets[variant_index]), probability_matrix)
                    target_dosage_vector = direct_dosage_by_variant[variant_index, :]
                    if use_full_sample_range:
                        if combination_count == 3 and not is_phased:
                            np.copyto(target_dosage_vector, probability_matrix[:, 2])
                            target_dosage_vector *= 2.0
                            target_dosage_vector += probability_matrix[:, 1]
                        else:
                            np.copyto(target_dosage_vector, probability_matrix[:, 1])
                            target_dosage_vector += probability_matrix[:, 3]
                    else:
                        if combination_count == 3 and not is_phased:
                            np.copyto(target_dosage_vector, probability_matrix[sample_index_array, 2])
                            target_dosage_vector *= 2.0
                            target_dosage_vector += probability_matrix[sample_index_array, 1]
                        else:
                            np.copyto(target_dosage_vector, probability_matrix[sample_index_array, 1])
                            target_dosage_vector += probability_matrix[sample_index_array, 3]
        except BaseException as error:  # noqa: BLE001
            with worker_error_lock:
                worker_errors.append(error)

    variants_per_thread = -(-variant_count // max(num_threads, 1))
    worker_threads: list[threading.Thread] = []
    variant_start_index = 0
    for _ in range(max(num_threads, 1)):
        variant_stop_index = min(variant_start_index + variants_per_thread, variant_count)
        if variant_start_index >= variant_stop_index:
            break
        worker_thread = threading.Thread(target=worker, args=(variant_start_index, variant_stop_index))
        worker_threads.append(worker_thread)
        worker_thread.start()
        variant_start_index = variant_stop_index
    for worker_thread in worker_threads:
        worker_thread.join()
    if worker_errors:
        raise worker_errors[0]
    return np.asarray(direct_dosage_by_variant.T, dtype=dtype, order=numpy_order)


def build_bgen_variant_table(bgen_handle: OpenBgenHandle) -> pl.DataFrame:
    """Build normalized variant metadata from an open BGEN handle."""
    variant_table_arrays = build_bgen_variant_table_arrays(bgen_handle)
    return pl.DataFrame(
        {
            "chromosome": variant_table_arrays.chromosome_values,
            "variant_identifier": variant_table_arrays.variant_identifier_values,
            "genetic_distance": np.zeros(int(bgen_handle.nvariants), dtype=np.float32),
            "position": variant_table_arrays.position_values,
            "allele_one": variant_table_arrays.allele_one_values,
            "allele_two": variant_table_arrays.allele_two_values,
        }
    )


def build_bgen_variant_table_arrays(bgen_handle: OpenBgenHandle) -> reader.VariantTableArrays:
    """Build normalized variant metadata arrays from an open BGEN handle."""
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


def resolve_sample_identifier_source(
    bgen_handle: OpenBgenHandle, sample_path: Path | None
) -> types.SampleIdentifierSource:
    """Resolve where sample identifiers originated for one open BGEN handle."""
    if sample_path is not None:
        return types.SampleIdentifierSource.EXTERNAL
    if bool(bgen_handle._cbgen.contain_samples):
        return types.SampleIdentifierSource.EMBEDDED
    return types.SampleIdentifierSource.GENERATED


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
        self.sample_identifier_array = self.resolve_sample_identifier_array()
        self.combination_count = int(np.asarray(self.backend_handle.ncombinations, dtype=np.int32)[0])
        self.is_phased = bool(np.asarray(self.backend_handle.phased, dtype=np.bool_)[0])
        self._variant_table: pl.DataFrame | None = None
        self._variant_table_arrays: reader.VariantTableArrays | None = None
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
        return self.sample_identifier_array

    @property
    def variant_table(self) -> pl.DataFrame:
        """Return normalized BGEN variant metadata."""
        if self._variant_table is None:
            self._variant_table = build_bgen_variant_table(self.backend_handle)
        return self._variant_table

    def get_variant_table_arrays(self, variant_start: int, variant_stop: int) -> reader.VariantTableArrays:
        """Return normalized metadata arrays for one BGEN variant slice."""
        if self._variant_table_arrays is None:
            self._variant_table_arrays = build_bgen_variant_table_arrays(self.backend_handle)
        return reader.VariantTableArrays(
            chromosome_values=self._variant_table_arrays.chromosome_values[variant_start:variant_stop],
            variant_identifier_values=self._variant_table_arrays.variant_identifier_values[variant_start:variant_stop],
            position_values=self._variant_table_arrays.position_values[variant_start:variant_stop],
            allele_one_values=self._variant_table_arrays.allele_one_values[variant_start:variant_stop],
            allele_two_values=self._variant_table_arrays.allele_two_values[variant_start:variant_stop],
        )

    def resolve_sample_identifier_array(self) -> npt.NDArray[np.str_]:
        """Resolve normalized individual identifiers for the open BGEN reader."""
        if self.sample_identifier_source == types.SampleIdentifierSource.EXTERNAL:
            assert self.sample_path is not None
            sample_table = load_sample_identifier_table(self.sample_path)
            return np.asarray(sample_table.get_column("individual_identifier").to_numpy(), dtype=np.str_)
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
        order: types.ArrayMemoryOrder = types.ArrayMemoryOrder.C_CONTIGUOUS,
    ) -> npt.NDArray[np.float32] | npt.NDArray[np.float64]:
        """Read BGEN dosages with the same calling convention as `bed_handle.read`."""
        if dtype in {np.float32, np.float64} and order in {
            types.ArrayMemoryOrder.C_CONTIGUOUS,
            types.ArrayMemoryOrder.FORTRAN_CONTIGUOUS,
        }:
            sample_index_selector, variant_index_selector = self.backend_handle._split_index(index)
            sample_index_array = np.asarray(self.backend_handle._sample_range[sample_index_selector], dtype=np.intp)
            variant_offsets = np.asarray(self.backend_handle._vaddr[variant_index_selector], dtype=np.uint64)
            num_threads = int(self.backend_handle._get_num_threads(None, len(variant_offsets)))
            return read_bgen_direct_dosage_matrix(
                bgen_path=self.bgen_path,
                variant_offsets=variant_offsets,
                sample_index_array=sample_index_array,
                total_sample_count=self.sample_count,
                combination_count=self.combination_count,
                is_phased=self.is_phased,
                dtype=dtype,
                order=order,
                num_threads=num_threads,
            )
        probability_tensor = self.backend_handle.read(index=index, dtype=dtype, order=order.value)
        return convert_probability_tensor_to_dosage(
            probability_tensor=np.asarray(probability_tensor, dtype=dtype, order=order.value),
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
        if resolve_sample_identifier_source(bgen_handle, None) == types.SampleIdentifierSource.GENERATED:
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
        order=types.ArrayMemoryOrder.C_CONTIGUOUS,
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


def iter_linear_genotype_chunks(
    bgen_path: Path,
    sample_indices: npt.NDArray[np.int64],
    expected_individual_identifiers: npt.NDArray[np.str_],
    chunk_size: int,
    variant_limit: int | None = None,
    *,
    sample_path: Path | None = None,
) -> collections.abc.Iterator[models.LinearGenotypeChunk]:
    """Yield linear-regression genotype chunks without missingness bookkeeping."""
    with open_bgen(bgen_path, sample_path=sample_path) as bgen_reader:
        if bgen_reader.sample_identifier_source == types.SampleIdentifierSource.GENERATED:
            message = "BGEN file does not contain samples and no .sample file was found."
            raise ValueError(message)
        yield from reader.iter_linear_genotype_chunks_from_reader(
            genotype_reader=bgen_reader,
            source_name="BGEN",
            sample_indices=sample_indices,
            expected_individual_identifiers=expected_individual_identifiers,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
        )
