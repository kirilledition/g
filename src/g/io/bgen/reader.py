from __future__ import annotations

from pathlib import Path
import typing

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import polars as pl

from g import models, types
from g.io import reader
from g.io.bgen import dosage, metadata, sample, selectors

if typing.TYPE_CHECKING:
    import collections.abc


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
        self.sample_path = sample.resolve_bgen_sample_path(
            self.bgen_path,
            Path(sample_path) if sample_path is not None else None,
        )
        core_module = dosage.load_backend_core()
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
            variant_metadata = metadata.build_core_variant_metadata(self.core_reader, 0, self.variant_count)
            self._variant_table = metadata.build_variant_table_from_core_metadata(variant_metadata)
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
            full_variant_metadata = metadata.build_core_variant_metadata(self.core_reader, 0, self.variant_count)
            self._variant_table_arrays = metadata.build_variant_table_arrays_from_core_metadata(full_variant_metadata)
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
            sample_table = sample.load_sample_identifier_table(self.sample_path)
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
        read_selection = selectors.normalize_read_selection(index, self.sample_count, self.variant_count)
        sample_index_array = np.ascontiguousarray(read_selection.sample_index_array, dtype=np.int64)
        variant_index_array = np.ascontiguousarray(read_selection.variant_index_array, dtype=np.int64)
        contiguous_variant_slice = selectors.resolve_contiguous_variant_slice(variant_index_array)

        if contiguous_variant_slice is not None:
            dosage_matrix = self.read_float32(
                sample_index_array,
                contiguous_variant_slice.variant_start,
                contiguous_variant_slice.variant_stop,
            )
        else:
            dosage_matrix = np.empty((sample_index_array.size, variant_index_array.size), dtype=np.float32, order="C")
            for variant_read_run in selectors.build_variant_read_runs(variant_index_array):
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
            message = "BGEN file does not contain samples and no .sample file was found."
            raise ValueError(message)
        return sample.build_sample_identifier_table(np.asarray(bgen_reader.samples, dtype=np.str_))


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
