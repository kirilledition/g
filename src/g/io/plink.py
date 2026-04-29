"""PLINK BED/BIM/FAM input and sample-alignment helpers."""

from __future__ import annotations

import typing
from dataclasses import dataclass
from pathlib import Path

import bed_reader
import bed_reader._open_bed
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import polars as pl

from g import models, types
from g.io import genotype_processing, reader

if typing.TYPE_CHECKING:
    import collections.abc


build_genotype_chunk = genotype_processing.build_genotype_chunk
preprocess_genotype_matrix = genotype_processing.preprocess_genotype_matrix
preprocess_genotype_matrix_arrays = genotype_processing.preprocess_genotype_matrix_arrays
open_bed = bed_reader.open_bed
read_f32 = bed_reader.read_f32
get_num_threads = bed_reader._open_bed.get_num_threads

VARIANT_TABLE_COLUMNS = (
    "chromosome",
    "variant_identifier",
    "genetic_distance",
    "position",
    "allele_one",
    "allele_two",
)

__all__ = (
    "FAMILY_TABLE_COLUMNS",
    "PlinkReader",
    "build_genotype_chunk",
    "convert_frame_to_float32_jax",
    "infer_covariate_names",
    "iter_genotype_chunks",
    "iter_linear_genotype_chunks",
    "load_aligned_sample_data",
    "load_aligned_sample_data_from_individual_identifier_table",
    "load_family_table",
    "load_phenotype_or_covariate_table",
    "load_variant_table",
    "preprocess_genotype_matrix",
    "preprocess_genotype_matrix_arrays",
    "read_bed_chunk",
    "read_bed_chunk_host",
    "recode_binary_phenotype",
    "validate_bed_sample_order",
)

FAMILY_TABLE_COLUMNS = (
    "family_identifier",
    "individual_identifier",
    "paternal_identifier",
    "maternal_identifier",
    "reported_sex",
    "placeholder_phenotype",
)
TABULAR_NULL_VALUES = ["NA", "NaN", "nan", "-9"]


@dataclass(frozen=True)
class VariantSliceBounds:
    """Normalized start and stop indices for one variant slice.

    Attributes:
        start: Start index of the variant slice.
        stop: Stop index of the variant slice.

    """

    start: int
    stop: int


def load_family_table(family_path: Path) -> pl.DataFrame:
    """Load a PLINK FAM file.

    Args:
        family_path: Path to the `.fam` file.

    Returns:
        Parsed family table.

    Raises:
        ValueError: If any row does not contain exactly six whitespace-delimited fields.

    """
    raw_line_table = pl.read_csv(
        family_path,
        has_header=False,
        separator="|",
        quote_char=None,
        new_columns=["raw_line"],
    ).with_row_index("line_number", offset=1)

    tokenized_table = (
        raw_line_table.filter(pl.col("raw_line").str.strip_chars() != "")
        .with_columns(pl.col("raw_line").str.extract_all(r"\S+").alias("field_values"))
        .with_columns(pl.col("field_values").list.len().alias("field_count"))
    )

    invalid_row_table = tokenized_table.filter(pl.col("field_count") != len(FAMILY_TABLE_COLUMNS))
    if invalid_row_table.height > 0:
        first_invalid_row = invalid_row_table.row(0, named=True)
        line_number = int(first_invalid_row["line_number"])
        field_count = int(first_invalid_row["field_count"])
        message = (
            "Invalid .fam row at line "
            f"{line_number}: expected {len(FAMILY_TABLE_COLUMNS)} whitespace-delimited fields, "
            f"found {field_count}."
        )
        raise ValueError(message)

    return tokenized_table.select(
        pl.col("field_values").list.get(0).alias("family_identifier"),
        pl.col("field_values").list.get(1).alias("individual_identifier"),
        pl.col("field_values").list.get(2).alias("paternal_identifier"),
        pl.col("field_values").list.get(3).alias("maternal_identifier"),
        pl.col("field_values").list.get(4).cast(pl.Int64).alias("reported_sex"),
        pl.col("field_values").list.get(5).cast(pl.Float32).alias("placeholder_phenotype"),
    ).with_row_index("sample_index")


def load_phenotype_or_covariate_table(table_path: Path) -> pl.DataFrame:
    """Load a tab-separated phenotype or covariate table.

    Args:
        table_path: Path to the tabular file.

    Returns:
        Parsed Polars table.

    """
    return pl.read_csv(table_path, separator="\t", null_values=TABULAR_NULL_VALUES)


def infer_covariate_names(covariate_table: pl.DataFrame) -> tuple[str, ...]:
    """Infer covariate names from a covariate table.

    Args:
        covariate_table: Parsed covariate table.

    Returns:
        Ordered covariate names excluding `FID` and `IID`.

    Raises:
        ValueError: If no covariate columns are available.

    """
    covariate_names = tuple(column_name for column_name in covariate_table.columns if column_name not in {"FID", "IID"})
    if not covariate_names:
        message = "Covariate table must contain at least one non-identifier column."
        raise ValueError(message)
    return covariate_names


def convert_frame_to_float32_jax(data_frame: pl.DataFrame) -> jax.Array:
    """Convert a numeric Polars DataFrame to a float32 JAX array.

    Args:
        data_frame: Numeric Polars DataFrame.

    Returns:
        JAX array exported from Polars.

    """
    host_array = data_frame.to_numpy(order="c")
    return jnp.asarray(host_array, dtype=jnp.float32)


def recode_binary_phenotype(phenotype_values: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Recode PLINK binary phenotypes from 1/2 encoding to 0/1.

    Args:
        phenotype_values: Binary phenotype values in PLINK encoding.

    Returns:
        Binary phenotype values in 0/1 encoding.

    Raises:
        ValueError: If values other than 1 and 2 are present.

    """
    unique_values = set(np.unique(phenotype_values))
    if not unique_values.issubset({1.0, 2.0}):
        message = f"Binary phenotype must contain only PLINK values 1 and 2, found {sorted(unique_values)}."
        raise ValueError(message)
    return phenotype_values - 1.0


def load_aligned_sample_data(
    bed_prefix: Path,
    phenotype_path: Path,
    phenotype_name: str,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    *,
    is_binary_trait: bool,
) -> models.AlignedSampleData:
    """Load and align FAM, phenotype, and covariate tables."""
    return load_aligned_sample_data_from_family_identifier_table(
        sample_table=load_family_table(bed_prefix.with_suffix(".fam")),
        phenotype_path=phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=is_binary_trait,
    )


def load_aligned_sample_data_from_family_identifier_table(
    sample_table: pl.DataFrame,
    phenotype_path: Path,
    phenotype_name: str,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    *,
    is_binary_trait: bool,
) -> models.AlignedSampleData:
    """Load and align a family-aware sample table, phenotype table, and covariate table.

    Args:
        sample_table: Sample table with `sample_index`, `family_identifier`, and
            `individual_identifier`.
        phenotype_path: Phenotype table path.
        phenotype_name: Phenotype column to select.
        covariate_path: Covariate table path.
        covariate_names: Optional explicit covariate names.
        is_binary_trait: Whether the selected phenotype is binary.

    Returns:
        Aligned sample data ready for computation.

    Raises:
        ValueError: If required columns are missing or if no samples remain.

    """
    phenotype_table = load_phenotype_or_covariate_table(phenotype_path).with_columns(
        pl.col("FID").cast(pl.String),
        pl.col("IID").cast(pl.String),
    )
    if phenotype_name not in phenotype_table.columns:
        message = f"Phenotype column '{phenotype_name}' was not found in {phenotype_path}."
        raise ValueError(message)

    aligned_table = sample_table.join(
        phenotype_table.select("FID", "IID", phenotype_name),
        left_on=["family_identifier", "individual_identifier"],
        right_on=["FID", "IID"],
        how="inner",
    )

    selected_covariate_names: tuple[str, ...]
    if covariate_path is None:
        if covariate_names is not None:
            message = "Covariate names cannot be provided without a covariate table."
            raise ValueError(message)
        selected_covariate_names = ()
    else:
        covariate_table = load_phenotype_or_covariate_table(covariate_path).with_columns(
            pl.col("FID").cast(pl.String),
            pl.col("IID").cast(pl.String),
        )
        selected_covariate_names = covariate_names or infer_covariate_names(covariate_table)
        missing_covariates = [name for name in selected_covariate_names if name not in covariate_table.columns]
        if missing_covariates:
            message = f"Covariate columns are missing from {covariate_path}: {missing_covariates}."
            raise ValueError(message)
        aligned_table = aligned_table.join(
            covariate_table.select("FID", "IID", *selected_covariate_names),
            left_on=["family_identifier", "individual_identifier"],
            right_on=["FID", "IID"],
            how="inner",
        )

    return build_aligned_sample_data(
        aligned_table=aligned_table,
        phenotype_name=phenotype_name,
        selected_covariate_names=selected_covariate_names,
        is_binary_trait=is_binary_trait,
    )


def load_aligned_sample_data_from_individual_identifier_table(
    sample_table: pl.DataFrame,
    phenotype_path: Path,
    phenotype_name: str,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    *,
    is_binary_trait: bool,
) -> models.AlignedSampleData:
    """Load and align a sample table by individual identifier only."""
    phenotype_table = load_phenotype_or_covariate_table(phenotype_path).with_columns(
        pl.col("FID").cast(pl.String),
        pl.col("IID").cast(pl.String),
    )
    if phenotype_name not in phenotype_table.columns:
        message = f"Phenotype column '{phenotype_name}' was not found in {phenotype_path}."
        raise ValueError(message)

    aligned_table = sample_table.join(
        phenotype_table.select("FID", "IID", phenotype_name),
        left_on=["individual_identifier"],
        right_on=["IID"],
        how="inner",
    )

    selected_covariate_names: tuple[str, ...]
    if covariate_path is None:
        if covariate_names is not None:
            message = "Covariate names cannot be provided without a covariate table."
            raise ValueError(message)
        selected_covariate_names = ()
    else:
        covariate_table = load_phenotype_or_covariate_table(covariate_path).with_columns(
            pl.col("FID").cast(pl.String),
            pl.col("IID").cast(pl.String),
        )
        selected_covariate_names = covariate_names or infer_covariate_names(covariate_table)
        missing_covariates = [name for name in selected_covariate_names if name not in covariate_table.columns]
        if missing_covariates:
            message = f"Covariate columns are missing from {covariate_path}: {missing_covariates}."
            raise ValueError(message)
        aligned_table = aligned_table.join(
            covariate_table.select("FID", "IID", *selected_covariate_names),
            left_on=["individual_identifier"],
            right_on=["IID"],
            how="inner",
        )

    return build_aligned_sample_data(
        aligned_table=aligned_table,
        phenotype_name=phenotype_name,
        selected_covariate_names=selected_covariate_names,
        is_binary_trait=is_binary_trait,
    )


def build_aligned_sample_data(
    aligned_table: pl.DataFrame,
    phenotype_name: str,
    selected_covariate_names: tuple[str, ...],
    *,
    is_binary_trait: bool,
) -> models.AlignedSampleData:
    """Build aligned sample outputs from an already-joined table."""
    aligned_table = aligned_table.drop_nulls(subset=[phenotype_name, *selected_covariate_names]).sort("sample_index")

    if aligned_table.height == 0:
        message = "No aligned samples remain after joining phenotype and covariate tables."
        raise ValueError(message)

    phenotype_values = aligned_table.get_column(phenotype_name).cast(pl.Float32).to_numpy()
    phenotype_array = recode_binary_phenotype(phenotype_values) if is_binary_trait else phenotype_values

    if selected_covariate_names:
        design_table = aligned_table.select(
            pl.lit(1.0).alias("intercept"),
            *[pl.col(column_name).cast(pl.Float32).alias(column_name) for column_name in selected_covariate_names],
        )
    else:
        design_table = pl.DataFrame(
            {"intercept": np.ones(aligned_table.height, dtype=np.float32)},
            schema={"intercept": pl.Float32},
        )
    phenotype_frame = pl.DataFrame({phenotype_name: phenotype_array}, schema={phenotype_name: pl.Float32})

    return models.AlignedSampleData(
        sample_indices=aligned_table.get_column("sample_index").cast(pl.Int64).to_numpy(),
        family_identifiers=aligned_table.get_column("family_identifier").cast(pl.String).to_numpy(),
        individual_identifiers=aligned_table.get_column("individual_identifier").cast(pl.String).to_numpy(),
        phenotype_name=phenotype_name,
        phenotype_vector=convert_frame_to_float32_jax(phenotype_frame).reshape((-1,)),
        covariate_names=("intercept", *selected_covariate_names),
        covariate_matrix=convert_frame_to_float32_jax(design_table),
        is_binary_trait=is_binary_trait,
    )


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
        self._variant_table = load_variant_table(self.bed_prefix)
        self._variant_table_arrays: reader.VariantTableArrays | None = None
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

    @property
    def variant_table(self) -> pl.DataFrame:
        """Return normalized PLINK variant metadata."""
        return self._variant_table

    def get_variant_table_arrays(self, variant_start: int, variant_stop: int) -> reader.VariantTableArrays:
        """Return normalized metadata arrays for one variant slice."""
        if self._variant_table_arrays is None:
            self._variant_table_arrays = reader.build_variant_table_arrays(self.variant_table)
        return reader.VariantTableArrays(
            chromosome_values=self._variant_table_arrays.chromosome_values[variant_start:variant_stop],
            variant_identifier_values=self._variant_table_arrays.variant_identifier_values[variant_start:variant_stop],
            position_values=self._variant_table_arrays.position_values[variant_start:variant_stop],
            allele_one_values=self._variant_table_arrays.allele_one_values[variant_start:variant_stop],
            allele_two_values=self._variant_table_arrays.allele_two_values[variant_start:variant_stop],
        )

    def read(
        self,
        index: object = None,
        dtype: type[np.float32] | type[np.float64] = np.float32,
        order: types.ArrayMemoryOrder = types.ArrayMemoryOrder.C_CONTIGUOUS,
    ) -> npt.NDArray[np.float32] | npt.NDArray[np.float64]:
        """Read PLINK genotypes with the same calling convention as `bed_handle.read`."""
        if (
            dtype is np.float32
            and order == types.ArrayMemoryOrder.C_CONTIGUOUS
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
        genotype_matrix = self.bed_handle.read(index=index, dtype=dtype, order=order.value)
        return np.asarray(genotype_matrix, dtype=dtype, order=order.value)

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
    bed_handle: typing.Any,
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
        order=types.ArrayMemoryOrder.C_CONTIGUOUS.value,
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
    bed_handle: typing.Any,
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
    reader.validate_sample_order(
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
) -> collections.abc.Iterator[models.GenotypeChunk]:
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
        yield from reader.iter_genotype_chunks_from_reader(
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
) -> collections.abc.Iterator[models.LinearGenotypeChunk]:
    """Yield linear-regression genotype chunks without missingness bookkeeping."""
    sample_index_array = np.ascontiguousarray(sample_indices, dtype=np.intp)
    validate_bed_sample_order(
        bed_prefix=bed_prefix,
        sample_index_array=sample_index_array,
        expected_individual_identifiers=expected_individual_identifiers,
    )
    with PlinkReader(bed_prefix) as plink_reader:
        yield from reader.iter_linear_genotype_chunks_from_reader(
            genotype_reader=plink_reader,
            source_name="BED",
            sample_indices=sample_indices,
            expected_individual_identifiers=expected_individual_identifiers,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
            validate_sample_order_flag=False,
        )
