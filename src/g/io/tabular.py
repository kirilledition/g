"""Tabular phenotype and covariate loading."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import polars as pl

from g.models import AlignedSampleData

if TYPE_CHECKING:
    from pathlib import Path

    import jax

FAMILY_TABLE_COLUMNS = (
    "family_identifier",
    "individual_identifier",
    "paternal_identifier",
    "maternal_identifier",
    "reported_sex",
    "placeholder_phenotype",
)
TABULAR_NULL_VALUES = ["NA", "NaN", "nan", "-9"]


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
) -> AlignedSampleData:
    """Load and align FAM, phenotype, and covariate tables."""
    return load_aligned_sample_data_from_sample_table(
        sample_table=load_family_table(bed_prefix.with_suffix(".fam")),
        phenotype_path=phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=is_binary_trait,
        match_family_and_individual_identifiers=True,
    )


def load_aligned_sample_data_from_sample_table(
    sample_table: pl.DataFrame,
    phenotype_path: Path,
    phenotype_name: str,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    *,
    is_binary_trait: bool,
    match_family_and_individual_identifiers: bool,
) -> AlignedSampleData:
    """Load and align a sample table, phenotype table, and covariate table.

    Args:
        sample_table: Sample table with `sample_index`, `family_identifier`, and
            `individual_identifier`.
        phenotype_path: Phenotype table path.
        phenotype_name: Phenotype column to select.
        covariate_path: Covariate table path.
        covariate_names: Optional explicit covariate names.
        is_binary_trait: Whether the selected phenotype is binary.
        match_family_and_individual_identifiers: Whether phenotype and covariate
            rows must match both family and individual identifiers. BGEN sources
            currently align by individual identifier only because the format
            does not expose PLINK-style family identifiers.

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

    if match_family_and_individual_identifiers:
        aligned_table = sample_table.join(
            phenotype_table.select("FID", "IID", phenotype_name),
            left_on=["family_identifier", "individual_identifier"],
            right_on=["FID", "IID"],
            how="inner",
        )
    else:
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
        if match_family_and_individual_identifiers:
            aligned_table = aligned_table.join(
                covariate_table.select("FID", "IID", *selected_covariate_names),
                left_on=["family_identifier", "individual_identifier"],
                right_on=["FID", "IID"],
                how="inner",
            )
        else:
            aligned_table = aligned_table.join(
                covariate_table.select("FID", "IID", *selected_covariate_names),
                left_on=["individual_identifier"],
                right_on=["IID"],
                how="inner",
            )

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

    return AlignedSampleData(
        sample_indices=aligned_table.get_column("sample_index").cast(pl.Int64).to_numpy(),
        family_identifiers=aligned_table.get_column("family_identifier").cast(pl.String).to_numpy(),
        individual_identifiers=aligned_table.get_column("individual_identifier").cast(pl.String).to_numpy(),
        phenotype_name=phenotype_name,
        phenotype_vector=convert_frame_to_float32_jax(phenotype_frame).reshape((-1,)),
        covariate_names=("intercept", *selected_covariate_names),
        covariate_matrix=convert_frame_to_float32_jax(design_table),
        is_binary_trait=is_binary_trait,
    )
