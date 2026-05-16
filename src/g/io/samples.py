"""Shared sample, phenotype, and covariate alignment helpers."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import polars as pl

from g.io import models

if typing.TYPE_CHECKING:
    from pathlib import Path

TABULAR_NULL_VALUES = ["NA", "NaN", "nan", "-9"]


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
    """Recode REGENIE binary phenotypes from 1/2 encoding to 0/1.

    Args:
        phenotype_values: Binary phenotype values in 1/2 encoding.

    Returns:
        Binary phenotype values in 0/1 encoding.

    Raises:
        ValueError: If values other than 1 and 2 are present.

    """
    unique_values = set(np.unique(phenotype_values))
    if not unique_values.issubset({1.0, 2.0}):
        message = f"Binary phenotype must contain only values 1 and 2, found {sorted(unique_values)}."
        raise ValueError(message)
    return phenotype_values - 1.0


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
        covariate_column_set = set(covariate_table.columns)
        missing_covariates = [name for name in selected_covariate_names if name not in covariate_column_set]
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
