#!/usr/bin/env python3
"""Generate deterministic phenotypes and covariates for benchmark data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

RANDOM_SEED = 42
CASE_PREVALENCE = 0.3
PLINK_CONTROL_VALUE = 1
PLINK_CASE_VALUE = 2
MEAN_AGE_YEARS = 50
AGE_STANDARD_DEVIATION_YEARS = 10
MINIMUM_AGE_YEARS = 18


@dataclass(frozen=True)
class PhenotypeTables:
    """Generated continuous, binary, and covariate tables."""

    continuous_table: pl.DataFrame
    binary_table: pl.DataFrame
    covariate_table: pl.DataFrame


def load_family_table(family_path: Path) -> pl.DataFrame:
    """Load the PLINK family file used for phenotype generation.

    Args:
        family_path: Path to the `.fam` file.

    Returns:
        A table containing family/sample identifiers and metadata.

    Raises:
        FileNotFoundError: The family file does not exist.

    """
    if not family_path.exists():
        raise FileNotFoundError(f"Could not find {family_path}. Run scripts/fetch_1kg.py first.")

    print(f"Reading {family_path}...")
    family_lines = [line.strip() for line in family_path.read_text().splitlines() if line.strip()]
    family_columns = [
        "family_identifier",
        "individual_identifier",
        "paternal_identifier",
        "maternal_identifier",
        "reported_sex",
        "placeholder_phenotype",
    ]
    rows: list[dict[str, str]] = []
    for line in family_lines:
        values = line.split()
        if len(values) != len(family_columns):
            raise ValueError(
                f"Unexpected column count in {family_path}: expected {len(family_columns)}, got {len(values)}"
            )
        rows.append(dict(zip(family_columns, values, strict=True)))
    return pl.DataFrame(rows).with_columns(
        pl.col("reported_sex").cast(pl.Int64),
        pl.col("placeholder_phenotype").cast(pl.Float64),
    )


def create_phenotype_and_covariate_tables(family_table: pl.DataFrame) -> PhenotypeTables:
    """Create deterministic continuous, binary, and covariate tables.

    Args:
        family_table: Input PLINK family table.

    Returns:
        Generated phenotype and covariate tables.

    """
    random_number_generator = np.random.default_rng(RANDOM_SEED)
    sample_count = len(family_table)
    print(f"Loaded {sample_count} samples.")

    continuous_trait = random_number_generator.standard_normal(sample_count)
    binary_case_indicator = random_number_generator.binomial(n=1, p=CASE_PREVALENCE, size=sample_count)
    binary_trait = np.where(
        binary_case_indicator == 1,
        PLINK_CASE_VALUE,
        PLINK_CONTROL_VALUE,
    )
    rounded_age = np.rint(
        random_number_generator.normal(
            loc=MEAN_AGE_YEARS,
            scale=AGE_STANDARD_DEVIATION_YEARS,
            size=sample_count,
        )
    ).astype(np.int64)
    age_years = np.maximum(rounded_age, MINIMUM_AGE_YEARS)

    sex_covariate = family_table.get_column("reported_sex").to_numpy().astype(np.int64, copy=True)
    unknown_sex_mask = sex_covariate == 0
    sex_covariate[unknown_sex_mask] = random_number_generator.choice([1, 2], size=int(unknown_sex_mask.sum()))

    identifier_columns = ["family_identifier", "individual_identifier"]
    identifier_table = family_table.select(identifier_columns)
    continuous_table = identifier_table.with_columns(pl.Series("phenotype_continuous", continuous_trait))
    binary_table = identifier_table.with_columns(pl.Series("phenotype_binary", binary_trait))
    covariate_table = identifier_table.with_columns(
        pl.Series("age", age_years),
        pl.Series("sex", sex_covariate),
    )

    return PhenotypeTables(
        continuous_table=continuous_table,
        binary_table=binary_table,
        covariate_table=covariate_table,
    )


def write_output_tables(
    data_directory: Path,
    continuous_table: pl.DataFrame,
    binary_table: pl.DataFrame,
    covariate_table: pl.DataFrame,
) -> None:
    """Write phenotype and covariate tables to the data directory.

    Args:
        data_directory: Output directory.
        continuous_table: Continuous phenotype table.
        binary_table: Binary phenotype table.
        covariate_table: Covariate table.

    """
    continuous_path = data_directory / "pheno_cont.txt"
    binary_path = data_directory / "pheno_bin.txt"
    covariate_path = data_directory / "covariates.txt"

    rename_columns = {
        "family_identifier": "FID",
        "individual_identifier": "IID",
    }
    continuous_output_table = continuous_table.rename(rename_columns)
    binary_output_table = binary_table.rename(rename_columns)
    covariate_output_table = covariate_table.rename(rename_columns)

    continuous_output_table.write_csv(continuous_path, separator="\t")
    binary_output_table.write_csv(binary_path, separator="\t")
    covariate_output_table.write_csv(covariate_path, separator="\t")

    print(f"Saved {continuous_path}")
    print(f"Saved {binary_path}")
    print(f"Saved {covariate_path}")


def main() -> None:
    """Generate deterministic phenotype and covariate files for Phase 0."""
    data_directory = Path("data")
    family_path = data_directory / "1kg_chr22_full.fam"
    family_table = load_family_table(family_path)
    phenotype_tables = create_phenotype_and_covariate_tables(family_table)
    write_output_tables(
        data_directory,
        phenotype_tables.continuous_table,
        phenotype_tables.binary_table,
        phenotype_tables.covariate_table,
    )
    print("Phenotype simulation complete.")


if __name__ == "__main__":
    main()
