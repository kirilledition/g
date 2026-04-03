#!/usr/bin/env python3
"""Generate deterministic phenotypes and covariates for benchmark data."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

RANDOM_SEED = 42
CASE_PREVALENCE = 0.3
PLINK_CONTROL_VALUE = 1
PLINK_CASE_VALUE = 2
MEAN_AGE_YEARS = 50
AGE_STANDARD_DEVIATION_YEARS = 10
MINIMUM_AGE_YEARS = 18


class PhenotypeTables(NamedTuple):
    """Generated continuous, binary, and covariate tables."""

    continuous_table: pd.DataFrame
    binary_table: pd.DataFrame
    covariate_table: pd.DataFrame


def load_family_table(family_path: Path) -> pd.DataFrame:
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
    return pd.read_csv(
        family_path,
        sep=r"\s+",
        header=None,
        names=[
            "family_identifier",
            "individual_identifier",
            "paternal_identifier",
            "maternal_identifier",
            "reported_sex",
            "placeholder_phenotype",
        ],
    )


def create_phenotype_and_covariate_tables(family_table: pd.DataFrame) -> PhenotypeTables:
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

    sex_covariate = family_table["reported_sex"].to_numpy(dtype=np.int64, copy=True)
    unknown_sex_mask = sex_covariate == 0
    sex_covariate[unknown_sex_mask] = random_number_generator.choice([1, 2], size=int(unknown_sex_mask.sum()))

    identifier_columns = ["family_identifier", "individual_identifier"]
    continuous_table = family_table[identifier_columns].copy()
    continuous_table["phenotype_continuous"] = continuous_trait

    binary_table = family_table[identifier_columns].copy()
    binary_table["phenotype_binary"] = binary_trait

    covariate_table = family_table[identifier_columns].copy()
    covariate_table["age"] = age_years
    covariate_table["sex"] = sex_covariate

    return PhenotypeTables(
        continuous_table=continuous_table,
        binary_table=binary_table,
        covariate_table=covariate_table,
    )


def write_output_tables(
    data_directory: Path,
    continuous_table: pd.DataFrame,
    binary_table: pd.DataFrame,
    covariate_table: pd.DataFrame,
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
    continuous_output_table = continuous_table.rename(columns=rename_columns)
    binary_output_table = binary_table.rename(columns=rename_columns)
    covariate_output_table = covariate_table.rename(columns=rename_columns)

    continuous_output_table.to_csv(continuous_path, sep="\t", index=False)
    binary_output_table.to_csv(binary_path, sep="\t", index=False)
    covariate_output_table.to_csv(covariate_path, sep="\t", index=False)

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
