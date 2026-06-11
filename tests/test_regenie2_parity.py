from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from g import api, types

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DATA_DIRECTORY = Path(os.environ.get("GWAS_ENGINE_DATA_DIR", str(REPOSITORY_ROOT / "data")))
BASELINE_DIRECTORY = DATA_DIRECTORY / "baselines"
PHENOTYPE_NAME = "phenotype_continuous"
PARITY_VARIANT_LIMIT = 1024

pytestmark = pytest.mark.phase0_data


@dataclasses.dataclass(frozen=True)
class Regenie2ParityResults:
    """Materialized output tables used by REGENIE parity tests."""

    observed_results: pl.DataFrame
    baseline_results: pl.DataFrame


def read_whitespace_table(table_path: Path) -> pl.DataFrame:
    """Read a whitespace-delimited table into a Polars frame."""
    table_lines = [line.strip() for line in table_path.read_text().splitlines() if line.strip()]
    if not table_lines:
        raise ValueError(f"Results table is empty: {table_path}")
    header_columns = table_lines[0].split()
    rows: list[dict[str, str]] = []
    for line in table_lines[1:]:
        values = line.split()
        if len(values) != len(header_columns):
            raise ValueError(
                f"Unexpected column count in {table_path}: expected {len(header_columns)}, got {len(values)}"
            )
        rows.append(dict(zip(header_columns, values, strict=True)))
    return pl.DataFrame(rows)


def load_regenie_baseline_results(variant_limit: int) -> pl.DataFrame:
    """Load and normalize the saved REGENIE baseline output."""
    baseline_frame = read_whitespace_table(BASELINE_DIRECTORY / "regenie_step2_qt_phenotype_continuous.regenie").head(
        variant_limit
    )
    return baseline_frame.rename(
        {
            "BETA": "baseline_beta",
            "SE": "baseline_standard_error",
            "CHISQ": "baseline_chi_squared",
            "LOG10P": "baseline_log10_p_value",
        },
    ).with_columns(
        pl.col("baseline_beta").cast(pl.Float64),
        pl.col("baseline_standard_error").cast(pl.Float64),
        pl.col("baseline_chi_squared").cast(pl.Float64),
        pl.col("baseline_log10_p_value").cast(pl.Float64),
    )


@pytest.fixture(scope="module")
def regenie2_parity_results(tmp_path_factory: pytest.TempPathFactory) -> Regenie2ParityResults:
    """Run one capped REGENIE step 2 scan and align it to the baseline output."""
    required_paths = [
        DATA_DIRECTORY / "1kg_chr22_full.bgen",
        DATA_DIRECTORY / "1kg_chr22_full.sample",
        DATA_DIRECTORY / "pheno_cont.txt",
        DATA_DIRECTORY / "covariates.txt",
        BASELINE_DIRECTORY / "regenie_step1_qt_pred.list",
        BASELINE_DIRECTORY / "regenie_step2_qt_phenotype_continuous.regenie",
    ]
    if not all(path.exists() for path in required_paths):
        pytest.skip("REGENIE phase-0 baseline data is not available.")

    output_directory = tmp_path_factory.mktemp("regenie2-parity")
    artifacts = api.regenie.from_options(
        {
            "step": 2,
            "qt": True,
            "bgen": DATA_DIRECTORY / "1kg_chr22_full.bgen",
            "sample": DATA_DIRECTORY / "1kg_chr22_full.sample",
            "phenoFile": DATA_DIRECTORY / "pheno_cont.txt",
            "phenoCol": PHENOTYPE_NAME,
            "out": output_directory / "regenie2_parity",
            "covarFile": DATA_DIRECTORY / "covariates.txt",
            "covarCol": ("age", "sex"),
            "pred": BASELINE_DIRECTORY / "regenie_step1_qt_pred.list",
            "bsize": 512,
            "device": types.Device.CPU.value,
            "variant_limit": PARITY_VARIANT_LIMIT,
            "staging_depth": 1,
            "output_run_directory": output_directory / "regenie2_parity",
            "format": types.OutputFormat.PARQUET.value,
        }
    )

    assert artifacts.final_parquet is not None
    observed_results = pl.read_parquet(artifacts.final_parquet)
    baseline_results = load_regenie_baseline_results(PARITY_VARIANT_LIMIT)

    return Regenie2ParityResults(
        observed_results=observed_results,
        baseline_results=baseline_results,
    )


@pytest.mark.parametrize(
    ("observed_column", "baseline_column", "absolute_tolerance"),
    [
        ("BETA", "baseline_beta", 1.0e-3),
        ("SE", "baseline_standard_error", 1.0e-3),
        ("CHISQ", "baseline_chi_squared", 1.5e-2),
        ("LOG10P", "baseline_log10_p_value", 1.5e-2),
    ],
)
def test_regenie2_linear_matches_regenie_baseline_statistics(
    regenie2_parity_results: Regenie2ParityResults,
    observed_column: str,
    baseline_column: str,
    absolute_tolerance: float,
) -> None:
    """Validate association statistics match REGENIE within tolerance."""
    merged_results = regenie2_parity_results.observed_results.join(
        regenie2_parity_results.baseline_results.select("ID", baseline_column),
        on="ID",
        how="inner",
    )

    assert merged_results.height == PARITY_VARIANT_LIMIT
    np.testing.assert_allclose(
        merged_results.get_column(observed_column).to_numpy(),
        merged_results.get_column(baseline_column).to_numpy(),
        atol=absolute_tolerance,
    )


def test_regenie2_linear_api_produces_valid_output(
    regenie2_parity_results: Regenie2ParityResults,
) -> None:
    """Validate the end-to-end API output shape and validity columns."""
    observed_results = regenie2_parity_results.observed_results

    assert observed_results.height == PARITY_VARIANT_LIMIT
    assert observed_results.get_column("ID").n_unique() == PARITY_VARIANT_LIMIT
    assert (observed_results.get_column("N") > 0).all()
    assert np.isfinite(observed_results.get_column("BETA").to_numpy()).all()
    assert np.isfinite(observed_results.get_column("SE").to_numpy()).all()
    assert np.isfinite(observed_results.get_column("CHISQ").to_numpy()).all()
    assert np.isfinite(observed_results.get_column("LOG10P").to_numpy()).all()
