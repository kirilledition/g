from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import numpy as np
import pandas as pd
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

    observed_results: pd.DataFrame
    baseline_results: pd.DataFrame


def load_regenie_baseline_results(variant_limit: int) -> pd.DataFrame:
    """Load and normalize the saved REGENIE baseline output."""
    baseline_frame = pd.read_csv(
        BASELINE_DIRECTORY / "regenie_step2_qt_phenotype_continuous.regenie",
        sep=r"\s+",
    ).head(variant_limit)
    return baseline_frame.rename(
        columns={
            "BETA": "baseline_beta",
            "SE": "baseline_standard_error",
            "CHISQ": "baseline_chi_squared",
            "LOG10P": "baseline_log10_p_value",
        }
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
    artifacts = api.regenie2_linear(
        bgen=DATA_DIRECTORY / "1kg_chr22_full.bgen",
        sample=DATA_DIRECTORY / "1kg_chr22_full.sample",
        pheno=DATA_DIRECTORY / "pheno_cont.txt",
        pheno_name=PHENOTYPE_NAME,
        out=output_directory / "regenie2_parity",
        covar=DATA_DIRECTORY / "covariates.txt",
        covar_names=("age", "sex"),
        pred=BASELINE_DIRECTORY / "regenie_step1_qt_pred.list",
        compute=api.ComputeConfig(
            chunk_size=512,
            device=types.Device.CPU,
            variant_limit=PARITY_VARIANT_LIMIT,
            prefetch_chunks=0,
            output_run_directory=output_directory / "regenie2_parity",
            finalize_parquet=True,
        ),
    )

    assert artifacts.final_parquet is not None
    observed_results = pd.read_parquet(artifacts.final_parquet)
    baseline_results = load_regenie_baseline_results(PARITY_VARIANT_LIMIT)

    return Regenie2ParityResults(
        observed_results=observed_results,
        baseline_results=baseline_results,
    )


def test_regenie2_linear_matches_regenie_baseline_beta(
    regenie2_parity_results: Regenie2ParityResults,
) -> None:
    """Validate beta estimates match REGENIE within tolerance."""
    merged_results = regenie2_parity_results.observed_results.merge(
        regenie2_parity_results.baseline_results[["ID", "baseline_beta"]],
        on="ID",
        how="inner",
    )

    assert len(merged_results) == PARITY_VARIANT_LIMIT
    np.testing.assert_allclose(
        merged_results["BETA"].to_numpy(),
        merged_results["baseline_beta"].to_numpy(),
        atol=1.0e-3,
    )


def test_regenie2_linear_matches_regenie_baseline_log10p(
    regenie2_parity_results: Regenie2ParityResults,
) -> None:
    """Validate -log10(p) values match REGENIE within tolerance."""
    merged_results = regenie2_parity_results.observed_results.merge(
        regenie2_parity_results.baseline_results[["ID", "baseline_log10_p_value"]],
        on="ID",
        how="inner",
    )

    assert len(merged_results) == PARITY_VARIANT_LIMIT
    np.testing.assert_allclose(
        merged_results["LOG10P"].to_numpy(),
        merged_results["baseline_log10_p_value"].to_numpy(),
        atol=1.5e-2,
    )


def test_regenie2_linear_api_produces_valid_output(
    regenie2_parity_results: Regenie2ParityResults,
) -> None:
    """Validate the end-to-end API output shape and validity columns."""
    observed_results = regenie2_parity_results.observed_results

    assert len(observed_results) == PARITY_VARIANT_LIMIT
    assert observed_results["ID"].is_unique
    assert (observed_results["N"] > 0).all()
    assert np.isfinite(observed_results["BETA"]).all()
    assert np.isfinite(observed_results["SE"]).all()
    assert np.isfinite(observed_results["CHISQ"]).all()
    assert np.isfinite(observed_results["LOG10P"]).all()
