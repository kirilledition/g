from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import polars as pl
import pytest

import tests.parity.harness
from g import api, types

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DATA_DIRECTORY = Path(os.environ.get("GWAS_ENGINE_DATA_DIR", str(REPOSITORY_ROOT / "data")))
BASELINE_DIRECTORY = DATA_DIRECTORY / "baselines"
QUANTITATIVE_PHENOTYPE_NAME = "phenotype_continuous"
BINARY_PHENOTYPE_NAME = "phenotype_binary"
PARITY_VARIANT_LIMIT = 1024
PARITY_METADATA = tests.parity.harness.load_golden_metadata()
QUANTITATIVE_WORKFLOW = PARITY_METADATA.workflow_by_identifier("quantitative_single_bgen_loco")
BINARY_SCORE_WORKFLOW = PARITY_METADATA.workflow_by_identifier("binary_score_only")

pytestmark = pytest.mark.phase0_data


@dataclasses.dataclass(frozen=True)
class Regenie2ParityResults:
    """Materialized output tables used by REGENIE parity tests."""

    observed_results: pl.DataFrame
    baseline_results: pl.DataFrame


def build_statistic_tolerance_parameters(
    workflow: tests.parity.harness.GoldenWorkflow,
) -> list[object]:
    """Build stable pytest parameters from the checked-in parity metadata."""
    return [
        pytest.param(tolerance, id=tolerance.observed_column)
        for tolerance in workflow.tolerances
    ]


def load_regenie_baseline_results(baseline_path: Path, workflow: tests.parity.harness.GoldenWorkflow) -> pl.DataFrame:
    """Load and normalize the saved REGENIE baseline output."""
    baseline_frame = tests.parity.harness.read_regenie_table(baseline_path).head(PARITY_VARIANT_LIMIT)
    baseline_rename_mapping = {
        tolerance.observed_column: tolerance.baseline_column
        for tolerance in workflow.tolerances
        if tolerance.observed_column != tolerance.baseline_column
    }
    baseline_statistic_columns = [tolerance.baseline_column for tolerance in workflow.tolerances]
    return baseline_frame.rename(baseline_rename_mapping).with_columns(
        pl.col(statistic_column).cast(pl.Float64) for statistic_column in baseline_statistic_columns
    )


@pytest.fixture(scope="module")
def quantitative_regenie2_parity_results(tmp_path_factory: pytest.TempPathFactory) -> Regenie2ParityResults:
    """Run one capped quantitative REGENIE step 2 scan and align it to the baseline output."""
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
            "phenoCol": QUANTITATIVE_PHENOTYPE_NAME,
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
            "finalize_parquet": True,
        }
    )

    assert artifacts.final_parquet is not None
    observed_results = pl.read_parquet(artifacts.final_parquet)
    baseline_results = load_regenie_baseline_results(
        BASELINE_DIRECTORY / "regenie_step2_qt_phenotype_continuous.regenie",
        QUANTITATIVE_WORKFLOW,
    )

    return Regenie2ParityResults(
        observed_results=observed_results,
        baseline_results=baseline_results,
    )


@pytest.fixture(scope="module")
def binary_score_regenie2_parity_results(tmp_path_factory: pytest.TempPathFactory) -> Regenie2ParityResults:
    """Run one capped binary score-only REGENIE step 2 scan and align it to the baseline output."""
    required_paths = [
        DATA_DIRECTORY / "1kg_chr22_full.bgen",
        DATA_DIRECTORY / "1kg_chr22_full.sample",
        DATA_DIRECTORY / "pheno_bin.txt",
        DATA_DIRECTORY / "covariates.txt",
        BASELINE_DIRECTORY / "regenie_step1_pred.list",
        BASELINE_DIRECTORY / "regenie_step2_score_only_phenotype_binary.regenie",
    ]
    if not all(path.exists() for path in required_paths):
        pytest.skip("REGENIE phase-0 binary score-only baseline data is not available.")

    output_directory = tmp_path_factory.mktemp("regenie2-binary-score-parity")
    artifacts = api.regenie.from_options(
        {
            "step": 2,
            "bt": True,
            "bgen": DATA_DIRECTORY / "1kg_chr22_full.bgen",
            "sample": DATA_DIRECTORY / "1kg_chr22_full.sample",
            "phenoFile": DATA_DIRECTORY / "pheno_bin.txt",
            "phenoCol": BINARY_PHENOTYPE_NAME,
            "out": output_directory / "regenie2_binary_score_parity",
            "covarFile": DATA_DIRECTORY / "covariates.txt",
            "covarCol": ("age", "sex"),
            "pred": BASELINE_DIRECTORY / "regenie_step1_pred.list",
            "bsize": 512,
            "device": types.Device.CPU.value,
            "variant_limit": PARITY_VARIANT_LIMIT,
            "staging_depth": 1,
            "output_run_directory": output_directory / "regenie2_binary_score_parity",
            "format": types.OutputFormat.PARQUET.value,
            "finalize_parquet": True,
        }
    )

    assert artifacts.final_parquet is not None
    observed_results = pl.read_parquet(artifacts.final_parquet)
    baseline_results = load_regenie_baseline_results(
        BASELINE_DIRECTORY / "regenie_step2_score_only_phenotype_binary.regenie",
        BINARY_SCORE_WORKFLOW,
    )

    return Regenie2ParityResults(
        observed_results=observed_results,
        baseline_results=baseline_results,
    )


@pytest.mark.parametrize(
    "tolerance",
    build_statistic_tolerance_parameters(QUANTITATIVE_WORKFLOW),
)
def test_regenie2_quantitative_linear_matches_regenie_baseline_statistics(
    quantitative_regenie2_parity_results: Regenie2ParityResults,
    tolerance: tests.parity.harness.StatisticTolerance,
) -> None:
    """Validate quantitative association statistics match REGENIE within tolerance."""
    tests.parity.harness.assert_statistic_columns_match(
        quantitative_regenie2_parity_results.observed_results,
        quantitative_regenie2_parity_results.baseline_results,
        join_column="ID",
        tolerance=tolerance,
        expected_row_count=PARITY_VARIANT_LIMIT,
    )


@pytest.mark.parametrize(
    "tolerance",
    build_statistic_tolerance_parameters(BINARY_SCORE_WORKFLOW),
)
def test_regenie2_binary_score_only_matches_regenie_baseline_statistics(
    binary_score_regenie2_parity_results: Regenie2ParityResults,
    tolerance: tests.parity.harness.StatisticTolerance,
) -> None:
    """Validate binary score-only association statistics match REGENIE within tolerance."""
    tests.parity.harness.assert_statistic_columns_match(
        binary_score_regenie2_parity_results.observed_results,
        binary_score_regenie2_parity_results.baseline_results,
        join_column="ID",
        tolerance=tolerance,
        expected_row_count=PARITY_VARIANT_LIMIT,
    )


def test_regenie2_linear_api_produces_valid_output(
    quantitative_regenie2_parity_results: Regenie2ParityResults,
) -> None:
    """Validate the end-to-end API output shape and validity columns."""
    observed_results = quantitative_regenie2_parity_results.observed_results

    assert observed_results.height == PARITY_VARIANT_LIMIT
    assert observed_results.get_column("ID").n_unique() == PARITY_VARIANT_LIMIT
    assert (observed_results.get_column("N") > 0).all()
    assert observed_results.get_column("BETA").is_finite().all()
    assert observed_results.get_column("SE").is_finite().all()
    assert observed_results.get_column("CHISQ").is_finite().all()
    assert observed_results.get_column("LOG10P").is_finite().all()
