from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import polars as pl
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
BGEN_PATH = REPOSITORY_ROOT / "tests" / "data" / "bgen" / "haplotypes.bgen"
CLI_SMOKE_DATA_DIRECTORY = REPOSITORY_ROOT / "tests" / "data" / "cli_smoke"
EXPECTED_VARIANT_IDENTIFIERS = ["RS1", "RS2", "RS3", "RS4"]
FINITE_RESULT_COLUMNS = ["BETA", "SE", "CHISQ", "LOG10P"]


@pytest.mark.cli_smoke
def test_installed_cli_runs_regenie2_linear_smoke(tmp_path: Path) -> None:
    """Run the installed console script through a tiny real BGEN scan."""
    g_executable = shutil.which("g")
    assert g_executable is not None

    prediction_list_path = tmp_path / "pred.list"
    prediction_list_path.write_text(
        f"trait {CLI_SMOKE_DATA_DIRECTORY / 'trait.loco'}\n",
        encoding="utf-8",
    )

    output_root = tmp_path / "smoke"
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"

    completed_process = subprocess.run(
        [
            g_executable,
            "regenie",
            "--step",
            "2",
            "--qt",
            "--bgen",
            str(BGEN_PATH),
            "--phenoFile",
            str(CLI_SMOKE_DATA_DIRECTORY / "phenotypes.tsv"),
            "--phenoCol",
            "trait",
            "--covarFile",
            str(CLI_SMOKE_DATA_DIRECTORY / "covariates.tsv"),
            "--covarColList",
            "age",
            "--pred",
            str(prediction_list_path),
            "--out",
            str(output_root),
            "--bsize",
            "2",
            "--g-device",
            "cpu",
            "--g-variant-limit",
            "4",
            "--g-staging-depth",
            "1",
            "--g-output-format",
            "parquet",
            "--g-output-chunks-per-arrow-file",
            "1",
            "--g-output-arrow-compression",
            "none",
            "--no-g-log-stderr",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed_process.returncode == 0, completed_process.stderr
    assert "Finalized Parquet saved" in completed_process.stdout

    final_parquet_paths = sorted(output_root.with_suffix(".g").glob("*.run/final.parquet"))
    assert len(final_parquet_paths) == 1

    result_frame = pl.read_parquet(final_parquet_paths[0])
    assert result_frame.height == 4
    assert result_frame.get_column("CHROM").to_list() == ["1"] * 4
    assert result_frame.get_column("ID").to_list() == EXPECTED_VARIANT_IDENTIFIERS
    assert result_frame.get_column("N").to_list() == [4] * 4
    for column_name in FINITE_RESULT_COLUMNS:
        assert result_frame.get_column(column_name).is_finite().all()
