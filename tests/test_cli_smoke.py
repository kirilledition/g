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
CLI_SMOKE_ARTIFACT_ENVIRONMENT_VARIABLE = "G_CLI_SMOKE_ARTIFACT_DIR"
EXPECTED_VARIANT_IDENTIFIERS = ["RS1", "RS2", "RS3", "RS4"]
FINITE_RESULT_COLUMNS = ["BETA", "SE", "CHISQ", "LOG10P"]


def resolve_cli_smoke_artifact_directory(tmp_path: Path) -> Path:
    configured_artifact_directory = os.environ.get(CLI_SMOKE_ARTIFACT_ENVIRONMENT_VARIABLE)
    if configured_artifact_directory is None:
        return tmp_path
    return Path(configured_artifact_directory)


def build_cli_failure_message(completed_process: subprocess.CompletedProcess[str]) -> str:
    return (
        f"g exited with {completed_process.returncode}\n"
        f"stdout:\n{completed_process.stdout}\n"
        f"stderr:\n{completed_process.stderr}"
    )


@pytest.mark.cli_smoke
def test_installed_cli_runs_regenie2_linear_smoke(tmp_path: Path) -> None:
    """Run the installed console script through a tiny real BGEN scan."""
    g_executable = shutil.which("g")
    assert g_executable is not None

    artifact_directory = resolve_cli_smoke_artifact_directory(tmp_path)
    artifact_directory.mkdir(parents=True, exist_ok=True)
    logs_directory = artifact_directory / "logs"
    output_run_root = artifact_directory / "output-runs"

    prediction_list_path = artifact_directory / "pred.list"
    prediction_list_path.write_text(
        f"trait {CLI_SMOKE_DATA_DIRECTORY / 'trait.loco'}\n",
        encoding="utf-8",
    )

    output_root = artifact_directory / "smoke"
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
            "--g-output-run-directory",
            str(output_run_root),
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
            "--g-telemetry",
            "profile",
            "--g-log-dir",
            str(logs_directory),
            "--g-log-file",
            str(logs_directory / "events-and-tracing.jsonl"),
            "--g-stage-timings-json",
            str(logs_directory / "stage-timings.json"),
            "--g-profile-summary-json",
            str(logs_directory / "profile.summary.json"),
            "--no-g-log-stderr",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    (artifact_directory / "g-stdout.txt").write_text(completed_process.stdout, encoding="utf-8")
    (artifact_directory / "g-stderr.txt").write_text(completed_process.stderr, encoding="utf-8")

    assert completed_process.returncode == 0, build_cli_failure_message(completed_process)
    assert "Parquet dataset saved" in completed_process.stdout

    part_paths = sorted(output_run_root.glob("*.run/parts/*.parquet"))
    assert len(part_paths) == 2

    result_frame = pl.concat([pl.read_parquet(part_path) for part_path in part_paths])
    assert result_frame.height == 4
    assert result_frame.get_column("CHROM").to_list() == ["1"] * 4
    assert result_frame.get_column("ID").to_list() == EXPECTED_VARIANT_IDENTIFIERS
    assert result_frame.get_column("N").to_list() == [4] * 4
    for column_name in FINITE_RESULT_COLUMNS:
        assert result_frame.get_column(column_name).is_finite().all()
