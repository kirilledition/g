#!/usr/bin/env python3
"""Quick GPU-only benchmark for updated g app."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import typing

import polars as pl

if typing.TYPE_CHECKING:
    from pathlib import Path

    from benchmark import BaselinePaths


def build_baseline_paths() -> BaselinePaths:
    """Build the standard Phase 0 file paths."""
    from benchmark import build_baseline_paths as original_build_baseline_paths

    return original_build_baseline_paths()


def resolve_required_executable(environment_name: str, default_command: str) -> str:
    """Resolve a required external executable from the environment or PATH."""
    from benchmark import resolve_required_executable as original_resolve_required_executable

    return original_resolve_required_executable(environment_name, default_command)


def build_plink1_continuous_command(plink_executable: str, baseline_paths: BaselinePaths) -> list[str]:
    """Build the PLINK 1.9 continuous trait command."""
    from benchmark import build_plink1_continuous_command as original_build_command

    return original_build_command(plink_executable, baseline_paths)


def build_plink1_binary_command(plink_executable: str, baseline_paths: BaselinePaths) -> list[str]:
    """Build the PLINK 1.9 binary trait command."""
    from benchmark import build_plink1_binary_command as original_build_command

    return original_build_command(plink_executable, baseline_paths)


def build_plink2_continuous_command(plink_executable: str, baseline_paths: BaselinePaths) -> list[str]:
    """Build the PLINK2 continuous trait command."""
    from benchmark import build_plink2_continuous_command as original_build_command

    return original_build_command(plink_executable, baseline_paths)


def build_plink2_binary_command(plink_executable: str, baseline_paths: BaselinePaths) -> list[str]:
    """Build the PLINK2 binary trait command."""
    from benchmark import build_plink2_binary_command as original_build_command

    return original_build_command(plink_executable, baseline_paths)


def time_command(command_arguments: list[str]) -> float:
    """Run a command and return elapsed seconds."""
    start_time = time.perf_counter()
    completed_process = subprocess.run(command_arguments, check=False, capture_output=True, text=True)
    duration_seconds = time.perf_counter() - start_time
    if completed_process.returncode != 0:
        message = completed_process.stderr.strip() or f"Command failed: {' '.join(command_arguments)}"
        raise RuntimeError(message)
    return duration_seconds


def run_phase1_gpu(
    baseline_paths: BaselinePaths,
    association_mode: str,
    output_prefix: Path,
) -> tuple[float, pl.DataFrame]:
    """Run Phase 1 command on GPU only."""
    if association_mode == "linear":
        command_arguments = [
            sys.executable,
            "-c",
            "from g.cli import main; main()",
            "linear",
            "--bfile",
            str(baseline_paths.bed_prefix),
            "--pheno",
            str(baseline_paths.continuous_phenotype_path),
            "--pheno-name",
            "phenotype_continuous",
            "--covar",
            str(baseline_paths.covariate_path),
            "--covar-names",
            "age,sex",
            "--chunk-size",
            "2048",
            "--device",
            "gpu",
            "--out",
            str(output_prefix),
        ]
    else:
        command_arguments = [
            sys.executable,
            "-c",
            "from g.cli import main; main()",
            "logistic",
            "--bfile",
            str(baseline_paths.bed_prefix),
            "--pheno",
            str(baseline_paths.binary_phenotype_path),
            "--pheno-name",
            "phenotype_binary",
            "--covar",
            str(baseline_paths.covariate_path),
            "--covar-names",
            "age,sex",
            "--chunk-size",
            "512",
            "--device",
            "gpu",
            "--max-iterations",
            "50",
            "--tolerance",
            "1.0e-8",
            "--out",
            str(output_prefix),
        ]

    command_environment = os.environ.copy()
    command_environment["JAX_PLATFORMS"] = "gpu"

    print(f"\nRunning Phase 1 {association_mode} on GPU...", end=" ", flush=True)
    start_time = time.perf_counter()
    completed_process = subprocess.run(
        command_arguments,
        check=False,
        capture_output=True,
        text=True,
        env=command_environment,
    )
    duration_seconds = time.perf_counter() - start_time

    if completed_process.returncode != 0:
        raise RuntimeError(f"Phase 1 {association_mode} GPU run failed: {completed_process.stderr.strip()}")

    print(f"{duration_seconds:.2f}s")

    output_suffix = ".linear.tsv" if association_mode == "linear" else ".logistic.tsv"
    output_path = output_prefix.with_suffix(output_suffix)

    if not output_path.exists():
        raise RuntimeError(f"Expected output file was not written: {output_path}")

    return duration_seconds, pl.read_csv(output_path, separator="\t")


def main() -> None:
    """Run GPU-only benchmark."""
    baseline_paths = build_baseline_paths()
    plink1_executable = resolve_required_executable("PLINK1_BIN", "plink")
    plink2_executable = resolve_required_executable("PLINK2_BIN", "plink2")

    print("=" * 70)
    print("GPU-ONLY BENCHMARK (Updated g app)")
    print("=" * 70)

    # Run PLINK baselines
    print("\n--- PLINK Baselines ---")
    plink1_linear = time_command(build_plink1_continuous_command(plink1_executable, baseline_paths))
    print(f"PLINK 1.9 Linear: {plink1_linear:.2f}s")

    plink1_logistic = time_command(build_plink1_binary_command(plink1_executable, baseline_paths))
    print(f"PLINK 1.9 Logistic: {plink1_logistic:.2f}s")

    plink2_linear = time_command(build_plink2_continuous_command(plink2_executable, baseline_paths))
    print(f"PLINK 2 Linear: {plink2_linear:.2f}s")

    plink2_logistic = time_command(build_plink2_binary_command(plink2_executable, baseline_paths))
    print(f"PLINK 2 Logistic: {plink2_logistic:.2f}s")

    # Run Phase 1 GPU only
    print("\n--- Phase 1 GPU ---")
    linear_gpu_time, _ = run_phase1_gpu(
        baseline_paths=baseline_paths,
        association_mode="linear",
        output_prefix=baseline_paths.data_directory / "phase1_linear_gpu_new",
    )

    logistic_gpu_time, _ = run_phase1_gpu(
        baseline_paths=baseline_paths,
        association_mode="logistic",
        output_prefix=baseline_paths.data_directory / "phase1_logistic_gpu_new",
    )

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\n--- Linear Regression ---")
    print(f"PLINK 1.9:     {plink1_linear:6.2f}s")
    print(f"PLINK 2:       {plink2_linear:6.2f}s")
    print(f"Phase 1 GPU:   {linear_gpu_time:6.2f}s")
    print(f"  vs PLINK 1.9: {plink1_linear / linear_gpu_time:5.2f}x")
    print(f"  vs PLINK 2:   {plink2_linear / linear_gpu_time:5.2f}x")

    print("\n--- Logistic Regression ---")
    print(f"PLINK 1.9:     {plink1_logistic:6.2f}s")
    print(f"PLINK 2:       {plink2_logistic:6.2f}s")
    print(f"Phase 1 GPU:   {logistic_gpu_time:6.2f}s")
    print(f"  vs PLINK 1.9: {plink1_logistic / logistic_gpu_time:5.2f}x")
    print(f"  vs PLINK 2:   {plink2_logistic / logistic_gpu_time:5.2f}x")

    # Save report
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "baselines": {
            "plink1_linear": plink1_linear,
            "plink1_logistic": plink1_logistic,
            "plink2_linear": plink2_linear,
            "plink2_logistic": plink2_logistic,
        },
        "phase1_gpu": {
            "linear_seconds": linear_gpu_time,
            "logistic_seconds": logistic_gpu_time,
        },
        "speed_ratios": {
            "linear_vs_plink1": plink1_linear / linear_gpu_time,
            "linear_vs_plink2": plink2_linear / linear_gpu_time,
            "logistic_vs_plink1": plink1_logistic / logistic_gpu_time,
            "logistic_vs_plink2": plink2_logistic / logistic_gpu_time,
        },
    }

    report_path = baseline_paths.data_directory / "benchmark_gpu_only.json"
    report_path.write_text(f"{json.dumps(report, indent=2)}\n")
    print(f"\nFull report saved to: {report_path}")


if __name__ == "__main__":
    main()
