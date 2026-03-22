#!/usr/bin/env python3
"""Run PLINK2 and Regenie Phase 0 baselines and save a benchmark report."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cpuinfo
import psutil

GPUtilModule: Any | None
try:
    import GPUtil as GPUtilModuleImport
except ImportError:  # pragma: no cover - optional dependency branch
    GPUtilModule = None
else:
    GPUtilModule = GPUtilModuleImport


@dataclass(frozen=True)
class HardwareGpu:
    """Summary information for a detected GPU device."""

    name: str
    memory_total_megabytes: float


@dataclass(frozen=True)
class HardwareSummary:
    """Machine hardware summary used in the benchmark report."""

    cpu_model_name: str
    physical_cpu_core_count: int | None
    logical_cpu_core_count: int | None
    total_memory_gigabytes: float
    gpus: list[HardwareGpu]


@dataclass(frozen=True)
class CommandResult:
    """Execution metadata for a benchmark command."""

    success: bool
    command: str
    duration_seconds: float
    stdout: str
    stderr: str
    output_prefix: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class BaselinePaths:
    """Paths required for running the Phase 0 benchmark suite."""

    data_directory: Path
    baseline_directory: Path
    bed_prefix: Path
    bgen_path: Path
    sample_path: Path
    continuous_phenotype_path: Path
    binary_phenotype_path: Path
    covariate_path: Path
    regenie_prediction_list_path: Path


def resolve_required_executable(environment_name: str, default_command: str) -> str:
    """Resolve a required external executable from the environment or PATH.

    Args:
        environment_name: Environment variable override name.
        default_command: Default executable name.

    Returns:
        The resolved command.

    Raises:
        RuntimeError: The executable cannot be found.

    """
    executable_name = os.environ.get(environment_name, default_command)
    if shutil.which(executable_name) is None:
        raise RuntimeError(
            f"Required executable '{executable_name}' is not available. "
            f"Set {environment_name} or enter the project nix shell."
        )
    return executable_name


def collect_hardware_summary() -> HardwareSummary:
    """Collect CPU, memory, and optional GPU information."""
    gpu_summaries: list[HardwareGpu] = []
    if GPUtilModule is not None:
        for gpu in GPUtilModule.getGPUs():
            gpu_summaries.append(
                HardwareGpu(
                    name=gpu.name,
                    memory_total_megabytes=float(gpu.memoryTotal),
                )
            )

    return HardwareSummary(
        cpu_model_name=cpuinfo.get_cpu_info().get("brand_raw", "Unknown CPU"),
        physical_cpu_core_count=psutil.cpu_count(logical=False),
        logical_cpu_core_count=psutil.cpu_count(logical=True),
        total_memory_gigabytes=round(psutil.virtual_memory().total / (1024**3), 2),
        gpus=gpu_summaries,
    )


def run_command(command_name: str, command_arguments: list[str], output_prefix: Path | None = None) -> CommandResult:
    """Run an external benchmark command and capture structured output.

    Args:
        command_name: Human-readable command label.
        command_arguments: Command line arguments.
        output_prefix: Optional output prefix associated with this command.

    Returns:
        Structured command execution details.

    """
    command_line = " ".join(command_arguments)
    print(f"\n--- Running {command_name} ---")
    print(f"Command: {command_line}")

    start_time = time.perf_counter()
    try:
        completed_process = subprocess.run(
            command_arguments,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        duration_seconds = time.perf_counter() - start_time
        error_message = str(error)
        print(error_message)
        return CommandResult(
            success=False,
            command=command_line,
            duration_seconds=duration_seconds,
            stdout="",
            stderr=error_message,
            output_prefix=str(output_prefix) if output_prefix is not None else None,
            error=error_message,
        )

    duration_seconds = time.perf_counter() - start_time
    success = completed_process.returncode == 0
    if success:
        print(f"Success: {command_name} completed in {duration_seconds:.2f} seconds.")
        error_message = None
    else:
        error_message = completed_process.stderr.strip() or f"Command exited with code {completed_process.returncode}."
        print(f"Error running {command_name}:\n{error_message}")

    return CommandResult(
        success=success,
        command=command_line,
        duration_seconds=duration_seconds,
        stdout=completed_process.stdout,
        stderr=completed_process.stderr,
        output_prefix=str(output_prefix) if output_prefix is not None else None,
        error=error_message,
    )


def build_baseline_paths() -> BaselinePaths:
    """Build the standard Phase 0 file paths."""
    data_directory = Path(os.environ.get("GWAS_ENGINE_DATA_DIR", "data"))
    baseline_directory = data_directory / "baselines"
    bed_prefix = data_directory / "1kg_chr22_full"
    return BaselinePaths(
        data_directory=data_directory,
        baseline_directory=baseline_directory,
        bed_prefix=bed_prefix,
        bgen_path=data_directory / "1kg_chr22_full.bgen",
        sample_path=data_directory / "1kg_chr22_full.sample",
        continuous_phenotype_path=data_directory / "pheno_cont.txt",
        binary_phenotype_path=data_directory / "pheno_bin.txt",
        covariate_path=data_directory / "covariates.txt",
        regenie_prediction_list_path=baseline_directory / "regenie_step1_pred.list",
    )


def validate_input_files(baseline_paths: BaselinePaths) -> None:
    """Ensure all required input files exist before benchmarking.

    Args:
        baseline_paths: Standard benchmark file paths.

    Raises:
        FileNotFoundError: One or more required inputs are missing.

    """
    required_paths = [
        baseline_paths.bed_prefix.with_suffix(".bed"),
        baseline_paths.bed_prefix.with_suffix(".bim"),
        baseline_paths.bed_prefix.with_suffix(".fam"),
        baseline_paths.bgen_path,
        baseline_paths.sample_path,
        baseline_paths.continuous_phenotype_path,
        baseline_paths.binary_phenotype_path,
        baseline_paths.covariate_path,
    ]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        missing_path_lines = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Required benchmark inputs are missing:\n{missing_path_lines}")


def build_plink_continuous_command(plink_executable: str, baseline_paths: BaselinePaths) -> list[str]:
    """Build the PLINK2 continuous trait command."""
    return [
        plink_executable,
        "--bfile",
        str(baseline_paths.bed_prefix),
        "--pheno",
        str(baseline_paths.continuous_phenotype_path),
        "--covar",
        str(baseline_paths.covariate_path),
        "--glm",
        "allow-no-covars",
        "--out",
        str(baseline_paths.baseline_directory / "plink_cont"),
    ]


def build_plink_binary_command(plink_executable: str, baseline_paths: BaselinePaths) -> list[str]:
    """Build the PLINK2 binary trait command."""
    return [
        plink_executable,
        "--bfile",
        str(baseline_paths.bed_prefix),
        "--pheno",
        str(baseline_paths.binary_phenotype_path),
        "--covar",
        str(baseline_paths.covariate_path),
        "--glm",
        "firth-fallback",
        "allow-no-covars",
        "--out",
        str(baseline_paths.baseline_directory / "plink_bin"),
    ]


def build_regenie_step1_command(regenie_executable: str, baseline_paths: BaselinePaths) -> list[str]:
    """Build the Regenie step 1 command."""
    return [
        regenie_executable,
        "--step",
        "1",
        "--bed",
        str(baseline_paths.bed_prefix),
        "--phenoFile",
        str(baseline_paths.binary_phenotype_path),
        "--covarFile",
        str(baseline_paths.covariate_path),
        "--bt",
        "--cc12",
        "--force-step1",
        "--bsize",
        "1000",
        "--out",
        str(baseline_paths.baseline_directory / "regenie_step1"),
    ]


def build_regenie_step2_command(regenie_executable: str, baseline_paths: BaselinePaths) -> list[str]:
    """Build the Regenie step 2 command."""
    return [
        regenie_executable,
        "--step",
        "2",
        "--bgen",
        str(baseline_paths.bgen_path),
        "--sample",
        str(baseline_paths.sample_path),
        "--ref-first",
        "--phenoFile",
        str(baseline_paths.binary_phenotype_path),
        "--covarFile",
        str(baseline_paths.covariate_path),
        "--bt",
        "--cc12",
        "--firth",
        "--approx",
        "--bsize",
        "400",
        "--pred",
        str(baseline_paths.regenie_prediction_list_path),
        "--out",
        str(baseline_paths.baseline_directory / "regenie_step2"),
    ]


def serialize_results(results_by_name: dict[str, CommandResult]) -> dict[str, dict[str, Any]]:
    """Convert command results into JSON-serializable dictionaries."""
    return {result_name: asdict(command_result) for result_name, command_result in results_by_name.items()}


def main() -> None:
    """Run all Phase 0 baseline commands and save the benchmark report."""
    plink_executable = resolve_required_executable("PLINK2_BIN", "plink2")
    regenie_executable = resolve_required_executable("REGENIE_BIN", "regenie")

    baseline_paths = build_baseline_paths()
    baseline_paths.baseline_directory.mkdir(exist_ok=True)
    validate_input_files(baseline_paths)

    print("Gathering hardware specs...")
    hardware_summary = collect_hardware_summary()

    results_by_name: dict[str, CommandResult] = {}
    results_by_name["plink_cont"] = run_command(
        "PLINK2 Continuous",
        build_plink_continuous_command(plink_executable, baseline_paths),
        baseline_paths.baseline_directory / "plink_cont",
    )
    results_by_name["plink_bin"] = run_command(
        "PLINK2 Binary",
        build_plink_binary_command(plink_executable, baseline_paths),
        baseline_paths.baseline_directory / "plink_bin",
    )
    results_by_name["regenie_step1"] = run_command(
        "Regenie Step 1",
        build_regenie_step1_command(regenie_executable, baseline_paths),
        baseline_paths.baseline_directory / "regenie_step1",
    )

    if results_by_name["regenie_step1"].success and baseline_paths.regenie_prediction_list_path.exists():
        results_by_name["regenie_step2"] = run_command(
            "Regenie Step 2",
            build_regenie_step2_command(regenie_executable, baseline_paths),
            baseline_paths.baseline_directory / "regenie_step2",
        )
    else:
        error_message = (
            f"Missing Regenie prediction list: {baseline_paths.regenie_prediction_list_path}"
            if results_by_name["regenie_step1"].success
            else "Regenie step 1 failed."
        )
        print(f"\nSkipping Regenie Step 2: {error_message}")
        results_by_name["regenie_step2"] = CommandResult(
            success=False,
            command="",
            duration_seconds=0.0,
            stdout="",
            stderr=error_message,
            output_prefix=str(baseline_paths.baseline_directory / "regenie_step2"),
            error=error_message,
        )

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "hardware": asdict(hardware_summary),
        "results": serialize_results(results_by_name),
    }
    report_path = baseline_paths.data_directory / "benchmark_report.json"
    report_path.write_text(f"{json.dumps(report, indent=2)}\n")
    print(f"\nBenchmark complete. Report saved to {report_path}")


if __name__ == "__main__":
    main()
