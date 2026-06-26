#!/usr/bin/env python3
"""Login-node-safe performance harness smoke benchmark."""

from __future__ import annotations

import dataclasses
import os
import platform
import subprocess
import sys
import time
import tracemalloc
import typing
from pathlib import Path

import hydra

from tooling.common import hydra_arguments as tooling_hydra_arguments
from tooling.common import hydra_compat as tooling_hydra_compat
from tooling.common import reports as tooling_reports

if typing.TYPE_CHECKING:
    import omegaconf

DEFAULT_OUTPUT_ROOT = Path("results/perf/smoke")
DEFAULT_ITERATION_COUNT = 48
DEFAULT_ITEM_COUNT = 4096
SMOKE_SCHEMA_VERSION = 1
SMOKE_REPORT_CONTRACT = tooling_reports.VersionedReportContract(
    schema_version=SMOKE_SCHEMA_VERSION,
    required_fields=("schema", "metadata", "configuration", "metrics"),
    optional_fields=(),
    schema_field_name="schema_version",
    reject_unknown_fields=True,
)


@dataclasses.dataclass(frozen=True)
class SmokeBenchmarkResult:
    """Result from the login-node-safe smoke workload.

    Attributes:
        wall_time_seconds: Elapsed workload time.
        peak_memory_bytes: Peak memory measured by tracemalloc.
        checksum: Deterministic numeric checksum.
        iteration_count: Workload iteration count.
        item_count: Integer values processed per iteration.

    """

    wall_time_seconds: float
    peak_memory_bytes: int
    checksum: float
    iteration_count: int
    item_count: int


@dataclasses.dataclass(frozen=True)
class PerformanceSmokeArguments:
    """Resolved parameters for the performance smoke benchmark.

    Attributes:
        output_root: Parent directory for smoke runs.
        output_dir: Exact output directory for this smoke run.
        iterations: Tiny workload iteration count.
        items: Tiny workload item count.

    """

    output_root: Path
    output_dir: Path | None
    iterations: int
    items: int


def timestamped_output_directory(output_root: Path) -> Path:
    """Build a timestamped smoke output directory.

    Args:
        output_root: Parent output directory.

    Returns:
        Timestamped output directory.

    """
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return output_root / f"smoke_{timestamp}"


def run_smoke_workload(iteration_count: int, item_count: int) -> SmokeBenchmarkResult:
    """Run a tiny deterministic workload to validate benchmark plumbing.

    Args:
        iteration_count: Number of timed iterations.
        item_count: Integer values processed per iteration.

    Returns:
        Smoke benchmark result.

    Raises:
        ValueError: If the workload size is invalid.

    """
    if iteration_count <= 0:
        message = "--iterations must be positive."
        raise ValueError(message)
    if item_count <= 0:
        message = "--items must be positive."
        raise ValueError(message)

    tracemalloc.start()
    start_time = time.perf_counter()
    checksum = 0.0
    for iteration_index in range(iteration_count):
        values = [((item_index * 17) + iteration_index) % 251 for item_index in range(item_count)]
        centered_square_sum = sum((value - 125) * (value - 125) for value in values)
        checksum += centered_square_sum / item_count
    wall_time_seconds = time.perf_counter() - start_time
    peak_memory_bytes = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return SmokeBenchmarkResult(
        wall_time_seconds=wall_time_seconds,
        peak_memory_bytes=peak_memory_bytes,
        checksum=checksum,
        iteration_count=iteration_count,
        item_count=item_count,
    )


def command_output(command_arguments: list[str]) -> dict[str, typing.Any]:
    """Run a metadata command and return captured output.

    Args:
        command_arguments: Command and arguments.

    Returns:
        JSON-serializable command result.

    """
    completed_process = subprocess.run(command_arguments, check=False, capture_output=True, text=True)
    return {
        "return_code": completed_process.returncode,
        "stdout": completed_process.stdout.strip(),
        "stderr": completed_process.stderr.strip(),
    }


def build_summary(result: SmokeBenchmarkResult) -> dict[str, typing.Any]:
    """Build the smoke benchmark JSON summary.

    Args:
        result: Smoke benchmark result.

    Returns:
        JSON-serializable summary.

    """
    relevant_environment = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(("G_", "GWAS_ENGINE_", "JAX_", "XLA_", "CUDA_", "RAYON_", "SLURM_"))
    }
    return {
        "schema_version": SMOKE_SCHEMA_VERSION,
        "schema": "g.performance_smoke.v1",
        "metadata": {
            "git_head": command_output(["git", "rev-parse", "HEAD"]),
            "git_status": command_output(["git", "status", "--short"]),
            "hostname": command_output(["hostname"]),
            "platform": platform.platform(),
            "python": sys.version,
            "environment": relevant_environment,
        },
        "configuration": {
            "iteration_count": result.iteration_count,
            "item_count": result.item_count,
        },
        "metrics": {
            "smoke.wall_time_seconds": {
                "category": "speed",
                "unit": "seconds",
                "value": result.wall_time_seconds,
            },
            "smoke.peak_memory_bytes": {
                "category": "memory",
                "unit": "bytes",
                "value": result.peak_memory_bytes,
            },
            "smoke.checksum": {
                "category": "numerical",
                "unit": "checksum",
                "value": result.checksum,
            },
        },
    }


def run_tool(arguments: PerformanceSmokeArguments) -> None:
    """Run the performance smoke benchmark CLI."""
    output_directory = arguments.output_dir or timestamped_output_directory(arguments.output_root)
    result = run_smoke_workload(iteration_count=arguments.iterations, item_count=arguments.items)
    summary_path = output_directory / "performance_smoke_summary.json"
    tooling_reports.write_versioned_json_report(
        summary_path, build_summary(result), SMOKE_REPORT_CONTRACT, sort_keys=True
    )
    print(f"Smoke benchmark wall_time_seconds={result.wall_time_seconds:.6g}")
    print(f"Smoke benchmark peak_memory_bytes={result.peak_memory_bytes}")
    print(f"Smoke benchmark checksum={result.checksum:.6g}")
    print(f"Wrote summary: {summary_path}")


def build_arguments_from_config(config: omegaconf.DictConfig) -> PerformanceSmokeArguments:
    """Resolve performance smoke parameters from Hydra config."""
    tool_values = tooling_hydra_arguments.tool_config_to_dictionary(config)
    return PerformanceSmokeArguments(
        output_root=tooling_hydra_arguments.path_or_none(tool_values["output_root"]) or DEFAULT_OUTPUT_ROOT,
        output_dir=tooling_hydra_arguments.path_or_none(tool_values["output_dir"]),
        iterations=int(tool_values["iterations"]),
        items=int(tool_values["items"]),
    )


@hydra.main(version_base=None, config_path="../configs", config_name="performance_smoke")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Run the performance smoke benchmark from Hydra configuration."""
    try:
        run_tool(build_arguments_from_config(config))
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


def main() -> None:
    """Run the performance smoke benchmark from default Hydra configuration."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()


if __name__ == "__main__":
    main()
