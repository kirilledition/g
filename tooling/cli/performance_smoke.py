#!/usr/bin/env python3
"""Login-node-safe performance harness smoke benchmark."""

from __future__ import annotations

import argparse
import dataclasses
import os
import platform
import subprocess
import sys
import time
import tracemalloc
import typing
from pathlib import Path

from tooling.common import reports as tooling_reports

DEFAULT_OUTPUT_ROOT = Path("results/perf/smoke")
DEFAULT_ITERATION_COUNT = 48
DEFAULT_ITEM_COUNT = 4096


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


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        Configured argument parser.

    """
    parser = argparse.ArgumentParser(description="Run a login-node-safe benchmark harness smoke test.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Parent directory for smoke runs.",
    )
    parser.add_argument("--output-dir", type=Path, help="Exact output directory for this smoke run.")
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATION_COUNT,
        help="Tiny workload iteration count.",
    )
    parser.add_argument("--items", type=int, default=DEFAULT_ITEM_COUNT, help="Tiny workload item count.")
    return parser


def main() -> None:
    """Run the performance smoke benchmark CLI."""
    parser = build_argument_parser()
    arguments = parser.parse_args()
    output_directory = arguments.output_dir or timestamped_output_directory(arguments.output_root)
    try:
        result = run_smoke_workload(iteration_count=int(arguments.iterations), item_count=int(arguments.items))
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    summary_path = output_directory / "performance_smoke_summary.json"
    tooling_reports.write_json_report(summary_path, build_summary(result), sort_keys=True)
    print(f"Smoke benchmark wall_time_seconds={result.wall_time_seconds:.6g}")
    print(f"Smoke benchmark peak_memory_bytes={result.peak_memory_bytes}")
    print(f"Smoke benchmark checksum={result.checksum:.6g}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
