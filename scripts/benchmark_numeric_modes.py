#!/usr/bin/env python3
"""Run benchmark and parity commands across JAX numeric modes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class NumericModeRun:
    """One benchmark/parity run under a numeric mode."""

    numeric_mode: str
    benchmark_report: dict[str, object] | None
    phase1_evaluation_report: dict[str, object] | None


@dataclass(frozen=True)
class NumericModeSweepReport:
    """Combined numeric-mode experiment report."""

    runs: list[NumericModeRun]


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Benchmark JAX numeric modes against existing GWAS checks.")
    parser.add_argument(
        "--modes",
        default="float32,bfloat16",
        help="Comma-separated JAX numeric modes to evaluate.",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip scripts/benchmark_jax_execution.py.",
    )
    parser.add_argument(
        "--run-phase1-evaluate",
        action="store_true",
        help="Run scripts/evaluate_phase1.py for each numeric mode.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path for the combined JSON report.",
    )
    return parser


def parse_modes(raw_modes: str) -> list[str]:
    """Parse and validate the numeric mode list."""
    modes = [mode.strip() for mode in raw_modes.split(",") if mode.strip()]
    if not modes:
        message = "At least one numeric mode must be specified."
        raise ValueError(message)
    return modes


def run_json_command(command_arguments: list[str], environment: dict[str, str]) -> dict[str, object]:
    """Run a command that emits JSON to stdout."""
    completed_process = subprocess.run(
        command_arguments,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed_process.returncode != 0:
        message = completed_process.stderr.strip() or completed_process.stdout.strip()
        raise RuntimeError(message)
    return json.loads(completed_process.stdout)


def run_phase1_evaluate(environment: dict[str, str]) -> dict[str, object]:
    """Run the Phase 1 parity evaluation and load its report."""
    completed_process = subprocess.run(
        ["uv", "run", "python", "scripts/evaluate_phase1.py"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed_process.returncode != 0:
        message = completed_process.stderr.strip() or completed_process.stdout.strip()
        raise RuntimeError(message)
    report_path = Path(completed_process.stdout.strip())
    return json.loads(report_path.read_text())


def main() -> None:
    """Run benchmark/parity commands across the requested numeric modes."""
    arguments = build_argument_parser().parse_args()
    modes = parse_modes(arguments.modes)
    report_runs: list[NumericModeRun] = []

    for numeric_mode in modes:
        environment = os.environ.copy()
        environment["G_JAX_NUMERIC_MODE"] = numeric_mode
        benchmark_report = None
        phase1_evaluation_report = None

        if not arguments.skip_benchmark:
            benchmark_report = run_json_command(
                ["uv", "run", "python", "scripts/benchmark_jax_execution.py"],
                environment,
            )
        if arguments.run_phase1_evaluate:
            phase1_evaluation_report = run_phase1_evaluate(environment)

        report_runs.append(
            NumericModeRun(
                numeric_mode=numeric_mode,
                benchmark_report=benchmark_report,
                phase1_evaluation_report=phase1_evaluation_report,
            )
        )

    report = NumericModeSweepReport(runs=report_runs)
    report_json = json.dumps(asdict(report), indent=2)
    if arguments.output is not None:
        arguments.output.write_text(f"{report_json}\n")
    print(report_json)


if __name__ == "__main__":
    main()
