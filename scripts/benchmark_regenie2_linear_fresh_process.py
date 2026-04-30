#!/usr/bin/env python3
"""Benchmark REGENIE step 2 in fresh Python processes."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_DATA_DIRECTORY = Path("data")
DEFAULT_OUTPUT_DIRECTORY = Path("data/benchmarks/regenie2_linear_fresh_process")


@dataclass(frozen=True)
class TrialResult:
    """One fresh-process benchmark trial result."""

    trial_index: int
    wall_time_seconds: float
    output_path: str


@dataclass(frozen=True)
class BenchmarkSummary:
    """Aggregate summary for one fresh-process benchmark run."""

    device: str
    chunk_size: int
    finalize_parquet: bool
    arrow_payload_batch_size: int
    trial_count: int
    warmup_count: int
    mean_wall_time_seconds: float
    median_wall_time_seconds: float
    min_wall_time_seconds: float
    max_wall_time_seconds: float
    trial_results: list[TrialResult]


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Benchmark g REGENIE step 2 in fresh Python processes.")
    parser.add_argument("--device", default="gpu", choices=("cpu", "gpu"), help="Execution device.")
    parser.add_argument("--chunk-size", type=int, default=8192, help="Variants per chunk.")
    parser.add_argument(
        "--finalize-parquet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Finalize Arrow chunks into Parquet before finishing the trial.",
    )
    parser.add_argument("--arrow-payload-batch-size", type=int, default=1, help="Arrow IPC payload batch size.")
    parser.add_argument("--trials", type=int, default=3, help="Measured fresh-process trial count.")
    parser.add_argument("--warmup-trials", type=int, default=1, help="Unreported fresh-process warmup trials.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIRECTORY, help="Input data directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for benchmark outputs and summary files.",
    )
    parser.add_argument(
        "--json-summary-path",
        type=Path,
        help="Optional explicit JSON summary output path.",
    )
    return parser


def build_child_command(
    *,
    data_directory: Path,
    output_path: Path,
    device: str,
    chunk_size: int,
    finalize_parquet: bool,
    arrow_payload_batch_size: int,
) -> list[str]:
    """Build the child Python command for one isolated trial."""
    child_code = textwrap.dedent(
        """
        import json
        import time

        from g import api, types

        start_time = time.perf_counter()
        artifacts = api.regenie2_linear(
            bgen={bgen_path!r},
            sample={sample_path!r},
            pheno={phenotype_path!r},
            pheno_name="phenotype_continuous",
            out={output_path!r},
            covar={covariate_path!r},
            covar_names="age,sex",
            pred={prediction_path!r},
            compute=api.ComputeConfig(
                device=types.Device({device!r}),
                chunk_size={chunk_size},
                finalize_parquet={finalize_parquet},
                arrow_payload_batch_size={arrow_payload_batch_size},
            ),
        )
        wall_time_seconds = time.perf_counter() - start_time
        artifact_path = artifacts.final_parquet or artifacts.output_run_directory
        print(json.dumps({{"wall_time_seconds": wall_time_seconds, "output_path": str(artifact_path)}}))
        """
    ).format(
        bgen_path=str(data_directory / "1kg_chr22_full.bgen"),
        sample_path=str(data_directory / "1kg_chr22_full.sample"),
        phenotype_path=str(data_directory / "pheno_cont.txt"),
        output_path=str(output_path),
        covariate_path=str(data_directory / "covariates.txt"),
        prediction_path=str(data_directory / "baselines/regenie_step1_qt_pred.list"),
        device=device,
        chunk_size=chunk_size,
        finalize_parquet="True" if finalize_parquet else "False",
        arrow_payload_batch_size=arrow_payload_batch_size,
    )
    return [sys.executable, "-c", child_code]


def run_fresh_process_trial(
    *,
    trial_index: int,
    data_directory: Path,
    output_directory: Path,
    device: str,
    chunk_size: int,
    finalize_parquet: bool,
    arrow_payload_batch_size: int,
) -> TrialResult:
    """Run one isolated fresh-process trial."""
    output_prefix = output_directory / (
        f"{device}_finalize{int(finalize_parquet)}_"
        f"chunk{chunk_size}_arrowbatch{arrow_payload_batch_size}_"
        f"trial{trial_index:02d}"
    )
    command_arguments = build_child_command(
        data_directory=data_directory,
        output_path=output_prefix,
        device=device,
        chunk_size=chunk_size,
        finalize_parquet=finalize_parquet,
        arrow_payload_batch_size=arrow_payload_batch_size,
    )
    child_environment = os.environ.copy()
    child_environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    child_environment.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".50")
    completed_process = subprocess.run(
        command_arguments,
        check=True,
        capture_output=True,
        text=True,
        env=child_environment,
    )
    result_line = completed_process.stdout.strip().splitlines()[-1]
    result_payload = json.loads(result_line)
    return TrialResult(
        trial_index=trial_index,
        wall_time_seconds=float(result_payload["wall_time_seconds"]),
        output_path=str(result_payload["output_path"]),
    )


def build_summary(
    *,
    device: str,
    chunk_size: int,
    finalize_parquet: bool,
    arrow_payload_batch_size: int,
    warmup_count: int,
    trial_results: list[TrialResult],
) -> BenchmarkSummary:
    """Build an aggregate summary from measured trials."""
    wall_time_values = [trial_result.wall_time_seconds for trial_result in trial_results]
    return BenchmarkSummary(
        device=device,
        chunk_size=chunk_size,
        finalize_parquet=finalize_parquet,
        arrow_payload_batch_size=arrow_payload_batch_size,
        trial_count=len(trial_results),
        warmup_count=warmup_count,
        mean_wall_time_seconds=statistics.fmean(wall_time_values),
        median_wall_time_seconds=statistics.median(wall_time_values),
        min_wall_time_seconds=min(wall_time_values),
        max_wall_time_seconds=max(wall_time_values),
        trial_results=trial_results,
    )


def main() -> None:
    """Run the fresh-process benchmark."""
    argument_parser = build_argument_parser()
    arguments = argument_parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    for warmup_index in range(arguments.warmup_trials):
        _ = run_fresh_process_trial(
            trial_index=-(warmup_index + 1),
            data_directory=arguments.data_dir,
            output_directory=arguments.output_dir,
            device=arguments.device,
            chunk_size=arguments.chunk_size,
            finalize_parquet=arguments.finalize_parquet,
            arrow_payload_batch_size=arguments.arrow_payload_batch_size,
        )

    measured_trial_results = [
        run_fresh_process_trial(
            trial_index=trial_index,
            data_directory=arguments.data_dir,
            output_directory=arguments.output_dir,
            device=arguments.device,
            chunk_size=arguments.chunk_size,
            finalize_parquet=arguments.finalize_parquet,
            arrow_payload_batch_size=arguments.arrow_payload_batch_size,
        )
        for trial_index in range(arguments.trials)
    ]

    benchmark_summary = build_summary(
        device=arguments.device,
        chunk_size=arguments.chunk_size,
        finalize_parquet=arguments.finalize_parquet,
        arrow_payload_batch_size=arguments.arrow_payload_batch_size,
        warmup_count=arguments.warmup_trials,
        trial_results=measured_trial_results,
    )
    default_summary_filename = (
        f"{arguments.device}_finalize{int(arguments.finalize_parquet)}_"
        f"chunk{arguments.chunk_size}_"
        f"arrowbatch{arguments.arrow_payload_batch_size}.json"
    )
    json_summary_path = arguments.json_summary_path or (
        arguments.output_dir / default_summary_filename
    )
    json_summary_path.write_text(json.dumps(asdict(benchmark_summary), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(asdict(benchmark_summary), indent=2))


if __name__ == "__main__":
    main()
