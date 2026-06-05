#!/usr/bin/env python3
"""Benchmark REGENIE step 2 in fresh Python processes."""

from __future__ import annotations

import argparse
import enum
import json
import os
import statistics
import subprocess
import sys
import textwrap
import typing
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_DATA_DIRECTORY = Path("data")
DEFAULT_OUTPUT_DIRECTORY = Path("data/benchmarks/regenie2_linear_fresh_process")


class RunnerMode(enum.StrEnum):
    """Fresh-process runner to benchmark."""

    PYTHON_JAX = "python-jax"
    NATIVE_CUDA_KERNEL = "native-cuda-kernel"


@dataclass(frozen=True)
class TrialResult:
    """One fresh-process benchmark trial result."""

    trial_index: int
    wall_time_seconds: float
    output_path: str
    output_row_count: int
    chunk_file_count: int
    chunk_bytes: int
    final_parquet_bytes: int | None


@dataclass(frozen=True)
class BenchmarkSummary:
    """Aggregate summary for one fresh-process benchmark run."""

    runner: str
    device: str
    chunk_size: int
    finalize_parquet: bool
    output_writer_thread_count: int
    cuda_block_size: int | None
    trial_count: int
    warmup_count: int
    mean_wall_time_seconds: float
    median_wall_time_seconds: float
    min_wall_time_seconds: float
    max_wall_time_seconds: float
    mean_rows_per_second: float
    mean_chunk_file_count: float
    mean_chunk_bytes: float
    mean_final_parquet_bytes: float | None
    trial_results: list[TrialResult]


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Benchmark g REGENIE step 2 in fresh isolated processes.")
    parser.add_argument(
        "--runner",
        default=RunnerMode.PYTHON_JAX.value,
        choices=tuple(runner_mode.value for runner_mode in RunnerMode),
        help="Execution runner.",
    )
    parser.add_argument("--device", default="gpu", choices=("cpu", "gpu"), help="Execution device.")
    parser.add_argument("--chunk-size", type=int, default=8192, help="Variants per chunk.")
    parser.add_argument(
        "--finalize-parquet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Finalize Arrow chunks into Parquet before finishing the trial.",
    )
    parser.add_argument(
        "--output-writer-thread-count",
        type=int,
        default=1,
        help="Background writer thread count.",
    )
    parser.add_argument("--trials", type=int, default=3, help="Measured fresh-process trial count.")
    parser.add_argument("--warmup-trials", type=int, default=1, help="Unreported fresh-process warmup trials.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIRECTORY, help="Input data directory.")
    parser.add_argument(
        "--native-binary",
        type=Path,
        help="Optional prebuilt regenie2-linear-native binary. Defaults to cargo run.",
    )
    parser.add_argument(
        "--cuda-block-size",
        type=int,
        default=256,
        help="CUDA kernel block size for the native runner.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for benchmark outputs and summary files.",
    )
    parser.add_argument("--json-summary-path", type=Path, help="Optional explicit JSON summary output path.")
    return parser


def build_python_jax_child_command(
    *,
    data_directory: Path,
    output_path: Path,
    device: str,
    chunk_size: int,
    finalize_parquet: bool,
    output_writer_thread_count: int,
) -> list[str]:
    """Build the child Python/JAX command for one isolated trial."""
    child_code = textwrap.dedent(
        """
        import json
        import time

        import polars as pl

        from g import api

        start_time = time.perf_counter()
        artifacts = api.regenie.from_options({{
            "step": 2,
            "qt": True,
            "bgen": {bgen_path!r},
            "sample": {sample_path!r},
            "phenoFile": {phenotype_path!r},
            "phenoCol": "phenotype_continuous",
            "out": {output_path!r},
            "covarFile": {covariate_path!r},
            "covarColList": "age,sex",
            "pred": {prediction_path!r},
            "g-device": {device!r},
            "bsize": {chunk_size},
            "g-output-format": "parquet" if {finalize_parquet} else "arrow",
            "g-writer-threads": {output_writer_thread_count},
        }})
        wall_time_seconds = time.perf_counter() - start_time
        artifact_path = artifacts.final_parquet or artifacts.output_run_directory
        output_row_count = (
            pl.scan_parquet(artifacts.final_parquet).select(pl.len()).collect().item()
            if artifacts.final_parquet is not None
            else 0
        )
        output_run_directory = artifacts.output_run_directory
        chunk_file_paths = (
            list((output_run_directory / "chunks").glob("chunk_*.arrow")) if output_run_directory is not None else []
        )
        chunk_bytes = sum(chunk_file_path.stat().st_size for chunk_file_path in chunk_file_paths)
        final_parquet_bytes = artifacts.final_parquet.stat().st_size if artifacts.final_parquet is not None else None
        print(
            json.dumps(
                {{
                    "wall_time_seconds": wall_time_seconds,
                    "output_path": str(artifact_path),
                    "output_row_count": output_row_count,
                    "chunk_file_count": len(chunk_file_paths),
                    "chunk_bytes": chunk_bytes,
                    "final_parquet_bytes": final_parquet_bytes,
                }}
            )
        )
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
        output_writer_thread_count=output_writer_thread_count,
    )
    return [sys.executable, "-c", child_code]


def build_native_cuda_child_command(
    *,
    data_directory: Path,
    output_path: Path,
    chunk_size: int,
    cuda_block_size: int,
    finalize_parquet: bool,
    output_writer_thread_count: int,
    native_binary: Path | None,
    report_json_path: Path,
) -> list[str]:
    """Build the native CUDA kernel command for one isolated trial."""
    if native_binary is None:
        command_arguments = [
            "cargo",
            "run",
            "--profile",
            "perf-dev",
            "--features",
            "cuda-kernel",
            "--bin",
            "regenie2-linear-native",
            "--",
        ]
    else:
        command_arguments = [str(native_binary)]
    command_arguments.extend(
        [
            "--bgen",
            str(data_directory / "1kg_chr22_full.bgen"),
            "--sample",
            str(data_directory / "1kg_chr22_full.sample"),
            "--pheno",
            str(data_directory / "pheno_cont.txt"),
            "--pheno-name",
            "phenotype_continuous",
            "--covar",
            str(data_directory / "covariates.txt"),
            "--covar-names",
            "age,sex",
            "--pred",
            str(data_directory / "baselines/regenie_step1_qt_pred.list"),
            "--out",
            str(output_path),
            "--chunk-size",
            str(chunk_size),
            "--cuda-block-size",
            str(cuda_block_size),
            "--writer-threads",
            str(output_writer_thread_count),
            "--output-mode",
            "full-parquet" if finalize_parquet else "chunks-only",
            "--report-json",
            str(report_json_path),
        ],
    )
    return command_arguments


def build_child_command(
    *,
    runner: RunnerMode,
    data_directory: Path,
    output_path: Path,
    device: str,
    chunk_size: int,
    cuda_block_size: int,
    finalize_parquet: bool,
    output_writer_thread_count: int,
    native_binary: Path | None,
    report_json_path: Path,
) -> list[str]:
    """Build the command for one isolated trial."""
    if runner == RunnerMode.PYTHON_JAX:
        return build_python_jax_child_command(
            data_directory=data_directory,
            output_path=output_path,
            device=device,
            chunk_size=chunk_size,
            finalize_parquet=finalize_parquet,
            output_writer_thread_count=output_writer_thread_count,
        )
    return build_native_cuda_child_command(
        data_directory=data_directory,
        output_path=output_path,
        chunk_size=chunk_size,
        cuda_block_size=cuda_block_size,
        finalize_parquet=finalize_parquet,
        output_writer_thread_count=output_writer_thread_count,
        native_binary=native_binary,
        report_json_path=report_json_path,
    )


def read_native_cuda_result_payload(report_json_path: Path) -> dict[str, typing.Any]:
    """Read native CUDA report JSON and normalize it to the trial payload shape."""
    report_payload = json.loads(report_json_path.read_text(encoding="utf-8"))
    output_run_directory = Path(str(report_payload["output_run_directory"]))
    final_parquet = report_payload.get("final_parquet")
    final_parquet_path = Path(str(final_parquet)) if final_parquet is not None else None
    artifact_path = final_parquet_path or output_run_directory
    chunk_file_paths = list((output_run_directory / "chunks").glob("chunk_*.arrow"))
    chunk_bytes = sum(chunk_file_path.stat().st_size for chunk_file_path in chunk_file_paths)
    final_parquet_bytes = final_parquet_path.stat().st_size if final_parquet_path is not None else None
    return {
        "wall_time_seconds": float(report_payload["total_wall_seconds"]),
        "output_path": str(artifact_path),
        "output_row_count": int(report_payload["processed_variant_count"]),
        "chunk_file_count": len(chunk_file_paths),
        "chunk_bytes": chunk_bytes,
        "final_parquet_bytes": final_parquet_bytes,
    }


def run_fresh_process_trial(
    *,
    runner: RunnerMode,
    trial_index: int,
    data_directory: Path,
    output_directory: Path,
    device: str,
    chunk_size: int,
    cuda_block_size: int,
    finalize_parquet: bool,
    output_writer_thread_count: int,
    native_binary: Path | None,
) -> TrialResult:
    """Run one isolated fresh-process trial."""
    output_name_parts = [
        f"{runner.value}_{device}_finalize{int(finalize_parquet)}",
        f"chunk{chunk_size}",
        f"writer{output_writer_thread_count}",
    ]
    if runner == RunnerMode.NATIVE_CUDA_KERNEL:
        output_name_parts.append(f"cuda{cuda_block_size}")
    output_name_parts.append(f"trial{trial_index:02d}")
    output_prefix = output_directory / "_".join(output_name_parts)
    native_report_path = output_prefix.with_suffix(".native_report.json")
    command_arguments = build_child_command(
        runner=runner,
        data_directory=data_directory,
        output_path=output_prefix,
        device=device,
        chunk_size=chunk_size,
        cuda_block_size=cuda_block_size,
        finalize_parquet=finalize_parquet,
        output_writer_thread_count=output_writer_thread_count,
        native_binary=native_binary,
        report_json_path=native_report_path,
    )
    child_environment = os.environ.copy()
    if runner == RunnerMode.PYTHON_JAX:
        child_environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        child_environment.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".50")
    completed_process = subprocess.run(
        command_arguments,
        check=True,
        capture_output=True,
        text=True,
        env=child_environment,
    )
    if runner == RunnerMode.NATIVE_CUDA_KERNEL:
        result_payload = read_native_cuda_result_payload(native_report_path)
    else:
        result_line = completed_process.stdout.strip().splitlines()[-1]
        result_payload = json.loads(result_line)
    return TrialResult(
        trial_index=trial_index,
        wall_time_seconds=float(result_payload["wall_time_seconds"]),
        output_path=str(result_payload["output_path"]),
        output_row_count=int(result_payload["output_row_count"]),
        chunk_file_count=int(result_payload["chunk_file_count"]),
        chunk_bytes=int(result_payload["chunk_bytes"]),
        final_parquet_bytes=(
            int(result_payload["final_parquet_bytes"]) if result_payload["final_parquet_bytes"] is not None else None
        ),
    )


def build_summary(
    *,
    runner: RunnerMode,
    device: str,
    chunk_size: int,
    cuda_block_size: int,
    finalize_parquet: bool,
    output_writer_thread_count: int,
    warmup_count: int,
    trial_results: list[TrialResult],
) -> BenchmarkSummary:
    """Build an aggregate summary from measured trials."""
    wall_time_values = [trial_result.wall_time_seconds for trial_result in trial_results]
    row_rate_values = [trial_result.output_row_count / trial_result.wall_time_seconds for trial_result in trial_results]
    final_parquet_byte_values = [
        trial_result.final_parquet_bytes
        for trial_result in trial_results
        if trial_result.final_parquet_bytes is not None
    ]
    return BenchmarkSummary(
        runner=runner.value,
        device=device,
        chunk_size=chunk_size,
        finalize_parquet=finalize_parquet,
        output_writer_thread_count=output_writer_thread_count,
        cuda_block_size=(cuda_block_size if runner == RunnerMode.NATIVE_CUDA_KERNEL else None),
        trial_count=len(trial_results),
        warmup_count=warmup_count,
        mean_wall_time_seconds=statistics.fmean(wall_time_values),
        median_wall_time_seconds=statistics.median(wall_time_values),
        min_wall_time_seconds=min(wall_time_values),
        max_wall_time_seconds=max(wall_time_values),
        mean_rows_per_second=statistics.fmean(row_rate_values),
        mean_chunk_file_count=statistics.fmean([trial_result.chunk_file_count for trial_result in trial_results]),
        mean_chunk_bytes=statistics.fmean([trial_result.chunk_bytes for trial_result in trial_results]),
        mean_final_parquet_bytes=(statistics.fmean(final_parquet_byte_values) if final_parquet_byte_values else None),
        trial_results=trial_results,
    )


def main() -> None:
    """Run the fresh-process benchmark."""
    argument_parser = build_argument_parser()
    arguments = argument_parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    runner = RunnerMode(arguments.runner)

    for warmup_index in range(arguments.warmup_trials):
        _ = run_fresh_process_trial(
            runner=runner,
            trial_index=-(warmup_index + 1),
            data_directory=arguments.data_dir,
            output_directory=arguments.output_dir,
            device=arguments.device,
            chunk_size=arguments.chunk_size,
            cuda_block_size=arguments.cuda_block_size,
            finalize_parquet=arguments.finalize_parquet,
            output_writer_thread_count=arguments.output_writer_thread_count,
            native_binary=arguments.native_binary,
        )

    measured_trial_results = [
        run_fresh_process_trial(
            runner=runner,
            trial_index=trial_index,
            data_directory=arguments.data_dir,
            output_directory=arguments.output_dir,
            device=arguments.device,
            chunk_size=arguments.chunk_size,
            cuda_block_size=arguments.cuda_block_size,
            finalize_parquet=arguments.finalize_parquet,
            output_writer_thread_count=arguments.output_writer_thread_count,
            native_binary=arguments.native_binary,
        )
        for trial_index in range(arguments.trials)
    ]

    benchmark_summary = build_summary(
        runner=runner,
        device=arguments.device,
        chunk_size=arguments.chunk_size,
        cuda_block_size=arguments.cuda_block_size,
        finalize_parquet=arguments.finalize_parquet,
        output_writer_thread_count=arguments.output_writer_thread_count,
        warmup_count=arguments.warmup_trials,
        trial_results=measured_trial_results,
    )
    summary_name_parts = [
        f"{runner.value}_{arguments.device}_finalize{int(arguments.finalize_parquet)}",
        f"chunk{arguments.chunk_size}",
        f"writer{arguments.output_writer_thread_count}",
    ]
    if runner == RunnerMode.NATIVE_CUDA_KERNEL:
        summary_name_parts.append(f"cuda{arguments.cuda_block_size}")
    default_summary_filename = "_".join(summary_name_parts) + ".json"
    json_summary_path = arguments.json_summary_path or (arguments.output_dir / default_summary_filename)
    json_summary_path.write_text(json.dumps(asdict(benchmark_summary), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(asdict(benchmark_summary), indent=2))


if __name__ == "__main__":
    main()
