#!/usr/bin/env python3
"""Benchmark output-stage timings across finalization, phenotype count, and bsize."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import statistics
import time
import typing
from pathlib import Path

import polars as pl

from g import api, types

DEFAULT_DATA_DIRECTORY = Path("data")
DEFAULT_OUTPUT_DIRECTORY = Path("data/benchmarks/output_stages")
DEFAULT_SINGLE_PHENOTYPE_NAME = "phenotype_continuous"
OUTPUT_STAGE_TIMING_FILE_NAME = "output_stage_timings.json"


@dataclasses.dataclass(frozen=True)
class PhenotypeResources:
    """Phenotype and prediction files for one benchmark phenotype mode.

    Attributes:
        phenotype_path: Phenotype file containing all requested traits.
        phenotype_names: Trait names to pass to the REGENIE API.
        prediction_list_path: Prediction list containing all requested traits.

    """

    phenotype_path: Path
    phenotype_names: tuple[str, ...]
    prediction_list_path: Path


@dataclasses.dataclass(frozen=True)
class BenchmarkCase:
    """One output-stage benchmark case.

    Attributes:
        name: Stable case name for output paths.
        finalize_parquet: Whether to write final Parquet output.
        phenotype_count: Number of phenotypes to write.
        chunk_size: REGENIE bsize value.

    """

    name: str
    finalize_parquet: bool
    phenotype_count: int
    chunk_size: int


@dataclasses.dataclass(frozen=True)
class TrialResult:
    """Measured output-stage benchmark result.

    Attributes:
        case_name: Benchmark case name.
        trial_index: Measured trial index.
        finalize_parquet: Whether final Parquet output was enabled.
        phenotype_count: Number of phenotypes written.
        chunk_size: REGENIE bsize value.
        wall_time_seconds: End-to-end API wall time.
        python_stage_timing_path: Python stage timing JSON path.
        rust_stage_timing_paths: Rust output timing JSON paths.
        output_run_directories: Output run directories.
        final_parquet_paths: Final Parquet files, when written.
        chunk_file_count: Total Arrow chunk file count across phenotypes.
        chunk_bytes: Total Arrow chunk bytes across phenotypes.
        final_parquet_bytes: Total final Parquet bytes across phenotypes, when written.

    """

    case_name: str
    trial_index: int
    finalize_parquet: bool
    phenotype_count: int
    chunk_size: int
    wall_time_seconds: float
    python_stage_timing_path: str
    rust_stage_timing_paths: tuple[str, ...]
    output_run_directories: tuple[str, ...]
    final_parquet_paths: tuple[str, ...]
    chunk_file_count: int
    chunk_bytes: int
    final_parquet_bytes: int | None


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Benchmark g output-stage timings.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIRECTORY, help="Input data directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY, help="Benchmark output directory.")
    parser.add_argument("--device", default=types.Device.GPU.value, choices=[device.value for device in types.Device])
    parser.add_argument("--small-bsize", type=int, default=1024, help="Small REGENIE bsize to benchmark.")
    parser.add_argument("--large-bsize", type=int, default=8192, help="Large REGENIE bsize to benchmark.")
    parser.add_argument("--many-phenotype-count", type=int, default=8, help="Trait count for many-phenotype runs.")
    parser.add_argument("--variant-limit", type=int, help="Optional variant cap for smoke runs.")
    parser.add_argument("--trials", type=int, default=1, help="Measured trial count per case.")
    parser.add_argument("--writer-thread-count", type=int, default=8, help="Background writer threads.")
    parser.add_argument("--writer-queue-depth", type=int, default=8, help="Background writer queue depth.")
    parser.add_argument("--json-summary-path", type=Path, help="Optional explicit summary JSON path.")
    return parser


def build_benchmark_cases(
    *,
    small_chunk_size: int,
    large_chunk_size: int,
    many_phenotype_count: int,
) -> tuple[BenchmarkCase, ...]:
    """Build the output benchmark matrix."""
    cases: list[BenchmarkCase] = []
    for finalize_parquet in (False, True):
        output_mode = "parquet_final" if finalize_parquet else "arrow_chunks"
        for phenotype_count in (1, many_phenotype_count):
            phenotype_mode = "single_phenotype" if phenotype_count == 1 else f"{phenotype_count}_phenotypes"
            for chunk_size_name, chunk_size in (("small_bsize", small_chunk_size), ("large_bsize", large_chunk_size)):
                cases.append(
                    BenchmarkCase(
                        name=f"{output_mode}_{phenotype_mode}_{chunk_size_name}_{chunk_size}",
                        finalize_parquet=finalize_parquet,
                        phenotype_count=phenotype_count,
                        chunk_size=chunk_size,
                    )
                )
    return tuple(cases)


def parse_prediction_list_first_entry(prediction_list_path: Path) -> Path:
    """Return the first LOCO prediction file path from a REGENIE prediction list."""
    first_line = prediction_list_path.read_text(encoding="utf-8").splitlines()[0]
    fields = first_line.split()
    if len(fields) < 2:
        message = f"Prediction list {prediction_list_path} does not contain a trait and path."
        raise ValueError(message)
    return Path(fields[1])


def prepare_phenotype_resources(
    *,
    data_directory: Path,
    output_directory: Path,
    phenotype_count: int,
) -> PhenotypeResources:
    """Prepare single- or many-phenotype resources for one benchmark mode."""
    source_phenotype_path = data_directory / "pheno_cont.txt"
    source_prediction_list_path = data_directory / "baselines/regenie_step1_qt_pred.list"
    if phenotype_count == 1:
        return PhenotypeResources(
            phenotype_path=source_phenotype_path,
            phenotype_names=(DEFAULT_SINGLE_PHENOTYPE_NAME,),
            prediction_list_path=source_prediction_list_path,
        )

    generated_directory = output_directory / "generated_inputs"
    generated_directory.mkdir(parents=True, exist_ok=True)
    phenotype_names = tuple(f"output_trait_{phenotype_index:02d}" for phenotype_index in range(phenotype_count))
    phenotype_frame = pl.read_csv(source_phenotype_path, separator="\t")
    phenotype_frame = phenotype_frame.with_columns(
        [
            (pl.col(DEFAULT_SINGLE_PHENOTYPE_NAME) + float(phenotype_index) * 0.001).alias(phenotype_name)
            for phenotype_index, phenotype_name in enumerate(phenotype_names)
        ]
    )
    generated_phenotype_path = generated_directory / f"pheno_cont_{phenotype_count}_traits.tsv"
    phenotype_frame.select(["FID", "IID", *phenotype_names]).write_csv(
        generated_phenotype_path,
        separator="\t",
    )

    loco_prediction_path = parse_prediction_list_first_entry(source_prediction_list_path)
    generated_prediction_list_path = generated_directory / f"regenie_step1_qt_{phenotype_count}_traits_pred.list"
    generated_prediction_list_path.write_text(
        "".join(f"{phenotype_name} {loco_prediction_path}\n" for phenotype_name in phenotype_names),
        encoding="utf-8",
    )
    return PhenotypeResources(
        phenotype_path=generated_phenotype_path,
        phenotype_names=phenotype_names,
        prediction_list_path=generated_prediction_list_path,
    )


def flatten_artifacts(artifacts: api.RunArtifacts) -> tuple[api.RunArtifacts, ...]:
    """Return one artifact object per phenotype."""
    if artifacts.phenotype_artifacts:
        return artifacts.phenotype_artifacts
    return (artifacts,)


def collect_trial_output_metrics(artifacts: tuple[api.RunArtifacts, ...]) -> dict[str, typing.Any]:
    """Collect output artifact metrics across all phenotype runs."""
    output_run_directories = tuple(
        artifact.output_run_directory for artifact in artifacts if artifact.output_run_directory is not None
    )
    final_parquet_paths = tuple(artifact.final_parquet for artifact in artifacts if artifact.final_parquet is not None)
    chunk_file_paths = tuple(
        chunk_file_path
        for output_run_directory in output_run_directories
        for chunk_file_path in sorted((output_run_directory / "chunks").glob("chunk_*.arrow"))
    )
    rust_stage_timing_paths = tuple(
        output_run_directory / OUTPUT_STAGE_TIMING_FILE_NAME
        for output_run_directory in output_run_directories
        if (output_run_directory / OUTPUT_STAGE_TIMING_FILE_NAME).exists()
    )
    chunk_bytes = sum(chunk_file_path.stat().st_size for chunk_file_path in chunk_file_paths)
    final_parquet_byte_values = [final_parquet_path.stat().st_size for final_parquet_path in final_parquet_paths]
    return {
        "rust_stage_timing_paths": tuple(str(path) for path in rust_stage_timing_paths),
        "output_run_directories": tuple(str(path) for path in output_run_directories),
        "final_parquet_paths": tuple(str(path) for path in final_parquet_paths),
        "chunk_file_count": len(chunk_file_paths),
        "chunk_bytes": chunk_bytes,
        "final_parquet_bytes": sum(final_parquet_byte_values) if final_parquet_byte_values else None,
    }


def run_trial(
    *,
    data_directory: Path,
    output_directory: Path,
    device: types.Device,
    writer_thread_count: int,
    writer_queue_depth: int,
    variant_limit: int | None,
    benchmark_case: BenchmarkCase,
    trial_index: int,
) -> TrialResult:
    """Run one output-stage benchmark trial."""
    phenotype_resources = prepare_phenotype_resources(
        data_directory=data_directory,
        output_directory=output_directory,
        phenotype_count=benchmark_case.phenotype_count,
    )
    trial_name = f"{benchmark_case.name}_trial{trial_index:02d}"
    output_root = output_directory / "outputs" / trial_name
    python_stage_timing_path = output_directory / "stage_timings" / f"{trial_name}.json"
    python_stage_timing_path.parent.mkdir(parents=True, exist_ok=True)
    options: dict[str, object] = {
        "step": 2,
        "qt": True,
        "bgen": data_directory / "1kg_chr22_full.bgen",
        "sample": data_directory / "1kg_chr22_full.sample",
        "phenoFile": phenotype_resources.phenotype_path,
        "out": output_root,
        "covarFile": data_directory / "covariates.txt",
        "covarColList": "age,sex",
        "pred": phenotype_resources.prediction_list_path,
        "g-device": device.value,
        "bsize": benchmark_case.chunk_size,
        "g-variant-limit": variant_limit,
        "g-output-format": "parquet" if benchmark_case.finalize_parquet else "arrow",
        "g-stage-timings-json": python_stage_timing_path,
        "g-writer-threads": writer_thread_count,
        "g-writer-queue-depth": writer_queue_depth,
    }
    if benchmark_case.phenotype_count == 1:
        options["phenoCol"] = phenotype_resources.phenotype_names[0]
    else:
        options["phenoColList"] = ",".join(phenotype_resources.phenotype_names)

    start_time = time.perf_counter()
    artifacts = api.regenie.from_options(options)
    wall_time_seconds = time.perf_counter() - start_time
    output_metrics = collect_trial_output_metrics(flatten_artifacts(artifacts))
    return TrialResult(
        case_name=benchmark_case.name,
        trial_index=trial_index,
        finalize_parquet=benchmark_case.finalize_parquet,
        phenotype_count=benchmark_case.phenotype_count,
        chunk_size=benchmark_case.chunk_size,
        wall_time_seconds=wall_time_seconds,
        python_stage_timing_path=str(python_stage_timing_path),
        rust_stage_timing_paths=typing.cast("tuple[str, ...]", output_metrics["rust_stage_timing_paths"]),
        output_run_directories=typing.cast("tuple[str, ...]", output_metrics["output_run_directories"]),
        final_parquet_paths=typing.cast("tuple[str, ...]", output_metrics["final_parquet_paths"]),
        chunk_file_count=int(output_metrics["chunk_file_count"]),
        chunk_bytes=int(output_metrics["chunk_bytes"]),
        final_parquet_bytes=typing.cast("int | None", output_metrics["final_parquet_bytes"]),
    )


def summarize_trial_group(trial_results: tuple[TrialResult, ...]) -> dict[str, typing.Any]:
    """Summarize repeated trials for one benchmark case."""
    wall_time_values = [trial_result.wall_time_seconds for trial_result in trial_results]
    first_trial = trial_results[0]
    return {
        "case_name": first_trial.case_name,
        "finalize_parquet": first_trial.finalize_parquet,
        "phenotype_count": first_trial.phenotype_count,
        "chunk_size": first_trial.chunk_size,
        "trial_count": len(trial_results),
        "mean_wall_time_seconds": statistics.fmean(wall_time_values),
        "median_wall_time_seconds": statistics.median(wall_time_values),
        "min_wall_time_seconds": min(wall_time_values),
        "max_wall_time_seconds": max(wall_time_values),
        "mean_chunk_file_count": statistics.fmean([trial_result.chunk_file_count for trial_result in trial_results]),
        "mean_chunk_bytes": statistics.fmean([trial_result.chunk_bytes for trial_result in trial_results]),
    }


def build_summary(
    *,
    device: types.Device,
    benchmark_cases: tuple[BenchmarkCase, ...],
    trial_results: tuple[TrialResult, ...],
) -> dict[str, typing.Any]:
    """Build the benchmark summary payload."""
    grouped_results = {
        benchmark_case.name: tuple(
            trial_result for trial_result in trial_results if trial_result.case_name == benchmark_case.name
        )
        for benchmark_case in benchmark_cases
    }
    return {
        "metadata": {
            "device": device.value,
            "pid": os.getpid(),
        },
        "case_summaries": [
            summarize_trial_group(case_trial_results)
            for case_trial_results in grouped_results.values()
            if case_trial_results
        ],
        "trial_results": [dataclasses.asdict(trial_result) for trial_result in trial_results],
    }


def main() -> None:
    """Run the output-stage benchmark matrix."""
    argument_parser = build_argument_parser()
    arguments = argument_parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    device = types.Device(arguments.device)
    benchmark_cases = build_benchmark_cases(
        small_chunk_size=int(arguments.small_bsize),
        large_chunk_size=int(arguments.large_bsize),
        many_phenotype_count=int(arguments.many_phenotype_count),
    )
    trial_results = tuple(
        run_trial(
            data_directory=arguments.data_dir,
            output_directory=arguments.output_dir,
            device=device,
            writer_thread_count=int(arguments.writer_thread_count),
            writer_queue_depth=int(arguments.writer_queue_depth),
            variant_limit=arguments.variant_limit,
            benchmark_case=benchmark_case,
            trial_index=trial_index,
        )
        for benchmark_case in benchmark_cases
        for trial_index in range(arguments.trials)
    )
    summary = build_summary(device=device, benchmark_cases=benchmark_cases, trial_results=trial_results)
    summary_path = arguments.json_summary_path or (arguments.output_dir / "output_stage_benchmark_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
