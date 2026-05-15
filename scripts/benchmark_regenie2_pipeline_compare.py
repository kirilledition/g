"""Compare Python and Rust REGENIE step 2 orchestration on identical inputs."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import polars as pl

from g import api, types


@dataclasses.dataclass(frozen=True)
class BenchmarkArguments:
    """Command-line configuration for one benchmark run."""

    bgen_path: Path
    sample_path: Path
    phenotype_path: Path
    phenotype_name: str
    covariate_path: Path
    covariate_names: str
    prediction_list_path: Path
    output_directory: Path
    variant_limits: tuple[int, ...]
    repeat_count: int
    chunk_size: int
    prefetch_chunks: int
    device: types.Device
    finalize_parquet: bool
    warmup: bool


@dataclasses.dataclass(frozen=True)
class TrialResult:
    """Measured output for one pipeline trial."""

    mode: str
    variant_limit: int
    repeat_index: int
    warmup: bool
    wall_time_seconds: float
    row_count: int
    final_parquet_path: Path


@dataclasses.dataclass(frozen=True)
class FrameComparison:
    """Summary of Python-versus-Rust output agreement for one variant limit."""

    variant_limit: int
    row_count: int
    schema_matches: bool
    identifier_columns_match: bool
    max_beta_delta: float
    max_standard_error_delta: float
    max_chi_squared_delta: float
    max_log10_p_value_delta: float


def parse_arguments() -> BenchmarkArguments:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bgen", required=True, type=Path)
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--pheno", required=True, type=Path)
    parser.add_argument("--pheno-name", required=True)
    parser.add_argument("--covar", required=True, type=Path)
    parser.add_argument("--covar-names", required=True)
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--variant-limit", required=True, action="append", type=int)
    parser.add_argument("--repeat-count", default=3, type=int)
    parser.add_argument("--chunk-size", default=api.DEFAULT_REGENIE2_LINEAR_CHUNK_SIZE, type=int)
    parser.add_argument("--prefetch-chunks", default=1, type=int)
    parser.add_argument("--device", default=types.Device.GPU.value, choices=[device.value for device in types.Device])
    parser.add_argument("--no-finalize-parquet", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    namespace = parser.parse_args()
    return BenchmarkArguments(
        bgen_path=namespace.bgen,
        sample_path=namespace.sample,
        phenotype_path=namespace.pheno,
        phenotype_name=namespace.pheno_name,
        covariate_path=namespace.covar,
        covariate_names=namespace.covar_names,
        prediction_list_path=namespace.pred,
        output_directory=namespace.output_dir,
        variant_limits=tuple(namespace.variant_limit),
        repeat_count=namespace.repeat_count,
        chunk_size=namespace.chunk_size,
        prefetch_chunks=namespace.prefetch_chunks,
        device=types.Device(namespace.device),
        finalize_parquet=not bool(namespace.no_finalize_parquet),
        warmup=not bool(namespace.no_warmup),
    )


def build_output_prefix(arguments: BenchmarkArguments, mode: str, variant_limit: int, repeat_index: int) -> Path:
    """Build a unique output prefix for one benchmark trial."""
    timestamp = time.time_ns()
    return arguments.output_directory / f"{mode}_limit{variant_limit}_repeat{repeat_index}_{timestamp}"


def run_trial(
    arguments: BenchmarkArguments,
    mode: str,
    variant_limit: int,
    repeat_index: int,
    *,
    warmup: bool,
) -> TrialResult:
    """Run one REGENIE step 2 trial and return timing details."""
    if mode == "rust":
        os.environ[api.RUST_PIPELINE_ENVIRONMENT_VARIABLE] = "1"
    else:
        os.environ.pop(api.RUST_PIPELINE_ENVIRONMENT_VARIABLE, None)

    output_prefix = build_output_prefix(arguments, mode, variant_limit, repeat_index)
    start_time = time.perf_counter()
    artifacts = api.regenie2(
        bgen=arguments.bgen_path,
        sample=arguments.sample_path,
        pheno=arguments.phenotype_path,
        pheno_name=arguments.phenotype_name,
        out=output_prefix,
        covar=arguments.covariate_path,
        covar_names=arguments.covariate_names,
        pred=arguments.prediction_list_path,
        trait_type=types.RegenieTraitType.BINARY,
        compute=api.ComputeConfig(
            chunk_size=arguments.chunk_size,
            device=arguments.device,
            variant_limit=variant_limit,
            prefetch_chunks=arguments.prefetch_chunks,
            finalize_parquet=arguments.finalize_parquet,
        ),
        binary=api.Regenie2BinaryConfig(),
    )
    wall_time_seconds = time.perf_counter() - start_time
    if artifacts.final_parquet is None:
        message = "Benchmark requires finalized Parquet output."
        raise RuntimeError(message)
    row_count = pl.read_parquet(artifacts.final_parquet).height
    return TrialResult(
        mode=mode,
        variant_limit=variant_limit,
        repeat_index=repeat_index,
        warmup=warmup,
        wall_time_seconds=wall_time_seconds,
        row_count=int(row_count),
        final_parquet_path=artifacts.final_parquet,
    )


def calculate_max_delta(left_frame: pl.DataFrame, right_frame: pl.DataFrame, column_name: str) -> float:
    """Calculate the maximum absolute numeric delta for one output column."""
    left_values = left_frame.get_column(column_name).to_numpy()
    right_values = right_frame.get_column(column_name).to_numpy()
    return float(np.nanmax(np.abs(left_values - right_values)))


def compare_frames(variant_limit: int, python_path: Path, rust_path: Path) -> FrameComparison:
    """Compare finalized Python and Rust pipeline outputs."""
    python_frame = pl.read_parquet(python_path)
    rust_frame = pl.read_parquet(rust_path)
    identifier_columns = ("CHROM", "GENPOS", "ID", "ALLELE0", "ALLELE1", "TEST", "EXTRA")
    identifier_columns_match = all(
        python_frame.get_column(column_name).to_list() == rust_frame.get_column(column_name).to_list()
        for column_name in identifier_columns
    )
    return FrameComparison(
        variant_limit=variant_limit,
        row_count=rust_frame.height,
        schema_matches=python_frame.schema == rust_frame.schema,
        identifier_columns_match=identifier_columns_match,
        max_beta_delta=calculate_max_delta(python_frame, rust_frame, "BETA"),
        max_standard_error_delta=calculate_max_delta(python_frame, rust_frame, "SE"),
        max_chi_squared_delta=calculate_max_delta(python_frame, rust_frame, "CHISQ"),
        max_log10_p_value_delta=calculate_max_delta(python_frame, rust_frame, "LOG10P"),
    )


def print_json_record(record: TrialResult | FrameComparison) -> None:
    """Print one dataclass record as compact JSON."""
    print(json.dumps(dataclasses.asdict(record), default=str, sort_keys=True), flush=True)


def summarize_trials(results: list[TrialResult], variant_limit: int) -> None:
    """Print per-mode timing summaries for one variant limit."""
    measured_results = [
        result for result in results if result.variant_limit == variant_limit and not result.warmup
    ]
    for mode in ("python", "rust"):
        mode_times = [
            result.wall_time_seconds
            for result in measured_results
            if result.mode == mode
        ]
        print(
            json.dumps(
                {
                    "mode": mode,
                    "variant_limit": variant_limit,
                    "repeat_count": len(mode_times),
                    "median_seconds": statistics.median(mode_times),
                    "mean_seconds": statistics.fmean(mode_times),
                    "min_seconds": min(mode_times),
                    "max_seconds": max(mode_times),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    python_median = statistics.median(
        result.wall_time_seconds
        for result in measured_results
        if result.mode == "python"
    )
    rust_median = statistics.median(
        result.wall_time_seconds
        for result in measured_results
        if result.mode == "rust"
    )
    print(
        json.dumps(
            {
                "variant_limit": variant_limit,
                "rust_vs_python_median_ratio": rust_median / python_median,
                "rust_speedup": python_median / rust_median,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    """Run the benchmark."""
    arguments = parse_arguments()
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    results: list[TrialResult] = []

    for variant_limit in arguments.variant_limits:
        if arguments.warmup:
            for mode in ("python", "rust"):
                result = run_trial(arguments, mode, variant_limit, -1, warmup=True)
                results.append(result)
                print_json_record(result)
        for repeat_index in range(arguments.repeat_count):
            for mode in ("python", "rust"):
                result = run_trial(arguments, mode, variant_limit, repeat_index, warmup=False)
                results.append(result)
                print_json_record(result)
        summarize_trials(results, variant_limit)
        latest_python_result = next(
            result
            for result in reversed(results)
            if result.variant_limit == variant_limit and result.mode == "python" and not result.warmup
        )
        latest_rust_result = next(
            result
            for result in reversed(results)
            if result.variant_limit == variant_limit and result.mode == "rust" and not result.warmup
        )
        print_json_record(
            compare_frames(
                variant_limit,
                latest_python_result.final_parquet_path,
                latest_rust_result.final_parquet_path,
            )
        )


if __name__ == "__main__":
    main()
