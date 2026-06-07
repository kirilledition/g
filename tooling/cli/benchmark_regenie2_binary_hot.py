#!/usr/bin/env python3
"""Benchmark full chr22 binary REGENIE step 2 with comparable hot/cold modes."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import enum
import json
import os
import subprocess
import sys
import textwrap
import time
import typing
from pathlib import Path

import polars as pl

from g import api, types
from g.interface import config
from tooling.common import paths as tooling_paths
from tooling.common import reports as tooling_reports
from tooling.common import sweeps as tooling_sweeps

DEFAULT_DATA_DIRECTORY = tooling_paths.configured_data_directory()
DEFAULT_OUTPUT_PARENT = Path("data/profiles")
DEFAULT_VARIANT_COUNT = 418_943
REPOSITORY_ROOT = tooling_paths.find_repository_root(Path(__file__))
GPU_JAX_CACHE_PARENT_DEFAULT = "/tmp/g-jax-binary-hot-cache"
DEFAULT_BGEN_FILE = Path("1kg_chr22_full.bgen")
DEFAULT_SAMPLE_FILE = Path("1kg_chr22_full.sample")
DEFAULT_PHENOTYPE_FILE = Path("pheno_bin.txt")
DEFAULT_PREDICTION_LIST = Path("baselines/regenie_step1_pred.list")
DEFAULT_PHENOTYPE_COLUMNS = ("phenotype_binary",)
DEFAULT_LOW_FALLBACK_P_THRESHOLD = 1.0e-8
DEFAULT_HIGH_FALLBACK_P_THRESHOLD = 0.999999
ENABLE_XLA_AUTOTUNE_CACHE = os.environ.get("G_PROFILE_ENABLE_XLA_AUTOTUNE_CACHE") == "1"


class BenchmarkMode(enum.StrEnum):
    """Execution mode measured by the benchmark harness."""

    COLD_PROCESS_FINALIZED = "cold_process_finalized"
    WARM_SAME_PROCESS_NO_FINAL = "warm_same_process_no_final"
    HOT_SAME_PROCESS_NO_FINAL = "hot_same_process_no_final"
    WARM_SAME_PROCESS_FINALIZED = "warm_same_process_finalized"
    HOT_SAME_PROCESS_FINALIZED = "hot_same_process_finalized"


class BenchmarkStorageMode(enum.StrEnum):
    """Genotype storage mode measured by the benchmark harness."""

    VARIANT_MAJOR = "variant_major"
    PACKED8 = "packed8"


class FallbackDensityScenario(enum.StrEnum):
    """Approximate-Firth fallback density scenario."""

    DEFAULT = "default"
    LOW = "low"
    HIGH = "high"


class StageTimingMode(enum.StrEnum):
    """Stage timing collection mode for benchmark trials."""

    EXACT = "exact"
    OFF = "off"


@dataclasses.dataclass(frozen=True)
class BenchmarkConfiguration:
    """Shared configuration for a binary REGENIE benchmark run."""

    data_directory: Path
    bgen_path: Path
    sample_path: Path
    phenotype_file: Path
    prediction_list: Path
    output_directory: Path
    device: types.Device
    chunk_size: int
    staging_depth: int
    output_writer_thread_count: int
    output_writer_queue_depth: int
    trusted_no_missing_diploid: bool
    assume_trusted_validated: bool
    phenotype_columns: tuple[str, ...]
    binary_trait_counts: tuple[int, ...]
    firth_batch_sizes: tuple[int, ...]
    firth_candidate_capacities: tuple[int, ...]
    storage_modes: tuple[BenchmarkStorageMode, ...]
    fallback_density_scenarios: tuple[FallbackDensityScenario, ...]
    default_fallback_p_threshold: float
    low_fallback_p_threshold: float
    high_fallback_p_threshold: float
    variant_limit: int | None
    expected_variant_count: int | None
    stage_timing_mode: StageTimingMode
    python_executable: str
    jax_cache_directory: Path


@dataclasses.dataclass(frozen=True)
class BenchmarkCase:
    """One workload configuration in the benchmark matrix."""

    name: str
    phenotype_columns: tuple[str, ...]
    binary_trait_count: int
    firth_batch_size: int
    firth_candidate_capacity: int
    storage_mode: BenchmarkStorageMode
    gpu_genotype_format: types.GpuGenotypeFormat
    fallback_density: FallbackDensityScenario
    firth_p_threshold: float


@dataclasses.dataclass(frozen=True)
class TrialSpec:
    """One benchmark trial to execute."""

    name: str
    mode: BenchmarkMode
    finalize_parquet: bool
    fresh_process: bool
    same_process_group: str | None


@dataclasses.dataclass(frozen=True)
class ChildProcessCommand:
    """Child Python process command and environment overrides."""

    command_arguments: list[str]
    environment_overrides: dict[str, str]


@dataclasses.dataclass(frozen=True)
class OutputMetrics:
    """Output artifact metrics from one trial."""

    output_run_directory: str | None
    final_parquet: str | None
    output_row_count: int | None
    info_non_null_count: int | None
    chunk_file_count: int
    chunk_bytes: int
    final_parquet_bytes: int | None


@dataclasses.dataclass(frozen=True)
class TrialResult:
    """Measured result for one trial."""

    name: str
    benchmark_case: BenchmarkCase
    mode: BenchmarkMode
    fresh_process: bool
    finalize_parquet: bool
    same_process_group: str | None
    wall_time_seconds: float
    stage_timing_path: str | None
    output_metrics: OutputMetrics


def split_comma_separated_values(raw_value: str, option_name: str) -> list[str]:
    """Split a comma-separated CLI value into non-empty entries."""
    return tooling_sweeps.split_comma_separated_values(raw_value, option_name)


def parse_positive_integer_list(raw_value: str, option_name: str) -> tuple[int, ...]:
    """Parse a comma-separated list of positive integers."""
    return tooling_sweeps.parse_positive_integer_list(raw_value, option_name)


def parse_single_or_sweep_positive_integers(
    *,
    single_value: int | None,
    sweep_value: str | None,
    default_value: int,
    single_option_name: str,
    sweep_option_name: str,
) -> tuple[int, ...]:
    """Parse a single-value override or comma-separated positive integer sweep."""
    if single_value is not None and sweep_value is not None:
        message = f"Use either {single_option_name} or {sweep_option_name}, not both."
        raise ValueError(message)
    if sweep_value is not None:
        return parse_positive_integer_list(sweep_value, sweep_option_name)
    if single_value is not None:
        if single_value <= 0:
            message = f"{single_option_name} must be positive."
            raise ValueError(message)
        return (single_value,)
    return (default_value,)


def parse_phenotype_columns(raw_value: str) -> tuple[str, ...]:
    """Parse phenotype column names for benchmark cases."""
    return tuple(split_comma_separated_values(raw_value, "--phenotype-columns"))


def parse_storage_modes(raw_value: str) -> tuple[BenchmarkStorageMode, ...]:
    """Parse benchmark storage modes."""
    storage_modes: list[BenchmarkStorageMode] = []
    for value in split_comma_separated_values(raw_value, "--storage-modes"):
        normalized_value = value.strip().lower().replace("-", "_")
        if normalized_value in {BenchmarkStorageMode.VARIANT_MAJOR.value, types.GpuGenotypeFormat.DOSAGE.value}:
            storage_modes.append(BenchmarkStorageMode.VARIANT_MAJOR)
        elif normalized_value == BenchmarkStorageMode.PACKED8.value:
            storage_modes.append(BenchmarkStorageMode.PACKED8)
        else:
            accepted_values = ", ".join(
                [
                    BenchmarkStorageMode.VARIANT_MAJOR.value,
                    types.GpuGenotypeFormat.DOSAGE.value,
                    BenchmarkStorageMode.PACKED8.value,
                ]
            )
            message = f"--storage-modes values must be one of: {accepted_values}."
            raise ValueError(message)
    return tuple(storage_modes)


def parse_fallback_density_scenarios(raw_value: str) -> tuple[FallbackDensityScenario, ...]:
    """Parse approximate-Firth fallback density scenarios."""
    scenarios: list[FallbackDensityScenario] = []
    for value in split_comma_separated_values(raw_value, "--fallback-density-scenarios"):
        try:
            scenarios.append(FallbackDensityScenario(value.strip().lower().replace("-", "_")))
        except ValueError as error:
            accepted_values = ", ".join(scenario.value for scenario in FallbackDensityScenario)
            message = f"--fallback-density-scenarios values must be one of: {accepted_values}."
            raise ValueError(message) from error
    return tuple(scenarios)


def validate_firth_p_threshold(option_name: str, value: float) -> None:
    """Validate a Firth fallback p-value threshold."""
    if not (0.0 < value < 1.0):
        message = f"{option_name} must be in (0, 1)."
        raise ValueError(message)


def resolve_data_path(data_directory: Path, path: Path) -> Path:
    """Resolve a benchmark input path relative to the data directory when needed."""
    return tooling_paths.resolve_data_path(data_directory, path)


def gpu_genotype_format_for_storage_mode(storage_mode: BenchmarkStorageMode) -> types.GpuGenotypeFormat:
    """Map benchmark storage mode onto the g runtime option."""
    if storage_mode == BenchmarkStorageMode.PACKED8:
        return types.GpuGenotypeFormat.PACKED8
    return types.GpuGenotypeFormat.DOSAGE


def firth_p_threshold_for_scenario(
    configuration: BenchmarkConfiguration,
    scenario: FallbackDensityScenario,
) -> float:
    """Return the p-value threshold for a fallback-density scenario."""
    if scenario == FallbackDensityScenario.LOW:
        return configuration.low_fallback_p_threshold
    if scenario == FallbackDensityScenario.HIGH:
        return configuration.high_fallback_p_threshold
    return configuration.default_fallback_p_threshold


def build_case_name(
    *,
    binary_trait_count: int,
    storage_mode: BenchmarkStorageMode,
    fallback_density: FallbackDensityScenario,
    firth_batch_size: int,
    firth_candidate_capacity: int,
) -> str:
    """Build a stable benchmark case name."""
    return (
        f"traits{binary_trait_count}_{storage_mode.value}_{fallback_density.value}"
        f"_batch{firth_batch_size}_capacity{firth_candidate_capacity}"
    )


def build_benchmark_cases(configuration: BenchmarkConfiguration) -> list[BenchmarkCase]:
    """Expand benchmark sweep settings into concrete cases."""
    benchmark_cases: list[BenchmarkCase] = []
    for binary_trait_count in configuration.binary_trait_counts:
        if binary_trait_count > len(configuration.phenotype_columns):
            message = (
                f"Requested {binary_trait_count} binary traits, but only "
                f"{len(configuration.phenotype_columns)} phenotype columns were provided."
            )
            raise ValueError(message)
        phenotype_columns = configuration.phenotype_columns[:binary_trait_count]
        for storage_mode in configuration.storage_modes:
            for fallback_density in configuration.fallback_density_scenarios:
                for firth_batch_size in configuration.firth_batch_sizes:
                    for firth_candidate_capacity in configuration.firth_candidate_capacities:
                        benchmark_cases.append(
                            BenchmarkCase(
                                name=build_case_name(
                                    binary_trait_count=binary_trait_count,
                                    storage_mode=storage_mode,
                                    fallback_density=fallback_density,
                                    firth_batch_size=firth_batch_size,
                                    firth_candidate_capacity=firth_candidate_capacity,
                                ),
                                phenotype_columns=phenotype_columns,
                                binary_trait_count=binary_trait_count,
                                firth_batch_size=firth_batch_size,
                                firth_candidate_capacity=firth_candidate_capacity,
                                storage_mode=storage_mode,
                                gpu_genotype_format=gpu_genotype_format_for_storage_mode(storage_mode),
                                fallback_density=fallback_density,
                                firth_p_threshold=firth_p_threshold_for_scenario(configuration, fallback_density),
                            )
                        )
    return benchmark_cases


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    packaged_configuration = config.load_packaged_config()
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark binary REGENIE step 2 while separating cold process, same-process hot, "
            "chunk-only, and finalized-Parquet timings."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIRECTORY, help="Input data directory.")
    parser.add_argument(
        "--bgen",
        type=Path,
        default=DEFAULT_BGEN_FILE,
        help="BGEN file path. Relative paths resolve under --data-dir.",
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=DEFAULT_SAMPLE_FILE,
        help="Sample file path. Relative paths resolve under --data-dir.",
    )
    parser.add_argument(
        "--phenotype-file",
        type=Path,
        default=DEFAULT_PHENOTYPE_FILE,
        help="Phenotype file path. Relative paths resolve under --data-dir.",
    )
    parser.add_argument(
        "--prediction-list",
        type=Path,
        default=DEFAULT_PREDICTION_LIST,
        help="REGENIE step 1 prediction list. Relative paths resolve under --data-dir.",
    )
    parser.add_argument("--output-dir", type=Path, help="Benchmark output directory.")
    parser.add_argument("--device", default=types.Device.GPU.value, choices=[device.value for device in types.Device])
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=packaged_configuration.trait.bsize,
        help="Variants per chunk.",
    )
    parser.add_argument(
        "--staging-depth",
        "--prefetch-chunks",
        type=int,
        default=1,
        help="Native callback staging depth.",
    )
    parser.add_argument("--output-writer-thread-count", type=int, default=8, help="Background writer threads.")
    parser.add_argument("--output-writer-queue-depth", type=int, default=8, help="Background writer queue depth.")
    parser.add_argument(
        "--trusted-no-missing-diploid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the trusted no-missing diploid BGEN decode path.",
    )
    parser.add_argument(
        "--assume-trusted-validated",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip repeated trusted-path validation when the input has already been checked.",
    )
    parser.add_argument(
        "--phenotype-columns",
        default=",".join(DEFAULT_PHENOTYPE_COLUMNS),
        help="Comma-separated binary phenotype columns available for trait-count sweeps.",
    )
    parser.add_argument(
        "--binary-trait-counts",
        help=(
            "Comma-separated binary trait counts. Each case uses the first N --phenotype-columns. "
            "Defaults to all provided phenotype columns."
        ),
    )
    parser.add_argument(
        "--firth-batch-size",
        type=int,
        help="Single binary Firth candidate batch size. Use --firth-batch-sizes for sweeps.",
    )
    parser.add_argument(
        "--firth-batch-sizes",
        help="Comma-separated binary Firth candidate batch size sweep.",
    )
    parser.add_argument(
        "--firth-candidate-capacity",
        type=int,
        help="Single Firth candidate capacity. Use --firth-candidate-capacities for sweeps.",
    )
    parser.add_argument(
        "--firth-candidate-capacities",
        help="Comma-separated Firth candidate capacity sweep.",
    )
    parser.add_argument(
        "--storage-modes",
        default=BenchmarkStorageMode.VARIANT_MAJOR.value,
        help="Comma-separated storage modes: variant_major, dosage, packed8.",
    )
    parser.add_argument(
        "--fallback-density-scenarios",
        default=FallbackDensityScenario.DEFAULT.value,
        help="Comma-separated fallback-density scenarios: default, low, high.",
    )
    parser.add_argument(
        "--default-fallback-p-threshold",
        type=float,
        default=packaged_configuration.binary.p_threshold,
        help="Approximate-Firth pThresh for the default fallback-density scenario.",
    )
    parser.add_argument(
        "--low-fallback-p-threshold",
        type=float,
        default=DEFAULT_LOW_FALLBACK_P_THRESHOLD,
        help="Approximate-Firth pThresh for the low fallback-density scenario.",
    )
    parser.add_argument(
        "--high-fallback-p-threshold",
        type=float,
        default=DEFAULT_HIGH_FALLBACK_P_THRESHOLD,
        help="Approximate-Firth pThresh for the high fallback-density scenario.",
    )
    parser.add_argument("--variant-limit", type=int, help="Optional variant cap for smoke runs.")
    parser.add_argument(
        "--expected-variant-count",
        type=int,
        default=DEFAULT_VARIANT_COUNT,
        help="Expected full input variant count recorded in benchmark metadata.",
    )
    parser.add_argument(
        "--stage-timing-mode",
        choices=[stage_timing_mode.value for stage_timing_mode in StageTimingMode],
        default=StageTimingMode.EXACT.value,
        help="Use exact synchronized stage timings, or disable them for production-throughput timing.",
    )
    parser.add_argument(
        "--include-cold-process",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include one fresh Python process finalized trial.",
    )
    parser.add_argument(
        "--include-finalized-hot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include same-process warm/hot trials with Parquet finalization.",
    )
    parser.add_argument(
        "--include-no-final-hot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include same-process warm/hot trials that stop after Arrow chunk writes.",
    )
    parser.add_argument("--python-executable", default=sys.executable, help="Python executable for fresh trials.")
    parser.add_argument("--jax-cache-dir", type=Path, help="Explicit JAX compilation cache directory.")
    parser.add_argument("--json-summary-path", type=Path, help="Optional explicit summary JSON path.")
    return parser


def default_output_directory() -> Path:
    """Build a timestamped default output directory."""
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return DEFAULT_OUTPUT_PARENT / f"regenie2_binary_hot_{timestamp}"


def build_configuration(arguments: argparse.Namespace) -> BenchmarkConfiguration:
    """Build a benchmark configuration from parsed CLI arguments."""
    packaged_configuration = config.load_packaged_config()
    output_directory = arguments.output_dir or default_output_directory()
    jax_cache_directory = arguments.jax_cache_dir
    if jax_cache_directory is None:
        job_identifier = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
        cache_parent = os.environ.get("G_PROFILE_GPU_JAX_CACHE_PARENT", GPU_JAX_CACHE_PARENT_DEFAULT)
        jax_cache_directory = Path(cache_parent) / job_identifier / output_directory.name
    data_directory = Path(arguments.data_dir)
    phenotype_columns = parse_phenotype_columns(str(arguments.phenotype_columns))
    binary_trait_counts = (
        parse_positive_integer_list(str(arguments.binary_trait_counts), "--binary-trait-counts")
        if arguments.binary_trait_counts is not None
        else (len(phenotype_columns),)
    )
    firth_batch_sizes = parse_single_or_sweep_positive_integers(
        single_value=arguments.firth_batch_size,
        sweep_value=arguments.firth_batch_sizes,
        default_value=packaged_configuration.g_compute.firth_batch_size,
        single_option_name="--firth-batch-size",
        sweep_option_name="--firth-batch-sizes",
    )
    firth_candidate_capacities = parse_single_or_sweep_positive_integers(
        single_value=arguments.firth_candidate_capacity,
        sweep_value=arguments.firth_candidate_capacities,
        default_value=packaged_configuration.g_compute.firth_candidate_capacity,
        single_option_name="--firth-candidate-capacity",
        sweep_option_name="--firth-candidate-capacities",
    )
    storage_modes = parse_storage_modes(str(arguments.storage_modes))
    fallback_density_scenarios = parse_fallback_density_scenarios(str(arguments.fallback_density_scenarios))
    default_fallback_p_threshold = float(arguments.default_fallback_p_threshold)
    low_fallback_p_threshold = float(arguments.low_fallback_p_threshold)
    high_fallback_p_threshold = float(arguments.high_fallback_p_threshold)
    validate_firth_p_threshold("--default-fallback-p-threshold", default_fallback_p_threshold)
    validate_firth_p_threshold("--low-fallback-p-threshold", low_fallback_p_threshold)
    validate_firth_p_threshold("--high-fallback-p-threshold", high_fallback_p_threshold)
    return BenchmarkConfiguration(
        data_directory=data_directory,
        bgen_path=resolve_data_path(data_directory, Path(arguments.bgen)),
        sample_path=resolve_data_path(data_directory, Path(arguments.sample)),
        phenotype_file=resolve_data_path(data_directory, Path(arguments.phenotype_file)),
        prediction_list=resolve_data_path(data_directory, Path(arguments.prediction_list)),
        output_directory=output_directory,
        device=types.Device(arguments.device),
        chunk_size=int(arguments.chunk_size),
        staging_depth=int(arguments.staging_depth),
        output_writer_thread_count=int(arguments.output_writer_thread_count),
        output_writer_queue_depth=int(arguments.output_writer_queue_depth),
        trusted_no_missing_diploid=bool(arguments.trusted_no_missing_diploid),
        assume_trusted_validated=bool(arguments.assume_trusted_validated),
        phenotype_columns=phenotype_columns,
        binary_trait_counts=binary_trait_counts,
        firth_batch_sizes=firth_batch_sizes,
        firth_candidate_capacities=firth_candidate_capacities,
        storage_modes=storage_modes,
        fallback_density_scenarios=fallback_density_scenarios,
        default_fallback_p_threshold=default_fallback_p_threshold,
        low_fallback_p_threshold=low_fallback_p_threshold,
        high_fallback_p_threshold=high_fallback_p_threshold,
        variant_limit=arguments.variant_limit,
        expected_variant_count=(
            int(arguments.expected_variant_count) if arguments.expected_variant_count is not None else None
        ),
        stage_timing_mode=StageTimingMode(str(arguments.stage_timing_mode)),
        python_executable=str(arguments.python_executable),
        jax_cache_directory=jax_cache_directory,
    )


def build_trial_specs(
    *,
    include_cold_process: bool,
    include_no_final_hot: bool,
    include_finalized_hot: bool,
) -> list[TrialSpec]:
    """Build the requested trial sequence."""
    trial_specs: list[TrialSpec] = []
    if include_cold_process:
        trial_specs.append(
            TrialSpec(
                name="cold_process_finalized",
                mode=BenchmarkMode.COLD_PROCESS_FINALIZED,
                finalize_parquet=True,
                fresh_process=True,
                same_process_group=None,
            )
        )
    if include_no_final_hot:
        trial_specs.extend(
            [
                TrialSpec(
                    name="warm_same_process_no_final",
                    mode=BenchmarkMode.WARM_SAME_PROCESS_NO_FINAL,
                    finalize_parquet=False,
                    fresh_process=False,
                    same_process_group="no_final",
                ),
                TrialSpec(
                    name="hot_same_process_no_final",
                    mode=BenchmarkMode.HOT_SAME_PROCESS_NO_FINAL,
                    finalize_parquet=False,
                    fresh_process=False,
                    same_process_group="no_final",
                ),
            ]
        )
    if include_finalized_hot:
        trial_specs.extend(
            [
                TrialSpec(
                    name="warm_same_process_finalized",
                    mode=BenchmarkMode.WARM_SAME_PROCESS_FINALIZED,
                    finalize_parquet=True,
                    fresh_process=False,
                    same_process_group="finalized",
                ),
                TrialSpec(
                    name="hot_same_process_finalized",
                    mode=BenchmarkMode.HOT_SAME_PROCESS_FINALIZED,
                    finalize_parquet=True,
                    fresh_process=False,
                    same_process_group="finalized",
                ),
            ]
        )
    return trial_specs


def configuration_to_json_dict(configuration: BenchmarkConfiguration) -> dict[str, typing.Any]:
    """Convert configuration into a JSON-serializable dictionary."""
    return {
        "data_directory": str(configuration.data_directory),
        "bgen_path": str(configuration.bgen_path),
        "sample_path": str(configuration.sample_path),
        "phenotype_file": str(configuration.phenotype_file),
        "prediction_list": str(configuration.prediction_list),
        "output_directory": str(configuration.output_directory),
        "device": configuration.device.value,
        "chunk_size": configuration.chunk_size,
        "staging_depth": configuration.staging_depth,
        "output_writer_thread_count": configuration.output_writer_thread_count,
        "output_writer_queue_depth": configuration.output_writer_queue_depth,
        "trusted_no_missing_diploid": configuration.trusted_no_missing_diploid,
        "assume_trusted_validated": configuration.assume_trusted_validated,
        "phenotype_columns": list(configuration.phenotype_columns),
        "binary_trait_counts": list(configuration.binary_trait_counts),
        "firth_batch_sizes": list(configuration.firth_batch_sizes),
        "firth_candidate_capacities": list(configuration.firth_candidate_capacities),
        "storage_modes": [storage_mode.value for storage_mode in configuration.storage_modes],
        "fallback_density_scenarios": [scenario.value for scenario in configuration.fallback_density_scenarios],
        "default_fallback_p_threshold": configuration.default_fallback_p_threshold,
        "low_fallback_p_threshold": configuration.low_fallback_p_threshold,
        "high_fallback_p_threshold": configuration.high_fallback_p_threshold,
        "variant_limit": configuration.variant_limit,
        "expected_variant_count": configuration.expected_variant_count,
        "stage_timing_mode": configuration.stage_timing_mode.value,
        "python_executable": configuration.python_executable,
        "jax_cache_directory": str(configuration.jax_cache_directory),
    }


def configuration_from_json_dict(payload: dict[str, typing.Any]) -> BenchmarkConfiguration:
    """Build configuration from a JSON dictionary."""
    packaged_configuration = config.load_packaged_config()
    data_directory = Path(str(payload["data_directory"]))
    firth_batch_sizes = payload.get("firth_batch_sizes")
    firth_candidate_capacities = payload.get("firth_candidate_capacities")
    storage_modes = payload.get("storage_modes")
    fallback_density_scenarios = payload.get("fallback_density_scenarios")
    return BenchmarkConfiguration(
        data_directory=data_directory,
        bgen_path=(Path(str(payload["bgen_path"])) if "bgen_path" in payload else data_directory / DEFAULT_BGEN_FILE),
        sample_path=(
            Path(str(payload["sample_path"])) if "sample_path" in payload else data_directory / DEFAULT_SAMPLE_FILE
        ),
        phenotype_file=Path(str(payload.get("phenotype_file", DEFAULT_PHENOTYPE_FILE))),
        prediction_list=Path(str(payload.get("prediction_list", DEFAULT_PREDICTION_LIST))),
        output_directory=Path(str(payload["output_directory"])),
        device=types.Device(str(payload["device"])),
        chunk_size=int(payload["chunk_size"]),
        staging_depth=int(payload.get("staging_depth", payload.get("prefetch_chunks", 1))),
        output_writer_thread_count=int(payload["output_writer_thread_count"]),
        output_writer_queue_depth=int(payload["output_writer_queue_depth"]),
        trusted_no_missing_diploid=bool(payload["trusted_no_missing_diploid"]),
        assume_trusted_validated=bool(payload["assume_trusted_validated"]),
        phenotype_columns=tuple(str(value) for value in payload.get("phenotype_columns", DEFAULT_PHENOTYPE_COLUMNS)),
        binary_trait_counts=tuple(int(value) for value in payload.get("binary_trait_counts", [1])),
        firth_batch_sizes=(
            tuple(int(value) for value in firth_batch_sizes)
            if firth_batch_sizes is not None
            else (int(payload.get("firth_batch_size", packaged_configuration.g_compute.firth_batch_size)),)
        ),
        firth_candidate_capacities=(
            tuple(int(value) for value in firth_candidate_capacities)
            if firth_candidate_capacities is not None
            else (
                int(
                    payload.get(
                        "firth_candidate_capacity",
                        packaged_configuration.g_compute.firth_candidate_capacity,
                    )
                ),
            )
        ),
        storage_modes=(
            tuple(BenchmarkStorageMode(str(value)) for value in storage_modes)
            if storage_modes is not None
            else (BenchmarkStorageMode.VARIANT_MAJOR,)
        ),
        fallback_density_scenarios=(
            tuple(FallbackDensityScenario(str(value)) for value in fallback_density_scenarios)
            if fallback_density_scenarios is not None
            else (FallbackDensityScenario.DEFAULT,)
        ),
        default_fallback_p_threshold=float(
            payload.get("default_fallback_p_threshold", packaged_configuration.binary.p_threshold)
        ),
        low_fallback_p_threshold=float(payload.get("low_fallback_p_threshold", DEFAULT_LOW_FALLBACK_P_THRESHOLD)),
        high_fallback_p_threshold=float(payload.get("high_fallback_p_threshold", DEFAULT_HIGH_FALLBACK_P_THRESHOLD)),
        variant_limit=(int(payload["variant_limit"]) if payload["variant_limit"] is not None else None),
        expected_variant_count=(
            int(payload["expected_variant_count"]) if payload.get("expected_variant_count") is not None else None
        ),
        stage_timing_mode=StageTimingMode(str(payload.get("stage_timing_mode", StageTimingMode.EXACT.value))),
        python_executable=str(payload["python_executable"]),
        jax_cache_directory=Path(str(payload["jax_cache_directory"])),
    )


def benchmark_case_to_json_dict(benchmark_case: BenchmarkCase) -> dict[str, typing.Any]:
    """Convert a benchmark case into a JSON-serializable dictionary."""
    return {
        "name": benchmark_case.name,
        "phenotype_columns": list(benchmark_case.phenotype_columns),
        "binary_trait_count": benchmark_case.binary_trait_count,
        "firth_batch_size": benchmark_case.firth_batch_size,
        "firth_candidate_capacity": benchmark_case.firth_candidate_capacity,
        "storage_mode": benchmark_case.storage_mode.value,
        "gpu_genotype_format": benchmark_case.gpu_genotype_format.value,
        "fallback_density": benchmark_case.fallback_density.value,
        "firth_p_threshold": benchmark_case.firth_p_threshold,
    }


def benchmark_case_from_json_dict(payload: dict[str, typing.Any]) -> BenchmarkCase:
    """Build a benchmark case from a JSON dictionary."""
    return BenchmarkCase(
        name=str(payload["name"]),
        phenotype_columns=tuple(str(value) for value in payload["phenotype_columns"]),
        binary_trait_count=int(payload["binary_trait_count"]),
        firth_batch_size=int(payload["firth_batch_size"]),
        firth_candidate_capacity=int(payload["firth_candidate_capacity"]),
        storage_mode=BenchmarkStorageMode(str(payload["storage_mode"])),
        gpu_genotype_format=types.GpuGenotypeFormat(str(payload["gpu_genotype_format"])),
        fallback_density=FallbackDensityScenario(str(payload["fallback_density"])),
        firth_p_threshold=float(payload["firth_p_threshold"]),
    )


def trial_spec_to_json_dict(trial_spec: TrialSpec) -> dict[str, typing.Any]:
    """Convert a trial spec into a JSON-serializable dictionary."""
    return {
        "name": trial_spec.name,
        "mode": trial_spec.mode.value,
        "finalize_parquet": trial_spec.finalize_parquet,
        "fresh_process": trial_spec.fresh_process,
        "same_process_group": trial_spec.same_process_group,
    }


def trial_spec_from_json_dict(payload: dict[str, typing.Any]) -> TrialSpec:
    """Build a trial spec from a JSON dictionary."""
    return TrialSpec(
        name=str(payload["name"]),
        mode=BenchmarkMode(str(payload["mode"])),
        finalize_parquet=bool(payload["finalize_parquet"]),
        fresh_process=bool(payload["fresh_process"]),
        same_process_group=(str(payload["same_process_group"]) if payload["same_process_group"] is not None else None),
    )


def output_metrics_to_json_dict(output_metrics: OutputMetrics) -> dict[str, typing.Any]:
    """Convert output metrics into a JSON-serializable dictionary."""
    return dataclasses.asdict(output_metrics)


def output_metrics_from_json_dict(payload: dict[str, typing.Any]) -> OutputMetrics:
    """Build output metrics from a JSON dictionary."""
    output_run_directory = payload["output_run_directory"]
    final_parquet = payload["final_parquet"]
    output_row_count = payload["output_row_count"]
    info_non_null_count = payload["info_non_null_count"]
    final_parquet_bytes = payload["final_parquet_bytes"]
    return OutputMetrics(
        output_run_directory=(str(output_run_directory) if output_run_directory is not None else None),
        final_parquet=(str(final_parquet) if final_parquet is not None else None),
        output_row_count=(int(output_row_count) if output_row_count is not None else None),
        info_non_null_count=(int(info_non_null_count) if info_non_null_count is not None else None),
        chunk_file_count=int(payload["chunk_file_count"]),
        chunk_bytes=int(payload["chunk_bytes"]),
        final_parquet_bytes=(int(final_parquet_bytes) if final_parquet_bytes is not None else None),
    )


def trial_result_to_json_dict(trial_result: TrialResult) -> dict[str, typing.Any]:
    """Convert a trial result into a JSON-serializable dictionary."""
    return {
        "name": trial_result.name,
        "benchmark_case": benchmark_case_to_json_dict(trial_result.benchmark_case),
        "mode": trial_result.mode.value,
        "fresh_process": trial_result.fresh_process,
        "finalize_parquet": trial_result.finalize_parquet,
        "same_process_group": trial_result.same_process_group,
        "wall_time_seconds": trial_result.wall_time_seconds,
        "stage_timing_path": trial_result.stage_timing_path,
        "output_metrics": output_metrics_to_json_dict(trial_result.output_metrics),
    }


def trial_result_from_json_dict(payload: dict[str, typing.Any]) -> TrialResult:
    """Build a trial result from a JSON dictionary."""
    benchmark_case_payload = payload.get("benchmark_case")
    benchmark_case = (
        benchmark_case_from_json_dict(benchmark_case_payload)
        if benchmark_case_payload is not None
        else BenchmarkCase(
            name="default",
            phenotype_columns=DEFAULT_PHENOTYPE_COLUMNS,
            binary_trait_count=1,
            firth_batch_size=config.load_packaged_config().g_compute.firth_batch_size,
            firth_candidate_capacity=config.load_packaged_config().g_compute.firth_candidate_capacity,
            storage_mode=BenchmarkStorageMode.VARIANT_MAJOR,
            gpu_genotype_format=types.GpuGenotypeFormat.DOSAGE,
            fallback_density=FallbackDensityScenario.DEFAULT,
            firth_p_threshold=config.load_packaged_config().binary.p_threshold,
        )
    )
    return TrialResult(
        name=str(payload["name"]),
        benchmark_case=benchmark_case,
        mode=BenchmarkMode(str(payload["mode"])),
        fresh_process=bool(payload["fresh_process"]),
        finalize_parquet=bool(payload["finalize_parquet"]),
        same_process_group=(str(payload["same_process_group"]) if payload["same_process_group"] is not None else None),
        wall_time_seconds=float(payload["wall_time_seconds"]),
        stage_timing_path=(str(payload["stage_timing_path"]) if payload.get("stage_timing_path") is not None else None),
        output_metrics=output_metrics_from_json_dict(payload["output_metrics"]),
    )


def build_trial_environment(configuration: BenchmarkConfiguration, stage_timing_path: Path | None) -> dict[str, str]:
    """Build environment overrides for one benchmark trial."""
    del stage_timing_path
    python_path_entries = [str(REPOSITORY_ROOT)]
    existing_python_path = os.environ.get("PYTHONPATH")
    if existing_python_path:
        python_path_entries.append(existing_python_path)
    return {
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": ".50",
        "PYTHONPATH": os.pathsep.join(python_path_entries),
    }


@contextlib.contextmanager
def temporary_environment(overrides: dict[str, str]) -> typing.Iterator[None]:
    """Temporarily apply environment overrides."""
    previous_values = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, previous_value in previous_values.items():
            if previous_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous_value


def build_compute_config(
    *,
    configuration: BenchmarkConfiguration,
    benchmark_case: BenchmarkCase,
    output_root: Path,
    finalize_parquet: bool,
    stage_timing_path: Path | None,
) -> dict[str, object]:
    """Build g-specific options for one trial."""
    return {
        "g-device": configuration.device.value,
        "bsize": configuration.chunk_size,
        "g-variant-limit": configuration.variant_limit,
        "g-staging-depth": configuration.staging_depth,
        "g-output-run-directory": output_root,
        "g-output-format": "parquet" if finalize_parquet else "arrow",
        "g-telemetry": "off",
        "g-jax-cache-dir": configuration.jax_cache_directory,
        "g-jax-persistent-cache-min-entry-size-bytes": -1,
        "g-jax-persistent-cache-min-compile-time-seconds": 0,
        "g-jax-xla-autotune-cache": ENABLE_XLA_AUTOTUNE_CACHE,
        "g-firth-batch-size": benchmark_case.firth_batch_size,
        "g-firth-candidate-capacity": benchmark_case.firth_candidate_capacity,
        "g-gpu-genotype-format": benchmark_case.gpu_genotype_format.value,
        "g-stage-timings-json": stage_timing_path,
        "g-writer-threads": configuration.output_writer_thread_count,
        "g-writer-queue-depth": configuration.output_writer_queue_depth,
        "g-trusted-no-missing-diploid": configuration.trusted_no_missing_diploid,
        "g-trusted-bgen-validation-mode": (
            types.TrustedBgenValidationMode.ASSUME_VALIDATED
            if configuration.assume_trusted_validated
            else types.TrustedBgenValidationMode.CACHE_ON_MISS
        ).value,
    }


def run_regenie2_api_call(
    *,
    configuration: BenchmarkConfiguration,
    benchmark_case: BenchmarkCase,
    trial_spec: TrialSpec,
    output_root: Path,
    stage_timing_path: Path | None = None,
) -> api.RunArtifacts:
    """Run binary REGENIE step 2 through the public Python API."""
    phenotype_options: dict[str, object]
    if len(benchmark_case.phenotype_columns) == 1:
        phenotype_options = {"phenoCol": benchmark_case.phenotype_columns[0]}
    else:
        phenotype_options = {"phenoColList": ",".join(benchmark_case.phenotype_columns)}
    return api.regenie.from_options(
        {
            "step": 2,
            "bt": True,
            "bgen": configuration.bgen_path,
            "sample": configuration.sample_path,
            "phenoFile": configuration.phenotype_file,
            **phenotype_options,
            "out": output_root,
            "covarFile": configuration.data_directory / "covariates.txt",
            "covarColList": "age,sex",
            "pred": configuration.prediction_list,
            "firth": True,
            "approx": True,
            "pThresh": benchmark_case.firth_p_threshold,
            **build_compute_config(
                configuration=configuration,
                benchmark_case=benchmark_case,
                output_root=output_root,
                finalize_parquet=trial_spec.finalize_parquet,
                stage_timing_path=stage_timing_path,
            ),
        }
    )


def count_parquet_rows_and_info(final_parquet_path: Path) -> dict[str, int]:
    """Count rows and non-null INFO values in a finalized Parquet artifact."""
    frame = typing.cast(
        "pl.DataFrame",
        (
            pl.scan_parquet(final_parquet_path)
            .select(
                pl.len().alias("row_count"),
                pl.col("INFO").is_not_null().sum().alias("info_non_null_count"),
            )
            .collect()
        ),
    )
    return {
        "row_count": int(frame.item(row=0, column="row_count")),
        "info_non_null_count": int(frame.item(row=0, column="info_non_null_count")),
    }


def count_chunk_rows_and_info(chunk_file_paths: list[Path]) -> dict[str, int] | None:
    """Count rows and non-null INFO values across Arrow chunks."""
    if not chunk_file_paths:
        return None
    row_count = 0
    info_non_null_count = 0
    for chunk_file_path in chunk_file_paths:
        chunk_frame = pl.read_ipc(chunk_file_path)
        row_count += chunk_frame.height
        if "INFO" in chunk_frame.columns:
            info_non_null_count += int(chunk_frame["INFO"].is_not_null().sum())
    return {"row_count": row_count, "info_non_null_count": info_non_null_count}


def sum_optional_integer_metrics(values: typing.Iterable[int | None]) -> int | None:
    """Sum present integer metrics, preserving None when every input is absent."""
    total = 0
    has_value = False
    for value in values:
        if value is None:
            continue
        total += value
        has_value = True
    if not has_value:
        return None
    return total


def combine_phenotype_output_metrics(metrics: list[OutputMetrics]) -> OutputMetrics:
    """Aggregate per-phenotype output metrics for multi-trait benchmark runs."""
    return OutputMetrics(
        output_run_directory=None,
        final_parquet=None,
        output_row_count=sum_optional_integer_metrics(metric.output_row_count for metric in metrics),
        info_non_null_count=sum_optional_integer_metrics(metric.info_non_null_count for metric in metrics),
        chunk_file_count=sum(metric.chunk_file_count for metric in metrics),
        chunk_bytes=sum(metric.chunk_bytes for metric in metrics),
        final_parquet_bytes=sum_optional_integer_metrics(metric.final_parquet_bytes for metric in metrics),
    )


def measure_output_metrics(artifacts: api.RunArtifacts) -> OutputMetrics:
    """Measure emitted chunk and final output artifacts."""
    if artifacts.phenotype_artifacts:
        return combine_phenotype_output_metrics(
            [measure_output_metrics(phenotype_artifact) for phenotype_artifact in artifacts.phenotype_artifacts]
        )
    output_run_directory = artifacts.output_run_directory
    final_parquet_path = artifacts.final_parquet
    chunk_file_paths = (
        sorted((output_run_directory / "chunks").glob("chunk_*.arrow")) if output_run_directory is not None else []
    )
    chunk_bytes = sum(chunk_file_path.stat().st_size for chunk_file_path in chunk_file_paths)
    final_parquet_bytes = final_parquet_path.stat().st_size if final_parquet_path is not None else None
    if final_parquet_path is not None:
        row_metrics = count_parquet_rows_and_info(final_parquet_path)
    else:
        row_metrics = count_chunk_rows_and_info(chunk_file_paths)
    return OutputMetrics(
        output_run_directory=(str(output_run_directory) if output_run_directory is not None else None),
        final_parquet=(str(final_parquet_path) if final_parquet_path is not None else None),
        output_row_count=(row_metrics["row_count"] if row_metrics is not None else None),
        info_non_null_count=(row_metrics["info_non_null_count"] if row_metrics is not None else None),
        chunk_file_count=len(chunk_file_paths),
        chunk_bytes=chunk_bytes,
        final_parquet_bytes=final_parquet_bytes,
    )


def run_api_trial(
    *,
    configuration: BenchmarkConfiguration,
    benchmark_case: BenchmarkCase,
    trial_spec: TrialSpec,
    stage_timing_path: Path | None,
) -> TrialResult:
    """Run one in-process API trial and measure wall time plus artifacts."""
    if stage_timing_path is not None:
        stage_timing_path.parent.mkdir(parents=True, exist_ok=True)
    output_root = configuration.output_directory / "outputs" / benchmark_case.name / trial_spec.name
    environment_overrides = build_trial_environment(configuration, stage_timing_path)
    with temporary_environment(environment_overrides):
        start_time = time.perf_counter()
        artifacts = run_regenie2_api_call(
            configuration=configuration,
            benchmark_case=benchmark_case,
            trial_spec=trial_spec,
            output_root=output_root,
            stage_timing_path=stage_timing_path,
        )
        wall_time_seconds = time.perf_counter() - start_time
    return TrialResult(
        name=f"{benchmark_case.name}_{trial_spec.name}",
        benchmark_case=benchmark_case,
        mode=trial_spec.mode,
        fresh_process=trial_spec.fresh_process,
        finalize_parquet=trial_spec.finalize_parquet,
        same_process_group=trial_spec.same_process_group,
        wall_time_seconds=wall_time_seconds,
        stage_timing_path=str(stage_timing_path) if stage_timing_path is not None else None,
        output_metrics=measure_output_metrics(artifacts),
    )


def build_fresh_process_command(
    *,
    configuration: BenchmarkConfiguration,
    benchmark_case: BenchmarkCase,
    trial_spec: TrialSpec,
    stage_timing_path: Path | None,
) -> ChildProcessCommand:
    """Build a fresh Python process command for one trial."""
    child_code = textwrap.dedent(
        """
        import json
        from pathlib import Path

        import tooling.cli.benchmark_regenie2_binary_hot as benchmark

        configuration = benchmark.configuration_from_json_dict(json.loads({configuration_payload!r}))
        benchmark_case = benchmark.benchmark_case_from_json_dict(json.loads({benchmark_case_payload!r}))
        trial_spec = benchmark.trial_spec_from_json_dict(json.loads({trial_payload!r}))
        stage_timing_path_value = {stage_timing_path_payload!r}
        result = benchmark.run_api_trial(
            configuration=configuration,
            benchmark_case=benchmark_case,
            trial_spec=trial_spec,
            stage_timing_path=Path(stage_timing_path_value) if stage_timing_path_value is not None else None,
        )
        print(json.dumps(benchmark.trial_result_to_json_dict(result), sort_keys=True))
        """
    ).format(
        configuration_payload=json.dumps(configuration_to_json_dict(configuration), sort_keys=True),
        benchmark_case_payload=json.dumps(benchmark_case_to_json_dict(benchmark_case), sort_keys=True),
        trial_payload=json.dumps(trial_spec_to_json_dict(trial_spec), sort_keys=True),
        stage_timing_path_payload=str(stage_timing_path) if stage_timing_path is not None else None,
    )
    return ChildProcessCommand(
        command_arguments=[configuration.python_executable, "-c", child_code],
        environment_overrides=build_trial_environment(configuration, None),
    )


def run_fresh_process_trial(
    *,
    configuration: BenchmarkConfiguration,
    benchmark_case: BenchmarkCase,
    trial_spec: TrialSpec,
    stage_timing_path: Path | None,
) -> TrialResult:
    """Run one trial in a fresh Python process."""
    child_process_command = build_fresh_process_command(
        configuration=configuration,
        benchmark_case=benchmark_case,
        trial_spec=trial_spec,
        stage_timing_path=stage_timing_path,
    )
    environment = dict(os.environ)
    environment.update(child_process_command.environment_overrides)
    completed_process = subprocess.run(
        child_process_command.command_arguments,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        cwd=REPOSITORY_ROOT,
    )
    result_line = completed_process.stdout.strip().splitlines()[-1]
    return trial_result_from_json_dict(json.loads(result_line))


def command_output(command_arguments: list[str]) -> dict[str, typing.Any]:
    """Run a metadata command and return captured output."""
    try:
        completed_process = subprocess.run(command_arguments, check=False, capture_output=True, text=True)
    except FileNotFoundError as error:
        return {
            "command": command_arguments,
            "returncode": None,
            "stdout": "",
            "stderr": str(error),
        }
    return {
        "command": command_arguments,
        "returncode": completed_process.returncode,
        "stdout": completed_process.stdout,
        "stderr": completed_process.stderr,
    }


def collect_metadata(configuration: BenchmarkConfiguration) -> dict[str, typing.Any]:
    """Collect reproducibility metadata for the benchmark."""
    relevant_environment = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(("G_", "GWAS_ENGINE_", "JAX_", "XLA_", "CUDA_", "RAYON_", "SLURM_"))
    }
    return {
        "git_head": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short"]),
        "hostname": command_output(["hostname"]),
        "python": command_output([sys.executable, "--version"]),
        "jax": command_output([sys.executable, "-c", "import jax; print(jax.__version__); print(jax.devices())"]),
        "nvidia_smi": command_output(["nvidia-smi"]),
        "configuration": configuration_to_json_dict(configuration),
        "benchmark_cases": [
            benchmark_case_to_json_dict(benchmark_case) for benchmark_case in build_benchmark_cases(configuration)
        ],
        "environment": relevant_environment,
        "expected_full_variant_count": configuration.expected_variant_count,
    }


def build_headline_for_results(trial_results: list[TrialResult]) -> dict[str, float | None]:
    """Build headline mode timings for a set of trial results."""
    return {
        "cold_process_finalized_seconds": next(
            (
                trial_result.wall_time_seconds
                for trial_result in trial_results
                if trial_result.mode == BenchmarkMode.COLD_PROCESS_FINALIZED
            ),
            None,
        ),
        "hot_same_process_no_final_seconds": next(
            (
                trial_result.wall_time_seconds
                for trial_result in trial_results
                if trial_result.mode == BenchmarkMode.HOT_SAME_PROCESS_NO_FINAL
            ),
            None,
        ),
        "hot_same_process_finalized_seconds": next(
            (
                trial_result.wall_time_seconds
                for trial_result in trial_results
                if trial_result.mode == BenchmarkMode.HOT_SAME_PROCESS_FINALIZED
            ),
            None,
        ),
    }


def build_summary(
    *,
    configuration: BenchmarkConfiguration,
    trial_results: list[TrialResult],
) -> dict[str, typing.Any]:
    """Build a JSON-serializable benchmark summary."""
    return {
        "metadata": collect_metadata(configuration),
        "results": [trial_result_to_json_dict(trial_result) for trial_result in trial_results],
        "headline": build_headline_for_results(trial_results),
        "headline_by_case": {
            benchmark_case.name: build_headline_for_results(
                [
                    trial_result
                    for trial_result in trial_results
                    if trial_result.benchmark_case.name == benchmark_case.name
                ]
            )
            for benchmark_case in build_benchmark_cases(configuration)
        },
    }


def write_summary(summary_path: Path, summary: dict[str, typing.Any]) -> None:
    """Write a benchmark summary JSON file."""
    tooling_reports.write_json_report(summary_path, summary, sort_keys=True)


def run_benchmark(configuration: BenchmarkConfiguration, trial_specs: list[TrialSpec]) -> list[TrialResult]:
    """Run the requested benchmark trials."""
    configuration.output_directory.mkdir(parents=True, exist_ok=True)
    trial_results: list[TrialResult] = []
    for benchmark_case in build_benchmark_cases(configuration):
        for trial_spec in trial_specs:
            stage_timing_path = None
            if configuration.stage_timing_mode == StageTimingMode.EXACT:
                stage_timing_path = (
                    configuration.output_directory / "stage_timings" / benchmark_case.name / f"{trial_spec.name}.json"
                )
            if trial_spec.fresh_process:
                trial_result = run_fresh_process_trial(
                    configuration=configuration,
                    benchmark_case=benchmark_case,
                    trial_spec=trial_spec,
                    stage_timing_path=stage_timing_path,
                )
            else:
                trial_result = run_api_trial(
                    configuration=configuration,
                    benchmark_case=benchmark_case,
                    trial_spec=trial_spec,
                    stage_timing_path=stage_timing_path,
                )
            trial_results.append(trial_result)
            print(json.dumps(trial_result_to_json_dict(trial_result), sort_keys=True))
    return trial_results


def main() -> None:
    """Run the binary hot benchmark."""
    argument_parser = build_argument_parser()
    arguments = argument_parser.parse_args()
    configuration = build_configuration(arguments)
    trial_specs = build_trial_specs(
        include_cold_process=bool(arguments.include_cold_process),
        include_no_final_hot=bool(arguments.include_no_final_hot),
        include_finalized_hot=bool(arguments.include_finalized_hot),
    )
    if not trial_specs:
        message = "At least one benchmark mode must be enabled."
        raise ValueError(message)
    benchmark_cases = build_benchmark_cases(configuration)
    if not benchmark_cases:
        message = "At least one benchmark case must be enabled."
        raise ValueError(message)
    trial_results = run_benchmark(configuration, trial_specs)
    summary = build_summary(configuration=configuration, trial_results=trial_results)
    summary_path = arguments.json_summary_path or (configuration.output_directory / "regenie2_binary_hot_summary.json")
    write_summary(summary_path, summary)
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
