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

DEFAULT_DATA_DIRECTORY = Path("data")
DEFAULT_OUTPUT_PARENT = Path("data/profiles")
DEFAULT_VARIANT_COUNT = 418_943
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
JAX_XLA_AUTOTUNE_CACHE = "xla_gpu_per_fusion_autotune_cache_dir"
GPU_JAX_CACHE_PARENT_DEFAULT = "/tmp/g-jax-binary-hot-cache"
ENABLE_XLA_AUTOTUNE_CACHE = os.environ.get("G_PROFILE_ENABLE_XLA_AUTOTUNE_CACHE") == "1"


class BenchmarkMode(enum.StrEnum):
    """Execution mode measured by the benchmark harness."""

    COLD_PROCESS_FINALIZED = "cold_process_finalized"
    WARM_SAME_PROCESS_NO_FINAL = "warm_same_process_no_final"
    HOT_SAME_PROCESS_NO_FINAL = "hot_same_process_no_final"
    WARM_SAME_PROCESS_FINALIZED = "warm_same_process_finalized"
    HOT_SAME_PROCESS_FINALIZED = "hot_same_process_finalized"


@dataclasses.dataclass(frozen=True)
class BenchmarkConfiguration:
    """Shared configuration for a binary REGENIE benchmark run."""

    data_directory: Path
    output_directory: Path
    device: types.Device
    chunk_size: int
    staging_depth: int
    output_writer_thread_count: int
    output_writer_queue_depth: int
    trusted_no_missing_diploid: bool
    assume_trusted_validated: bool
    firth_batch_size: int
    variant_limit: int | None
    python_executable: str
    jax_cache_directory: Path


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
    mode: BenchmarkMode
    fresh_process: bool
    finalize_parquet: bool
    same_process_group: str | None
    wall_time_seconds: float
    stage_timing_path: str
    output_metrics: OutputMetrics


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark binary REGENIE step 2 while separating cold process, same-process hot, "
            "chunk-only, and finalized-Parquet timings."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIRECTORY, help="Input data directory.")
    parser.add_argument("--output-dir", type=Path, help="Benchmark output directory.")
    parser.add_argument("--device", default=types.Device.GPU.value, choices=[device.value for device in types.Device])
    parser.add_argument("--chunk-size", type=int, default=8192, help="Variants per chunk.")
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
        default=True,
        help="Skip repeated trusted-path validation when the input has already been checked.",
    )
    parser.add_argument("--firth-batch-size", type=int, default=64, help="Binary Firth candidate batch size.")
    parser.add_argument("--variant-limit", type=int, help="Optional variant cap for smoke runs.")
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
    output_directory = arguments.output_dir or default_output_directory()
    jax_cache_directory = arguments.jax_cache_dir
    if jax_cache_directory is None:
        job_identifier = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
        cache_parent = os.environ.get("G_PROFILE_GPU_JAX_CACHE_PARENT", GPU_JAX_CACHE_PARENT_DEFAULT)
        jax_cache_directory = Path(cache_parent) / job_identifier / output_directory.name
    return BenchmarkConfiguration(
        data_directory=arguments.data_dir,
        output_directory=output_directory,
        device=types.Device(arguments.device),
        chunk_size=int(arguments.chunk_size),
        staging_depth=int(arguments.staging_depth),
        output_writer_thread_count=int(arguments.output_writer_thread_count),
        output_writer_queue_depth=int(arguments.output_writer_queue_depth),
        trusted_no_missing_diploid=bool(arguments.trusted_no_missing_diploid),
        assume_trusted_validated=bool(arguments.assume_trusted_validated),
        firth_batch_size=int(arguments.firth_batch_size),
        variant_limit=arguments.variant_limit,
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
        "output_directory": str(configuration.output_directory),
        "device": configuration.device.value,
        "chunk_size": configuration.chunk_size,
        "staging_depth": configuration.staging_depth,
        "output_writer_thread_count": configuration.output_writer_thread_count,
        "output_writer_queue_depth": configuration.output_writer_queue_depth,
        "trusted_no_missing_diploid": configuration.trusted_no_missing_diploid,
        "assume_trusted_validated": configuration.assume_trusted_validated,
        "firth_batch_size": configuration.firth_batch_size,
        "variant_limit": configuration.variant_limit,
        "python_executable": configuration.python_executable,
        "jax_cache_directory": str(configuration.jax_cache_directory),
    }


def configuration_from_json_dict(payload: dict[str, typing.Any]) -> BenchmarkConfiguration:
    """Build configuration from a JSON dictionary."""
    return BenchmarkConfiguration(
        data_directory=Path(str(payload["data_directory"])),
        output_directory=Path(str(payload["output_directory"])),
        device=types.Device(str(payload["device"])),
        chunk_size=int(payload["chunk_size"]),
        staging_depth=int(payload.get("staging_depth", payload.get("prefetch_chunks", 1))),
        output_writer_thread_count=int(payload["output_writer_thread_count"]),
        output_writer_queue_depth=int(payload["output_writer_queue_depth"]),
        trusted_no_missing_diploid=bool(payload["trusted_no_missing_diploid"]),
        assume_trusted_validated=bool(payload["assume_trusted_validated"]),
        firth_batch_size=int(payload["firth_batch_size"]),
        variant_limit=(int(payload["variant_limit"]) if payload["variant_limit"] is not None else None),
        python_executable=str(payload["python_executable"]),
        jax_cache_directory=Path(str(payload["jax_cache_directory"])),
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
    return TrialResult(
        name=str(payload["name"]),
        mode=BenchmarkMode(str(payload["mode"])),
        fresh_process=bool(payload["fresh_process"]),
        finalize_parquet=bool(payload["finalize_parquet"]),
        same_process_group=(str(payload["same_process_group"]) if payload["same_process_group"] is not None else None),
        wall_time_seconds=float(payload["wall_time_seconds"]),
        stage_timing_path=str(payload["stage_timing_path"]),
        output_metrics=output_metrics_from_json_dict(payload["output_metrics"]),
    )


def build_trial_environment(configuration: BenchmarkConfiguration, stage_timing_path: Path | None) -> dict[str, str]:
    """Build environment overrides for one benchmark trial."""
    python_path_entries = [str(REPOSITORY_ROOT)]
    existing_python_path = os.environ.get("PYTHONPATH")
    if existing_python_path:
        python_path_entries.append(existing_python_path)
    environment = {
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": ".50",
        "JAX_COMPILATION_CACHE_DIR": str(configuration.jax_cache_directory),
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1",
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
        "G_REGENIE2_BINARY_FIRTH_BATCH_SIZE": str(configuration.firth_batch_size),
        "PYTHONPATH": os.pathsep.join(python_path_entries),
    }
    if ENABLE_XLA_AUTOTUNE_CACHE:
        environment["JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"] = JAX_XLA_AUTOTUNE_CACHE
    if configuration.device == types.Device.CPU:
        environment["JAX_PLATFORMS"] = "cpu"
    if configuration.assume_trusted_validated:
        environment["G_REGENIE2_ASSUME_TRUSTED_NO_MISSING_DIPLOID_VALIDATED"] = "1"
    if stage_timing_path is not None:
        environment["G_REGENIE2_STAGE_TIMINGS_JSON"] = str(stage_timing_path)
    return environment


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
    output_root: Path,
    finalize_parquet: bool,
) -> api.ComputeConfig:
    """Build the API compute configuration for one trial."""
    return api.ComputeConfig(
        device=configuration.device,
        chunk_size=configuration.chunk_size,
        variant_limit=configuration.variant_limit,
        staging_depth=configuration.staging_depth,
        output_run_directory=output_root,
        finalize_parquet=finalize_parquet,
        output_writer_thread_count=configuration.output_writer_thread_count,
        output_writer_queue_depth=configuration.output_writer_queue_depth,
        trusted_no_missing_diploid=configuration.trusted_no_missing_diploid,
    )


def run_regenie2_api_call(
    *,
    configuration: BenchmarkConfiguration,
    trial_spec: TrialSpec,
    output_root: Path,
) -> api.RunArtifacts:
    """Run binary REGENIE step 2 through the public Python API."""
    data_directory = configuration.data_directory
    return api.regenie2(
        bgen=data_directory / "1kg_chr22_full.bgen",
        sample=data_directory / "1kg_chr22_full.sample",
        pheno=data_directory / "pheno_bin.txt",
        pheno_name="phenotype_binary",
        out=output_root,
        covar=data_directory / "covariates.txt",
        covar_names="age,sex",
        pred=data_directory / "baselines/regenie_step1_pred.list",
        trait_type=types.RegenieTraitType.BINARY,
        compute=build_compute_config(
            configuration=configuration,
            output_root=output_root,
            finalize_parquet=trial_spec.finalize_parquet,
        ),
        binary=api.Regenie2BinaryConfig(firth=True, approx=True),
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


def measure_output_metrics(artifacts: api.RunArtifacts) -> OutputMetrics:
    """Measure emitted chunk and final output artifacts."""
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
    trial_spec: TrialSpec,
    stage_timing_path: Path,
) -> TrialResult:
    """Run one in-process API trial and measure wall time plus artifacts."""
    stage_timing_path.parent.mkdir(parents=True, exist_ok=True)
    output_root = configuration.output_directory / "outputs" / trial_spec.name
    environment_overrides = build_trial_environment(configuration, stage_timing_path)
    with temporary_environment(environment_overrides):
        start_time = time.perf_counter()
        artifacts = run_regenie2_api_call(
            configuration=configuration,
            trial_spec=trial_spec,
            output_root=output_root,
        )
        wall_time_seconds = time.perf_counter() - start_time
    return TrialResult(
        name=trial_spec.name,
        mode=trial_spec.mode,
        fresh_process=trial_spec.fresh_process,
        finalize_parquet=trial_spec.finalize_parquet,
        same_process_group=trial_spec.same_process_group,
        wall_time_seconds=wall_time_seconds,
        stage_timing_path=str(stage_timing_path),
        output_metrics=measure_output_metrics(artifacts),
    )


def build_fresh_process_command(
    *,
    configuration: BenchmarkConfiguration,
    trial_spec: TrialSpec,
    stage_timing_path: Path,
) -> ChildProcessCommand:
    """Build a fresh Python process command for one trial."""
    child_code = textwrap.dedent(
        """
        import json
        from pathlib import Path

        from scripts import benchmark_regenie2_binary_hot as benchmark

        configuration = benchmark.configuration_from_json_dict(json.loads({configuration_payload!r}))
        trial_spec = benchmark.trial_spec_from_json_dict(json.loads({trial_payload!r}))
        result = benchmark.run_api_trial(
            configuration=configuration,
            trial_spec=trial_spec,
            stage_timing_path=Path({stage_timing_path!r}),
        )
        print(json.dumps(benchmark.trial_result_to_json_dict(result), sort_keys=True))
        """
    ).format(
        configuration_payload=json.dumps(configuration_to_json_dict(configuration), sort_keys=True),
        trial_payload=json.dumps(trial_spec_to_json_dict(trial_spec), sort_keys=True),
        stage_timing_path=str(stage_timing_path),
    )
    return ChildProcessCommand(
        command_arguments=[configuration.python_executable, "-c", child_code],
        environment_overrides=build_trial_environment(configuration, None),
    )


def run_fresh_process_trial(
    *,
    configuration: BenchmarkConfiguration,
    trial_spec: TrialSpec,
    stage_timing_path: Path,
) -> TrialResult:
    """Run one trial in a fresh Python process."""
    child_process_command = build_fresh_process_command(
        configuration=configuration,
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
        "environment": relevant_environment,
        "expected_full_variant_count": DEFAULT_VARIANT_COUNT,
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
        "headline": {
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
        },
    }


def write_summary(summary_path: Path, summary: dict[str, typing.Any]) -> None:
    """Write a benchmark summary JSON file."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def run_benchmark(configuration: BenchmarkConfiguration, trial_specs: list[TrialSpec]) -> list[TrialResult]:
    """Run the requested benchmark trials."""
    configuration.output_directory.mkdir(parents=True, exist_ok=True)
    trial_results: list[TrialResult] = []
    for trial_spec in trial_specs:
        stage_timing_path = configuration.output_directory / "stage_timings" / f"{trial_spec.name}.json"
        if trial_spec.fresh_process:
            trial_result = run_fresh_process_trial(
                configuration=configuration,
                trial_spec=trial_spec,
                stage_timing_path=stage_timing_path,
            )
        else:
            trial_result = run_api_trial(
                configuration=configuration,
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
    trial_results = run_benchmark(configuration, trial_specs)
    summary = build_summary(configuration=configuration, trial_results=trial_results)
    summary_path = arguments.json_summary_path or (configuration.output_directory / "regenie2_binary_hot_summary.json")
    write_summary(summary_path, summary)
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
