#!/usr/bin/env python3
"""Deep landau profiling harness for original REGENIE and g REGENIE step 2."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import textwrap
import time
import typing
from datetime import UTC, datetime
from pathlib import Path

import scripts.benchmark as baseline_benchmark
import scripts.benchmark_regenie_comparison as comparison_benchmark
import tooling.cli.benchmark_bgen_reader as benchmark_bgen_reader
from tooling.common import paths as tooling_paths

REPOSITORY_ROOT = tooling_paths.find_repository_root(Path(__file__))
DEFAULT_OUTPUT_PARENT = Path("data/profiles")
DEFAULT_VARIANT_COUNT = 418_943
JAX_XLA_AUTOTUNE_CACHE = "xla_gpu_per_fusion_autotune_cache_dir"
ENABLE_XLA_AUTOTUNE_CACHE = os.environ.get("G_PROFILE_ENABLE_XLA_AUTOTUNE_CACHE") == "1"
GPU_JAX_CACHE_PARENT_DEFAULT = "/tmp/g-jax-profile-cache"


@dataclasses.dataclass(frozen=True)
class Step2Candidate:
    """One g REGENIE step 2 tuning candidate."""

    trait_type: str
    device: str
    chunk_size: int
    staging_depth: int
    output_writer_thread_count: int
    output_writer_queue_depth: int
    bgen_decode_tile_variant_count: int | None
    rayon_thread_count: int | None
    firth_batch_size: int | None


@dataclasses.dataclass(frozen=True)
class BgenCandidateSummary:
    """Measured BGEN reader candidate summary."""

    decode_tile_variant_count: int | None
    rayon_thread_count: int | None
    median_seconds: float
    mean_seconds: float
    durations_seconds: list[float]


@dataclasses.dataclass(frozen=True)
class TrialResult:
    """One measured process execution."""

    name: str
    implementation: str
    trait_type: str
    device: str
    status: str
    wall_time_seconds: float | None
    output_row_count: int | None
    stdout_log_path: str
    stderr_log_path: str
    command_arguments: list[str]
    environment_overrides: dict[str, str]
    output_path: str | None = None
    stage_timing_path: str | None = None
    regenie_profile_path: str | None = None
    device_diagnostics: dict[str, typing.Any] | None = None
    notes: str | None = None


@dataclasses.dataclass(frozen=True)
class AggregateResult:
    """Aggregate runtime statistics for one benchmark cell."""

    name: str
    implementation: str
    trait_type: str
    device: str
    status: str
    trial_count: int
    warmup_count: int
    median_wall_time_seconds: float | None
    mean_wall_time_seconds: float | None
    min_wall_time_seconds: float | None
    max_wall_time_seconds: float | None
    standard_deviation_seconds: float | None
    rows_per_second: float | None
    trials: list[TrialResult]


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Profile REGENIE step 2 deeply on landau.")
    parser.add_argument("--output-dir", type=Path, help="Explicit output directory.")
    parser.add_argument("--variant-limit", type=int, help="Optional variant cap for smoke runs.")
    parser.add_argument("--smoke", action="store_true", help="Use a fast smoke configuration.")
    parser.add_argument("--skip-deep-profiles", action="store_true", help="Skip perf/py-spy/cProfile/JAX trace runs.")
    parser.add_argument("--chunk-sizes", default="2048,4096,8192,16384")
    parser.add_argument("--staging-depths", "--prefetch-chunks", default="1,2")
    parser.add_argument("--output-writer-thread-counts", default="1,2,4,8")
    parser.add_argument("--writer-queue-depth-multipliers", default="1,2")
    parser.add_argument("--firth-batch-sizes", default="512,1024,2048")
    parser.add_argument("--bgen-decode-tile-variant-counts", default="32,64,128,256")
    parser.add_argument("--rayon-thread-counts", default="1,2,4,8")
    parser.add_argument("--bgen-benchmark-chunk-size", type=int, default=8192)
    parser.add_argument("--top-bgen-candidates", type=int, default=3)
    parser.add_argument("--top-finalists", type=int, default=3)
    parser.add_argument("--tuning-warmups", type=int, default=1)
    parser.add_argument("--tuning-trials", type=int, default=3)
    parser.add_argument("--finalist-warmups", type=int, default=2)
    parser.add_argument("--finalist-trials", type=int, default=7)
    parser.add_argument("--headline-warmups", type=int, default=1)
    parser.add_argument("--headline-trials", type=int, default=7)
    return parser


def parse_int_list(raw_values: str) -> tuple[int, ...]:
    """Parse a comma-separated list of integers."""
    parsed_values = tuple(int(value.strip()) for value in raw_values.split(",") if value.strip())
    if not parsed_values:
        message = "At least one integer is required."
        raise ValueError(message)
    return parsed_values


def build_output_directory(arguments: argparse.Namespace) -> Path:
    """Resolve the campaign output directory."""
    if arguments.output_dir is not None:
        return arguments.output_dir
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_PARENT / f"landau_deep_{timestamp}"


def command_output(
    command_arguments: list[str],
    environment_overrides: dict[str, str] | None = None,
) -> dict[str, typing.Any]:
    """Run a metadata command and return captured output."""
    environment = dict(os.environ)
    if environment_overrides is not None:
        environment.update(environment_overrides)
    try:
        completed_process = subprocess.run(
            command_arguments,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
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


def dirty_diff_sha256() -> str:
    """Hash the current dirty diff without writing it into the report."""
    completed_process = subprocess.run(["git", "diff"], check=False, capture_output=True)
    return hashlib.sha256(completed_process.stdout).hexdigest()


def collect_environment_metadata(baseline_paths: typing.Any) -> dict[str, typing.Any]:
    """Collect reproducibility metadata for a profiling campaign."""
    input_paths = [
        baseline_paths.bgen_path,
        baseline_paths.sample_path,
        baseline_paths.continuous_phenotype_path,
        baseline_paths.binary_phenotype_path,
        baseline_paths.covariate_path,
        baseline_paths.regenie_prediction_list_path,
        baseline_paths.regenie_qt_prediction_list_path,
    ]
    file_sizes = {
        str(input_path): input_path.stat().st_size
        for input_path in input_paths
        if input_path is not None and input_path.exists()
    }
    relevant_environment = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(("G_", "GWAS_ENGINE_", "JAX_", "XLA_", "CUDA_", "RAYON_", "SLURM_"))
    }
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_head": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short"]),
        "dirty_diff_sha256": dirty_diff_sha256(),
        "lscpu": command_output(["lscpu"]),
        "nvidia_smi": command_output(["nvidia-smi"]),
        "python": command_output([sys.executable, "--version"]),
        "jax": command_output([sys.executable, "-c", "import jax; print(jax.__version__); print(jax.devices())"]),
        "rustc": command_output(["rustc", "--version"]),
        "cargo": command_output(["cargo", "--version"]),
        "regenie": command_output(["regenie", "--version"]),
        "hardware": dataclasses.asdict(baseline_benchmark.collect_hardware_summary()),
        "environment": relevant_environment,
        "input_file_sizes": file_sizes,
        "expected_full_variant_count": DEFAULT_VARIANT_COUNT,
    }


def ensure_prediction_lists(
    *,
    baseline_paths: typing.Any,
    regenie_executable: str,
    log_directory: Path,
) -> list[TrialResult]:
    """Generate missing REGENIE step 1 prediction lists before profiling."""
    setup_results: list[TrialResult] = []
    prediction_specs = [
        (
            baseline_paths.regenie_prediction_list_path,
            "regenie_step1_binary_setup",
            baseline_benchmark.build_regenie_step1_command(regenie_executable, baseline_paths),
        ),
        (
            baseline_paths.regenie_qt_prediction_list_path,
            "regenie_step1_quantitative_setup",
            baseline_benchmark.build_regenie_step1_continuous_command(regenie_executable, baseline_paths),
        ),
    ]
    for prediction_path, name, command_arguments in prediction_specs:
        if prediction_path is not None and prediction_path.exists():
            continue
        setup_results.append(
            run_logged_command(
                name=name,
                implementation="regenie",
                trait_type="setup",
                device="external_cpu",
                command_arguments=command_arguments,
                environment_overrides={},
                log_directory=log_directory,
            )
        )
    return setup_results


def replace_command_output_prefix(command_arguments: list[str], output_prefix: Path) -> list[str]:
    """Return a command with its --out value replaced."""
    updated_arguments = list(command_arguments)
    output_index = updated_arguments.index("--out")
    updated_arguments[output_index + 1] = str(output_prefix)
    return updated_arguments


def build_regenie_step2_command(
    *,
    trait_type: str,
    regenie_executable: str,
    baseline_paths: typing.Any,
    output_prefix: Path,
) -> list[str]:
    """Build one original REGENIE step 2 command with an isolated output prefix."""
    if trait_type == "binary":
        base_command = baseline_benchmark.build_regenie_step2_command(regenie_executable, baseline_paths)
    else:
        base_command = baseline_benchmark.build_regenie_step2_continuous_command(regenie_executable, baseline_paths)
    return replace_command_output_prefix(base_command, output_prefix)


def build_queue_depth_values(writer_thread_count: int, queue_depth_multipliers: tuple[int, ...]) -> tuple[int, ...]:
    """Build queue depths from writer thread count and multipliers."""
    return tuple(sorted({max(1, writer_thread_count * multiplier) for multiplier in queue_depth_multipliers}))


def build_candidate_slug(candidate: Step2Candidate) -> str:
    """Build a stable filename slug for a tuning candidate."""
    candidate_parts = [
        candidate.trait_type,
        candidate.device,
        f"chunk{candidate.chunk_size}",
        f"staging{candidate.staging_depth}",
        f"writer{candidate.output_writer_thread_count}",
        f"queue{candidate.output_writer_queue_depth}",
        (
            f"tile{candidate.bgen_decode_tile_variant_count}"
            if candidate.bgen_decode_tile_variant_count is not None
            else "tiledefault"
        ),
        f"rayon{candidate.rayon_thread_count if candidate.rayon_thread_count is not None else 'default'}",
    ]
    if candidate.firth_batch_size is not None:
        candidate_parts.append(f"firth{candidate.firth_batch_size}")
    return "_".join(candidate_parts)


def build_step2_candidates(
    *,
    trait_type: str,
    device: str,
    bgen_candidates: tuple[BgenCandidateSummary, ...],
    chunk_sizes: tuple[int, ...],
    staging_depths: tuple[int, ...],
    writer_thread_counts: tuple[int, ...],
    queue_depth_multipliers: tuple[int, ...],
    firth_batch_sizes: tuple[int, ...],
) -> tuple[Step2Candidate, ...]:
    """Build the g step 2 candidate grid."""
    candidates: list[Step2Candidate] = []
    for bgen_candidate in bgen_candidates:
        for chunk_size in chunk_sizes:
            for staging_depth in staging_depths:
                for writer_thread_count in writer_thread_counts:
                    for queue_depth in build_queue_depth_values(writer_thread_count, queue_depth_multipliers):
                        if trait_type == "binary":
                            for firth_batch_size in firth_batch_sizes:
                                candidates.append(
                                    Step2Candidate(
                                        trait_type=trait_type,
                                        device=device,
                                        chunk_size=chunk_size,
                                        staging_depth=staging_depth,
                                        output_writer_thread_count=writer_thread_count,
                                        output_writer_queue_depth=queue_depth,
                                        bgen_decode_tile_variant_count=bgen_candidate.decode_tile_variant_count,
                                        rayon_thread_count=bgen_candidate.rayon_thread_count,
                                        firth_batch_size=firth_batch_size,
                                    )
                                )
                            continue
                        candidates.append(
                            Step2Candidate(
                                trait_type=trait_type,
                                device=device,
                                chunk_size=chunk_size,
                                staging_depth=staging_depth,
                                output_writer_thread_count=writer_thread_count,
                                output_writer_queue_depth=queue_depth,
                                bgen_decode_tile_variant_count=bgen_candidate.decode_tile_variant_count,
                                rayon_thread_count=bgen_candidate.rayon_thread_count,
                                firth_batch_size=None,
                            )
                        )
    return tuple(candidates)


def build_g_trial_environment(
    *,
    candidate: Step2Candidate,
    cache_directory: Path,
    stage_timing_path: Path | None,
) -> dict[str, str]:
    """Build child process environment overrides for one g trial."""
    del cache_directory, stage_timing_path
    return {
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": ".50",
    }


def build_g_step2_child_command(
    *,
    baseline_paths: typing.Any,
    candidate: Step2Candidate,
    output_prefix: Path,
    variant_limit: int | None,
    cache_directory: Path | None = None,
    stage_timing_path: Path | None = None,
    trace_directory: Path | None = None,
    memory_profile_path: Path | None = None,
) -> list[str]:
    """Build one isolated Python child command for a g REGENIE step 2 run."""
    phenotype_path = baseline_paths.continuous_phenotype_path
    phenotype_name = "phenotype_continuous"
    prediction_path = baseline_paths.regenie_qt_prediction_list_path
    binary_options_expression = "{}"
    if candidate.trait_type == "binary":
        phenotype_path = baseline_paths.binary_phenotype_path
        phenotype_name = "phenotype_binary"
        prediction_path = baseline_paths.regenie_prediction_list_path
        binary_options_expression = '{"firth": True, "approx": True}'
    variant_limit_expression = "None" if variant_limit is None else str(variant_limit)
    jax_cache_directory = cache_directory
    if jax_cache_directory is not None and candidate.device == "gpu":
        job_identifier = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
        gpu_cache_parent = os.environ.get("G_PROFILE_GPU_JAX_CACHE_PARENT", GPU_JAX_CACHE_PARENT_DEFAULT)
        jax_cache_directory = Path(gpu_cache_parent) / job_identifier / jax_cache_directory.name
    jax_cache_directory_expression = "None" if jax_cache_directory is None else repr(str(jax_cache_directory))
    bgen_tile_expression = (
        "64" if candidate.bgen_decode_tile_variant_count is None else str(candidate.bgen_decode_tile_variant_count)
    )
    firth_batch_expression = "1024" if candidate.firth_batch_size is None else str(candidate.firth_batch_size)
    rayon_thread_expression = "None" if candidate.rayon_thread_count is None else str(candidate.rayon_thread_count)
    child_code = textwrap.dedent(
        """
        import json
        import time

        import jax
        import polars as pl

        from g import api, types

        trace_directory = {trace_directory!r}
        memory_profile_path = {memory_profile_path!r}
        if trace_directory is not None:
            jax.profiler.start_trace(trace_directory)
        try:
            start_time = time.perf_counter()
            artifacts = api.regenie.from_options({{
                "step": 2,
                "bt" if {trait_type!r} == "binary" else "qt": True,
                "bgen": {bgen_path!r},
                "sample": {sample_path!r},
                "phenoFile": {phenotype_path!r},
                "phenoCol": {phenotype_name!r},
                "out": {output_prefix!r},
                "covarFile": {covariate_path!r},
                "covarColList": "age,sex",
                "pred": {prediction_path!r},
                "g-device": {device!r},
                "bsize": {chunk_size},
                "g-variant-limit": {variant_limit_expression},
                "g-staging-depth": {staging_depth},
                "g-output-format": "parquet",
                "g-writer-threads": {writer_thread_count},
                "g-writer-queue-depth": {writer_queue_depth},
                "g-bgen-decode-tile-variant-count": {bgen_tile_expression},
                "g-firth-batch-size": {firth_batch_expression},
                "threads": {rayon_thread_expression},
                "g-jax-cache-dir": {jax_cache_directory_expression},
                "g-jax-xla-autotune-cache": {enable_xla_autotune_cache},
                "g-stage-timings-json": {stage_timing_path!r},
                **{binary_options_expression},
            }})
            wall_time_seconds = time.perf_counter() - start_time
            output_row_count = pl.scan_parquet(artifacts.final_parquet).select(pl.len()).collect().item()
            probe_array = jax.device_put(0)
            probe_device = next(iter(probe_array.devices()))
            if memory_profile_path is not None:
                jax.profiler.save_device_memory_profile(memory_profile_path)
            print(json.dumps({{
                "wall_time_seconds": wall_time_seconds,
                "output_path": str(artifacts.final_parquet),
                "output_row_count": int(output_row_count),
                "jax_devices": [str(device) for device in jax.devices()],
                "jax_probe_device": str(probe_device),
                "jax_probe_device_platform": getattr(probe_device, "platform", None),
            }}))
        finally:
            if trace_directory is not None:
                jax.profiler.stop_trace()
        """
    ).format(
        trace_directory=str(trace_directory) if trace_directory is not None else None,
        memory_profile_path=str(memory_profile_path) if memory_profile_path is not None else None,
        bgen_path=str(baseline_paths.bgen_path),
        sample_path=str(baseline_paths.sample_path),
        phenotype_path=str(phenotype_path),
        phenotype_name=phenotype_name,
        output_prefix=str(output_prefix),
        covariate_path=str(baseline_paths.covariate_path),
        prediction_path=str(prediction_path),
        trait_type=candidate.trait_type,
        device=candidate.device,
        chunk_size=candidate.chunk_size,
        variant_limit_expression=variant_limit_expression,
        staging_depth=candidate.staging_depth,
        writer_thread_count=candidate.output_writer_thread_count,
        writer_queue_depth=candidate.output_writer_queue_depth,
        bgen_tile_expression=bgen_tile_expression,
        firth_batch_expression=firth_batch_expression,
        rayon_thread_expression=rayon_thread_expression,
        jax_cache_directory_expression=jax_cache_directory_expression,
        enable_xla_autotune_cache=ENABLE_XLA_AUTOTUNE_CACHE,
        stage_timing_path=str(stage_timing_path) if stage_timing_path is not None else None,
        binary_options_expression=binary_options_expression,
    )
    return [sys.executable, "-c", child_code]


def run_logged_command(
    *,
    name: str,
    implementation: str,
    trait_type: str,
    device: str,
    command_arguments: list[str],
    environment_overrides: dict[str, str],
    log_directory: Path,
) -> TrialResult:
    """Run one command and persist stdout/stderr logs."""
    log_directory.mkdir(parents=True, exist_ok=True)
    stdout_log_path = log_directory / f"{name}.stdout.log"
    stderr_log_path = log_directory / f"{name}.stderr.log"
    environment = dict(os.environ)
    environment.update(environment_overrides)
    start_time = time.perf_counter()
    completed_process = subprocess.run(
        command_arguments,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    wall_time_seconds = time.perf_counter() - start_time
    stdout_log_path.write_text(completed_process.stdout, encoding="utf-8")
    stderr_log_path.write_text(completed_process.stderr, encoding="utf-8")
    status = "success" if completed_process.returncode == 0 else "failed"
    notes = None
    if completed_process.returncode != 0:
        notes = completed_process.stderr.strip() or completed_process.stdout.strip()
    return TrialResult(
        name=name,
        implementation=implementation,
        trait_type=trait_type,
        device=device,
        status=status,
        wall_time_seconds=wall_time_seconds,
        output_row_count=None,
        stdout_log_path=str(stdout_log_path),
        stderr_log_path=str(stderr_log_path),
        command_arguments=command_arguments,
        environment_overrides=environment_overrides,
        notes=notes,
    )


def run_g_trial(
    *,
    name: str,
    baseline_paths: typing.Any,
    candidate: Step2Candidate,
    output_directory: Path,
    log_directory: Path,
    cache_directory: Path,
    variant_limit: int | None,
    emit_stage_timings: bool,
    trace_directory: Path | None = None,
    memory_profile_path: Path | None = None,
) -> TrialResult:
    """Run one g trial in a fresh Python process."""
    output_prefix = output_directory / name
    stage_timing_path = output_directory / f"{name}.stage_timings.json" if emit_stage_timings else None
    command_arguments = build_g_step2_child_command(
        baseline_paths=baseline_paths,
        candidate=candidate,
        output_prefix=output_prefix,
        variant_limit=variant_limit,
        cache_directory=cache_directory,
        stage_timing_path=stage_timing_path,
        trace_directory=trace_directory,
        memory_profile_path=memory_profile_path,
    )
    environment_overrides = build_g_trial_environment(
        candidate=candidate,
        cache_directory=cache_directory,
        stage_timing_path=stage_timing_path,
    )
    result = run_logged_command(
        name=name,
        implementation="g",
        trait_type=candidate.trait_type,
        device=candidate.device,
        command_arguments=command_arguments,
        environment_overrides=environment_overrides,
        log_directory=log_directory,
    )
    output_row_count = None
    output_path = None
    device_diagnostics = None
    if result.status == "success":
        output_payload = json.loads(Path(result.stdout_log_path).read_text(encoding="utf-8").strip().splitlines()[-1])
        output_row_count = int(output_payload["output_row_count"])
        output_path = str(output_payload["output_path"])
        device_diagnostics = {
            "jax_devices": output_payload.get("jax_devices"),
            "jax_probe_device": output_payload.get("jax_probe_device"),
            "jax_probe_device_platform": output_payload.get("jax_probe_device_platform"),
        }
    return dataclasses.replace(
        result,
        output_row_count=output_row_count,
        output_path=output_path,
        stage_timing_path=str(stage_timing_path) if stage_timing_path is not None else None,
        device_diagnostics=device_diagnostics,
    )


def run_regenie_trial(
    *,
    name: str,
    trait_type: str,
    regenie_executable: str,
    baseline_paths: typing.Any,
    output_directory: Path,
    log_directory: Path,
) -> TrialResult:
    """Run one original REGENIE step 2 trial."""
    output_directory.mkdir(parents=True, exist_ok=True)
    output_prefix = output_directory / name
    regenie_profile_path = output_directory / f"{name}.regenie_profile.json"
    command_arguments = build_regenie_step2_command(
        trait_type=trait_type,
        regenie_executable=regenie_executable,
        baseline_paths=baseline_paths,
        output_prefix=output_prefix,
    )
    result = run_logged_command(
        name=name,
        implementation="regenie",
        trait_type=trait_type,
        device="external_cpu",
        command_arguments=command_arguments,
        environment_overrides={"REGENIE_PROFILE_JSON": str(regenie_profile_path)},
        log_directory=log_directory,
    )
    output_row_count = comparison_benchmark.count_regenie_step2_rows(output_prefix)
    output_suffix = "phenotype_binary" if trait_type == "binary" else "phenotype_continuous"
    output_path = output_prefix.parent / f"{output_prefix.name}_{output_suffix}.regenie"
    return dataclasses.replace(
        result,
        output_row_count=output_row_count,
        output_path=str(output_path) if output_path.exists() else None,
        regenie_profile_path=str(regenie_profile_path) if regenie_profile_path.exists() else None,
    )


def aggregate_trial_results(
    *,
    name: str,
    implementation: str,
    trait_type: str,
    device: str,
    warmup_count: int,
    trial_results: list[TrialResult],
) -> AggregateResult:
    """Aggregate successful measured trial results."""
    successful_trials = [
        trial_result
        for trial_result in trial_results
        if trial_result.status == "success" and trial_result.wall_time_seconds is not None
    ]
    if not successful_trials:
        return AggregateResult(
            name=name,
            implementation=implementation,
            trait_type=trait_type,
            device=device,
            status="failed",
            trial_count=len(trial_results),
            warmup_count=warmup_count,
            median_wall_time_seconds=None,
            mean_wall_time_seconds=None,
            min_wall_time_seconds=None,
            max_wall_time_seconds=None,
            standard_deviation_seconds=None,
            rows_per_second=None,
            trials=trial_results,
        )
    wall_times = [typing.cast("float", trial_result.wall_time_seconds) for trial_result in successful_trials]
    row_counts = [
        trial_result.output_row_count for trial_result in successful_trials if trial_result.output_row_count is not None
    ]
    median_wall_time = statistics.median(wall_times)
    rows_per_second = None
    if row_counts and median_wall_time > 0.0:
        rows_per_second = statistics.median(row_counts) / median_wall_time
    return AggregateResult(
        name=name,
        implementation=implementation,
        trait_type=trait_type,
        device=device,
        status="success" if len(successful_trials) == len(trial_results) else "partial",
        trial_count=len(trial_results),
        warmup_count=warmup_count,
        median_wall_time_seconds=median_wall_time,
        mean_wall_time_seconds=statistics.fmean(wall_times),
        min_wall_time_seconds=min(wall_times),
        max_wall_time_seconds=max(wall_times),
        standard_deviation_seconds=statistics.stdev(wall_times) if len(wall_times) > 1 else 0.0,
        rows_per_second=rows_per_second,
        trials=trial_results,
    )


def run_repeated_g_trials(
    *,
    name: str,
    baseline_paths: typing.Any,
    candidate: Step2Candidate,
    output_directory: Path,
    log_directory: Path,
    cache_directory: Path,
    variant_limit: int | None,
    warmup_count: int,
    trial_count: int,
    emit_stage_timings: bool,
) -> AggregateResult:
    """Warm and measure one g candidate in fresh child processes."""
    for warmup_index in range(warmup_count):
        run_g_trial(
            name=f"{name}_warmup{warmup_index:02d}",
            baseline_paths=baseline_paths,
            candidate=candidate,
            output_directory=output_directory,
            log_directory=log_directory,
            cache_directory=cache_directory,
            variant_limit=variant_limit,
            emit_stage_timings=False,
        )
    trial_results = [
        run_g_trial(
            name=f"{name}_trial{trial_index:02d}",
            baseline_paths=baseline_paths,
            candidate=candidate,
            output_directory=output_directory,
            log_directory=log_directory,
            cache_directory=cache_directory,
            variant_limit=variant_limit,
            emit_stage_timings=emit_stage_timings,
        )
        for trial_index in range(trial_count)
    ]
    return aggregate_trial_results(
        name=name,
        implementation="g",
        trait_type=candidate.trait_type,
        device=candidate.device,
        warmup_count=warmup_count,
        trial_results=trial_results,
    )


def run_repeated_regenie_trials(
    *,
    name: str,
    trait_type: str,
    regenie_executable: str,
    baseline_paths: typing.Any,
    output_directory: Path,
    log_directory: Path,
    warmup_count: int,
    trial_count: int,
) -> AggregateResult:
    """Warm and measure original REGENIE step 2."""
    for warmup_index in range(warmup_count):
        run_regenie_trial(
            name=f"{name}_warmup{warmup_index:02d}",
            trait_type=trait_type,
            regenie_executable=regenie_executable,
            baseline_paths=baseline_paths,
            output_directory=output_directory,
            log_directory=log_directory,
        )
    trial_results = [
        run_regenie_trial(
            name=f"{name}_trial{trial_index:02d}",
            trait_type=trait_type,
            regenie_executable=regenie_executable,
            baseline_paths=baseline_paths,
            output_directory=output_directory,
            log_directory=log_directory,
        )
        for trial_index in range(trial_count)
    ]
    return aggregate_trial_results(
        name=name,
        implementation="regenie",
        trait_type=trait_type,
        device="external_cpu",
        warmup_count=warmup_count,
        trial_results=trial_results,
    )


def summarize_bgen_case(case_report: typing.Any) -> BgenCandidateSummary:
    """Summarize one BGEN reader benchmark case."""
    matching_results = [
        path_result
        for path_result in case_report.path_results
        if path_result.path_mode == benchmark_bgen_reader.BenchmarkPathMode.VARIANT_MAJOR_BUFFERED.value
    ]
    if len(matching_results) != 1:
        message = "Expected exactly one variant-major buffered BGEN result."
        raise ValueError(message)
    path_result = matching_results[0]
    return BgenCandidateSummary(
        decode_tile_variant_count=case_report.decode_tile_variant_count,
        rayon_thread_count=case_report.rayon_thread_count,
        median_seconds=statistics.median(path_result.durations_seconds),
        mean_seconds=path_result.mean_seconds,
        durations_seconds=list(path_result.durations_seconds),
    )


def run_bgen_sweep(
    *,
    arguments: argparse.Namespace,
    baseline_paths: typing.Any,
    output_directory: Path,
) -> tuple[BgenCandidateSummary, ...]:
    """Run BGEN reader sweeps over decode tile size and Rayon threads."""
    parser = benchmark_bgen_reader.build_argument_parser()
    summaries: list[BgenCandidateSummary] = []
    variant_limit = arguments.variant_limit or 16_384
    sweep_directory = output_directory / "bgen_sweep"
    sweep_directory.mkdir(parents=True, exist_ok=True)
    for decode_tile_variant_count in parse_int_list(arguments.bgen_decode_tile_variant_counts):
        for rayon_thread_count in parse_int_list(arguments.rayon_thread_counts):
            benchmark_arguments = parser.parse_args(
                [
                    "--bgen",
                    str(baseline_paths.bgen_path),
                    "--sample",
                    str(baseline_paths.sample_path),
                    "--chunk-size",
                    str(arguments.bgen_benchmark_chunk_size),
                    "--variant-limit",
                    str(variant_limit),
                    "--repeat-count",
                    str(arguments.tuning_trials),
                    "--path-modes",
                    benchmark_bgen_reader.BenchmarkPathMode.VARIANT_MAJOR_BUFFERED.value,
                ]
            )
            case_report = benchmark_bgen_reader.run_case_subprocess(
                benchmark_arguments,
                arguments.bgen_benchmark_chunk_size,
                decode_tile_variant_count,
                rayon_thread_count,
                trusted_no_missing_diploid=False,
                sample_selection_mode=benchmark_bgen_reader.SampleSelectionMode.FULL,
            )
            summaries.append(summarize_bgen_case(case_report))
    summaries = sorted(summaries, key=lambda summary: (summary.median_seconds, summary.mean_seconds))
    (sweep_directory / "bgen_sweep.json").write_text(
        json.dumps([dataclasses.asdict(summary) for summary in summaries], indent=2) + "\n",
        encoding="utf-8",
    )
    return tuple(summaries)


def run_candidate_tuning(
    *,
    arguments: argparse.Namespace,
    baseline_paths: typing.Any,
    bgen_summaries: tuple[BgenCandidateSummary, ...],
    output_directory: Path,
    cache_directory: Path,
) -> dict[str, AggregateResult]:
    """Tune g candidates for each trait/device and return winners."""
    winners: dict[str, AggregateResult] = {}
    chunk_sizes = parse_int_list(arguments.chunk_sizes)
    staging_depths = parse_int_list(arguments.staging_depths)
    writer_thread_counts = parse_int_list(arguments.output_writer_thread_counts)
    queue_depth_multipliers = parse_int_list(arguments.writer_queue_depth_multipliers)
    firth_batch_sizes = parse_int_list(arguments.firth_batch_sizes)
    selected_bgen_summaries = bgen_summaries[: arguments.top_bgen_candidates]
    for trait_type in ("quantitative", "binary"):
        for device in ("cpu", "gpu"):
            candidates = build_step2_candidates(
                trait_type=trait_type,
                device=device,
                bgen_candidates=selected_bgen_summaries,
                chunk_sizes=chunk_sizes,
                staging_depths=staging_depths,
                writer_thread_counts=writer_thread_counts,
                queue_depth_multipliers=queue_depth_multipliers,
                firth_batch_sizes=firth_batch_sizes,
            )
            if arguments.smoke:
                candidates = candidates[:1]
            initial_results = [
                run_repeated_g_trials(
                    name=f"tune_{build_candidate_slug(candidate)}",
                    baseline_paths=baseline_paths,
                    candidate=candidate,
                    output_directory=output_directory / "tuning_runs",
                    log_directory=output_directory / "logs",
                    cache_directory=cache_directory,
                    variant_limit=arguments.variant_limit,
                    warmup_count=arguments.tuning_warmups,
                    trial_count=arguments.tuning_trials,
                    emit_stage_timings=False,
                )
                for candidate in candidates
            ]
            successful_initial_results = [
                result for result in initial_results if result.median_wall_time_seconds is not None
            ]
            finalists = sorted(
                successful_initial_results,
                key=lambda result: typing.cast("float", result.median_wall_time_seconds),
            )[: arguments.top_finalists]
            finalist_results: list[AggregateResult] = []
            for finalist in finalists:
                candidate = recover_candidate_from_trial(finalist.trials[0], candidates)
                finalist_results.append(
                    run_repeated_g_trials(
                        name=f"finalist_{build_candidate_slug(candidate)}",
                        baseline_paths=baseline_paths,
                        candidate=candidate,
                        output_directory=output_directory / "finalist_runs",
                        log_directory=output_directory / "logs",
                        cache_directory=cache_directory,
                        variant_limit=arguments.variant_limit,
                        warmup_count=arguments.finalist_warmups,
                        trial_count=arguments.finalist_trials,
                        emit_stage_timings=True,
                    )
                )
            if finalist_results:
                winner = sorted(
                    finalist_results,
                    key=lambda result: typing.cast("float", result.median_wall_time_seconds),
                )[0]
                winners[f"{trait_type}_{device}"] = winner
            tuning_path = output_directory / f"tuning_{trait_type}_{device}.json"
            tuning_path.write_text(
                json.dumps(
                    {
                        "initial_results": [dataclasses.asdict(result) for result in initial_results],
                        "finalist_results": [dataclasses.asdict(result) for result in finalist_results],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    return winners


def recover_candidate_from_trial(trial_result: TrialResult, candidates: tuple[Step2Candidate, ...]) -> Step2Candidate:
    """Recover the tuning candidate that produced a trial by matching its command and env."""
    for candidate in candidates:
        if build_candidate_slug(candidate) in trial_result.name:
            return candidate
    message = f"Could not recover candidate from trial {trial_result.name}."
    raise ValueError(message)


def run_headline_trials(
    *,
    arguments: argparse.Namespace,
    baseline_paths: typing.Any,
    regenie_executable: str,
    winners: dict[str, AggregateResult],
    output_directory: Path,
    cache_directory: Path,
) -> list[AggregateResult]:
    """Run headline original REGENIE and winning g configurations."""
    headline_results: list[AggregateResult] = []
    for trait_type in ("quantitative", "binary"):
        headline_results.append(
            run_repeated_regenie_trials(
                name=f"headline_regenie_{trait_type}",
                trait_type=trait_type,
                regenie_executable=regenie_executable,
                baseline_paths=baseline_paths,
                output_directory=output_directory / "headline_runs",
                log_directory=output_directory / "logs",
                warmup_count=arguments.headline_warmups,
                trial_count=arguments.headline_trials,
            )
        )
    for winner_key, winner in sorted(winners.items()):
        if not winner.trials:
            continue
        candidate = candidate_from_aggregate_name(winner_key, winner)
        headline_results.append(
            run_repeated_g_trials(
                name=f"headline_g_{winner_key}",
                baseline_paths=baseline_paths,
                candidate=candidate,
                output_directory=output_directory / "headline_runs",
                log_directory=output_directory / "logs",
                cache_directory=cache_directory,
                variant_limit=arguments.variant_limit,
                warmup_count=arguments.headline_warmups,
                trial_count=arguments.headline_trials,
                emit_stage_timings=True,
            )
        )
    return headline_results


def candidate_from_aggregate_name(winner_key: str, aggregate_result: AggregateResult) -> Step2Candidate:
    """Reconstruct a winner candidate from its child environment and command."""
    trial = aggregate_result.trials[0]
    trait_type, device = winner_key.rsplit("_", maxsplit=1)
    command = trial.command_arguments
    code = command[2] if len(command) >= 3 and command[1] == "-c" else ""

    def read_int(marker: str, default_value: int) -> int:
        marker_index = code.find(marker)
        if marker_index < 0:
            return default_value
        value_start = marker_index + len(marker)
        value_end = code.find(",", value_start)
        return int(code[value_start:value_end].strip())

    return Step2Candidate(
        trait_type=trait_type,
        device=device,
        chunk_size=read_int('"bsize": ', 8192),
        staging_depth=read_int('"g-staging-depth": ', 1),
        output_writer_thread_count=read_int('"g-writer-threads": ', 8),
        output_writer_queue_depth=read_int('"g-writer-queue-depth": ', 4),
        bgen_decode_tile_variant_count=read_int('"g-bgen-decode-tile-variant-count": ', 64),
        rayon_thread_count=read_int('"threads": ', 0) or None,
        firth_batch_size=read_int('"g-firth-batch-size": ', 1024),
    )


def build_runtime_comparisons(aggregate_results: list[AggregateResult]) -> dict[str, dict[str, float]]:
    """Build speedup/slowdown comparisons against original REGENIE."""
    by_name = {result.name: result for result in aggregate_results}
    comparisons: dict[str, dict[str, float]] = {}
    for trait_type in ("quantitative", "binary"):
        baseline = by_name.get(f"headline_regenie_{trait_type}")
        if baseline is None or baseline.median_wall_time_seconds is None:
            continue
        for result in aggregate_results:
            if (
                result.implementation != "g"
                or result.trait_type != trait_type
                or result.median_wall_time_seconds is None
            ):
                continue
            comparison_name = f"{result.name}_vs_regenie_{trait_type}"
            comparisons[comparison_name] = {
                "speedup_ratio": baseline.median_wall_time_seconds / result.median_wall_time_seconds,
                "absolute_delta_seconds": result.median_wall_time_seconds - baseline.median_wall_time_seconds,
            }
    return comparisons


REGENIE_STAGE_GROUPS: dict[str, tuple[str, ...]] = {
    "input_setup": (
        "input_file_initialization",
        "phenotype_covariate_load",
        "prediction_load",
        "step2_setup",
        "null_residual_setup",
    ),
    "bgen_decode": ("block_read", "bgen_raw_read", "bgen_decode_impute_filter"),
    "association_compute": (
        "association_compute",
        "sparse_check",
        "covariate_projection",
        "qt_score",
        "bt_score",
        "correction_candidate_check",
        "firth_correction",
        "spa_correction",
    ),
    "output": ("output_setup", "block_output"),
}

G_STAGE_GROUPS: dict[str, tuple[str, ...]] = {
    "input_setup": (
        "bgen_engine_open_index_setup",
        "sample_phenotype_covariate_alignment",
        "prediction_source_load",
        "preflight_validation",
        "chromosome_state_preparation",
        "output_run_preparation",
        "output_writer_preparation",
    ),
    "bgen_decode": ("native_engine_delivery",),
    "association_compute": (
        "host_to_device_transfer",
        "jax_compute",
        "device_to_host_materialization",
    ),
    "output": (
        "output_write",
        "single_trait_output_write",
        "multi_trait_output_write_total",
        "writer_finish_and_parquet_finalization",
        "callback_drain",
    ),
}


def read_json_file(path: str | None) -> dict[str, typing.Any] | None:
    """Read a JSON file when the path exists."""
    if path is None:
        return None
    json_path = Path(path)
    if not json_path.exists():
        return None
    return typing.cast("dict[str, typing.Any]", json.loads(json_path.read_text(encoding="utf-8")))


def collect_trial_stage_totals(trial: TrialResult) -> dict[str, float]:
    """Collect raw stage totals for one g or REGENIE trial."""
    stage_payload = read_json_file(trial.stage_timing_path)
    if stage_payload is None:
        stage_payload = read_json_file(trial.regenie_profile_path)
    if stage_payload is None:
        return {}
    return {stage_name: float(seconds) for stage_name, seconds in stage_payload.get("stage_totals_seconds", {}).items()}


def collect_stage_totals(aggregate_results: list[AggregateResult]) -> dict[str, float]:
    """Collect representative stage totals from g and REGENIE trials."""
    stage_totals: dict[str, float] = {}
    for aggregate_result in aggregate_results:
        for trial in aggregate_result.trials:
            for stage_name, seconds in collect_trial_stage_totals(trial).items():
                key = f"{aggregate_result.name}:{stage_name}"
                stage_totals[key] = seconds
    return stage_totals


def build_grouped_stage_totals(
    stage_totals: dict[str, float],
    stage_groups: dict[str, tuple[str, ...]],
) -> dict[str, float]:
    """Map raw stage totals into common comparison groups."""
    grouped_totals: dict[str, float] = {}
    for group_name, raw_stage_names in stage_groups.items():
        grouped_totals[group_name] = sum(stage_totals.get(stage_name, 0.0) for stage_name in raw_stage_names)
    return grouped_totals


def representative_stage_totals(aggregate_result: AggregateResult) -> dict[str, float]:
    """Return stage totals from the median-ish successful trial for an aggregate."""
    successful_trials = [
        trial for trial in aggregate_result.trials if trial.status == "success" and trial.wall_time_seconds is not None
    ]
    if not successful_trials:
        return {}
    sorted_trials = sorted(successful_trials, key=lambda trial: typing.cast("float", trial.wall_time_seconds))
    median_index = len(sorted_trials) // 2
    return collect_trial_stage_totals(sorted_trials[median_index])


def build_stage_comparison_rows(aggregate_results: list[AggregateResult]) -> list[dict[str, float | str]]:
    """Build stage-by-stage REGENIE versus g comparison rows."""
    rows: list[dict[str, float | str]] = []
    results_by_trait = {
        result.trait_type: result
        for result in aggregate_results
        if result.implementation == "regenie" and result.median_wall_time_seconds is not None
    }
    for regenie_trait_type, regenie_result in results_by_trait.items():
        regenie_grouped_totals = build_grouped_stage_totals(
            representative_stage_totals(regenie_result),
            REGENIE_STAGE_GROUPS,
        )
        for g_result in aggregate_results:
            if (
                g_result.implementation != "g"
                or g_result.trait_type != regenie_trait_type
                or g_result.median_wall_time_seconds is None
            ):
                continue
            g_grouped_totals = build_grouped_stage_totals(representative_stage_totals(g_result), G_STAGE_GROUPS)
            for stage_group in REGENIE_STAGE_GROUPS:
                regenie_seconds = regenie_grouped_totals.get(stage_group, 0.0)
                g_seconds = g_grouped_totals.get(stage_group, 0.0)
                speedup_ratio = regenie_seconds / g_seconds if g_seconds > 0.0 else 0.0
                rows.append(
                    {
                        "trait_type": regenie_trait_type,
                        "g_device": g_result.device,
                        "stage_group": stage_group,
                        "regenie_seconds": regenie_seconds,
                        "g_seconds": g_seconds,
                        "g_speedup_ratio": speedup_ratio,
                    }
                )
    return rows


def build_algorithmic_findings(stage_comparison_rows: list[dict[str, float | str]]) -> list[str]:
    """Explain likely source-level reasons for measured stage gaps."""
    findings: list[str] = []
    for row in stage_comparison_rows:
        stage_group = str(row["stage_group"])
        speedup_ratio = float(row["g_speedup_ratio"])
        trait_type = str(row["trait_type"])
        device = str(row["g_device"])
        if stage_group == "bgen_decode" and speedup_ratio > 1.0:
            findings.append(
                f"{trait_type}/{device}: g is faster in BGEN delivery, consistent with its native buffered decoder "
                "and larger chunk pipeline versus REGENIE's per-block BGEN read/decode path."
            )
        elif stage_group == "association_compute" and speedup_ratio > 1.0:
            findings.append(
                f"{trait_type}/{device}: g is faster in association compute, consistent with vectorized chunk scoring "
                "and GPU/JAX execution when available."
            )
        elif stage_group == "output" and speedup_ratio > 1.0:
            findings.append(
                f"{trait_type}/{device}: g is faster in output, consistent with chunked Arrow/Parquet writes instead "
                "of REGENIE's per-variant text formatting loop."
            )
        elif stage_group == "bgen_decode" and speedup_ratio > 0.0 and speedup_ratio < 1.0:
            findings.append(
                f"{trait_type}/{device}: REGENIE remains faster in measured BGEN delivery; the g timing group maps "
                "to inclusive native engine delivery, so it can also include downstream compute and correction work. "
                "Split that timer before attributing the whole gap to decoding."
            )
        elif stage_group == "association_compute" and speedup_ratio > 0.0 and speedup_ratio < 1.0:
            findings.append(
                f"{trait_type}/{device}: REGENIE remains faster in association compute; likely gaps include "
                "REGENIE's sparse genotype checks, OpenMP score/correction scheduling, and correction thresholds "
                "that avoid expensive fallback work on most variants."
            )
        elif stage_group == "output" and speedup_ratio > 0.0 and speedup_ratio < 1.0:
            findings.append(
                f"{trait_type}/{device}: REGENIE remains faster in output; likely gaps include g callback draining, "
                "Parquet finalization, and writer queue overhead for this workload."
            )
        elif speedup_ratio > 1.0:
            findings.append(
                f"{trait_type}/{device}: g is faster in {stage_group}; this points to lower orchestration overhead "
                "or more compact data movement in the measured g path."
            )
        elif speedup_ratio > 0.0 and speedup_ratio < 1.0:
            findings.append(
                f"{trait_type}/{device}: REGENIE remains faster in {stage_group}; likely gaps include REGENIE's "
                "sparse genotype paths, OpenMP scheduling, null-model reuse, correction thresholds, or lower "
                "formatting overhead for this workload."
            )
        elif speedup_ratio == 1.0:
            findings.append(f"{trait_type}/{device}: {stage_group} is effectively tied between g and REGENIE.")
    return sorted(set(findings))


def build_summary_markdown(
    *,
    aggregate_results: list[AggregateResult],
    comparisons: dict[str, dict[str, float]],
    stage_totals: dict[str, float],
    stage_comparison_rows: list[dict[str, float | str]],
    algorithmic_findings: list[str],
) -> str:
    """Build the human-readable campaign summary."""
    lines = ["# Landau Deep REGENIE Step 2 Profile", ""]
    lines.append("## Headline Runtimes")
    lines.append("")
    lines.append("| name | trait | device | median s | mean s | min s | max s | std s | rows/s |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for result in aggregate_results:
        lines.append(
            "| "
            f"{result.name} | {result.trait_type} | {result.device} | "
            f"{format_optional_float(result.median_wall_time_seconds)} | "
            f"{format_optional_float(result.mean_wall_time_seconds)} | "
            f"{format_optional_float(result.min_wall_time_seconds)} | "
            f"{format_optional_float(result.max_wall_time_seconds)} | "
            f"{format_optional_float(result.standard_deviation_seconds)} | "
            f"{format_optional_float(result.rows_per_second)} |"
        )
    lines.extend(["", "## Runtime Comparisons", ""])
    if comparisons:
        for comparison_name, comparison in comparisons.items():
            lines.append(
                f"- {comparison_name}: speedup={comparison['speedup_ratio']:.4f}x, "
                f"delta={comparison['absolute_delta_seconds']:.4f}s"
            )
    else:
        lines.append("- No successful direct comparisons were available.")
    lines.extend(["", "## Stage Comparisons", ""])
    if stage_comparison_rows:
        lines.append(
            "_Stage timings are inclusive cumulative seconds from each profiler, so nested stages can overlap and "
            "can exceed headline wall time._"
        )
        lines.append("")
        lines.append("| trait | g device | stage | REGENIE s | g s | g speedup |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: |")
        for row in stage_comparison_rows:
            lines.append(
                "| "
                f"{row['trait_type']} | {row['g_device']} | {row['stage_group']} | "
                f"{float(row['regenie_seconds']):.6f} | "
                f"{float(row['g_seconds']):.6f} | "
                f"{float(row['g_speedup_ratio']):.4f}x |"
            )
    else:
        lines.append("- No paired stage timing JSON files were available.")
    lines.extend(["", "## Algorithmic Findings", ""])
    if algorithmic_findings:
        for finding in algorithmic_findings:
            lines.append(f"- {finding}")
    else:
        lines.append("- Re-run with successful REGENIE and g profile JSON files to generate source-level findings.")
    lines.extend(["", "## Ranked Bottlenecks", ""])
    if stage_totals:
        for stage_name, seconds in sorted(stage_totals.items(), key=lambda item: item[1], reverse=True)[:20]:
            lines.append(f"- {stage_name}: {seconds:.6f}s")
    else:
        lines.append("- No stage timing JSON files were available.")
    lines.extend(["", "## Next Optimization Targets", ""])
    if stage_totals:
        for stage_name, seconds in sorted(stage_totals.items(), key=lambda item: item[1], reverse=True)[:5]:
            lines.append(f"- Reduce `{stage_name}` first; it is one of the largest measured wall-time shares.")
    else:
        lines.append("- Re-run with successful g diagnostic trials to rank measured stage shares.")
    return "\n".join(lines) + "\n"


def format_optional_float(value: float | None) -> str:
    """Format optional floats for markdown tables."""
    if value is None:
        return ""
    return f"{value:.6f}"


def run_deep_profiles(
    *,
    baseline_paths: typing.Any,
    winners: dict[str, AggregateResult],
    output_directory: Path,
    cache_directory: Path,
    variant_limit: int | None,
) -> dict[str, typing.Any]:
    """Run optional profiler commands for representative g winners."""
    profile_directory = output_directory / "deep_profiles"
    profile_directory.mkdir(parents=True, exist_ok=True)
    results: dict[str, typing.Any] = {
        "criterion_bgen": command_output(
            ["cargo", "bench", "--bench", "bgen_read"],
            environment_overrides={"RUSTFLAGS": "-C target-cpu=native"},
        ),
        "sampling_profiles": [],
    }
    for winner_key, winner in sorted(winners.items()):
        if not winner.trials:
            continue
        candidate = candidate_from_aggregate_name(winner_key, winner)
        trace_directory = profile_directory / f"{winner_key}_jax_trace"
        memory_profile_path = profile_directory / f"{winner_key}_device_memory.prof"
        profile_result = run_g_trial(
            name=f"profile_{winner_key}_jax",
            baseline_paths=baseline_paths,
            candidate=candidate,
            output_directory=profile_directory,
            log_directory=output_directory / "logs",
            cache_directory=cache_directory,
            variant_limit=variant_limit,
            emit_stage_timings=True,
            trace_directory=trace_directory,
            memory_profile_path=memory_profile_path,
        )
        results["sampling_profiles"].append(dataclasses.asdict(profile_result))
        if shutil.which("py-spy") is not None:
            speedscope_path = profile_directory / f"{winner_key}.speedscope.json"
            command_arguments = [
                "py-spy",
                "record",
                "--format",
                "speedscope",
                "--output",
                str(speedscope_path),
                "--",
                *profile_result.command_arguments,
            ]
            sampling_result = run_logged_command(
                name=f"profile_{winner_key}_py_spy",
                implementation="py-spy",
                trait_type=candidate.trait_type,
                device=candidate.device,
                command_arguments=command_arguments,
                environment_overrides=profile_result.environment_overrides,
                log_directory=output_directory / "logs",
            )
            results["sampling_profiles"].append(dataclasses.asdict(sampling_result))
        else:
            cprofile_script_path = profile_directory / f"{winner_key}_cprofile_child.py"
            cprofile_output_path = profile_directory / f"{winner_key}.cprofile"
            cprofile_script_path.write_text(profile_result.command_arguments[2], encoding="utf-8")
            cprofile_result = run_logged_command(
                name=f"profile_{winner_key}_cprofile",
                implementation="cProfile",
                trait_type=candidate.trait_type,
                device=candidate.device,
                command_arguments=[
                    sys.executable,
                    "-m",
                    "cProfile",
                    "-o",
                    str(cprofile_output_path),
                    str(cprofile_script_path),
                ],
                environment_overrides=profile_result.environment_overrides,
                log_directory=output_directory / "logs",
            )
            results["sampling_profiles"].append(dataclasses.asdict(cprofile_result))
        if shutil.which("perf") is not None:
            perf_path = profile_directory / f"{winner_key}.perf.data"
            command_arguments = [
                "perf",
                "record",
                "-g",
                "-o",
                str(perf_path),
                "--",
                *profile_result.command_arguments,
            ]
            perf_result = run_logged_command(
                name=f"profile_{winner_key}_perf",
                implementation="perf",
                trait_type=candidate.trait_type,
                device=candidate.device,
                command_arguments=command_arguments,
                environment_overrides=profile_result.environment_overrides,
                log_directory=output_directory / "logs",
            )
            results["sampling_profiles"].append(dataclasses.asdict(perf_result))
    return results


def apply_smoke_overrides(arguments: argparse.Namespace) -> None:
    """Reduce the campaign size for a landau smoke profile."""
    if not arguments.smoke:
        return
    if arguments.variant_limit is None:
        arguments.variant_limit = 1000
    arguments.chunk_sizes = "2048"
    arguments.staging_depths = "1"
    arguments.output_writer_thread_counts = "1"
    arguments.writer_queue_depth_multipliers = "1"
    arguments.firth_batch_sizes = "32"
    arguments.bgen_decode_tile_variant_counts = "64"
    arguments.rayon_thread_counts = "1"
    arguments.top_bgen_candidates = 1
    arguments.top_finalists = 1
    arguments.tuning_warmups = 0
    arguments.tuning_trials = 1
    arguments.finalist_warmups = 0
    arguments.finalist_trials = 1
    arguments.headline_warmups = 0
    arguments.headline_trials = 1


def main() -> None:
    """Run the landau deep profiling campaign."""
    arguments = build_argument_parser().parse_args()
    apply_smoke_overrides(arguments)
    output_directory = build_output_directory(arguments)
    output_directory.mkdir(parents=True, exist_ok=True)
    log_directory = output_directory / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    cache_directory = output_directory / "jax_cache"
    cache_directory.mkdir(parents=True, exist_ok=True)

    baseline_paths = baseline_benchmark.build_baseline_paths()
    baseline_benchmark.validate_input_files(baseline_paths)
    regenie_executable = baseline_benchmark.resolve_required_executable("REGENIE_BIN", "regenie")
    setup_results = ensure_prediction_lists(
        baseline_paths=baseline_paths,
        regenie_executable=regenie_executable,
        log_directory=log_directory,
    )
    preflight_metadata = collect_environment_metadata(baseline_paths)
    (output_directory / "preflight.json").write_text(json.dumps(preflight_metadata, indent=2) + "\n", encoding="utf-8")

    bgen_summaries = run_bgen_sweep(
        arguments=arguments,
        baseline_paths=baseline_paths,
        output_directory=output_directory,
    )
    winners = run_candidate_tuning(
        arguments=arguments,
        baseline_paths=baseline_paths,
        bgen_summaries=bgen_summaries,
        output_directory=output_directory,
        cache_directory=cache_directory,
    )
    headline_results = run_headline_trials(
        arguments=arguments,
        baseline_paths=baseline_paths,
        regenie_executable=regenie_executable,
        winners=winners,
        output_directory=output_directory,
        cache_directory=cache_directory,
    )
    deep_profile_results: dict[str, typing.Any] = {}
    if not arguments.skip_deep_profiles:
        deep_profile_results = run_deep_profiles(
            baseline_paths=baseline_paths,
            winners=winners,
            output_directory=output_directory,
            cache_directory=cache_directory,
            variant_limit=arguments.variant_limit,
        )
    comparisons = build_runtime_comparisons(headline_results)
    stage_totals = collect_stage_totals(headline_results)
    stage_comparison_rows = build_stage_comparison_rows(headline_results)
    algorithmic_findings = build_algorithmic_findings(stage_comparison_rows)
    summary_payload = {
        "preflight": preflight_metadata,
        "setup_results": [dataclasses.asdict(result) for result in setup_results],
        "bgen_summaries": [dataclasses.asdict(summary) for summary in bgen_summaries],
        "winners": {key: dataclasses.asdict(value) for key, value in winners.items()},
        "headline_results": [dataclasses.asdict(result) for result in headline_results],
        "comparisons": comparisons,
        "stage_totals": stage_totals,
        "stage_comparisons": stage_comparison_rows,
        "algorithmic_findings": algorithmic_findings,
        "deep_profiles": deep_profile_results,
    }
    (output_directory / "summary.json").write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    (output_directory / "summary.md").write_text(
        build_summary_markdown(
            aggregate_results=headline_results,
            comparisons=comparisons,
            stage_totals=stage_totals,
            stage_comparison_rows=stage_comparison_rows,
            algorithmic_findings=algorithmic_findings,
        ),
        encoding="utf-8",
    )
    print(f"Wrote deep profile artifacts under {output_directory}")


if __name__ == "__main__":
    main()
