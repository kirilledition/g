#!/usr/bin/env python3
"""Benchmark output-stage timings across finalization, phenotype count, and bsize."""

from __future__ import annotations

import dataclasses
import json
import os
import statistics
import time
import typing
from pathlib import Path

import hydra
import polars as pl

import tooling.configuration as tooling_configuration
from g import api, types
from tooling.common import hydra_arguments as tooling_hydra_arguments
from tooling.common import hydra_compat as tooling_hydra_compat
from tooling.common import paths as tooling_paths
from tooling.common import reports as tooling_reports
from tooling.common import sweeps as tooling_sweeps

if typing.TYPE_CHECKING:
    import omegaconf

DEFAULT_DATA_DIRECTORY = tooling_paths.configured_data_directory()
DEFAULT_OUTPUT_DIRECTORY = Path("data/benchmarks/output_stages")
DEFAULT_SINGLE_PHENOTYPE_NAME = "phenotype_continuous"
OUTPUT_STAGE_TIMING_FILE_NAME = "output_stage_timings.json"
DEVICE_TO_HOST_MATERIALIZATION_METRIC = "device_to_host_materialization"
PYTHON_OUTPUT_WRITE_METRIC = "python_output_write"
RUST_OUTPUT_METADATA_CLONE_METRIC = "rust_output_metadata_clone"
RUST_OUTPUT_RESULT_BUFFER_COPY_METRIC = "rust_output_result_buffer_copy"
RUST_OUTPUT_ENQUEUE_METRIC = "rust_output_enqueue"
RUST_OUTPUT_WRITER_RECORD_BATCH_BUILD_METRIC = "rust_output_writer_record_batch_build"
RUST_OUTPUT_WRITER_ARROW_FILE_WRITE_METRIC = "rust_output_writer_arrow_file_write"
RUST_OUTPUT_WRITER_TOTAL_METRIC = "rust_output_writer_total"
RUST_OUTPUT_FINALIZATION_TOTAL_METRIC = "rust_output_finalization_total"
BRIDGE_RESIDUAL_METRIC = "bridge_residual"
MEASURED_OUTPUT_PATH_METRIC = "measured_output_path"
MEBIBYTE = 1024.0 * 1024.0


@dataclasses.dataclass(frozen=True)
class BenchmarkArguments:
    """Resolved output-stage benchmark parameters.

    Attributes:
        data_dir: Input data directory.
        output_dir: Benchmark output directory.
        device: Runtime device.
        small_bsize: Small bsize value.
        large_bsize: Large bsize value.
        extra_bsizes: Comma-separated extra bsize values.
        many_phenotype_count: Trait count for many-phenotype cases.
        variant_limit: Optional variant cap.
        trials: Measured trials per case.
        writer_thread_counts: Comma-separated writer thread-count values.
        writer_queue_depth_multipliers: Comma-separated queue-depth multipliers.
        chunks_per_arrow_file_values: Comma-separated chunk grouping values.
        arrow_compressions: Comma-separated Arrow compression codecs.
        json_summary_path: Optional JSON summary path.
        markdown_summary_path: Optional Markdown summary path.
        enable_jax_trace: Whether to capture representative JAX traces.
        jax_trace_case_limit: Maximum traced case count.

    """

    data_dir: Path
    output_dir: Path
    device: str
    small_bsize: int
    large_bsize: int
    extra_bsizes: str
    many_phenotype_count: int
    variant_limit: int | None
    trials: int
    writer_thread_counts: str
    writer_queue_depth_multipliers: str
    chunks_per_arrow_file_values: str
    arrow_compressions: str
    json_summary_path: Path | None
    markdown_summary_path: Path | None
    enable_jax_trace: bool
    jax_trace_case_limit: int


@dataclasses.dataclass(frozen=True)
class OutputHandoffTimingMetrics:
    """Derived timing metrics for the JAX-host-Rust-Arrow output handoff.

    Attributes:
        seconds_by_metric: Absolute seconds for each measured or derived metric.
        wall_time_percentage_by_metric: Metric share of total trial wall time.
        output_path_percentage_by_metric: Metric share of measured output path time.

    """

    seconds_by_metric: dict[str, float]
    wall_time_percentage_by_metric: dict[str, float]
    output_path_percentage_by_metric: dict[str, float]


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
        writer_thread_count: Background writer thread count.
        writer_queue_depth: Background writer queue depth.
        chunks_per_arrow_file: Number of compute chunks per Arrow file.
        arrow_compression: Arrow IPC chunk compression codec.

    """

    name: str
    finalize_parquet: bool
    phenotype_count: int
    chunk_size: int
    writer_thread_count: int
    writer_queue_depth: int
    chunks_per_arrow_file: int
    arrow_compression: types.ArrowCompression


@dataclasses.dataclass(frozen=True)
class TrialResult:
    """Measured output-stage benchmark result.

    Attributes:
        case_name: Benchmark case name.
        trial_index: Measured trial index.
        finalize_parquet: Whether final Parquet output was enabled.
        phenotype_count: Number of phenotypes written.
        chunk_size: REGENIE bsize value.
        writer_thread_count: Background writer thread count.
        writer_queue_depth: Background writer queue depth.
        chunks_per_arrow_file: Number of compute chunks per Arrow file.
        arrow_compression: Arrow IPC chunk compression codec.
        wall_time_seconds: End-to-end API wall time.
        python_stage_timing_path: Python stage timing JSON path.
        rust_stage_timing_paths: Rust output timing JSON paths.
        output_run_directories: Output run directories.
        final_parquet_paths: Final Parquet files, when written.
        chunk_file_count: Total Arrow chunk file count across phenotypes.
        chunk_bytes: Total Arrow chunk bytes across phenotypes.
        final_parquet_bytes: Total final Parquet bytes across phenotypes, when written.
        rust_output_metrics: Summed Rust output counters across phenotype run directories.
        handoff_timing: Derived handoff timing metrics.
        jax_trace_directory: Optional JAX profiler trace artifact directory.

    """

    case_name: str
    trial_index: int
    finalize_parquet: bool
    phenotype_count: int
    chunk_size: int
    writer_thread_count: int
    writer_queue_depth: int
    chunks_per_arrow_file: int
    arrow_compression: str
    wall_time_seconds: float
    python_stage_timing_path: str
    rust_stage_timing_paths: tuple[str, ...]
    output_run_directories: tuple[str, ...]
    final_parquet_paths: tuple[str, ...]
    chunk_file_count: int
    chunk_bytes: int
    final_parquet_bytes: int | None
    handoff_timing: OutputHandoffTimingMetrics
    rust_output_metrics: dict[str, float] = dataclasses.field(default_factory=dict)
    jax_trace_directory: str | None = None


def parse_positive_int_list(text: str) -> tuple[int, ...]:
    """Parse a comma-separated list of positive integers."""
    return tooling_sweeps.parse_positive_integer_list(text, "positive integer list")


def parse_arrow_compressions(text: str) -> tuple[types.ArrowCompression, ...]:
    """Parse a comma-separated list of Arrow compression codecs."""
    values = tuple(types.ArrowCompression(field.strip()) for field in text.split(",") if field.strip())
    if not values:
        message = "At least one Arrow compression codec is required."
        raise ValueError(message)
    return values


def build_benchmark_cases(
    *,
    small_chunk_size: int,
    large_chunk_size: int,
    many_phenotype_count: int,
    writer_thread_counts: tuple[int, ...],
    writer_queue_depth_multipliers: tuple[int, ...],
    chunks_per_arrow_file_values: tuple[int, ...],
    arrow_compressions: tuple[types.ArrowCompression, ...],
    extra_chunk_sizes: tuple[int, ...] = (),
) -> tuple[BenchmarkCase, ...]:
    """Build the output benchmark matrix."""
    cases: list[BenchmarkCase] = []
    chunk_size_modes = (
        ("small_bsize", small_chunk_size),
        ("large_bsize", large_chunk_size),
        *((f"extra_bsize_{chunk_size}", chunk_size) for chunk_size in extra_chunk_sizes),
    )
    for finalize_parquet in (False, True):
        output_mode = "parquet_final" if finalize_parquet else "arrow_chunks"
        for phenotype_count in (1, many_phenotype_count):
            phenotype_mode = "single_phenotype" if phenotype_count == 1 else f"{phenotype_count}_phenotypes"
            for chunk_size_name, chunk_size in chunk_size_modes:
                for writer_thread_count in writer_thread_counts:
                    for writer_queue_depth_multiplier in writer_queue_depth_multipliers:
                        writer_queue_depth = writer_thread_count * writer_queue_depth_multiplier
                        for chunks_per_arrow_file in chunks_per_arrow_file_values:
                            for arrow_compression in arrow_compressions:
                                cases.append(
                                    BenchmarkCase(
                                        name=(
                                            f"{output_mode}_{phenotype_mode}_{chunk_size_name}_{chunk_size}_"
                                            f"writer{writer_thread_count}_queue{writer_queue_depth}_"
                                            f"chunks{chunks_per_arrow_file}_{arrow_compression.value}"
                                        ),
                                        finalize_parquet=finalize_parquet,
                                        phenotype_count=phenotype_count,
                                        chunk_size=chunk_size,
                                        writer_thread_count=writer_thread_count,
                                        writer_queue_depth=writer_queue_depth,
                                        chunks_per_arrow_file=chunks_per_arrow_file,
                                        arrow_compression=arrow_compression,
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


def load_stage_totals(stage_timing_path: Path) -> dict[str, float]:
    """Load stage totals from a timing JSON file."""
    if not stage_timing_path.exists():
        return {}
    timing_payload = json.loads(stage_timing_path.read_text(encoding="utf-8"))
    stage_totals = timing_payload.get("stage_totals_seconds", {})
    if not isinstance(stage_totals, dict):
        return {}
    return {
        str(stage_name): float(stage_seconds)
        for stage_name, stage_seconds in stage_totals.items()
        if isinstance(stage_seconds, int | float)
    }


def load_output_metrics(stage_timing_path: Path) -> dict[str, float]:
    """Load output counters from a Rust timing JSON file."""
    if not stage_timing_path.exists():
        return {}
    timing_payload = json.loads(stage_timing_path.read_text(encoding="utf-8"))
    output_metrics = timing_payload.get("output_metrics", {})
    if not isinstance(output_metrics, dict):
        return {}
    return {
        str(metric_name): float(metric_value)
        for metric_name, metric_value in output_metrics.items()
        if isinstance(metric_value, int | float)
    }


def sum_stage_totals(stage_timing_paths: tuple[Path, ...]) -> dict[str, float]:
    """Sum stage totals across zero or more timing JSON files."""
    total_stage_seconds: dict[str, float] = {}
    for stage_timing_path in stage_timing_paths:
        for stage_name, stage_seconds in load_stage_totals(stage_timing_path).items():
            total_stage_seconds[stage_name] = total_stage_seconds.get(stage_name, 0.0) + stage_seconds
    return total_stage_seconds


def sum_output_metrics(stage_timing_paths: tuple[Path, ...]) -> dict[str, float]:
    """Sum Rust output counters across zero or more timing JSON files."""
    total_metrics: dict[str, float] = {}
    for stage_timing_path in stage_timing_paths:
        for metric_name, metric_value in load_output_metrics(stage_timing_path).items():
            total_metrics[metric_name] = total_metrics.get(metric_name, 0.0) + metric_value
    return total_metrics


def build_metric_percentages(seconds_by_metric: dict[str, float], denominator_seconds: float) -> dict[str, float]:
    """Build percentage shares for each metric against one denominator."""
    if denominator_seconds <= 0.0:
        return dict.fromkeys(seconds_by_metric, 0.0)
    return {
        metric_name: (metric_seconds / denominator_seconds) * 100.0
        for metric_name, metric_seconds in seconds_by_metric.items()
    }


def build_output_handoff_timing_metrics(
    *,
    python_stage_timing_path: Path,
    rust_stage_timing_paths: tuple[Path, ...],
    wall_time_seconds: float,
) -> OutputHandoffTimingMetrics:
    """Build derived output handoff timing metrics for one trial."""
    python_stage_totals = load_stage_totals(python_stage_timing_path)
    rust_stage_totals = sum_stage_totals(rust_stage_timing_paths)
    rust_metadata_clone_seconds = rust_stage_totals.get(RUST_OUTPUT_METADATA_CLONE_METRIC, 0.0)
    rust_result_buffer_copy_seconds = rust_stage_totals.get(RUST_OUTPUT_RESULT_BUFFER_COPY_METRIC, 0.0)
    rust_enqueue_seconds = rust_stage_totals.get(RUST_OUTPUT_ENQUEUE_METRIC, 0.0)
    python_output_write_seconds = python_stage_totals.get("output_write", 0.0)
    bridge_residual_seconds = python_output_write_seconds - (
        rust_metadata_clone_seconds + rust_result_buffer_copy_seconds + rust_enqueue_seconds
    )
    rust_writer_total_seconds = rust_stage_totals.get(RUST_OUTPUT_WRITER_TOTAL_METRIC, 0.0)
    rust_finalization_total_seconds = rust_stage_totals.get(RUST_OUTPUT_FINALIZATION_TOTAL_METRIC, 0.0)
    measured_output_path_seconds = (
        python_stage_totals.get(DEVICE_TO_HOST_MATERIALIZATION_METRIC, 0.0)
        + python_output_write_seconds
        + rust_writer_total_seconds
        + rust_finalization_total_seconds
    )
    seconds_by_metric = {
        DEVICE_TO_HOST_MATERIALIZATION_METRIC: python_stage_totals.get(
            DEVICE_TO_HOST_MATERIALIZATION_METRIC,
            0.0,
        ),
        PYTHON_OUTPUT_WRITE_METRIC: python_output_write_seconds,
        RUST_OUTPUT_METADATA_CLONE_METRIC: rust_metadata_clone_seconds,
        RUST_OUTPUT_RESULT_BUFFER_COPY_METRIC: rust_result_buffer_copy_seconds,
        RUST_OUTPUT_ENQUEUE_METRIC: rust_enqueue_seconds,
        RUST_OUTPUT_WRITER_RECORD_BATCH_BUILD_METRIC: rust_stage_totals.get(
            RUST_OUTPUT_WRITER_RECORD_BATCH_BUILD_METRIC,
            0.0,
        ),
        RUST_OUTPUT_WRITER_ARROW_FILE_WRITE_METRIC: rust_stage_totals.get(
            RUST_OUTPUT_WRITER_ARROW_FILE_WRITE_METRIC,
            0.0,
        ),
        RUST_OUTPUT_WRITER_TOTAL_METRIC: rust_writer_total_seconds,
        RUST_OUTPUT_FINALIZATION_TOTAL_METRIC: rust_finalization_total_seconds,
        BRIDGE_RESIDUAL_METRIC: bridge_residual_seconds,
        MEASURED_OUTPUT_PATH_METRIC: measured_output_path_seconds,
    }
    for stage_name, stage_seconds in rust_stage_totals.items():
        if stage_name.startswith("rust_output_writer_") or stage_name.startswith("rust_output_finalization_"):
            seconds_by_metric.setdefault(stage_name, stage_seconds)
    return OutputHandoffTimingMetrics(
        seconds_by_metric=seconds_by_metric,
        wall_time_percentage_by_metric=build_metric_percentages(seconds_by_metric, wall_time_seconds),
        output_path_percentage_by_metric=build_metric_percentages(seconds_by_metric, measured_output_path_seconds),
    )


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
    variant_limit: int | None,
    benchmark_case: BenchmarkCase,
    trial_index: int,
    enable_jax_trace: bool,
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
        "g-writer-threads": benchmark_case.writer_thread_count,
        "g-writer-queue-depth": benchmark_case.writer_queue_depth,
        "g-output-chunks-per-arrow-file": benchmark_case.chunks_per_arrow_file,
        "g-output-arrow-compression": benchmark_case.arrow_compression.value,
    }
    if benchmark_case.phenotype_count == 1:
        options["phenoCol"] = phenotype_resources.phenotype_names[0]
    else:
        options["phenoColList"] = ",".join(phenotype_resources.phenotype_names)

    jax_trace_directory = output_directory / "jax_traces" / trial_name if enable_jax_trace else None
    if jax_trace_directory is not None:
        import jax

        jax_trace_directory.mkdir(parents=True, exist_ok=True)
        jax.profiler.start_trace(str(jax_trace_directory))

    start_time = time.perf_counter()
    try:
        artifacts = api.regenie.from_options(options)
    finally:
        if jax_trace_directory is not None:
            jax.profiler.stop_trace()
    wall_time_seconds = time.perf_counter() - start_time
    output_metrics = collect_trial_output_metrics(flatten_artifacts(artifacts))
    rust_stage_timing_path_strings = typing.cast("tuple[str, ...]", output_metrics["rust_stage_timing_paths"])
    rust_stage_timing_paths = tuple(Path(path) for path in rust_stage_timing_path_strings)
    handoff_timing = build_output_handoff_timing_metrics(
        python_stage_timing_path=python_stage_timing_path,
        rust_stage_timing_paths=rust_stage_timing_paths,
        wall_time_seconds=wall_time_seconds,
    )
    return TrialResult(
        case_name=benchmark_case.name,
        trial_index=trial_index,
        finalize_parquet=benchmark_case.finalize_parquet,
        phenotype_count=benchmark_case.phenotype_count,
        chunk_size=benchmark_case.chunk_size,
        writer_thread_count=benchmark_case.writer_thread_count,
        writer_queue_depth=benchmark_case.writer_queue_depth,
        chunks_per_arrow_file=benchmark_case.chunks_per_arrow_file,
        arrow_compression=benchmark_case.arrow_compression.value,
        wall_time_seconds=wall_time_seconds,
        python_stage_timing_path=str(python_stage_timing_path),
        rust_stage_timing_paths=rust_stage_timing_path_strings,
        output_run_directories=typing.cast("tuple[str, ...]", output_metrics["output_run_directories"]),
        final_parquet_paths=typing.cast("tuple[str, ...]", output_metrics["final_parquet_paths"]),
        chunk_file_count=int(output_metrics["chunk_file_count"]),
        chunk_bytes=int(output_metrics["chunk_bytes"]),
        final_parquet_bytes=typing.cast("int | None", output_metrics["final_parquet_bytes"]),
        handoff_timing=handoff_timing,
        rust_output_metrics=sum_output_metrics(rust_stage_timing_paths),
        jax_trace_directory=None if jax_trace_directory is None else str(jax_trace_directory),
    )


def summarize_metric_maps(metric_maps: tuple[dict[str, float], ...]) -> dict[str, float]:
    """Return per-metric means across trial metric maps."""
    metric_names = sorted({metric_name for metric_map in metric_maps for metric_name in metric_map})
    return {
        metric_name: statistics.fmean(metric_map.get(metric_name, 0.0) for metric_map in metric_maps)
        for metric_name in metric_names
    }


def calculate_rate_per_second(numerator: float, denominator_seconds: float) -> float:
    """Calculate a throughput rate with a zero-safe denominator."""
    if denominator_seconds <= 0.0:
        return 0.0
    return numerator / denominator_seconds


def summarize_trial_group(trial_results: tuple[TrialResult, ...]) -> dict[str, typing.Any]:
    """Summarize repeated trials for one benchmark case."""
    wall_time_values = [trial_result.wall_time_seconds for trial_result in trial_results]
    first_trial = trial_results[0]
    mean_wall_time_seconds = statistics.fmean(wall_time_values)
    mean_chunk_bytes = statistics.fmean([trial_result.chunk_bytes for trial_result in trial_results])
    mean_chunk_file_count = statistics.fmean([trial_result.chunk_file_count for trial_result in trial_results])
    mean_final_parquet_bytes = statistics.fmean(
        [trial_result.final_parquet_bytes or 0 for trial_result in trial_results]
    )
    mean_rust_output_metrics = summarize_metric_maps(
        tuple(trial_result.rust_output_metrics for trial_result in trial_results)
    )
    mean_writer_seconds = statistics.fmean(
        [
            trial_result.handoff_timing.seconds_by_metric.get(RUST_OUTPUT_WRITER_TOTAL_METRIC, 0.0)
            for trial_result in trial_results
        ]
    )
    mean_finalization_seconds = statistics.fmean(
        [
            trial_result.handoff_timing.seconds_by_metric.get(RUST_OUTPUT_FINALIZATION_TOTAL_METRIC, 0.0)
            for trial_result in trial_results
        ]
    )
    mean_writer_row_count = mean_rust_output_metrics.get("writer_row_count", 0.0)
    mean_writer_arrow_array_memory_bytes = mean_rust_output_metrics.get("writer_arrow_array_memory_bytes", 0.0)
    mean_writer_arrow_file_bytes = mean_rust_output_metrics.get("writer_arrow_file_bytes", mean_chunk_bytes)
    mean_finalization_row_count = mean_rust_output_metrics.get("finalization_row_count", 0.0)
    mean_finalization_arrow_file_bytes = mean_rust_output_metrics.get("finalization_arrow_file_bytes", mean_chunk_bytes)
    mean_finalization_parquet_file_bytes = mean_rust_output_metrics.get(
        "finalization_parquet_file_bytes",
        mean_final_parquet_bytes,
    )
    return {
        "case_name": first_trial.case_name,
        "finalize_parquet": first_trial.finalize_parquet,
        "phenotype_count": first_trial.phenotype_count,
        "chunk_size": first_trial.chunk_size,
        "writer_thread_count": first_trial.writer_thread_count,
        "writer_queue_depth": first_trial.writer_queue_depth,
        "chunks_per_arrow_file": first_trial.chunks_per_arrow_file,
        "arrow_compression": first_trial.arrow_compression,
        "trial_count": len(trial_results),
        "mean_wall_time_seconds": mean_wall_time_seconds,
        "median_wall_time_seconds": statistics.median(wall_time_values),
        "min_wall_time_seconds": min(wall_time_values),
        "max_wall_time_seconds": max(wall_time_values),
        "mean_chunk_file_count": mean_chunk_file_count,
        "mean_chunk_bytes": mean_chunk_bytes,
        "mean_final_parquet_bytes": mean_final_parquet_bytes,
        "mean_rust_output_metrics": mean_rust_output_metrics,
        "mean_throughput": {
            "wall_rows_per_second": calculate_rate_per_second(mean_writer_row_count, mean_wall_time_seconds),
            "wall_arrow_physical_mib_per_second": calculate_rate_per_second(
                mean_writer_arrow_file_bytes / MEBIBYTE,
                mean_wall_time_seconds,
            ),
            "writer_rows_per_second": calculate_rate_per_second(mean_writer_row_count, mean_writer_seconds),
            "writer_logical_mib_per_second": calculate_rate_per_second(
                mean_writer_arrow_array_memory_bytes / MEBIBYTE,
                mean_writer_seconds,
            ),
            "writer_physical_mib_per_second": calculate_rate_per_second(
                mean_writer_arrow_file_bytes / MEBIBYTE,
                mean_writer_seconds,
            ),
            "writer_files_per_second": calculate_rate_per_second(mean_chunk_file_count, mean_writer_seconds),
            "finalization_rows_per_second": calculate_rate_per_second(
                mean_finalization_row_count,
                mean_finalization_seconds,
            ),
            "finalization_arrow_physical_mib_per_second": calculate_rate_per_second(
                mean_finalization_arrow_file_bytes / MEBIBYTE,
                mean_finalization_seconds,
            ),
            "finalization_parquet_physical_mib_per_second": calculate_rate_per_second(
                mean_finalization_parquet_file_bytes / MEBIBYTE,
                mean_finalization_seconds,
            ),
        },
        "mean_handoff_timing_seconds": summarize_metric_maps(
            tuple(trial_result.handoff_timing.seconds_by_metric for trial_result in trial_results)
        ),
        "mean_handoff_wall_time_percentages": summarize_metric_maps(
            tuple(trial_result.handoff_timing.wall_time_percentage_by_metric for trial_result in trial_results)
        ),
        "mean_handoff_output_path_percentages": summarize_metric_maps(
            tuple(trial_result.handoff_timing.output_path_percentage_by_metric for trial_result in trial_results)
        ),
    }


def build_ranked_bottlenecks(case_summary: dict[str, typing.Any], limit: int = 8) -> list[dict[str, float | str]]:
    """Rank the largest measured timing buckets for one summarized case."""
    mean_timing_seconds = typing.cast("dict[str, float]", case_summary["mean_handoff_timing_seconds"])
    candidate_seconds = {
        metric_name: metric_seconds
        for metric_name, metric_seconds in mean_timing_seconds.items()
        if metric_name != MEASURED_OUTPUT_PATH_METRIC and metric_seconds > 0.0
    }
    wall_time_seconds = float(case_summary["mean_wall_time_seconds"])
    return [
        {
            "metric": metric_name,
            "seconds": metric_seconds,
            "wall_time_percentage": calculate_rate_per_second(metric_seconds * 100.0, wall_time_seconds),
        }
        for metric_name, metric_seconds in sorted(candidate_seconds.items(), key=lambda item: item[1], reverse=True)[
            :limit
        ]
    ]


def choose_recommended_optimization(case_summaries: list[dict[str, typing.Any]]) -> str:
    """Choose a short recommendation from the observed case summaries."""
    if not case_summaries:
        return "No benchmark cases completed."
    largest_case = max(case_summaries, key=lambda summary: float(summary["mean_wall_time_seconds"]))
    bottlenecks = build_ranked_bottlenecks(largest_case, limit=1)
    if not bottlenecks:
        return "No dominant measured output bottleneck was observed."
    top_metric = str(bottlenecks[0]["metric"])
    if "finalization" in top_metric:
        return "Prioritize direct Parquet or removing Arrow reread finalization overhead."
    if top_metric == RUST_OUTPUT_WRITER_TOTAL_METRIC:
        return (
            "Prioritize writer internals, then use the substage timings to choose array build versus Arrow write work."
        )
    if "writer_arrow" in top_metric:
        return "Prioritize Arrow writer throughput, chunk/file layout, and compression policy experiments."
    if "writer_result_arrays" in top_metric or "writer_metadata_arrays" in top_metric:
        return "Prioritize writer-side array construction and wide multi-trait output experiments."
    return f"Prioritize the largest measured bucket: {top_metric}."


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
    case_summaries = [
        summarize_trial_group(case_trial_results)
        for case_trial_results in grouped_results.values()
        if case_trial_results
    ]
    return {
        "metadata": {
            "device": device.value,
            "pid": os.getpid(),
        },
        "case_summaries": case_summaries,
        "ranked_bottlenecks": {
            str(case_summary["case_name"]): build_ranked_bottlenecks(case_summary) for case_summary in case_summaries
        },
        "recommendation": choose_recommended_optimization(case_summaries),
        "trial_results": [dataclasses.asdict(trial_result) for trial_result in trial_results],
    }


def build_markdown_summary(summary: dict[str, typing.Any]) -> str:
    """Build a short Markdown summary for benchmark output."""
    case_summaries = typing.cast("list[dict[str, typing.Any]]", summary["case_summaries"])
    ranked_cases = sorted(case_summaries, key=lambda case_summary: float(case_summary["mean_wall_time_seconds"]))
    lines = [
        "# Output Hotspot Profile Summary",
        "",
        f"Recommendation: {summary['recommendation']}",
        "",
        "## Fastest Cases",
        "",
    ]
    for case_summary in ranked_cases[:5]:
        throughput = typing.cast("dict[str, float]", case_summary["mean_throughput"])
        lines.append(
            "- "
            f"{case_summary['case_name']}: "
            f"{float(case_summary['mean_wall_time_seconds']):.3f}s wall, "
            f"{throughput['writer_physical_mib_per_second']:.2f} MiB/s writer physical"
        )
    lines.extend(["", "## Slowest Bottlenecks", ""])
    ranked_bottlenecks = typing.cast("dict[str, list[dict[str, typing.Any]]]", summary["ranked_bottlenecks"])
    for case_summary in ranked_cases[-5:]:
        case_name = str(case_summary["case_name"])
        bottlenecks = ranked_bottlenecks.get(case_name, [])
        top_bottleneck = bottlenecks[0] if bottlenecks else {"metric": "none", "seconds": 0.0}
        lines.append(
            "- "
            f"{case_name}: "
            f"{float(case_summary['mean_wall_time_seconds']):.3f}s wall, "
            f"top bucket {top_bottleneck['metric']}={float(top_bottleneck['seconds']):.3f}s"
        )
    lines.append("")
    return "\n".join(lines)


def build_arguments_from_config(config: omegaconf.DictConfig) -> BenchmarkArguments:
    """Build benchmark parameters from a composed Hydra config."""
    tool_values = tooling_hydra_arguments.tool_config_to_dictionary(config)
    return BenchmarkArguments(
        data_dir=Path(str(tool_values["data_dir"])),
        output_dir=Path(str(tool_values["output_dir"])),
        device=str(tool_values["device"]),
        small_bsize=int(tool_values["small_bsize"]),
        large_bsize=int(tool_values["large_bsize"]),
        extra_bsizes=tooling_hydra_arguments.comma_join(tool_values["extra_bsizes"]),
        many_phenotype_count=int(tool_values["many_phenotype_count"]),
        variant_limit=tooling_hydra_arguments.integer_or_none(tool_values.get("variant_limit")),
        trials=int(tool_values["trials"]),
        writer_thread_counts=tooling_hydra_arguments.comma_join(tool_values["writer_thread_counts"]),
        writer_queue_depth_multipliers=tooling_hydra_arguments.comma_join(
            tool_values["writer_queue_depth_multipliers"]
        ),
        chunks_per_arrow_file_values=tooling_hydra_arguments.comma_join(
            tool_values["chunks_per_arrow_file_values"]
        ),
        arrow_compressions=tooling_hydra_arguments.comma_join(tool_values["arrow_compressions"]),
        json_summary_path=tooling_hydra_arguments.path_or_none(tool_values.get("json_summary_path")),
        markdown_summary_path=tooling_hydra_arguments.path_or_none(tool_values.get("markdown_summary_path")),
        enable_jax_trace=bool(tool_values["enable_jax_trace"]),
        jax_trace_case_limit=int(tool_values["jax_trace_case_limit"]),
    )


def build_arguments_from_overrides(overrides: typing.Sequence[str] | None = None) -> BenchmarkArguments:
    """Compose the output-stage config and return resolved parameters."""
    config = tooling_configuration.compose_config(config_name="benchmark_output_stages", overrides=overrides)
    return build_arguments_from_config(config)


def run_tool(arguments: BenchmarkArguments) -> None:
    """Run the output-stage benchmark matrix."""
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    device = types.Device(arguments.device)
    benchmark_cases = build_benchmark_cases(
        small_chunk_size=int(arguments.small_bsize),
        large_chunk_size=int(arguments.large_bsize),
        many_phenotype_count=int(arguments.many_phenotype_count),
        writer_thread_counts=parse_positive_int_list(arguments.writer_thread_counts),
        writer_queue_depth_multipliers=parse_positive_int_list(arguments.writer_queue_depth_multipliers),
        chunks_per_arrow_file_values=parse_positive_int_list(arguments.chunks_per_arrow_file_values),
        arrow_compressions=parse_arrow_compressions(arguments.arrow_compressions),
        extra_chunk_sizes=parse_positive_int_list(arguments.extra_bsizes) if arguments.extra_bsizes else (),
    )
    trial_results = tuple(
        run_trial(
            data_directory=arguments.data_dir,
            output_directory=arguments.output_dir,
            device=device,
            variant_limit=arguments.variant_limit,
            benchmark_case=benchmark_case,
            trial_index=trial_index,
            enable_jax_trace=(
                arguments.enable_jax_trace
                and trial_index == 0
                and benchmark_case_index < int(arguments.jax_trace_case_limit)
            ),
        )
        for benchmark_case_index, benchmark_case in enumerate(benchmark_cases)
        for trial_index in range(arguments.trials)
    )
    summary = build_summary(device=device, benchmark_cases=benchmark_cases, trial_results=trial_results)
    summary_path = arguments.json_summary_path or (arguments.output_dir / "summary.json")
    tooling_reports.write_json_report(summary_path, summary, sort_keys=True)
    markdown_summary_path = arguments.markdown_summary_path or (arguments.output_dir / "summary.md")
    tooling_reports.write_markdown_report(markdown_summary_path, build_markdown_summary(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote Markdown summary: {markdown_summary_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="benchmark_output_stages")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Run the output-stage benchmark through Hydra."""
    run_tool(build_arguments_from_config(config))


def main() -> None:
    """Run the output-stage benchmark matrix."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()


if __name__ == "__main__":
    main()
