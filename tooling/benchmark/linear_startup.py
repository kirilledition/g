#!/usr/bin/env python3
"""Benchmark REGENIE step 2 in fresh Python processes."""

from __future__ import annotations

import dataclasses
import json
import os
import statistics
import subprocess
import sys
import textwrap
import time
import typing
from dataclasses import dataclass
from pathlib import Path

import hydra

from tooling.common import hydra_arguments as tooling_hydra_arguments
from tooling.common import hydra_compat as tooling_hydra_compat

if typing.TYPE_CHECKING:
    import omegaconf

DEFAULT_DATA_DIRECTORY = Path("data")
DEFAULT_OUTPUT_DIRECTORY = Path("data/benchmarks/regenie2_linear_fresh_process")
SINGLE_PHENOTYPE_NAME = "phenotype_continuous"


@dataclass(frozen=True)
class BenchmarkInputs:
    """Input paths and phenotype columns used by the benchmark."""

    bgen_path: Path
    sample_path: Path
    phenotype_path: Path
    phenotype_names: tuple[str, ...]
    covariate_path: Path
    prediction_list_path: Path


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
    mode: str = "fresh_process"
    phenotype_count: int = 1
    child_wall_time_seconds: float | None = None
    stage_timing_path: str | None = None
    output_paths: list[str] | None = None


@dataclass(frozen=True)
class BenchmarkSummary:
    """Aggregate summary for one fresh-process benchmark run."""

    device: str
    chunk_size: int
    finalize_parquet: bool
    output_writer_thread_count: int
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
    mode: str = "fresh_process"
    phenotype_count: int = 1
    mean_child_wall_time_seconds: float | None = None
    stage_timing_paths: list[str] = dataclasses.field(default_factory=list)


@dataclass(frozen=True)
class BenchmarkReport:
    """Combined fresh-process and same-process benchmark report."""

    fresh_process: BenchmarkSummary
    same_process: BenchmarkSummary
    comparisons: dict[str, float]


@dataclass(frozen=True)
class LinearStartupArguments:
    """Resolved fresh-process benchmark parameters.

    Attributes:
        device: Execution device.
        chunk_size: Variants per chunk.
        finalize_parquet: Whether each trial finalizes Parquet output.
        output_writer_thread_count: Background writer thread count.
        trials: Measured fresh-process trial count.
        warmup_trials: Unreported fresh-process warmup count.
        same_process_trials: Measured same-process trial count.
        same_process_warmup_trials: Unreported same-process warmup count.
        multi_phenotype_count: Number of cloned quantitative phenotypes.
        multi_phenotype_sample_mode: Multi-trait sample handling mode.
        emit_stage_timings: Whether measured trials write stage timing JSON.
        data_dir: Input data directory.
        output_dir: Benchmark output directory.
        json_summary_path: Optional explicit JSON summary path.

    """

    device: str
    chunk_size: int
    finalize_parquet: bool
    output_writer_thread_count: int
    trials: int
    warmup_trials: int
    same_process_trials: int
    same_process_warmup_trials: int
    multi_phenotype_count: int
    multi_phenotype_sample_mode: str
    emit_stage_timings: bool
    data_dir: Path
    output_dir: Path
    json_summary_path: Path | None


def prepare_benchmark_inputs(
    *,
    data_directory: Path,
    output_directory: Path,
    phenotype_count: int,
) -> BenchmarkInputs:
    """Resolve benchmark inputs, generating cloned phenotype files when requested."""
    if phenotype_count < 1:
        message = "--multi-phenotype-count must be at least 1."
        raise ValueError(message)
    bgen_path = data_directory / "1kg_chr22_full.bgen"
    sample_path = data_directory / "1kg_chr22_full.sample"
    phenotype_path = data_directory / "pheno_cont.txt"
    covariate_path = data_directory / "covariates.txt"
    prediction_list_path = data_directory / "baselines/regenie_step1_qt_pred.list"
    if phenotype_count == 1:
        return BenchmarkInputs(
            bgen_path=bgen_path,
            sample_path=sample_path,
            phenotype_path=phenotype_path,
            phenotype_names=(SINGLE_PHENOTYPE_NAME,),
            covariate_path=covariate_path,
            prediction_list_path=prediction_list_path,
        )

    generated_directory = output_directory / "generated_inputs"
    generated_directory.mkdir(parents=True, exist_ok=True)
    phenotype_names = tuple(
        f"{SINGLE_PHENOTYPE_NAME}_{phenotype_index + 1}" for phenotype_index in range(phenotype_count)
    )
    generated_phenotype_path = generated_directory / f"pheno_cont_{phenotype_count}_traits.txt"
    write_cloned_phenotype_table(
        source_phenotype_path=phenotype_path,
        generated_phenotype_path=generated_phenotype_path,
        phenotype_names=phenotype_names,
    )
    generated_prediction_list_path = generated_directory / f"regenie_step1_qt_{phenotype_count}_traits_pred.list"
    write_cloned_prediction_list(
        source_prediction_list_path=prediction_list_path,
        generated_prediction_list_path=generated_prediction_list_path,
        phenotype_names=phenotype_names,
    )
    return BenchmarkInputs(
        bgen_path=bgen_path,
        sample_path=sample_path,
        phenotype_path=generated_phenotype_path,
        phenotype_names=phenotype_names,
        covariate_path=covariate_path,
        prediction_list_path=generated_prediction_list_path,
    )


def write_cloned_phenotype_table(
    *,
    source_phenotype_path: Path,
    generated_phenotype_path: Path,
    phenotype_names: tuple[str, ...],
) -> None:
    """Write a quantitative phenotype table with cloned trait columns."""
    source_lines = source_phenotype_path.read_text(encoding="utf-8").splitlines()
    if not source_lines:
        message = f"Phenotype file is empty: {source_phenotype_path}"
        raise ValueError(message)
    header_values = source_lines[0].split("\t")
    try:
        family_identifier_index = header_values.index("FID")
        individual_identifier_index = header_values.index("IID")
        phenotype_index = header_values.index(SINGLE_PHENOTYPE_NAME)
    except ValueError as error:
        message = f"Phenotype file must contain FID, IID, and {SINGLE_PHENOTYPE_NAME}: {source_phenotype_path}"
        raise ValueError(message) from error

    generated_lines = ["\t".join(("FID", "IID", *phenotype_names))]
    for line_number, source_line in enumerate(source_lines[1:], start=2):
        if not source_line:
            continue
        row_values = source_line.split("\t")
        required_index = max(family_identifier_index, individual_identifier_index, phenotype_index)
        if len(row_values) <= required_index:
            message = f"Phenotype file line {line_number} has fewer columns than the header."
            raise ValueError(message)
        phenotype_value = row_values[phenotype_index]
        generated_lines.append(
            "\t".join(
                (
                    row_values[family_identifier_index],
                    row_values[individual_identifier_index],
                    *(phenotype_value for _ in phenotype_names),
                )
            )
        )
    generated_phenotype_path.write_text("\n".join(generated_lines) + "\n", encoding="utf-8")


def write_cloned_prediction_list(
    *,
    source_prediction_list_path: Path,
    generated_prediction_list_path: Path,
    phenotype_names: tuple[str, ...],
) -> None:
    """Write a prediction list that maps cloned traits to the source LOCO file."""
    source_lines = [
        source_line
        for source_line in source_prediction_list_path.read_text(encoding="utf-8").splitlines()
        if source_line.strip()
    ]
    if not source_lines:
        message = f"Prediction list is empty: {source_prediction_list_path}"
        raise ValueError(message)
    first_line_fields = source_lines[0].split()
    if len(first_line_fields) != 2:
        message = f"Prediction list line must contain phenotype and LOCO path: {source_prediction_list_path}"
        raise ValueError(message)
    raw_loco_path = Path(first_line_fields[1])
    loco_path = raw_loco_path if raw_loco_path.is_absolute() else source_prediction_list_path.parent / raw_loco_path
    generated_prediction_list_path.write_text(
        "".join(f"{phenotype_name} {loco_path}\n" for phenotype_name in phenotype_names),
        encoding="utf-8",
    )


def build_regenie_options(
    *,
    benchmark_inputs: BenchmarkInputs,
    output_path: Path,
    device: str,
    chunk_size: int,
    finalize_parquet: bool,
    output_writer_thread_count: int,
    stage_timing_path: Path | None,
    multi_phenotype_sample_mode: str,
    disable_telemetry: bool,
) -> dict[str, object]:
    """Build g API options for one benchmark trial."""
    regenie_options: dict[str, object] = {
        "step": 2,
        "qt": True,
        "bgen": str(benchmark_inputs.bgen_path),
        "sample": str(benchmark_inputs.sample_path),
        "phenoFile": str(benchmark_inputs.phenotype_path),
        "out": str(output_path),
        "covarFile": str(benchmark_inputs.covariate_path),
        "covarColList": "age,sex",
        "pred": str(benchmark_inputs.prediction_list_path),
        "g-device": device,
        "bsize": chunk_size,
        "g-output-format": "parquet" if finalize_parquet else "arrow",
        "g-writer-threads": output_writer_thread_count,
    }
    if len(benchmark_inputs.phenotype_names) == 1:
        regenie_options["phenoCol"] = benchmark_inputs.phenotype_names[0]
    else:
        regenie_options["phenoColList"] = ",".join(benchmark_inputs.phenotype_names)
        regenie_options["g-multi-phenotype-sample-mode"] = multi_phenotype_sample_mode
    if stage_timing_path is not None:
        regenie_options["g-stage-timings-json"] = str(stage_timing_path)
    if disable_telemetry:
        regenie_options["g-telemetry"] = "off"
    return regenie_options


def build_child_metrics_code() -> str:
    """Return inline child helper code for collecting output metrics."""
    return textwrap.dedent(
        """
        def collect_artifact_metrics(artifacts):
            artifact_values = artifacts.phenotype_artifacts or (artifacts,)
            output_paths = []
            output_row_count = 0
            chunk_file_count = 0
            chunk_bytes = 0
            final_parquet_bytes = 0
            for artifact in artifact_values:
                if artifact.final_parquet is not None:
                    output_paths.append(str(artifact.final_parquet))
                    output_row_count += pl.scan_parquet(artifact.final_parquet).select(pl.len()).collect().item()
                    final_parquet_bytes += artifact.final_parquet.stat().st_size
                if artifact.output_run_directory is None:
                    continue
                output_run_directory = Path(artifact.output_run_directory)
                chunk_file_paths = sorted((output_run_directory / "chunks").glob("*.arrow"))
                part_file_paths = sorted((output_run_directory / "parts").glob("*.parquet"))
                chunk_file_count += len(chunk_file_paths)
                chunk_bytes += sum(chunk_file_path.stat().st_size for chunk_file_path in chunk_file_paths)
                for chunk_file_path in chunk_file_paths:
                    if artifact.final_parquet is None:
                        output_paths.append(str(chunk_file_path))
                        output_row_count += pl.scan_ipc(chunk_file_path).select(pl.len()).collect().item()
                for part_file_path in part_file_paths:
                    output_paths.append(str(part_file_path))
                    output_row_count += pl.scan_parquet(part_file_path).select(pl.len()).collect().item()
                    final_parquet_bytes += part_file_path.stat().st_size
            if not output_paths:
                raise RuntimeError("No readable output artifacts were produced.")
            return {
                "output_path": output_paths[0],
                "output_paths": output_paths,
                "output_row_count": int(output_row_count),
                "chunk_file_count": int(chunk_file_count),
                "chunk_bytes": int(chunk_bytes),
                "final_parquet_bytes": int(final_parquet_bytes) if final_parquet_bytes else None,
            }
        """
    )


def collect_artifact_metrics(artifacts: typing.Any) -> dict[str, object]:
    """Collect row and byte metrics from single- or multi-phenotype artifacts."""
    import polars as pl

    def count_rows(lazy_frame: typing.Any) -> int:
        collected_frame = lazy_frame.select(pl.len()).collect()
        return int(collected_frame.item())

    artifact_values = artifacts.phenotype_artifacts or (artifacts,)
    output_paths: list[str] = []
    output_row_count = 0
    chunk_file_count = 0
    chunk_bytes = 0
    final_parquet_bytes = 0
    for artifact in artifact_values:
        if artifact.final_parquet is not None:
            output_paths.append(str(artifact.final_parquet))
            output_row_count += count_rows(pl.scan_parquet(artifact.final_parquet))
            final_parquet_bytes += artifact.final_parquet.stat().st_size
        if artifact.output_run_directory is None:
            continue
        output_run_directory = Path(artifact.output_run_directory)
        chunk_file_paths = sorted((output_run_directory / "chunks").glob("*.arrow"))
        part_file_paths = sorted((output_run_directory / "parts").glob("*.parquet"))
        chunk_file_count += len(chunk_file_paths)
        chunk_bytes += sum(chunk_file_path.stat().st_size for chunk_file_path in chunk_file_paths)
        for chunk_file_path in chunk_file_paths:
            if artifact.final_parquet is None:
                output_paths.append(str(chunk_file_path))
                output_row_count += count_rows(pl.scan_ipc(chunk_file_path))
        for part_file_path in part_file_paths:
            output_paths.append(str(part_file_path))
            output_row_count += count_rows(pl.scan_parquet(part_file_path))
            final_parquet_bytes += part_file_path.stat().st_size
    if not output_paths:
        message = "No readable output artifacts were produced."
        raise RuntimeError(message)
    return {
        "output_path": output_paths[0],
        "output_paths": output_paths,
        "output_row_count": int(output_row_count),
        "chunk_file_count": int(chunk_file_count),
        "chunk_bytes": int(chunk_bytes),
        "final_parquet_bytes": int(final_parquet_bytes) if final_parquet_bytes else None,
    }


def payload_int(payload: dict[str, object], key: str) -> int:
    """Read an integer-compatible payload field."""
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        message = f"Expected numeric payload field: {key}"
        raise TypeError(message)
    return int(value)


def payload_optional_int(payload: dict[str, object], key: str) -> int | None:
    """Read an optional integer-compatible payload field."""
    value = payload[key]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        message = f"Expected optional numeric payload field: {key}"
        raise TypeError(message)
    return int(value)


def payload_string_list(payload: dict[str, object], key: str) -> list[str]:
    """Read a list of strings from a payload field."""
    value = payload.get(key, [])
    if not isinstance(value, list):
        message = f"Expected list payload field: {key}"
        raise TypeError(message)
    return [str(item) for item in value]


def build_child_command(
    *,
    benchmark_inputs: BenchmarkInputs,
    output_path: Path,
    device: str,
    chunk_size: int,
    finalize_parquet: bool,
    output_writer_thread_count: int,
    stage_timing_path: Path | None = None,
    multi_phenotype_sample_mode: str = "complete-case",
) -> list[str]:
    """Build the child Python command for one isolated trial."""
    regenie_options = build_regenie_options(
        benchmark_inputs=benchmark_inputs,
        output_path=output_path,
        device=device,
        chunk_size=chunk_size,
        finalize_parquet=finalize_parquet,
        output_writer_thread_count=output_writer_thread_count,
        stage_timing_path=stage_timing_path,
        multi_phenotype_sample_mode=multi_phenotype_sample_mode,
        disable_telemetry=False,
    )
    child_imports_code = textwrap.dedent(
        """
        import json
        import time
        from pathlib import Path

        import polars as pl

        from g import api
        """
    )
    child_run_code = textwrap.dedent(
        f"""
        start_time = time.perf_counter()
        artifacts = api.regenie.from_options({regenie_options!r})
        child_wall_time_seconds = time.perf_counter() - start_time
        metrics = collect_artifact_metrics(artifacts)
        metrics["child_wall_time_seconds"] = child_wall_time_seconds
        print(json.dumps(metrics))
        """
    )
    child_code = "\n".join(
        (
            child_imports_code,
            build_child_metrics_code(),
            child_run_code,
        )
    )
    return [sys.executable, "-c", child_code]


def run_fresh_process_trial(
    *,
    trial_index: int,
    benchmark_inputs: BenchmarkInputs,
    output_directory: Path,
    device: str,
    chunk_size: int,
    finalize_parquet: bool,
    output_writer_thread_count: int,
    emit_stage_timings: bool = False,
    multi_phenotype_sample_mode: str = "complete-case",
) -> TrialResult:
    """Run one isolated fresh-process trial."""
    stage_timing_path = build_stage_timing_path(
        output_directory=output_directory,
        mode="fresh_process",
        device=device,
        trial_index=trial_index,
        emit_stage_timings=emit_stage_timings,
    )
    output_prefix = output_directory / (
        f"{device}_finalize{int(finalize_parquet)}_"
        f"chunk{chunk_size}_"
        f"writer{output_writer_thread_count}_"
        f"phenotypes{len(benchmark_inputs.phenotype_names)}_"
        f"trial{trial_index:02d}"
    )
    command_arguments = build_child_command(
        benchmark_inputs=benchmark_inputs,
        output_path=output_prefix,
        device=device,
        chunk_size=chunk_size,
        finalize_parquet=finalize_parquet,
        output_writer_thread_count=output_writer_thread_count,
        stage_timing_path=stage_timing_path,
        multi_phenotype_sample_mode=multi_phenotype_sample_mode,
    )
    child_environment = os.environ.copy()
    child_environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    child_environment.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".50")
    start_time = time.perf_counter()
    completed_process = subprocess.run(
        command_arguments,
        check=True,
        capture_output=True,
        text=True,
        env=child_environment,
    )
    wall_time_seconds = time.perf_counter() - start_time
    result_line = completed_process.stdout.strip().splitlines()[-1]
    result_payload = json.loads(result_line)
    return TrialResult(
        trial_index=trial_index,
        wall_time_seconds=wall_time_seconds,
        output_path=str(result_payload["output_path"]),
        output_row_count=int(result_payload["output_row_count"]),
        chunk_file_count=int(result_payload["chunk_file_count"]),
        chunk_bytes=int(result_payload["chunk_bytes"]),
        final_parquet_bytes=(
            int(result_payload["final_parquet_bytes"]) if result_payload["final_parquet_bytes"] is not None else None
        ),
        mode="fresh_process",
        phenotype_count=len(benchmark_inputs.phenotype_names),
        child_wall_time_seconds=float(result_payload["child_wall_time_seconds"]),
        stage_timing_path=str(stage_timing_path) if stage_timing_path is not None else None,
        output_paths=[str(output_path) for output_path in result_payload.get("output_paths", [])],
    )


def build_stage_timing_path(
    *,
    output_directory: Path,
    mode: str,
    device: str,
    trial_index: int,
    emit_stage_timings: bool,
) -> Path | None:
    """Build a measured-trial stage timing path when requested."""
    if not emit_stage_timings or trial_index < 0:
        return None
    stage_timing_directory = output_directory / "stage_timings"
    stage_timing_directory.mkdir(parents=True, exist_ok=True)
    return stage_timing_directory / f"{mode}_{device}_trial{trial_index:02d}.json"


def run_same_process_trial(
    *,
    api_module: typing.Any,
    trial_index: int,
    benchmark_inputs: BenchmarkInputs,
    output_directory: Path,
    device: str,
    chunk_size: int,
    finalize_parquet: bool,
    output_writer_thread_count: int,
    emit_stage_timings: bool = False,
    multi_phenotype_sample_mode: str = "complete-case",
) -> TrialResult:
    """Run one repeated trial inside the current Python process."""
    stage_timing_path = build_stage_timing_path(
        output_directory=output_directory,
        mode="same_process",
        device=device,
        trial_index=trial_index,
        emit_stage_timings=emit_stage_timings,
    )
    output_prefix = output_directory / (
        f"same_process_{device}_finalize{int(finalize_parquet)}_"
        f"chunk{chunk_size}_"
        f"writer{output_writer_thread_count}_"
        f"phenotypes{len(benchmark_inputs.phenotype_names)}_"
        f"trial{trial_index:02d}"
    )
    regenie_options = build_regenie_options(
        benchmark_inputs=benchmark_inputs,
        output_path=output_prefix,
        device=device,
        chunk_size=chunk_size,
        finalize_parquet=finalize_parquet,
        output_writer_thread_count=output_writer_thread_count,
        stage_timing_path=stage_timing_path,
        multi_phenotype_sample_mode=multi_phenotype_sample_mode,
        disable_telemetry=True,
    )
    start_time = time.perf_counter()
    artifacts = api_module.regenie.from_options(regenie_options)
    wall_time_seconds = time.perf_counter() - start_time
    result_payload = collect_artifact_metrics(artifacts)
    return TrialResult(
        trial_index=trial_index,
        wall_time_seconds=wall_time_seconds,
        output_path=str(result_payload["output_path"]),
        output_row_count=payload_int(result_payload, "output_row_count"),
        chunk_file_count=payload_int(result_payload, "chunk_file_count"),
        chunk_bytes=payload_int(result_payload, "chunk_bytes"),
        final_parquet_bytes=payload_optional_int(result_payload, "final_parquet_bytes"),
        mode="same_process",
        phenotype_count=len(benchmark_inputs.phenotype_names),
        child_wall_time_seconds=None,
        stage_timing_path=str(stage_timing_path) if stage_timing_path is not None else None,
        output_paths=payload_string_list(result_payload, "output_paths"),
    )


def run_same_process_trials(
    *,
    benchmark_inputs: BenchmarkInputs,
    output_directory: Path,
    device: str,
    chunk_size: int,
    finalize_parquet: bool,
    output_writer_thread_count: int,
    warmup_count: int,
    trial_count: int,
    emit_stage_timings: bool = False,
    multi_phenotype_sample_mode: str = "complete-case",
) -> list[TrialResult]:
    """Run warm and measured trials inside one Python process."""
    from g import api as g_api

    for warmup_index in range(warmup_count):
        run_same_process_trial(
            api_module=g_api,
            trial_index=-(warmup_index + 1),
            benchmark_inputs=benchmark_inputs,
            output_directory=output_directory,
            device=device,
            chunk_size=chunk_size,
            finalize_parquet=finalize_parquet,
            output_writer_thread_count=output_writer_thread_count,
            emit_stage_timings=False,
            multi_phenotype_sample_mode=multi_phenotype_sample_mode,
        )
    return [
        run_same_process_trial(
            api_module=g_api,
            trial_index=trial_index,
            benchmark_inputs=benchmark_inputs,
            output_directory=output_directory,
            device=device,
            chunk_size=chunk_size,
            finalize_parquet=finalize_parquet,
            output_writer_thread_count=output_writer_thread_count,
            emit_stage_timings=emit_stage_timings,
            multi_phenotype_sample_mode=multi_phenotype_sample_mode,
        )
        for trial_index in range(trial_count)
    ]


def run_fresh_process_trials(
    *,
    benchmark_inputs: BenchmarkInputs,
    output_directory: Path,
    device: str,
    chunk_size: int,
    finalize_parquet: bool,
    output_writer_thread_count: int,
    warmup_count: int,
    trial_count: int,
    emit_stage_timings: bool = False,
    multi_phenotype_sample_mode: str = "complete-case",
) -> list[TrialResult]:
    """Run warm and measured trials in isolated Python child processes."""
    for warmup_index in range(warmup_count):
        run_fresh_process_trial(
            trial_index=-(warmup_index + 1),
            benchmark_inputs=benchmark_inputs,
            output_directory=output_directory,
            device=device,
            chunk_size=chunk_size,
            finalize_parquet=finalize_parquet,
            output_writer_thread_count=output_writer_thread_count,
            emit_stage_timings=False,
            multi_phenotype_sample_mode=multi_phenotype_sample_mode,
        )
    return [
        run_fresh_process_trial(
            trial_index=trial_index,
            benchmark_inputs=benchmark_inputs,
            output_directory=output_directory,
            device=device,
            chunk_size=chunk_size,
            finalize_parquet=finalize_parquet,
            output_writer_thread_count=output_writer_thread_count,
            emit_stage_timings=emit_stage_timings,
            multi_phenotype_sample_mode=multi_phenotype_sample_mode,
        )
        for trial_index in range(trial_count)
    ]


def require_positive_count(argument_name: str, argument_value: int) -> None:
    """Reject non-positive trial or phenotype counts."""
    if argument_value < 1:
        message = f"{argument_name} must be at least 1."
        raise ValueError(message)


def require_non_negative_count(argument_name: str, argument_value: int) -> None:
    """Reject negative counts."""
    if argument_value < 0:
        message = f"{argument_name} must be non-negative."
        raise ValueError(message)


def build_summary(
    *,
    device: str,
    chunk_size: int,
    finalize_parquet: bool,
    output_writer_thread_count: int,
    warmup_count: int,
    trial_results: list[TrialResult],
    mode: str = "fresh_process",
    phenotype_count: int = 1,
) -> BenchmarkSummary:
    """Build an aggregate summary from measured trials."""
    if not trial_results:
        message = f"No measured {mode} trials were provided."
        raise ValueError(message)
    wall_time_values = [trial_result.wall_time_seconds for trial_result in trial_results]
    row_rate_values = [trial_result.output_row_count / trial_result.wall_time_seconds for trial_result in trial_results]
    final_parquet_byte_values = [
        trial_result.final_parquet_bytes
        for trial_result in trial_results
        if trial_result.final_parquet_bytes is not None
    ]
    child_wall_time_values = [
        trial_result.child_wall_time_seconds
        for trial_result in trial_results
        if trial_result.child_wall_time_seconds is not None
    ]
    stage_timing_paths = [
        trial_result.stage_timing_path for trial_result in trial_results if trial_result.stage_timing_path is not None
    ]
    return BenchmarkSummary(
        device=device,
        chunk_size=chunk_size,
        finalize_parquet=finalize_parquet,
        output_writer_thread_count=output_writer_thread_count,
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
        mode=mode,
        phenotype_count=phenotype_count,
        mean_child_wall_time_seconds=(statistics.fmean(child_wall_time_values) if child_wall_time_values else None),
        stage_timing_paths=stage_timing_paths,
    )


def build_benchmark_report(
    *,
    fresh_process_summary: BenchmarkSummary,
    same_process_summary: BenchmarkSummary,
) -> BenchmarkReport:
    """Build a combined fresh versus same-process report."""
    speedup_ratio = fresh_process_summary.median_wall_time_seconds / same_process_summary.median_wall_time_seconds
    return BenchmarkReport(
        fresh_process=fresh_process_summary,
        same_process=same_process_summary,
        comparisons={
            "fresh_process_to_same_process_median_speedup_ratio": speedup_ratio,
            "fresh_process_minus_same_process_median_seconds": (
                fresh_process_summary.median_wall_time_seconds - same_process_summary.median_wall_time_seconds
            ),
        },
    )


def run_tool(arguments: LinearStartupArguments) -> None:
    """Run the fresh-process benchmark."""
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    require_positive_count("tool.trials", arguments.trials)
    require_non_negative_count("tool.warmup_trials", arguments.warmup_trials)
    require_non_negative_count("tool.same_process_trials", arguments.same_process_trials)
    require_non_negative_count("tool.same_process_warmup_trials", arguments.same_process_warmup_trials)
    require_positive_count("tool.multi_phenotype_count", arguments.multi_phenotype_count)
    benchmark_inputs = prepare_benchmark_inputs(
        data_directory=arguments.data_dir,
        output_directory=arguments.output_dir,
        phenotype_count=arguments.multi_phenotype_count,
    )

    measured_trial_results = run_fresh_process_trials(
        benchmark_inputs=benchmark_inputs,
        output_directory=arguments.output_dir,
        device=arguments.device,
        chunk_size=arguments.chunk_size,
        finalize_parquet=arguments.finalize_parquet,
        output_writer_thread_count=arguments.output_writer_thread_count,
        warmup_count=arguments.warmup_trials,
        trial_count=arguments.trials,
        emit_stage_timings=arguments.emit_stage_timings,
        multi_phenotype_sample_mode=arguments.multi_phenotype_sample_mode,
    )

    benchmark_summary = build_summary(
        device=arguments.device,
        chunk_size=arguments.chunk_size,
        finalize_parquet=arguments.finalize_parquet,
        output_writer_thread_count=arguments.output_writer_thread_count,
        warmup_count=arguments.warmup_trials,
        trial_results=measured_trial_results,
        mode="fresh_process",
        phenotype_count=len(benchmark_inputs.phenotype_names),
    )
    output_payload: dict[str, object]
    if arguments.same_process_trials:
        same_process_trial_results = run_same_process_trials(
            benchmark_inputs=benchmark_inputs,
            output_directory=arguments.output_dir,
            device=arguments.device,
            chunk_size=arguments.chunk_size,
            finalize_parquet=arguments.finalize_parquet,
            output_writer_thread_count=arguments.output_writer_thread_count,
            warmup_count=arguments.same_process_warmup_trials,
            trial_count=arguments.same_process_trials,
            emit_stage_timings=arguments.emit_stage_timings,
            multi_phenotype_sample_mode=arguments.multi_phenotype_sample_mode,
        )
        same_process_summary = build_summary(
            device=arguments.device,
            chunk_size=arguments.chunk_size,
            finalize_parquet=arguments.finalize_parquet,
            output_writer_thread_count=arguments.output_writer_thread_count,
            warmup_count=arguments.same_process_warmup_trials,
            trial_results=same_process_trial_results,
            mode="same_process",
            phenotype_count=len(benchmark_inputs.phenotype_names),
        )
        output_payload = dataclasses.asdict(
            build_benchmark_report(
                fresh_process_summary=benchmark_summary,
                same_process_summary=same_process_summary,
            )
        )
    else:
        output_payload = dataclasses.asdict(benchmark_summary)

    default_summary_filename = (
        f"{arguments.device}_finalize{int(arguments.finalize_parquet)}_"
        f"chunk{arguments.chunk_size}_"
        f"writer{arguments.output_writer_thread_count}_"
        f"phenotypes{len(benchmark_inputs.phenotype_names)}"
        f"{'_with_same_process' if arguments.same_process_trials else ''}.json"
    )
    json_summary_path = arguments.json_summary_path or (arguments.output_dir / default_summary_filename)
    json_summary_path.write_text(json.dumps(output_payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output_payload, indent=2))


def build_arguments_from_config(config: omegaconf.DictConfig) -> LinearStartupArguments:
    """Resolve fresh-process benchmark parameters from Hydra config."""
    tool_values = tooling_hydra_arguments.tool_config_to_dictionary(config)
    return LinearStartupArguments(
        device=str(tool_values["device"]),
        chunk_size=int(tool_values["chunk_size"]),
        finalize_parquet=tooling_hydra_arguments.boolean_value(tool_values["finalize_parquet"]),
        output_writer_thread_count=int(tool_values["output_writer_thread_count"]),
        trials=int(tool_values["trials"]),
        warmup_trials=int(tool_values["warmup_trials"]),
        same_process_trials=int(tool_values["same_process_trials"]),
        same_process_warmup_trials=int(tool_values["same_process_warmup_trials"]),
        multi_phenotype_count=int(tool_values["multi_phenotype_count"]),
        multi_phenotype_sample_mode=str(tool_values["multi_phenotype_sample_mode"]),
        emit_stage_timings=tooling_hydra_arguments.boolean_value(tool_values["emit_stage_timings"]),
        data_dir=tooling_hydra_arguments.path_or_none(tool_values["data_dir"]) or DEFAULT_DATA_DIRECTORY,
        output_dir=tooling_hydra_arguments.path_or_none(tool_values["output_dir"]) or DEFAULT_OUTPUT_DIRECTORY,
        json_summary_path=tooling_hydra_arguments.path_or_none(tool_values["json_summary_path"]),
    )


@hydra.main(version_base=None, config_path="../configs", config_name="benchmark_linear_startup")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Run the fresh-process benchmark from Hydra configuration."""
    run_tool(build_arguments_from_config(config))


def main() -> None:
    """Run the fresh-process benchmark from default Hydra configuration."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()


if __name__ == "__main__":
    main()
