#!/usr/bin/env python3
"""Run a standard chromosome REGENIE step 2 CPU/GPU/cache comparison matrix."""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import logging
import os
import shlex
import subprocess
import sys
import threading
import time
import typing
from pathlib import Path

import hydra

import tooling.configuration as tooling_configuration
from tooling.benchmark import native_lifecycle
from tooling.common import artifact_format as tooling_artifact_format
from tooling.common import g_regenie as tooling_g_regenie
from tooling.common import hydra_arguments as tooling_hydra_arguments
from tooling.common import hydra_compat as tooling_hydra_compat
from tooling.common import jax_cache as tooling_jax_cache
from tooling.common import logging as tooling_logging
from tooling.common import paths as tooling_paths
from tooling.common import reports as tooling_reports

logger = logging.getLogger(__name__)
REPOSITORY_ROOT = tooling_paths.find_repository_root(Path(__file__))
DEFAULT_OUTPUT_PARENT = Path("data/benchmarks")
MATRIX_MANIFEST_SCHEMA_VERSION = 1
MATRIX_MANIFEST_CONTRACT = tooling_reports.VersionedReportContract(
    schema_version=MATRIX_MANIFEST_SCHEMA_VERSION,
    required_fields=(
        "tool",
        "created_at_utc",
        "dry_run",
        "configuration",
        "compatibility_scope",
        "implementation_provenance",
        "previous_manifest_path",
        "runs",
        "comparisons",
    ),
    optional_fields=(),
    schema_field_name="schema_version",
    reject_unknown_fields=True,
)

if typing.TYPE_CHECKING:
    import omegaconf


class TraitKind(enum.StrEnum):
    """Trait family executed by the matrix runner."""

    BINARY = "binary"
    LINEAR = "linear"


class ExecutionMode(enum.StrEnum):
    """Execution mode executed by the matrix runner."""

    CPU = "cpu"
    GPU = "gpu"
    GPU_CACHED = "gpu_cached"


class CacheState(enum.StrEnum):
    """Persistent-cache state for one matrix run."""

    DISABLED = "disabled"
    ENABLED = "enabled"
    COLD = "cold"
    WARM = "warm"


class RunStatus(enum.StrEnum):
    """Status for one matrix run."""

    DRY_RUN = "dry_run"
    SUCCESS = "success"
    FAILED = "failed"


@dataclasses.dataclass(frozen=True)
class MatrixArguments:
    """Resolved parameters for one chromosome step 2 matrix.

    Attributes:
        data_directory: Directory containing chromosome inputs and phenotypes.
        bgen_path: Chromosome BGEN path.
        sample_path: Chromosome sample path.
        covariate_path: Covariate table path.
        covariate_columns: Comma-separated covariate column list.
        linear_phenotype_path: Quantitative phenotype table path.
        linear_phenotype_column: Quantitative phenotype column.
        linear_prediction_list_path: Quantitative step 1 prediction list.
        binary_phenotype_path: Binary phenotype table path.
        binary_phenotype_column: Binary phenotype column.
        binary_prediction_list_path: Binary step 1 prediction list.
        chromosome_label: Chromosome label shown in reports and logs.
        run_directory_prefix: Output directory prefix used for timestamped runs and previous-run discovery.
        output_parent: Parent directory used when output_dir is not explicit.
        output_directory: Output directory for this matrix run.
        previous_manifest_path: Optional explicit manifest to compare against.
        dry_run: Whether to only materialize commands and reports.
        validate_inputs: Whether real runs validate required input files first.
        chunk_size: REGENIE bsize.
        cpu_threads: Optional REGENIE thread count.
        output_writer_thread_count: Output writer thread count.
        cpu_jax_persistent_cache: Whether CPU runs enable JAX persistent cache.
        gpu_jax_persistent_cache: Whether GPU runs enable JAX persistent cache.
        jax_cache_directory: Base persistent JAX cache directory.
        binary_fallback_method: Binary fallback method.
        binary_p_threshold: Binary fallback p-value threshold.
        binary_firth_batch_size: Optional binary Firth batch size override.
        binary_firth_candidate_capacity: Optional binary Firth candidate capacity override.
        telemetry_mode: g telemetry mode.
        runner_prefix: Command prefix used to invoke g regenie.

    """

    data_directory: Path
    bgen_path: Path
    sample_path: Path
    covariate_path: Path
    covariate_columns: str
    linear_phenotype_path: Path
    linear_phenotype_column: str
    linear_prediction_list_path: Path
    binary_phenotype_path: Path
    binary_phenotype_column: str
    binary_prediction_list_path: Path
    chromosome_label: str
    run_directory_prefix: str
    output_parent: Path
    output_directory: Path
    previous_manifest_path: Path | None
    dry_run: bool
    validate_inputs: bool
    chunk_size: int
    cpu_threads: int | None
    output_writer_thread_count: int
    cpu_jax_persistent_cache: bool
    gpu_jax_persistent_cache: bool
    jax_cache_directory: Path
    binary_fallback_method: tooling_g_regenie.RegenieBinaryFallback
    binary_p_threshold: float
    binary_firth_batch_size: int | None
    binary_firth_candidate_capacity: int | None
    telemetry_mode: tooling_g_regenie.RegenieTelemetry
    runner_prefix: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class RunSpec:
    """One concrete g regenie command in the matrix."""

    name: str
    trait: TraitKind
    mode: ExecutionMode
    command_arguments: list[str]
    output_prefix: Path
    output_run_root: Path
    profile_summary_path: Path | None
    event_log_path: Path | None
    cache_enabled: bool
    cache_state: CacheState
    cache_directory: Path | None
    environment_overrides: dict[str, str]


@dataclasses.dataclass(frozen=True)
class RunResult:
    """Measured result for one matrix run."""

    name: str
    trait: TraitKind
    mode: ExecutionMode
    status: RunStatus
    return_code: int | None
    wall_time_seconds: float | None
    command_arguments: list[str]
    output_prefix: str
    output_run_directory: str | None
    profile_summary_path: str | None
    event_log_path: str | None
    cache_enabled: bool
    cache_state: CacheState
    cache_before: native_lifecycle.CacheSnapshot | None
    cache_after: native_lifecycle.CacheSnapshot | None
    output_row_count: int | None
    committed_chunk_count: int | None
    output_file_count: int | None
    output_total_bytes: int | None
    stage_seconds: dict[str, float]


@dataclasses.dataclass(frozen=True)
class StreamingCommandResult:
    """Exit status and retained output from one streamed subprocess."""

    return_code: int
    stdout_chunks: tuple[str, ...]
    stderr_chunks: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class MetricComparison:
    """Comparison of one metric against a previous matrix run."""

    run_name: str
    metric: str
    current_value: float | None
    previous_value: float | None
    delta: float | None
    ratio: float | None


@dataclasses.dataclass(frozen=True)
class ComparisonIdentity:
    """Compatibility key and implementation provenance for one matrix campaign."""

    compatibility_scope: dict[str, typing.Any]
    implementation_provenance: dict[str, typing.Any]


def timestamped_output_directory(output_parent: Path, run_directory_prefix: str) -> Path:
    """Build the default timestamped output directory."""
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return output_parent / f"{run_directory_prefix}_{timestamp}"


def resolve_output_directory(
    output_parent: Path,
    explicit_output_directory: Path | None,
    run_directory_prefix: str,
) -> Path:
    """Resolve the output directory for a matrix run."""
    if explicit_output_directory is not None:
        return explicit_output_directory
    return timestamped_output_directory(output_parent, run_directory_prefix)


def resolve_data_path(data_directory: Path, value: typing.Any) -> Path:
    """Resolve one input path relative to the data directory."""
    return tooling_paths.resolve_data_path(data_directory, Path(str(value)))


def resolve_repo_path(value: typing.Any) -> Path:
    """Resolve one path relative to the repository root."""
    return tooling_paths.resolve_repo_relative_path(Path(str(value)), REPOSITORY_ROOT)


def build_arguments_from_config(config: omegaconf.DictConfig) -> MatrixArguments:
    """Build matrix arguments from a composed Hydra config."""
    tool_values = tooling_hydra_arguments.tool_config_to_dictionary(config)
    run_directory_prefix = str(tool_values["run_directory_prefix"])
    output_parent = resolve_repo_path(tool_values.get("output_parent", DEFAULT_OUTPUT_PARENT))
    explicit_output_directory = tooling_hydra_arguments.path_or_none(tool_values.get("output_dir"))
    if explicit_output_directory is not None:
        explicit_output_directory = tooling_paths.resolve_repo_relative_path(explicit_output_directory, REPOSITORY_ROOT)
        output_parent = explicit_output_directory.parent
    output_directory = resolve_output_directory(output_parent, explicit_output_directory, run_directory_prefix)
    data_directory = resolve_repo_path(tool_values["data_dir"])
    jax_cache_directory = tooling_hydra_arguments.path_or_none(tool_values.get("jax_cache_dir"))
    if jax_cache_directory is None:
        jax_cache_directory = output_directory / "jax-cache"
    else:
        jax_cache_directory = tooling_paths.resolve_repo_relative_path(jax_cache_directory, REPOSITORY_ROOT)
    previous_manifest_path = tooling_hydra_arguments.path_or_none(tool_values.get("previous_manifest_path"))
    if previous_manifest_path is not None:
        previous_manifest_path = tooling_paths.resolve_repo_relative_path(previous_manifest_path, REPOSITORY_ROOT)
    runner_prefix = tuple(str(value) for value in typing.cast("list[typing.Any]", tool_values.get("runner_prefix", [])))
    return MatrixArguments(
        data_directory=data_directory,
        bgen_path=resolve_data_path(data_directory, tool_values["bgen"]),
        sample_path=resolve_data_path(data_directory, tool_values["sample"]),
        covariate_path=resolve_data_path(data_directory, tool_values["covariate_file"]),
        covariate_columns=str(tool_values["covariate_columns"]),
        linear_phenotype_path=resolve_data_path(data_directory, tool_values["linear_phenotype_file"]),
        linear_phenotype_column=str(tool_values["linear_phenotype_column"]),
        linear_prediction_list_path=resolve_data_path(data_directory, tool_values["linear_prediction_list"]),
        binary_phenotype_path=resolve_data_path(data_directory, tool_values["binary_phenotype_file"]),
        binary_phenotype_column=str(tool_values["binary_phenotype_column"]),
        binary_prediction_list_path=resolve_data_path(data_directory, tool_values["binary_prediction_list"]),
        chromosome_label=str(tool_values["chromosome_label"]),
        run_directory_prefix=run_directory_prefix,
        output_parent=output_parent,
        output_directory=output_directory,
        previous_manifest_path=previous_manifest_path,
        dry_run=bool(tool_values["dry_run"]),
        validate_inputs=bool(tool_values["validate_inputs"]),
        chunk_size=int(tool_values["chunk_size"]),
        cpu_threads=tooling_hydra_arguments.integer_or_none(tool_values.get("cpu_threads")),
        output_writer_thread_count=int(tool_values["output_writer_thread_count"]),
        cpu_jax_persistent_cache=bool(tool_values["cpu_jax_persistent_cache"]),
        gpu_jax_persistent_cache=bool(tool_values["gpu_jax_persistent_cache"]),
        jax_cache_directory=jax_cache_directory,
        binary_fallback_method=tooling_g_regenie.RegenieBinaryFallback(str(tool_values["binary_fallback_method"])),
        binary_p_threshold=float(tool_values["binary_p_threshold"]),
        binary_firth_batch_size=tooling_hydra_arguments.integer_or_none(tool_values.get("binary_firth_batch_size")),
        binary_firth_candidate_capacity=tooling_hydra_arguments.integer_or_none(
            tool_values.get("binary_firth_candidate_capacity")
        ),
        telemetry_mode=tooling_g_regenie.RegenieTelemetry(str(tool_values["telemetry_mode"])),
        runner_prefix=runner_prefix,
    )


def build_arguments_from_overrides(
    overrides: typing.Sequence[str] | None = None,
    *,
    config_name: str = "matrix_chr10",
) -> MatrixArguments:
    """Build matrix arguments from Hydra overrides."""
    config = tooling_configuration.compose_config(config_name=config_name, overrides=overrides)
    return build_arguments_from_config(config)


def resolve_jax_cache_directory_for_mode(arguments: MatrixArguments, mode: ExecutionMode, trait: TraitKind) -> Path:
    """Resolve the persistent JAX cache directory for one matrix execution mode."""
    if mode == ExecutionMode.CPU:
        return tooling_jax_cache.resolve_cpu_feature_aware_cache_directory(
            arguments.jax_cache_directory / "cpu" / trait.value
        )
    return arguments.jax_cache_directory / "gpu" / trait.value


def build_environment_overrides() -> dict[str, str]:
    """Build child-process environment overrides."""
    python_path_entries = [str(REPOSITORY_ROOT)]
    existing_python_path = os.environ.get("PYTHONPATH")
    if existing_python_path:
        python_path_entries.append(existing_python_path)
    return {
        "PYTHONPATH": os.pathsep.join(python_path_entries),
    }


def build_run_command(arguments: MatrixArguments, spec: RunSpec) -> list[str]:
    """Build the full command arguments for a run spec."""
    regenie_run_spec = build_regenie_run_spec(
        arguments=arguments,
        trait=spec.trait,
        mode=spec.mode,
        output_prefix=spec.output_prefix,
    )
    config_path = arguments.output_directory / "configs" / f"{spec.name}.toml"
    tooling_g_regenie.write_regenie_toml(regenie_run_spec, config_path)
    return tooling_g_regenie.render_g_regenie_command(regenie_run_spec, config_path)


def build_regenie_run_spec(
    *,
    arguments: MatrixArguments,
    trait: TraitKind,
    mode: ExecutionMode,
    output_prefix: Path,
) -> tooling_g_regenie.RegenieRunSpec:
    """Build the shared REGENIE run spec for one matrix entry."""
    persistent_cache_enabled = (
        arguments.cpu_jax_persistent_cache if mode == ExecutionMode.CPU else arguments.gpu_jax_persistent_cache
    )
    cache_directory = resolve_jax_cache_directory_for_mode(arguments, mode, trait) if persistent_cache_enabled else None
    is_binary_trait = trait == TraitKind.BINARY
    phenotype_path = arguments.binary_phenotype_path if is_binary_trait else arguments.linear_phenotype_path
    phenotype_column = arguments.binary_phenotype_column if is_binary_trait else arguments.linear_phenotype_column
    prediction_list_path = (
        arguments.binary_prediction_list_path if is_binary_trait else arguments.linear_prediction_list_path
    )
    binary_options = (
        tooling_g_regenie.RegenieBinaryOptions(
            fallback_method=arguments.binary_fallback_method,
            firth_se=None,
            p_threshold=arguments.binary_p_threshold,
        )
        if is_binary_trait
        else None
    )
    return tooling_g_regenie.RegenieRunSpec(
        trait_kind=(
            tooling_g_regenie.RegenieTraitKind.BINARY
            if is_binary_trait
            else tooling_g_regenie.RegenieTraitKind.QUANTITATIVE
        ),
        command_prefix=arguments.runner_prefix,
        inputs=tooling_g_regenie.RegenieInputSpec(
            bgen_path=arguments.bgen_path,
            sample_path=arguments.sample_path,
            phenotype_path=phenotype_path,
            phenotype_columns=(phenotype_column,),
            covariate_path=arguments.covariate_path,
            covariate_columns=tuple(
                column.strip() for column in arguments.covariate_columns.split(",") if column.strip()
            ),
            prediction_list_path=prediction_list_path,
            output_prefix=output_prefix,
        ),
        compute=tooling_g_regenie.RegenieComputeOptions(
            device=tooling_g_regenie.RegenieDevice.CPU
            if mode == ExecutionMode.CPU
            else tooling_g_regenie.RegenieDevice.GPU,
            bsize=arguments.chunk_size,
            cpu_threads=arguments.cpu_threads,
            firth_batch_size=arguments.binary_firth_batch_size if is_binary_trait else None,
            firth_candidate_capacity=arguments.binary_firth_candidate_capacity if is_binary_trait else None,
            jax_cache_dir=cache_directory,
        ),
        output=tooling_g_regenie.RegenieOutputOptions(
            output_run_directory=None,
            writer_threads=arguments.output_writer_thread_count,
            resume=False,
        ),
        diagnostics=tooling_g_regenie.RegenieDiagnosticsOptions(
            telemetry=arguments.telemetry_mode,
        ),
        binary=binary_options,
    )


def build_run_specs(arguments: MatrixArguments) -> list[RunSpec]:
    """Build the configured CPU/GPU cache comparison run specs."""
    run_specs: list[RunSpec] = []
    environment_overrides = build_environment_overrides()
    for trait in (TraitKind.BINARY, TraitKind.LINEAR):
        modes = [ExecutionMode.CPU, ExecutionMode.GPU]
        if arguments.gpu_jax_persistent_cache:
            modes.append(ExecutionMode.GPU_CACHED)
        for mode in modes:
            name = f"{trait.value}_{mode.value}"
            output_prefix = arguments.output_directory / "runs" / name
            output_run_root = Path(f"{output_prefix}.g")
            telemetry_directory = output_run_root / "logs"
            cache_enabled = (
                arguments.cpu_jax_persistent_cache if mode == ExecutionMode.CPU else arguments.gpu_jax_persistent_cache
            )
            if mode == ExecutionMode.GPU:
                cache_state = CacheState.COLD if cache_enabled else CacheState.DISABLED
            elif mode == ExecutionMode.GPU_CACHED:
                cache_state = CacheState.WARM
            else:
                cache_state = CacheState.ENABLED if cache_enabled else CacheState.DISABLED
            run_spec = RunSpec(
                name=name,
                trait=trait,
                mode=mode,
                command_arguments=[],
                output_prefix=output_prefix,
                output_run_root=output_run_root,
                profile_summary_path=(
                    telemetry_directory / "profile.summary.json"
                    if arguments.telemetry_mode == tooling_g_regenie.RegenieTelemetry.PROFILE
                    else None
                ),
                event_log_path=(
                    telemetry_directory / "events.jsonl"
                    if arguments.telemetry_mode != tooling_g_regenie.RegenieTelemetry.OFF
                    else None
                ),
                cache_enabled=cache_enabled,
                cache_state=cache_state,
                cache_directory=(
                    resolve_jax_cache_directory_for_mode(arguments, mode, trait) if cache_enabled else None
                ),
                environment_overrides=environment_overrides,
            )
            run_specs.append(dataclasses.replace(run_spec, command_arguments=build_run_command(arguments, run_spec)))
    return run_specs


def validate_input_paths(arguments: MatrixArguments) -> None:
    """Validate required inputs before a real matrix run."""
    required_paths = [
        arguments.bgen_path,
        arguments.sample_path,
        arguments.covariate_path,
        arguments.linear_phenotype_path,
        arguments.linear_prediction_list_path,
        arguments.binary_phenotype_path,
        arguments.binary_prediction_list_path,
    ]
    missing_paths = [path for path in required_paths if not path.is_file()]
    if missing_paths:
        formatted_paths = "\n".join(f"- {path}" for path in missing_paths)
        message = f"Required {arguments.chromosome_label} matrix inputs are missing:\n{formatted_paths}"
        raise FileNotFoundError(message)


def run_streaming_command(spec: RunSpec) -> StreamingCommandResult:
    """Run one subprocess while retaining authoritative stdout separately."""
    environment = dict(os.environ)
    environment.update(spec.environment_overrides)
    logger.info("Starting %s: %s", spec.name, shlex.join(spec.command_arguments))
    process = subprocess.Popen(
        spec.command_arguments,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=REPOSITORY_ROOT,
        env=environment,
        bufsize=1,
    )
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    stream_threads = [
        threading.Thread(
            target=stream_process_pipe,
            kwargs={
                "stream": stream,
                "chunks": chunks,
                "run_name": spec.name,
                "stream_name": stream_name,
            },
        )
        for stream, chunks, stream_name in (
            (process.stdout, stdout_chunks, "stdout"),
            (process.stderr, stderr_chunks, "stderr"),
        )
        if stream is not None
    ]
    for stream_thread in stream_threads:
        stream_thread.start()
    return_code = process.wait()
    for stream_thread in stream_threads:
        stream_thread.join()
    return StreamingCommandResult(
        return_code=return_code,
        stdout_chunks=tuple(stdout_chunks),
        stderr_chunks=tuple(stderr_chunks),
    )


def stream_process_pipe(
    *,
    stream: typing.TextIO,
    chunks: list[str],
    run_name: str,
    stream_name: str,
) -> None:
    """Retain and log one subprocess stream without mixing provenance."""
    for raw_line in stream:
        chunks.append(raw_line)
        line = raw_line.rstrip()
        if line:
            logger.info("[%s:%s] %s", run_name, stream_name, line)


def load_json_mapping(path: Path) -> dict[str, typing.Any] | None:
    """Load a JSON object if it exists."""
    if not path.is_file():
        return None
    return typing.cast("dict[str, typing.Any]", json.loads(path.read_text(encoding="utf-8")))


def measure_run_outputs(
    arguments: MatrixArguments,
    spec: RunSpec,
    stdout_chunks: typing.Sequence[str],
) -> dict[str, typing.Any]:
    """Measure run-output metadata from manifests and files."""
    verified_outputs = native_lifecycle.collect_completed_output_evidence(
        stdout_chunks,
        output_root=spec.output_run_root,
        expected_phenotype_count=1,
        run_label=spec.name,
    )
    output_measurement = verified_outputs.runs[0]
    output_run_directory = Path(output_measurement.run_directory)
    diagnostic_evidence = native_lifecycle.collect_diagnostic_evidence(
        telemetry=arguments.telemetry_mode,
        telemetry_root=spec.output_run_root,
        run_directories=(output_run_directory,),
    )
    return {
        "output_run_directory": str(output_run_directory),
        "output_row_count": output_measurement.row_count,
        "committed_chunk_count": output_measurement.committed_chunk_count,
        "output_file_count": output_measurement.parquet_file_count,
        "output_total_bytes": output_measurement.parquet_total_bytes,
        "stage_seconds": diagnostic_evidence.profile_stage_totals_seconds,
    }


def run_one_spec(arguments: MatrixArguments, spec: RunSpec) -> RunResult:
    """Run or dry-run one spec."""
    if arguments.dry_run:
        logger.info("Dry-run %s: %s", spec.name, shlex.join(spec.command_arguments))
        return RunResult(
            name=spec.name,
            trait=spec.trait,
            mode=spec.mode,
            status=RunStatus.DRY_RUN,
            return_code=None,
            wall_time_seconds=None,
            command_arguments=spec.command_arguments,
            output_prefix=str(spec.output_prefix),
            output_run_directory=None,
            profile_summary_path=str(spec.profile_summary_path) if spec.profile_summary_path is not None else None,
            event_log_path=str(spec.event_log_path) if spec.event_log_path is not None else None,
            cache_enabled=spec.cache_enabled,
            cache_state=spec.cache_state,
            cache_before=None,
            cache_after=None,
            output_row_count=None,
            committed_chunk_count=None,
            output_file_count=None,
            output_total_bytes=None,
            stage_seconds={},
        )
    cache_before = native_lifecycle.snapshot_tree(spec.cache_directory) if spec.cache_directory is not None else None
    if spec.cache_state == CacheState.COLD and cache_before is not None and cache_before.file_count != 0:
        raise RuntimeError(f"Cold GPU cache is not empty for {spec.name}: {spec.cache_directory}")
    if spec.cache_state == CacheState.WARM and (cache_before is None or cache_before.file_count == 0):
        raise RuntimeError(f"Warm GPU cache is empty for {spec.name}: {spec.cache_directory}")
    start_time = time.perf_counter()
    command_result = run_streaming_command(spec)
    wall_time_seconds = time.perf_counter() - start_time
    status = RunStatus.SUCCESS if command_result.return_code == 0 else RunStatus.FAILED
    cache_after = native_lifecycle.snapshot_tree(spec.cache_directory) if spec.cache_directory is not None else None
    output_metrics = (
        measure_run_outputs(arguments, spec, command_result.stdout_chunks)
        if status == RunStatus.SUCCESS
        else {"stage_seconds": {}}
    )
    if (
        status == RunStatus.SUCCESS
        and spec.cache_state == CacheState.COLD
        and cache_after is not None
        and cache_after.file_count == 0
    ):
        raise RuntimeError(f"Cold GPU run did not populate its cache for {spec.name}: {spec.cache_directory}")
    if status == RunStatus.SUCCESS and spec.cache_state == CacheState.WARM and cache_before != cache_after:
        raise RuntimeError(f"Warm GPU run changed its cache tree for {spec.name}: {spec.cache_directory}")
    logger.info("Finished %s with status=%s elapsed=%.3fs", spec.name, status.value, wall_time_seconds)
    return RunResult(
        name=spec.name,
        trait=spec.trait,
        mode=spec.mode,
        status=status,
        return_code=command_result.return_code,
        wall_time_seconds=wall_time_seconds,
        command_arguments=spec.command_arguments,
        output_prefix=str(spec.output_prefix),
        output_run_directory=typing.cast("str | None", output_metrics.get("output_run_directory")),
        profile_summary_path=str(spec.profile_summary_path) if spec.profile_summary_path is not None else None,
        event_log_path=str(spec.event_log_path) if spec.event_log_path is not None else None,
        cache_enabled=spec.cache_enabled,
        cache_state=spec.cache_state,
        cache_before=cache_before,
        cache_after=cache_after,
        output_row_count=typing.cast("int | None", output_metrics.get("output_row_count")),
        committed_chunk_count=typing.cast("int | None", output_metrics.get("committed_chunk_count")),
        output_file_count=typing.cast("int | None", output_metrics.get("output_file_count")),
        output_total_bytes=typing.cast("int | None", output_metrics.get("output_total_bytes")),
        stage_seconds=typing.cast("dict[str, float]", output_metrics["stage_seconds"]),
    )


def find_previous_manifest(
    arguments: MatrixArguments, compatibility_scope: dict[str, typing.Any] | None
) -> Path | None:
    """Find the previous matrix manifest for comparison."""
    if compatibility_scope is None:
        return None
    if arguments.previous_manifest_path is not None:
        payload = load_json_mapping(arguments.previous_manifest_path)
        if payload is not None and manifest_matches_compatibility_scope(compatibility_scope, payload):
            return arguments.previous_manifest_path
        return None
    if not arguments.output_parent.is_dir():
        return None
    current_manifest = arguments.output_directory / "manifest.json"
    manifest_paths = sorted(
        (
            path
            for path in arguments.output_parent.glob(f"{arguments.run_directory_prefix}_*/manifest.json")
            if path != current_manifest and path.is_file()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for manifest_path in manifest_paths:
        payload = load_json_mapping(manifest_path)
        if payload is not None and manifest_matches_compatibility_scope(compatibility_scope, payload):
            return manifest_path
    return None


def run_result_from_json_dict(payload: dict[str, typing.Any]) -> RunResult:
    """Build a run result from a manifest payload."""
    return RunResult(
        name=str(payload["name"]),
        trait=TraitKind(str(payload["trait"])),
        mode=ExecutionMode(str(payload["mode"])),
        status=RunStatus(str(payload["status"])),
        return_code=(int(payload["return_code"]) if payload["return_code"] is not None else None),
        wall_time_seconds=(float(payload["wall_time_seconds"]) if payload["wall_time_seconds"] is not None else None),
        command_arguments=[str(value) for value in payload.get("command_arguments", [])],
        output_prefix=str(payload["output_prefix"]),
        output_run_directory=(
            str(payload["output_run_directory"]) if payload.get("output_run_directory") is not None else None
        ),
        profile_summary_path=(
            str(payload["profile_summary_path"]) if payload.get("profile_summary_path") is not None else None
        ),
        event_log_path=str(payload["event_log_path"]) if payload.get("event_log_path") is not None else None,
        cache_enabled=bool(payload.get("cache_enabled", False)),
        cache_state=CacheState(str(payload.get("cache_state", CacheState.DISABLED.value))),
        cache_before=(
            native_lifecycle.CacheSnapshot(**typing.cast("dict[str, typing.Any]", payload["cache_before"]))
            if isinstance(payload.get("cache_before"), dict)
            else None
        ),
        cache_after=(
            native_lifecycle.CacheSnapshot(**typing.cast("dict[str, typing.Any]", payload["cache_after"]))
            if isinstance(payload.get("cache_after"), dict)
            else None
        ),
        output_row_count=(int(payload["output_row_count"]) if payload["output_row_count"] is not None else None),
        committed_chunk_count=(
            int(payload["committed_chunk_count"]) if payload["committed_chunk_count"] is not None else None
        ),
        output_file_count=(int(payload["output_file_count"]) if payload["output_file_count"] is not None else None),
        output_total_bytes=(int(payload["output_total_bytes"]) if payload["output_total_bytes"] is not None else None),
        stage_seconds={str(key): float(value) for key, value in payload.get("stage_seconds", {}).items()},
    )


def load_previous_run_results(previous_manifest_path: Path | None) -> dict[str, RunResult]:
    """Load previous run results keyed by run name."""
    if previous_manifest_path is None:
        return {}
    payload = load_json_mapping(previous_manifest_path)
    if payload is None:
        return {}
    raw_runs = payload.get("runs", [])
    if not isinstance(raw_runs, list):
        return {}
    run_results = [run_result_from_json_dict(run_payload) for run_payload in raw_runs if isinstance(run_payload, dict)]
    return {run_result.name: run_result for run_result in run_results}


def manifest_matches_compatibility_scope(
    compatibility_scope: dict[str, typing.Any], payload: dict[str, typing.Any]
) -> bool:
    """Return whether a previous manifest is comparable to the current run."""
    if payload.get("dry_run") is True:
        return False
    return payload.get("compatibility_scope") == compatibility_scope


def normalized_runner_protocol(runner_prefix: tuple[str, ...]) -> list[str]:
    """Normalize path-bearing runner tokens while preserving command semantics."""
    return [Path(token).name if os.sep in token and not token.startswith("-") else token for token in runner_prefix]


def build_comparison_identity(arguments: MatrixArguments) -> ComparisonIdentity:
    """Build matrix compatibility and implementation evidence."""
    import g._core

    input_paths = {
        "bgen": arguments.bgen_path,
        "sample": arguments.sample_path,
        "covariate": arguments.covariate_path,
        "linear_phenotype": arguments.linear_phenotype_path,
        "linear_prediction_list": arguments.linear_prediction_list_path,
        "binary_phenotype": arguments.binary_phenotype_path,
        "binary_prediction_list": arguments.binary_prediction_list_path,
    }
    for name, path in native_lifecycle.prediction_dependency_paths(arguments.linear_prediction_list_path).items():
        input_paths[f"linear_{name}"] = path
    for name, path in native_lifecycle.prediction_dependency_paths(arguments.binary_prediction_list_path).items():
        input_paths[f"binary_{name}"] = path
    native_library_path = Path(g._core.__file__)
    compatibility_payload: dict[str, typing.Any] = {
        "input_sha256": {name: native_lifecycle.sha256_file(path) for name, path in input_paths.items()},
        "rustflags": os.environ.get("RUSTFLAGS"),
        "rustc": native_lifecycle.command_output(["rustc", "--version", "--verbose"]),
        "cpu": native_lifecycle.command_output(["lscpu", "--json", "--output=MODELNAME,FLAGS"]),
        "gpu": native_lifecycle.command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "python_version": sys.version,
        "jax_version": native_lifecycle.distribution_version("jax"),
        "jaxlib_version": native_lifecycle.distribution_version("jaxlib"),
        "cuda_runtime_version": native_lifecycle.distribution_version("nvidia-cuda-runtime-cu12"),
        "nvcomp_version": native_lifecycle.distribution_version("nvidia-libnvcomp-cu12"),
        "configuration": {
            "chromosome_label": arguments.chromosome_label,
            "chunk_size": arguments.chunk_size,
            "cpu_threads": arguments.cpu_threads,
            "output_writer_thread_count": arguments.output_writer_thread_count,
            "cpu_jax_persistent_cache": arguments.cpu_jax_persistent_cache,
            "gpu_jax_persistent_cache": arguments.gpu_jax_persistent_cache,
            "binary_fallback_method": arguments.binary_fallback_method.value,
            "binary_p_threshold": arguments.binary_p_threshold,
            "binary_firth_batch_size": arguments.binary_firth_batch_size,
            "binary_firth_candidate_capacity": arguments.binary_firth_candidate_capacity,
            "telemetry_mode": arguments.telemetry_mode.value,
            "runner_protocol": normalized_runner_protocol(arguments.runner_prefix),
        },
    }
    canonical_payload = json.dumps(compatibility_payload, sort_keys=True, separators=(",", ":"))
    compatibility_scope = {
        "sha256": hashlib.sha256(canonical_payload.encode()).hexdigest(),
        **compatibility_payload,
    }
    source_paths = {
        "run_regenie2_matrix": Path(__file__),
        "native_lifecycle": Path(str(native_lifecycle.__file__)),
        "g_regenie": Path(str(tooling_g_regenie.__file__)),
    }
    implementation_provenance = {
        "git_commit": native_lifecycle.command_output(["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"]),
        "native_library_path": str(native_library_path),
        "native_library_sha256": native_lifecycle.sha256_file(native_library_path),
        "dependency_lock_sha256": native_lifecycle.sha256_file(REPOSITORY_ROOT / "uv.lock"),
        "cargo_lock_sha256": native_lifecycle.sha256_file(REPOSITORY_ROOT / "Cargo.lock"),
        "runner_prefix": list(arguments.runner_prefix),
        "python_source_sha256": {name: native_lifecycle.sha256_file(path) for name, path in source_paths.items()},
    }
    return ComparisonIdentity(
        compatibility_scope=compatibility_scope,
        implementation_provenance=implementation_provenance,
    )


def numeric_result_metrics(run_result: RunResult) -> dict[str, float | None]:
    """Return comparable numeric metrics for one run result."""
    return {
        "wall_time_seconds": run_result.wall_time_seconds,
        "output_row_count": float(run_result.output_row_count) if run_result.output_row_count is not None else None,
        "output_total_bytes": (
            float(run_result.output_total_bytes) if run_result.output_total_bytes is not None else None
        ),
        **{
            f"stage.{stage_name}": run_result.stage_seconds.get(stage_name)
            for stage_name in native_lifecycle.NATIVE_PROFILE_STAGE_NAMES
        },
    }


def compare_run_results(
    *,
    current_results: list[RunResult],
    previous_results_by_name: dict[str, RunResult],
) -> list[MetricComparison]:
    """Compare current run metrics against previous results."""
    comparisons: list[MetricComparison] = []
    for current_result in current_results:
        previous_result = previous_results_by_name.get(current_result.name)
        if previous_result is None:
            continue
        previous_metrics = numeric_result_metrics(previous_result)
        for metric, current_value in numeric_result_metrics(current_result).items():
            previous_value = previous_metrics.get(metric)
            delta = current_value - previous_value if current_value is not None and previous_value is not None else None
            ratio = None
            if current_value is not None and previous_value is not None and previous_value != 0.0:
                ratio = current_value / previous_value
            comparisons.append(
                MetricComparison(
                    run_name=current_result.name,
                    metric=metric,
                    current_value=current_value,
                    previous_value=previous_value,
                    delta=delta,
                    ratio=ratio,
                )
            )
    return comparisons


def arguments_to_json_dict(arguments: MatrixArguments) -> dict[str, typing.Any]:
    """Convert matrix arguments into a JSON-serializable dictionary."""
    return {
        "chromosome_label": arguments.chromosome_label,
        "run_directory_prefix": arguments.run_directory_prefix,
        "data_directory": str(arguments.data_directory),
        "bgen_path": str(arguments.bgen_path),
        "sample_path": str(arguments.sample_path),
        "output_parent": str(arguments.output_parent),
        "output_directory": str(arguments.output_directory),
        "chunk_size": arguments.chunk_size,
        "cpu_threads": arguments.cpu_threads,
        "output_writer_thread_count": arguments.output_writer_thread_count,
        "cpu_jax_persistent_cache": arguments.cpu_jax_persistent_cache,
        "gpu_jax_persistent_cache": arguments.gpu_jax_persistent_cache,
        "jax_cache_directory": str(arguments.jax_cache_directory),
        "binary_fallback_method": arguments.binary_fallback_method.value,
        "binary_p_threshold": arguments.binary_p_threshold,
        "binary_firth_batch_size": arguments.binary_firth_batch_size,
        "binary_firth_candidate_capacity": arguments.binary_firth_candidate_capacity,
        "telemetry_mode": arguments.telemetry_mode.value,
        "runner_prefix": list(arguments.runner_prefix),
    }


def run_result_to_json_dict(run_result: RunResult) -> dict[str, typing.Any]:
    """Convert a run result into a JSON-serializable dictionary."""
    return {
        "name": run_result.name,
        "trait": run_result.trait.value,
        "mode": run_result.mode.value,
        "status": run_result.status.value,
        "return_code": run_result.return_code,
        "wall_time_seconds": run_result.wall_time_seconds,
        "command_arguments": run_result.command_arguments,
        "output_prefix": run_result.output_prefix,
        "output_run_directory": run_result.output_run_directory,
        "profile_summary_path": run_result.profile_summary_path,
        "event_log_path": run_result.event_log_path,
        "cache_enabled": run_result.cache_enabled,
        "cache_state": run_result.cache_state.value,
        "cache_before": dataclasses.asdict(run_result.cache_before) if run_result.cache_before is not None else None,
        "cache_after": dataclasses.asdict(run_result.cache_after) if run_result.cache_after is not None else None,
        "output_row_count": run_result.output_row_count,
        "committed_chunk_count": run_result.committed_chunk_count,
        "output_file_count": run_result.output_file_count,
        "output_total_bytes": run_result.output_total_bytes,
        "stage_seconds": run_result.stage_seconds,
    }


def comparison_to_json_dict(comparison: MetricComparison) -> dict[str, typing.Any]:
    """Convert a metric comparison into a JSON-serializable dictionary."""
    return dataclasses.asdict(comparison)


def run_status_to_artifact_status(run_status: RunStatus) -> tooling_artifact_format.ToolArtifactStatus:
    """Convert matrix run status to the shared artifact status enum."""
    if run_status == RunStatus.DRY_RUN:
        return tooling_artifact_format.ToolArtifactStatus.DRY_RUN
    if run_status == RunStatus.FAILED:
        return tooling_artifact_format.ToolArtifactStatus.FAILED
    return tooling_artifact_format.ToolArtifactStatus.SUCCESS


def matrix_artifact_status(
    arguments: MatrixArguments, run_results: list[RunResult]
) -> tooling_artifact_format.ToolArtifactStatus:
    """Determine the overall matrix artifact status."""
    if arguments.dry_run:
        return tooling_artifact_format.ToolArtifactStatus.DRY_RUN
    if any(run_result.status == RunStatus.FAILED for run_result in run_results):
        return tooling_artifact_format.ToolArtifactStatus.FAILED
    return tooling_artifact_format.ToolArtifactStatus.SUCCESS


def standard_metric_unit(metric_name: str) -> str:
    """Return the standard unit for a matrix metric."""
    if metric_name == "wall_time_seconds" or metric_name.startswith("stage."):
        return tooling_artifact_format.MetricUnit.SECONDS.value
    if metric_name == "output_row_count":
        return tooling_artifact_format.MetricUnit.ROW.value
    if metric_name.endswith("_bytes"):
        return tooling_artifact_format.MetricUnit.BYTES.value
    if metric_name.endswith("_count"):
        return tooling_artifact_format.MetricUnit.COUNT.value
    return tooling_artifact_format.MetricUnit.COUNT.value


def standard_metric_name(metric_name: str) -> str:
    """Normalize matrix metric names for Tooling Artifact Format v1."""
    if metric_name.startswith("stage.") and not metric_name.endswith(".seconds"):
        return f"{metric_name}.seconds"
    return metric_name


def build_matrix_metric_dimensions(arguments: MatrixArguments, run_result: RunResult) -> dict[str, object]:
    """Build shared metric dimensions for one matrix run."""
    return {
        "chromosome_label": arguments.chromosome_label,
        "trait_type": run_result.trait.value,
        "mode": run_result.mode.value,
        "device": "cpu" if run_result.mode == ExecutionMode.CPU else "gpu",
        "jax_persistent_cache": run_result.cache_enabled,
        "cache_state": run_result.cache_state.value,
        "chunk_size": arguments.chunk_size,
        "output_format": "parquet_parts",
    }


def build_matrix_metrics(
    *,
    arguments: MatrixArguments,
    run_id: str,
    run_results: list[RunResult],
) -> list[tooling_artifact_format.MetricRecord]:
    """Build long-form matrix metrics."""
    metric_records: list[tooling_artifact_format.MetricRecord] = []
    for run_index, run_result in enumerate(run_results):
        metrics = numeric_result_metrics(run_result)
        metrics.update(
            {
                "output_file_count": (
                    float(run_result.output_file_count) if run_result.output_file_count is not None else None
                ),
                "committed_chunk_count": (
                    float(run_result.committed_chunk_count) if run_result.committed_chunk_count is not None else None
                ),
            }
        )
        for metric_name, metric_value in metrics.items():
            normalized_metric_name = standard_metric_name(metric_name)
            json_pointer = f"/trials/{run_index}/{metric_name.replace('.', '/')}"
            metric_records.append(
                tooling_artifact_format.build_metric_record(
                    run_id=run_id,
                    case_id=run_result.name,
                    trial_id=run_result.name,
                    phase=run_result.mode.value,
                    metric_name=normalized_metric_name,
                    value=metric_value,
                    unit=standard_metric_unit(normalized_metric_name),
                    aggregation=tooling_artifact_format.MetricAggregation.EXACT.value,
                    higher_is_better=False
                    if normalized_metric_name == "wall_time_seconds" or normalized_metric_name.startswith("stage.")
                    else None,
                    dimensions=build_matrix_metric_dimensions(arguments, run_result),
                    source=tooling_artifact_format.MetricSource(
                        artifact_path="report.json",
                        json_pointer=json_pointer,
                    ),
                )
            )
    return metric_records


def build_matrix_events(
    *,
    arguments: MatrixArguments,
    run_id: str,
    run_results: list[RunResult],
) -> list[tooling_artifact_format.ToolEventRecord]:
    """Build matrix event records."""
    events = [
        tooling_artifact_format.build_tool_event(
            tool_name="run_regenie2_matrix",
            run_id=run_id,
            phase="matrix",
            event="matrix_completed",
            message=f"{arguments.chromosome_label} matrix completed.",
            fields={
                "run_count": len(run_results),
                "dry_run": arguments.dry_run,
                "status": matrix_artifact_status(arguments, run_results).value,
            },
        )
    ]
    for run_result in run_results:
        events.append(
            tooling_artifact_format.build_tool_event(
                tool_name="run_regenie2_matrix",
                run_id=run_id,
                phase=run_result.mode.value,
                event="matrix_run_completed",
                message=f"Matrix run {run_result.name} finished with status {run_result.status.value}.",
                fields={
                    "run_name": run_result.name,
                    "trait_type": run_result.trait.value,
                    "mode": run_result.mode.value,
                    "return_code": run_result.return_code,
                    "wall_time_seconds": run_result.wall_time_seconds,
                },
            )
        )
    return events


def build_matrix_command_records(
    *,
    run_id: str,
    output_directory: Path,
    run_results: list[RunResult],
) -> list[tooling_artifact_format.CommandRecord]:
    """Build command ledger records for matrix subprocesses."""
    command_records: list[tooling_artifact_format.CommandRecord] = []
    environment_overrides = build_environment_overrides()
    for run_result in run_results:
        command_records.append(
            tooling_artifact_format.build_command_record(
                command_id=run_result.name,
                tool_name="run_regenie2_matrix",
                run_id=run_id,
                phase=run_result.mode.value,
                args=run_result.command_arguments,
                output_directory=output_directory,
                cwd=REPOSITORY_ROOT,
                environment_overrides=environment_overrides,
                status=run_status_to_artifact_status(run_result.status),
                return_code=run_result.return_code,
                wall_time_seconds=run_result.wall_time_seconds,
            )
        )
    return command_records


def build_matrix_input_records(arguments: MatrixArguments) -> list[tooling_artifact_format.InputFileRecord]:
    """Build input-file records for a matrix run."""
    return [
        tooling_artifact_format.build_input_file_record(path=arguments.bgen_path, kind="bgen"),
        tooling_artifact_format.build_input_file_record(path=arguments.sample_path, kind="sample"),
        tooling_artifact_format.build_input_file_record(path=arguments.covariate_path, kind="covariates"),
        tooling_artifact_format.build_input_file_record(
            path=arguments.linear_phenotype_path,
            kind="linear_phenotype",
        ),
        tooling_artifact_format.build_input_file_record(
            path=arguments.linear_prediction_list_path,
            kind="linear_prediction_list",
        ),
        tooling_artifact_format.build_input_file_record(
            path=arguments.binary_phenotype_path,
            kind="binary_phenotype",
        ),
        tooling_artifact_format.build_input_file_record(
            path=arguments.binary_prediction_list_path,
            kind="binary_prediction_list",
        ),
    ]


def build_matrix_failure_records(run_results: list[RunResult]) -> list[tooling_artifact_format.FailureRecord]:
    """Build structured failure records for failed matrix runs."""
    failures: list[tooling_artifact_format.FailureRecord] = []
    for failure_index, run_result in enumerate(
        (result for result in run_results if result.status == RunStatus.FAILED),
        start=1,
    ):
        failures.append(
            tooling_artifact_format.FailureRecord(
                failure_id=f"F{failure_index:03d}",
                phase=run_result.mode.value,
                status=tooling_artifact_format.ToolArtifactStatus.FAILED,
                message=f"Matrix run {run_result.name} failed.",
                exception_type=None,
                stderr_excerpt=None,
                stdout_log=None,
                stderr_log=None,
                command_id=run_result.name,
            )
        )
    return failures


def comparison_judgement(comparison: MetricComparison) -> tooling_artifact_format.ComparisonJudgement:
    """Classify a matrix metric comparison."""
    if comparison.current_value is None or comparison.previous_value is None or comparison.ratio is None:
        return tooling_artifact_format.ComparisonJudgement.INCONCLUSIVE
    if comparison.metric == "output_row_count" and comparison.current_value != comparison.previous_value:
        return tooling_artifact_format.ComparisonJudgement.REGRESSION
    if comparison.metric != "wall_time_seconds" and not comparison.metric.startswith("stage."):
        return tooling_artifact_format.ComparisonJudgement.NEUTRAL
    percent_change = (comparison.ratio - 1.0) * 100.0
    if percent_change <= -2.0:
        return tooling_artifact_format.ComparisonJudgement.IMPROVEMENT
    if percent_change >= 2.0:
        return tooling_artifact_format.ComparisonJudgement.REGRESSION
    return tooling_artifact_format.ComparisonJudgement.NEUTRAL


def build_standard_comparison_rows(comparisons: list[MetricComparison]) -> list[dict[str, object]]:
    """Build first-class comparison rows."""
    rows: list[dict[str, object]] = []
    for comparison in comparisons:
        normalized_metric_name = standard_metric_name(comparison.metric)
        percent_change = None
        if comparison.ratio is not None:
            percent_change = (comparison.ratio - 1.0) * 100.0
        rows.append(
            {
                "metric_name": normalized_metric_name,
                "case_id": comparison.run_name,
                "dimensions": {},
                "baseline_value": comparison.previous_value,
                "current_value": comparison.current_value,
                "unit": standard_metric_unit(normalized_metric_name),
                "delta": comparison.delta,
                "ratio": comparison.ratio,
                "percent_change": percent_change,
                "higher_is_better": False
                if normalized_metric_name == "wall_time_seconds" or normalized_metric_name.startswith("stage.")
                else None,
                "judgement": comparison_judgement(comparison).value,
            }
        )
    return rows


def build_comparison_report(
    *,
    producer: tooling_artifact_format.ToolProducer,
    run: tooling_artifact_format.ToolRunIdentity,
    previous_manifest_path: Path | None,
    comparisons: list[MetricComparison],
) -> tooling_artifact_format.ComparisonReport | None:
    """Build a standard comparison report when previous results exist."""
    if not comparisons:
        return None
    comparison_rows = build_standard_comparison_rows(comparisons)
    regression_count = sum(
        1
        for comparison_row in comparison_rows
        if comparison_row["judgement"] == tooling_artifact_format.ComparisonJudgement.REGRESSION.value
    )
    improvement_count = sum(
        1
        for comparison_row in comparison_rows
        if comparison_row["judgement"] == tooling_artifact_format.ComparisonJudgement.IMPROVEMENT.value
    )
    neutral_count = sum(
        1
        for comparison_row in comparison_rows
        if comparison_row["judgement"] == tooling_artifact_format.ComparisonJudgement.NEUTRAL.value
    )
    return tooling_artifact_format.ComparisonReport(
        schema_name="g.tooling.comparison",
        schema_version=tooling_artifact_format.SCHEMA_VERSION,
        producer=producer,
        run=run,
        baseline={
            "label": "previous",
            "report_path": str(previous_manifest_path) if previous_manifest_path is not None else None,
            "git_head": None,
        },
        current={
            "label": "current",
            "report_path": "report.json",
            "git_head": producer.git_head,
        },
        thresholds=[
            {
                "metric_name": "wall_time_seconds",
                "max_regression_percent": 2.0,
                "scope": {},
            }
        ],
        comparisons=comparison_rows,
        summary={
            "status": run.status.value,
            "regression_count": regression_count,
            "improvement_count": improvement_count,
            "neutral_count": neutral_count,
        },
    )


def build_matrix_agent_summary(
    *,
    arguments: MatrixArguments,
    run_results: list[RunResult],
    comparisons: list[MetricComparison],
) -> dict[str, object]:
    """Build an agent-oriented matrix summary."""
    failed_runs = [run_result.name for run_result in run_results if run_result.status == RunStatus.FAILED]
    key_observations = [
        f"Recorded {len(run_results)} matrix run results.",
        f"Dry run: {str(arguments.dry_run).lower()}.",
    ]
    if comparisons:
        key_observations.append(f"Compared {len(comparisons)} metric values against the previous manifest.")
    risks = [f"Failed runs: {', '.join(failed_runs)}."] if failed_runs else []
    return {
        "one_sentence": f"{arguments.chromosome_label} matrix finished with {len(failed_runs)} failed runs.",
        "key_observations": key_observations,
        "risks": risks,
        "next_actions": [],
    }


def format_optional_float(value: float | None) -> str:
    """Format an optional float for Markdown."""
    if value is None:
        return "-"
    return f"{value:.6g}"


def format_optional_integer(value: int | None) -> str:
    """Format an optional integer for Markdown."""
    if value is None:
        return "-"
    return str(value)


def format_optional_path(value: str | None) -> str:
    """Format an optional path for Markdown."""
    return f"`{value}`" if value is not None else "-"


def build_markdown_report(
    *,
    arguments: MatrixArguments,
    run_results: list[RunResult],
    implementation_provenance: dict[str, typing.Any] | None,
    previous_implementation_provenance: dict[str, typing.Any] | None,
    previous_manifest_path: Path | None,
    comparisons: list[MetricComparison],
) -> str:
    """Build the Markdown summary report."""
    lines = [
        f"# {arguments.chromosome_label} REGENIE Step 2 Matrix",
        "",
        f"- Output directory: `{arguments.output_directory}`",
        (
            f"- Previous manifest: `{previous_manifest_path}`"
            if previous_manifest_path is not None
            else "- Previous manifest: none"
        ),
        (
            f"- Current implementation commit: `{implementation_provenance.get('git_commit')}`"
            if implementation_provenance is not None
            else "- Current implementation commit: dry run"
        ),
        (
            f"- Previous implementation commit: `{previous_implementation_provenance.get('git_commit')}`"
            if previous_implementation_provenance is not None
            else "- Previous implementation commit: none"
        ),
        f"- Dry run: `{str(arguments.dry_run).lower()}`",
        f"- Chunk size: `{arguments.chunk_size}`",
        f"- JAX cache directory: `{arguments.jax_cache_directory}`",
        "",
        "## Runs",
        "",
        "| Run | Status | Wall seconds | Rows | Files | Bytes | Profile summary | Events |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for run_result in run_results:
        lines.append(
            "| "
            f"{run_result.name} | "
            f"{run_result.status.value} | "
            f"{format_optional_float(run_result.wall_time_seconds)} | "
            f"{format_optional_integer(run_result.output_row_count)} | "
            f"{format_optional_integer(run_result.output_file_count)} | "
            f"{format_optional_integer(run_result.output_total_bytes)} | "
            f"{format_optional_path(run_result.profile_summary_path)} | "
            f"{format_optional_path(run_result.event_log_path)} |"
        )
    lines.extend(["", "## Previous-Run Comparison", ""])
    if not comparisons:
        lines.append("No previous comparable manifest was found.")
    else:
        lines.extend(
            [
                "| Run | Metric | Current | Previous | Delta | Ratio |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for comparison in comparisons:
            lines.append(
                "| "
                f"{comparison.run_name} | "
                f"{comparison.metric} | "
                f"{format_optional_float(comparison.current_value)} | "
                f"{format_optional_float(comparison.previous_value)} | "
                f"{format_optional_float(comparison.delta)} | "
                f"{format_optional_float(comparison.ratio)} |"
            )
    lines.extend(["", "## Commands", ""])
    for run_result in run_results:
        lines.extend(
            [
                f"### `{run_result.name}`",
                "",
                "```bash",
                shlex.join(run_result.command_arguments),
                "```",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def write_reports(
    *,
    arguments: MatrixArguments,
    run_results: list[RunResult],
    comparison_identity: ComparisonIdentity | None,
    previous_manifest_path: Path | None,
    comparisons: list[MetricComparison],
    hydra_config: omegaconf.DictConfig | None = None,
) -> None:
    """Write JSON and Markdown reports for the matrix run."""
    manifest_path = arguments.output_directory / "manifest.json"
    report_path = arguments.output_directory / "report.md"
    previous_manifest = load_json_mapping(previous_manifest_path) if previous_manifest_path is not None else None
    previous_implementation_provenance = (
        typing.cast("dict[str, typing.Any]", previous_manifest.get("implementation_provenance"))
        if previous_manifest is not None and isinstance(previous_manifest.get("implementation_provenance"), dict)
        else None
    )
    manifest_payload = {
        "schema_version": MATRIX_MANIFEST_SCHEMA_VERSION,
        "tool": "tooling.cli.run_regenie2_matrix",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dry_run": arguments.dry_run,
        "configuration": arguments_to_json_dict(arguments),
        "compatibility_scope": (comparison_identity.compatibility_scope if comparison_identity is not None else None),
        "implementation_provenance": (
            comparison_identity.implementation_provenance if comparison_identity is not None else None
        ),
        "previous_manifest_path": str(previous_manifest_path) if previous_manifest_path is not None else None,
        "runs": [run_result_to_json_dict(run_result) for run_result in run_results],
        "comparisons": [comparison_to_json_dict(comparison) for comparison in comparisons],
    }
    tooling_reports.write_versioned_json_report(
        manifest_path,
        manifest_payload,
        MATRIX_MANIFEST_CONTRACT,
        sort_keys=True,
    )
    producer = tooling_artifact_format.build_producer(
        tool_name="run_regenie2_matrix",
        repository_root=REPOSITORY_ROOT,
    )
    status = matrix_artifact_status(arguments, run_results)
    status_reason = None
    if status == tooling_artifact_format.ToolArtifactStatus.FAILED:
        status_reason = "One or more matrix runs failed."
    run = tooling_artifact_format.build_run_identity(
        tool_name="run_regenie2_matrix",
        output_directory=arguments.output_directory,
        status=status,
        status_reason=status_reason,
    )
    context_snapshot = tooling_artifact_format.build_context_snapshot(
        output_directory=arguments.output_directory,
        repository_root=REPOSITORY_ROOT,
    )
    metrics = build_matrix_metrics(arguments=arguments, run_id=run.run_id, run_results=run_results)
    comparison_report = build_comparison_report(
        producer=producer,
        run=run,
        previous_manifest_path=previous_manifest_path,
        comparisons=comparisons,
    )
    report = tooling_artifact_format.build_report_envelope(
        producer=producer,
        run=run,
        context=context_snapshot,
        title=f"{arguments.chromosome_label} REGENIE Step 2 Matrix",
        configuration=arguments_to_json_dict(arguments),
        summary={
            "headline": f"{arguments.chromosome_label} matrix finished with status {status.value}.",
            "agent_summary": build_matrix_agent_summary(
                arguments=arguments,
                run_results=run_results,
                comparisons=comparisons,
            ),
            "legacy_manifest": manifest_payload,
        },
        cases=[
            {
                "case_id": run_result.name,
                "trait_type": run_result.trait.value,
                "mode": run_result.mode.value,
            }
            for run_result in run_results
        ],
        trials=[run_result_to_json_dict(run_result) for run_result in run_results],
        metrics=metrics,
        comparisons=build_standard_comparison_rows(comparisons),
        diagnostics={
            "previous_manifest_path": str(previous_manifest_path) if previous_manifest_path is not None else None,
            "legacy_comparison_count": len(comparisons),
            "current_implementation_provenance": (
                comparison_identity.implementation_provenance if comparison_identity is not None else None
            ),
            "previous_implementation_provenance": previous_implementation_provenance,
        },
        failures=build_matrix_failure_records(run_results),
    )
    markdown_report = build_markdown_report(
        arguments=arguments,
        run_results=run_results,
        implementation_provenance=(
            comparison_identity.implementation_provenance if comparison_identity is not None else None
        ),
        previous_implementation_provenance=previous_implementation_provenance,
        previous_manifest_path=previous_manifest_path,
        comparisons=comparisons,
    )
    tooling_artifact_format.write_standard_artifact_bundle(
        output_directory=arguments.output_directory,
        report=report,
        events=build_matrix_events(arguments=arguments, run_id=run.run_id, run_results=run_results),
        commands=build_matrix_command_records(
            run_id=run.run_id,
            output_directory=arguments.output_directory,
            run_results=run_results,
        ),
        input_files=build_matrix_input_records(arguments),
        summary_markdown=markdown_report,
        comparisons=comparison_report,
        hydra_config=hydra_config,
        tool_payload=arguments_to_json_dict(arguments),
        legacy_markdown_aliases=(report_path,),
        notes=["Legacy manifest.json preserves the pre-v1 matrix manifest shape."],
    )
    logger.info("Wrote manifest: %s", manifest_path)
    logger.info("Wrote report: %s", report_path)
    logger.info("Wrote standard report: %s", arguments.output_directory / "report.json")


def run_matrix(arguments: MatrixArguments, hydra_config: omegaconf.DictConfig | None = None) -> list[RunResult]:
    """Run the matrix and write reports."""
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    if arguments.validate_inputs and not arguments.dry_run:
        validate_input_paths(arguments)
    comparison_identity = None if arguments.dry_run else build_comparison_identity(arguments)
    run_specs = build_run_specs(arguments)
    previous_manifest_path = find_previous_manifest(
        arguments,
        comparison_identity.compatibility_scope if comparison_identity is not None else None,
    )
    previous_results_by_name = load_previous_run_results(previous_manifest_path)
    logger.info("Output directory: %s", arguments.output_directory)
    logger.info("Previous manifest: %s", previous_manifest_path if previous_manifest_path is not None else "none")
    run_results: list[RunResult] = []
    for index, run_spec in enumerate(run_specs, start=1):
        logger.info("Matrix run %d/%d: %s", index, len(run_specs), run_spec.name)
        run_result = run_one_spec(arguments, run_spec)
        run_results.append(run_result)
        if run_result.status == RunStatus.FAILED:
            logger.error("Stopping matrix after failed run %s.", run_result.name)
            break
    comparisons = compare_run_results(
        current_results=run_results,
        previous_results_by_name=previous_results_by_name,
    )
    write_reports(
        arguments=arguments,
        run_results=run_results,
        comparison_identity=comparison_identity,
        previous_manifest_path=previous_manifest_path,
        comparisons=comparisons,
        hydra_config=hydra_config,
    )
    if any(run_result.status == RunStatus.FAILED for run_result in run_results):
        message = f"One or more {arguments.chromosome_label} matrix runs failed."
        raise SystemExit(message)
    return run_results


@hydra.main(version_base=None, config_path="../configs", config_name="matrix_chr10")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Hydra entrypoint for the chromosome matrix runner."""
    arguments = build_arguments_from_config(config)
    tooling_logging.configure_tool_logging(arguments.output_directory / "tooling.log")
    run_matrix(arguments, hydra_config=config)


def main() -> None:
    """Run the Hydra entrypoint."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()


if __name__ == "__main__":
    main()
