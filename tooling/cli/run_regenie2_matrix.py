#!/usr/bin/env python3
"""Run a standard chromosome REGENIE step 2 CPU/GPU/cache comparison matrix."""

from __future__ import annotations

import dataclasses
import enum
import json
import logging
import os
import shlex
import subprocess
import time
import typing
from pathlib import Path

import hydra

import tooling.configuration as tooling_configuration
from tooling.common import hydra_arguments as tooling_hydra_arguments
from tooling.common import hydra_compat as tooling_hydra_compat
from tooling.common import logging as tooling_logging
from tooling.common import paths as tooling_paths
from tooling.common import reports as tooling_reports

logger = logging.getLogger(__name__)
REPOSITORY_ROOT = tooling_paths.find_repository_root(Path(__file__))
DEFAULT_OUTPUT_PARENT = Path("data/benchmarks")

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
        variant_limit: Optional variant cap.
        cpu_threads: Optional REGENIE thread count.
        staging_depth: Native callback staging depth.
        output_writer_thread_count: Output writer thread count.
        output_writer_queue_depth: Output writer queue depth.
        trusted_no_missing_diploid: Whether to use the trusted BGEN path.
        trusted_bgen_validation_mode: Trusted BGEN validation mode.
        gpu_genotype_format: GPU genotype transfer format.
        cpu_jax_persistent_cache: Whether CPU runs enable JAX persistent cache.
        gpu_jax_persistent_cache: Whether GPU runs enable JAX persistent cache.
        jax_cache_directory: Base persistent JAX cache directory.
        jax_persistent_cache_min_entry_size_bytes: Minimum JAX cache entry size.
        jax_persistent_cache_min_compile_time_seconds: Minimum JAX compile time for cache writes.
        jax_xla_autotune_cache: Whether to enable XLA autotune caches.
        binary_firth: Whether binary runs enable Firth fallback.
        binary_approx: Whether binary runs enable approximate Firth fallback.
        binary_p_threshold: Binary fallback p-value threshold.
        binary_firth_batch_size: Optional binary Firth batch size override.
        binary_firth_candidate_capacity: Optional binary Firth candidate capacity override.
        output_format: g output format.
        finalize_parquet: Whether to finalize Parquet output.
        telemetry_mode: g telemetry mode.
        log_filter: g log filter.
        log_stderr: Whether g writes compact logs to stderr.
        progress_interval_seconds: Minimum seconds between progress events.
        progress_interval_chunks: Minimum chunks between progress events.
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
    variant_limit: int | None
    cpu_threads: int | None
    staging_depth: int
    output_writer_thread_count: int
    output_writer_queue_depth: int
    trusted_no_missing_diploid: bool
    trusted_bgen_validation_mode: str
    gpu_genotype_format: str
    cpu_jax_persistent_cache: bool
    gpu_jax_persistent_cache: bool
    jax_cache_directory: Path
    jax_persistent_cache_min_entry_size_bytes: int
    jax_persistent_cache_min_compile_time_seconds: int
    jax_xla_autotune_cache: bool
    binary_firth: bool
    binary_approx: bool
    binary_p_threshold: float
    binary_firth_batch_size: int | None
    binary_firth_candidate_capacity: int | None
    output_format: str
    finalize_parquet: bool
    telemetry_mode: str
    log_filter: str
    log_stderr: bool
    progress_interval_seconds: float
    progress_interval_chunks: int
    runner_prefix: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class RunSpec:
    """One concrete g regenie command in the matrix."""

    name: str
    trait: TraitKind
    mode: ExecutionMode
    command_arguments: list[str]
    output_prefix: Path
    output_run_directory: Path
    stage_timing_path: Path
    profile_summary_path: Path
    event_log_path: Path
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
    output_run_directory: str
    stage_timing_path: str
    profile_summary_path: str
    event_log_path: str
    output_row_count: int | None
    committed_chunk_count: int | None
    output_file_count: int | None
    output_total_bytes: int | None
    final_parquet_path: str | None
    final_parquet_bytes: int | None
    stage_seconds: dict[str, float]


@dataclasses.dataclass(frozen=True)
class MetricComparison:
    """Comparison of one metric against a previous matrix run."""

    run_name: str
    metric: str
    current_value: float | None
    previous_value: float | None
    delta: float | None
    ratio: float | None


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
    runner_prefix = tuple(
        str(value) for value in typing.cast("list[typing.Any]", tool_values.get("runner_prefix", []))
    )
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
        variant_limit=tooling_hydra_arguments.integer_or_none(tool_values.get("variant_limit")),
        cpu_threads=tooling_hydra_arguments.integer_or_none(tool_values.get("cpu_threads")),
        staging_depth=int(tool_values["staging_depth"]),
        output_writer_thread_count=int(tool_values["output_writer_thread_count"]),
        output_writer_queue_depth=int(tool_values["output_writer_queue_depth"]),
        trusted_no_missing_diploid=bool(tool_values["trusted_no_missing_diploid"]),
        trusted_bgen_validation_mode=str(tool_values["trusted_bgen_validation_mode"]),
        gpu_genotype_format=str(tool_values["gpu_genotype_format"]),
        cpu_jax_persistent_cache=bool(tool_values["cpu_jax_persistent_cache"]),
        gpu_jax_persistent_cache=bool(tool_values["gpu_jax_persistent_cache"]),
        jax_cache_directory=jax_cache_directory,
        jax_persistent_cache_min_entry_size_bytes=int(tool_values["jax_persistent_cache_min_entry_size_bytes"]),
        jax_persistent_cache_min_compile_time_seconds=int(
            tool_values["jax_persistent_cache_min_compile_time_seconds"]
        ),
        jax_xla_autotune_cache=bool(tool_values["jax_xla_autotune_cache"]),
        binary_firth=bool(tool_values["binary_firth"]),
        binary_approx=bool(tool_values["binary_approx"]),
        binary_p_threshold=float(tool_values["binary_p_threshold"]),
        binary_firth_batch_size=tooling_hydra_arguments.integer_or_none(
            tool_values.get("binary_firth_batch_size")
        ),
        binary_firth_candidate_capacity=tooling_hydra_arguments.integer_or_none(
            tool_values.get("binary_firth_candidate_capacity")
        ),
        output_format=str(tool_values["output_format"]),
        finalize_parquet=bool(tool_values["finalize_parquet"]),
        telemetry_mode=str(tool_values["telemetry_mode"]),
        log_filter=str(tool_values["log_filter"]),
        log_stderr=bool(tool_values["log_stderr"]),
        progress_interval_seconds=float(tool_values["progress_interval_seconds"]),
        progress_interval_chunks=int(tool_values["progress_interval_chunks"]),
        runner_prefix=runner_prefix,
    )


def build_arguments_from_overrides(
    overrides: typing.Sequence[str] | None = None,
    *,
    config_name: str = "run_regenie2_chr10_matrix",
) -> MatrixArguments:
    """Build matrix arguments from Hydra overrides."""
    config = tooling_configuration.compose_config(config_name=config_name, overrides=overrides)
    return build_arguments_from_config(config)


def append_optional_option(command_arguments: list[str], option_name: str, value: object | None) -> None:
    """Append an option and value when the value is present."""
    if value is None:
        return
    command_arguments.extend([option_name, str(value)])


def append_boolean_flag(
    command_arguments: list[str],
    *,
    enabled: bool,
    enabled_flag: str,
    disabled_flag: str | None = None,
) -> None:
    """Append a Click boolean flag."""
    if enabled:
        command_arguments.append(enabled_flag)
    elif disabled_flag is not None:
        command_arguments.append(disabled_flag)


def build_trait_arguments(arguments: MatrixArguments, trait: TraitKind) -> list[str]:
    """Build trait-specific g regenie arguments."""
    if trait == TraitKind.BINARY:
        command_arguments = [
            "--bt",
            "--phenoFile",
            str(arguments.binary_phenotype_path),
            "--phenoCol",
            arguments.binary_phenotype_column,
            "--pred",
            str(arguments.binary_prediction_list_path),
            "--pThresh",
            str(arguments.binary_p_threshold),
        ]
        append_boolean_flag(command_arguments, enabled=arguments.binary_firth, enabled_flag="--firth")
        append_boolean_flag(command_arguments, enabled=arguments.binary_approx, enabled_flag="--approx")
        append_optional_option(command_arguments, "--g-firth-batch-size", arguments.binary_firth_batch_size)
        append_optional_option(
            command_arguments,
            "--g-firth-candidate-capacity",
            arguments.binary_firth_candidate_capacity,
        )
        return command_arguments
    return [
        "--qt",
        "--phenoFile",
        str(arguments.linear_phenotype_path),
        "--phenoCol",
        arguments.linear_phenotype_column,
        "--pred",
        str(arguments.linear_prediction_list_path),
    ]


def output_run_directory_for_spec(arguments: MatrixArguments, output_prefix: Path, trait: TraitKind) -> Path:
    """Infer the single-phenotype output run directory for a spec."""
    if trait == TraitKind.BINARY:
        return Path(f"{output_prefix}.g") / f"{arguments.binary_phenotype_column}.regenie2_binary.run"
    return Path(f"{output_prefix}.g") / f"{arguments.linear_phenotype_column}.regenie2_linear.run"


def build_environment_overrides() -> dict[str, str]:
    """Build child-process environment overrides."""
    python_path_entries = [str(REPOSITORY_ROOT)]
    existing_python_path = os.environ.get("PYTHONPATH")
    if existing_python_path:
        python_path_entries.append(existing_python_path)
    return {
        "PYTHONPATH": os.pathsep.join(python_path_entries),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": ".50",
    }


def build_run_command(arguments: MatrixArguments, spec: RunSpec) -> list[str]:
    """Build the full command arguments for a run spec."""
    command_arguments = list(arguments.runner_prefix)
    command_arguments.extend(
        [
            "--step",
            "2",
            "--bgen",
            str(arguments.bgen_path),
            "--sample",
            str(arguments.sample_path),
            "--covarFile",
            str(arguments.covariate_path),
            "--covarColList",
            arguments.covariate_columns,
            "--out",
            str(spec.output_prefix),
            "--g-device",
            "cpu" if spec.mode == ExecutionMode.CPU else "gpu",
            "--bsize",
            str(arguments.chunk_size),
            "--g-staging-depth",
            str(arguments.staging_depth),
            "--g-output-format",
            arguments.output_format,
            "--g-writer-threads",
            str(arguments.output_writer_thread_count),
            "--g-writer-queue-depth",
            str(arguments.output_writer_queue_depth),
            "--g-stage-timings-json",
            str(spec.stage_timing_path),
            "--g-profile-summary-json",
            str(spec.profile_summary_path),
            "--g-log-dir",
            str(spec.event_log_path.parent),
            "--g-log-file",
            str(spec.event_log_path),
            "--g-telemetry",
            arguments.telemetry_mode,
            "--g-log-filter",
            arguments.log_filter,
            "--g-progress-interval-seconds",
            str(arguments.progress_interval_seconds),
            "--g-progress-interval-chunks",
            str(arguments.progress_interval_chunks),
            "--g-trusted-bgen-validation-mode",
            arguments.trusted_bgen_validation_mode,
        ]
    )
    command_arguments.extend(build_trait_arguments(arguments, spec.trait))
    append_optional_option(command_arguments, "--g-variant-limit", arguments.variant_limit)
    append_optional_option(command_arguments, "--threads", arguments.cpu_threads)
    append_boolean_flag(
        command_arguments,
        enabled=arguments.trusted_no_missing_diploid,
        enabled_flag="--g-trusted-no-missing-diploid",
        disabled_flag="--no-g-trusted-no-missing-diploid",
    )
    append_boolean_flag(
        command_arguments,
        enabled=arguments.finalize_parquet,
        enabled_flag="--g-finalize-parquet",
        disabled_flag="--no-g-finalize-parquet",
    )
    append_boolean_flag(
        command_arguments,
        enabled=arguments.log_stderr,
        enabled_flag="--g-log-stderr",
        disabled_flag="--no-g-log-stderr",
    )
    persistent_cache_enabled = (
        arguments.cpu_jax_persistent_cache if spec.mode == ExecutionMode.CPU else arguments.gpu_jax_persistent_cache
    )
    append_boolean_flag(
        command_arguments,
        enabled=persistent_cache_enabled,
        enabled_flag="--g-jax-persistent-cache",
        disabled_flag="--no-g-jax-persistent-cache",
    )
    if persistent_cache_enabled:
        command_arguments.extend(
            [
                "--g-jax-cache-dir",
                str(arguments.jax_cache_directory / "gpu"),
                "--g-jax-persistent-cache-min-entry-size-bytes",
                str(arguments.jax_persistent_cache_min_entry_size_bytes),
                "--g-jax-persistent-cache-min-compile-time-seconds",
                str(arguments.jax_persistent_cache_min_compile_time_seconds),
            ]
        )
    if spec.mode != ExecutionMode.CPU:
        command_arguments.extend(["--g-gpu-genotype-format", arguments.gpu_genotype_format])
        append_boolean_flag(
            command_arguments,
            enabled=arguments.jax_xla_autotune_cache,
            enabled_flag="--g-jax-xla-autotune-cache",
            disabled_flag="--no-g-jax-xla-autotune-cache",
        )
    return command_arguments


def build_run_specs(arguments: MatrixArguments) -> list[RunSpec]:
    """Build the six standard run specs."""
    run_specs: list[RunSpec] = []
    environment_overrides = build_environment_overrides()
    for trait in (TraitKind.BINARY, TraitKind.LINEAR):
        for mode in (ExecutionMode.CPU, ExecutionMode.GPU, ExecutionMode.GPU_CACHED):
            name = f"{trait.value}_{mode.value}"
            output_prefix = arguments.output_directory / "runs" / name
            log_directory = arguments.output_directory / "logs" / name
            run_spec = RunSpec(
                name=name,
                trait=trait,
                mode=mode,
                command_arguments=[],
                output_prefix=output_prefix,
                output_run_directory=output_run_directory_for_spec(arguments, output_prefix, trait),
                stage_timing_path=log_directory / "stage_timings.json",
                profile_summary_path=log_directory / "profile_summary.json",
                event_log_path=log_directory / "events.jsonl",
                environment_overrides=environment_overrides,
            )
            run_specs.append(
                dataclasses.replace(run_spec, command_arguments=build_run_command(arguments, run_spec))
            )
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


def run_streaming_command(spec: RunSpec) -> int:
    """Run one subprocess and stream output through logging."""
    environment = dict(os.environ)
    environment.update(spec.environment_overrides)
    logger.info("Starting %s: %s", spec.name, shlex.join(spec.command_arguments))
    process = subprocess.Popen(
        spec.command_arguments,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=REPOSITORY_ROOT,
        env=environment,
        bufsize=1,
    )
    stdout_pipe = process.stdout
    if stdout_pipe is not None:
        for raw_line in stdout_pipe:
            line = raw_line.rstrip()
            if line:
                logger.info("[%s] %s", spec.name, line)
    return process.wait()


def load_json_mapping(path: Path) -> dict[str, typing.Any] | None:
    """Load a JSON object if it exists."""
    if not path.is_file():
        return None
    return typing.cast("dict[str, typing.Any]", json.loads(path.read_text(encoding="utf-8")))


def load_stage_seconds(path: Path) -> dict[str, float]:
    """Load stage totals from a stage-timing JSON file."""
    payload = load_json_mapping(path)
    if payload is None:
        return {}
    raw_stage_seconds = payload.get("stage_totals_seconds")
    if not isinstance(raw_stage_seconds, dict):
        return {}
    return {str(key): float(value) for key, value in raw_stage_seconds.items()}


def measure_output_files(output_run_directory: Path) -> tuple[int, int, Path | None, int | None]:
    """Measure output file count and byte size."""
    output_files: list[Path] = []
    for relative_pattern in ("chunks/*.arrow", "parts/*.parquet", "final.parquet"):
        output_files.extend(path for path in output_run_directory.glob(relative_pattern) if path.is_file())
    final_parquet_path = output_run_directory / "final.parquet"
    final_parquet_bytes = final_parquet_path.stat().st_size if final_parquet_path.is_file() else None
    return (
        len(output_files),
        sum(path.stat().st_size for path in output_files),
        final_parquet_path if final_parquet_path.is_file() else None,
        final_parquet_bytes,
    )


def discover_output_run_directory(spec: RunSpec) -> Path:
    """Discover the actual production output run directory for a completed spec."""
    if spec.output_run_directory.is_dir():
        return spec.output_run_directory
    output_root = Path(f"{spec.output_prefix}.g")
    association_suffix = "regenie2_binary.run" if spec.trait == TraitKind.BINARY else "regenie2_linear.run"
    discovered_directories = sorted(
        path
        for path in output_root.glob(f"*.{association_suffix}")
        if path.is_dir() and (path / "run_manifest.json").is_file()
    )
    if len(discovered_directories) == 1:
        return discovered_directories[0]
    return spec.output_run_directory


def measure_run_outputs(spec: RunSpec) -> dict[str, typing.Any]:
    """Measure run-output metadata from manifests and files."""
    output_run_directory = discover_output_run_directory(spec)
    manifest_payload = load_json_mapping(output_run_directory / "run_manifest.json")
    committed_chunks = []
    if manifest_payload is not None:
        raw_committed_chunks = manifest_payload.get("committed_chunks", [])
        if isinstance(raw_committed_chunks, list):
            committed_chunks = raw_committed_chunks
    output_row_count = 0
    for chunk_payload in committed_chunks:
        if isinstance(chunk_payload, dict) and chunk_payload.get("row_count") is not None:
            output_row_count += int(chunk_payload["row_count"])
    output_file_count, output_total_bytes, final_parquet_path, final_parquet_bytes = measure_output_files(
        output_run_directory
    )
    return {
        "output_run_directory": str(output_run_directory),
        "output_row_count": output_row_count if committed_chunks else None,
        "committed_chunk_count": len(committed_chunks) if committed_chunks else None,
        "output_file_count": output_file_count,
        "output_total_bytes": output_total_bytes,
        "final_parquet_path": str(final_parquet_path) if final_parquet_path is not None else None,
        "final_parquet_bytes": final_parquet_bytes,
        "stage_seconds": load_stage_seconds(spec.stage_timing_path),
    }


def run_one_spec(arguments: MatrixArguments, spec: RunSpec) -> RunResult:
    """Run or dry-run one spec."""
    spec.stage_timing_path.parent.mkdir(parents=True, exist_ok=True)
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
            output_run_directory=str(spec.output_run_directory),
            stage_timing_path=str(spec.stage_timing_path),
            profile_summary_path=str(spec.profile_summary_path),
            event_log_path=str(spec.event_log_path),
            output_row_count=None,
            committed_chunk_count=None,
            output_file_count=None,
            output_total_bytes=None,
            final_parquet_path=None,
            final_parquet_bytes=None,
            stage_seconds={},
        )
    start_time = time.perf_counter()
    return_code = run_streaming_command(spec)
    wall_time_seconds = time.perf_counter() - start_time
    status = RunStatus.SUCCESS if return_code == 0 else RunStatus.FAILED
    output_metrics = measure_run_outputs(spec) if status == RunStatus.SUCCESS else {"stage_seconds": {}}
    logger.info("Finished %s with status=%s elapsed=%.3fs", spec.name, status.value, wall_time_seconds)
    return RunResult(
        name=spec.name,
        trait=spec.trait,
        mode=spec.mode,
        status=status,
        return_code=return_code,
        wall_time_seconds=wall_time_seconds,
        command_arguments=spec.command_arguments,
        output_prefix=str(spec.output_prefix),
        output_run_directory=str(
            output_metrics.get(
                "output_run_directory",
                str(spec.output_run_directory),
            )
        ),
        stage_timing_path=str(spec.stage_timing_path),
        profile_summary_path=str(spec.profile_summary_path),
        event_log_path=str(spec.event_log_path),
        output_row_count=typing.cast("int | None", output_metrics.get("output_row_count")),
        committed_chunk_count=typing.cast("int | None", output_metrics.get("committed_chunk_count")),
        output_file_count=typing.cast("int | None", output_metrics.get("output_file_count")),
        output_total_bytes=typing.cast("int | None", output_metrics.get("output_total_bytes")),
        final_parquet_path=typing.cast("str | None", output_metrics.get("final_parquet_path")),
        final_parquet_bytes=typing.cast("int | None", output_metrics.get("final_parquet_bytes")),
        stage_seconds=typing.cast("dict[str, float]", output_metrics["stage_seconds"]),
    )


def find_previous_manifest(arguments: MatrixArguments) -> Path | None:
    """Find the previous matrix manifest for comparison."""
    if arguments.previous_manifest_path is not None:
        return arguments.previous_manifest_path if arguments.previous_manifest_path.is_file() else None
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
        if payload is not None and manifest_matches_comparison_scope(arguments, payload):
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
        wall_time_seconds=(
            float(payload["wall_time_seconds"]) if payload["wall_time_seconds"] is not None else None
        ),
        command_arguments=[str(value) for value in payload.get("command_arguments", [])],
        output_prefix=str(payload["output_prefix"]),
        output_run_directory=str(payload["output_run_directory"]),
        stage_timing_path=str(payload["stage_timing_path"]),
        profile_summary_path=str(payload["profile_summary_path"]),
        event_log_path=str(payload["event_log_path"]),
        output_row_count=(int(payload["output_row_count"]) if payload["output_row_count"] is not None else None),
        committed_chunk_count=(
            int(payload["committed_chunk_count"]) if payload["committed_chunk_count"] is not None else None
        ),
        output_file_count=(int(payload["output_file_count"]) if payload["output_file_count"] is not None else None),
        output_total_bytes=(int(payload["output_total_bytes"]) if payload["output_total_bytes"] is not None else None),
        final_parquet_path=(
            str(payload["final_parquet_path"]) if payload["final_parquet_path"] is not None else None
        ),
        final_parquet_bytes=(
            int(payload["final_parquet_bytes"]) if payload["final_parquet_bytes"] is not None else None
        ),
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


def manifest_matches_comparison_scope(arguments: MatrixArguments, payload: dict[str, typing.Any]) -> bool:
    """Return whether a previous manifest is comparable to the current run."""
    if payload.get("dry_run") is True:
        return False
    raw_configuration = payload.get("configuration")
    if not isinstance(raw_configuration, dict):
        return True
    return raw_configuration.get("variant_limit") == arguments.variant_limit


def numeric_result_metrics(run_result: RunResult) -> dict[str, float | None]:
    """Return comparable numeric metrics for one run result."""
    return {
        "wall_time_seconds": run_result.wall_time_seconds,
        "output_row_count": float(run_result.output_row_count) if run_result.output_row_count is not None else None,
        "output_total_bytes": (
            float(run_result.output_total_bytes) if run_result.output_total_bytes is not None else None
        ),
        "stage.python_api_entry": run_result.stage_seconds.get("python_api_entry"),
        "stage.jax_device_configuration_backend_init": run_result.stage_seconds.get(
            "jax_device_configuration_backend_init"
        ),
        "stage.native_engine_delivery": run_result.stage_seconds.get("native_engine_delivery"),
        "stage.host_to_device_transfer": run_result.stage_seconds.get("host_to_device_transfer"),
        "stage.jax_compute": run_result.stage_seconds.get("jax_compute"),
        "stage.device_to_host_materialization": run_result.stage_seconds.get("device_to_host_materialization"),
        "stage.output_write": run_result.stage_seconds.get("output_write"),
        "stage.writer_finish_and_parquet_finalization": run_result.stage_seconds.get(
            "writer_finish_and_parquet_finalization"
        ),
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
            delta = (
                current_value - previous_value if current_value is not None and previous_value is not None else None
            )
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
        "variant_limit": arguments.variant_limit,
        "cpu_threads": arguments.cpu_threads,
        "staging_depth": arguments.staging_depth,
        "output_writer_thread_count": arguments.output_writer_thread_count,
        "output_writer_queue_depth": arguments.output_writer_queue_depth,
        "trusted_no_missing_diploid": arguments.trusted_no_missing_diploid,
        "trusted_bgen_validation_mode": arguments.trusted_bgen_validation_mode,
        "gpu_genotype_format": arguments.gpu_genotype_format,
        "cpu_jax_persistent_cache": arguments.cpu_jax_persistent_cache,
        "gpu_jax_persistent_cache": arguments.gpu_jax_persistent_cache,
        "jax_cache_directory": str(arguments.jax_cache_directory),
        "binary_firth": arguments.binary_firth,
        "binary_approx": arguments.binary_approx,
        "binary_p_threshold": arguments.binary_p_threshold,
        "binary_firth_batch_size": arguments.binary_firth_batch_size,
        "binary_firth_candidate_capacity": arguments.binary_firth_candidate_capacity,
        "output_format": arguments.output_format,
        "finalize_parquet": arguments.finalize_parquet,
        "telemetry_mode": arguments.telemetry_mode,
        "log_filter": arguments.log_filter,
        "progress_interval_seconds": arguments.progress_interval_seconds,
        "progress_interval_chunks": arguments.progress_interval_chunks,
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
        "stage_timing_path": run_result.stage_timing_path,
        "profile_summary_path": run_result.profile_summary_path,
        "event_log_path": run_result.event_log_path,
        "output_row_count": run_result.output_row_count,
        "committed_chunk_count": run_result.committed_chunk_count,
        "output_file_count": run_result.output_file_count,
        "output_total_bytes": run_result.output_total_bytes,
        "final_parquet_path": run_result.final_parquet_path,
        "final_parquet_bytes": run_result.final_parquet_bytes,
        "stage_seconds": run_result.stage_seconds,
    }


def comparison_to_json_dict(comparison: MetricComparison) -> dict[str, typing.Any]:
    """Convert a metric comparison into a JSON-serializable dictionary."""
    return dataclasses.asdict(comparison)


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


def build_markdown_report(
    *,
    arguments: MatrixArguments,
    run_results: list[RunResult],
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
        f"- Dry run: `{str(arguments.dry_run).lower()}`",
        f"- Variant limit: `{arguments.variant_limit}`",
        f"- Chunk size: `{arguments.chunk_size}`",
        f"- JAX cache directory: `{arguments.jax_cache_directory}`",
        "",
        "## Runs",
        "",
        "| Run | Status | Wall seconds | Rows | Files | Bytes | Stage JSON | Events |",
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
            f"`{run_result.stage_timing_path}` | "
            f"`{run_result.event_log_path}` |"
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
    previous_manifest_path: Path | None,
    comparisons: list[MetricComparison],
) -> None:
    """Write JSON and Markdown reports for the matrix run."""
    manifest_path = arguments.output_directory / "manifest.json"
    report_path = arguments.output_directory / "report.md"
    manifest_payload = {
        "tool": "tooling.cli.run_regenie2_matrix",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dry_run": arguments.dry_run,
        "configuration": arguments_to_json_dict(arguments),
        "previous_manifest_path": str(previous_manifest_path) if previous_manifest_path is not None else None,
        "runs": [run_result_to_json_dict(run_result) for run_result in run_results],
        "comparisons": [comparison_to_json_dict(comparison) for comparison in comparisons],
    }
    tooling_reports.write_json_report(manifest_path, manifest_payload, sort_keys=True)
    tooling_reports.write_markdown_report(
        report_path,
        build_markdown_report(
            arguments=arguments,
            run_results=run_results,
            previous_manifest_path=previous_manifest_path,
            comparisons=comparisons,
        ),
    )
    logger.info("Wrote manifest: %s", manifest_path)
    logger.info("Wrote report: %s", report_path)


def run_matrix(arguments: MatrixArguments) -> list[RunResult]:
    """Run the matrix and write reports."""
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    if arguments.validate_inputs and not arguments.dry_run:
        validate_input_paths(arguments)
    run_specs = build_run_specs(arguments)
    previous_manifest_path = find_previous_manifest(arguments)
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
        previous_manifest_path=previous_manifest_path,
        comparisons=comparisons,
    )
    if any(run_result.status == RunStatus.FAILED for run_result in run_results):
        message = f"One or more {arguments.chromosome_label} matrix runs failed."
        raise SystemExit(message)
    return run_results


@hydra.main(version_base=None, config_path="../configs", config_name="run_regenie2_chr10_matrix")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Hydra entrypoint for the chromosome matrix runner."""
    arguments = build_arguments_from_config(config)
    tooling_logging.configure_tool_logging(arguments.output_directory / "tooling.log")
    run_matrix(arguments)


def main() -> None:
    """Run the Hydra entrypoint."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()


if __name__ == "__main__":
    main()
