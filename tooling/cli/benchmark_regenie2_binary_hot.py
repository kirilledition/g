#!/usr/bin/env python3
"""Benchmark the supported native binary REGENIE production lifecycle."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import typing
from pathlib import Path

import hydra
import pyarrow as pa
import pyarrow.parquet as pq

import g._core
import tooling.configuration as tooling_configuration
from tooling.common import hydra_arguments as tooling_hydra_arguments
from tooling.common import hydra_compat as tooling_hydra_compat
from tooling.common import paths as tooling_paths

if typing.TYPE_CHECKING:
    import omegaconf

REPOSITORY_ROOT = tooling_paths.find_repository_root(Path(__file__))
DEFAULT_OUTPUT_PARENT = Path("data/profiles")
SUMMARY_SCHEMA_VERSION = 2
CHILD_RUN_SOURCE = """
import json
import sys
import time

import g._core

started_at = time.perf_counter()
result = g._core.cli.run(["regenie", "--config", sys.argv[1]])
elapsed_seconds = time.perf_counter() - started_at
print(json.dumps({
    "elapsed_seconds": elapsed_seconds,
    "exit_code": result.exit_code,
    "stdout_chunks": result.stdout_chunks,
    "stderr_chunks": result.stderr_chunks,
}))
"""


@dataclasses.dataclass(frozen=True)
class BenchmarkArguments:
    """Resolved production benchmark arguments."""

    data_directory: Path
    bgen_path: Path
    sample_path: Path
    phenotype_path: Path
    covariate_path: Path
    prediction_list_path: Path
    phenotype_column: str
    covariate_columns: tuple[str, ...]
    output_directory: Path
    device: str
    chunk_size: int
    firth_batch_size: int
    firth_candidate_capacity: int
    writer_thread_count: int
    p_threshold: float
    expected_variant_count: int | None
    jax_cache_directory: Path
    include_fresh_process: bool
    hot_run_count: int
    diagnostic_run_count: int
    python_executable: str
    summary_path: Path | None


@dataclasses.dataclass(frozen=True)
class CacheSnapshot:
    """Content snapshot of a persistent JAX cache tree."""

    file_count: int
    total_size_bytes: int
    sha256: str


@dataclasses.dataclass(frozen=True)
class NativeRunResult:
    """Result returned by one native CLI lifecycle."""

    elapsed_seconds: float
    process_elapsed_seconds: float | None
    exit_code: int
    stdout_chunks: tuple[str, ...]
    stderr_chunks: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class OutputEvidence:
    """Correctness evidence collected from one production output."""

    output_root: str
    run_directory: str
    parquet_file_count: int
    parquet_size_bytes: int
    parquet_sha256: str
    row_count: int
    schema: str
    schema_metadata: dict[str, str]
    parquet_metadata: tuple[dict[str, str], ...]
    manifest_sha256: str
    manifest: dict[str, typing.Any]


@dataclasses.dataclass(frozen=True)
class TrialResult:
    """Measurement and evidence for one complete lifecycle."""

    name: str
    role: str
    headline: bool
    telemetry: str
    native: NativeRunResult
    output: OutputEvidence
    cache_before: CacheSnapshot
    cache_after: CacheSnapshot
    cache_state: str


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    with path.open("rb") as file_handle:
        return hashlib.file_digest(file_handle, "sha256").hexdigest()


def snapshot_tree(path: Path) -> CacheSnapshot:
    """Hash regular files below a directory in stable relative-path order."""
    digest = hashlib.sha256()
    file_count = 0
    total_size_bytes = 0
    if path.exists():
        for file_path in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
            relative_path = file_path.relative_to(path).as_posix()
            file_size = file_path.stat().st_size
            digest.update(relative_path.encode())
            digest.update(b"\0")
            digest.update(bytes.fromhex(sha256_file(file_path)))
            file_count += 1
            total_size_bytes += file_size
    return CacheSnapshot(file_count=file_count, total_size_bytes=total_size_bytes, sha256=digest.hexdigest())


def cache_state(before: CacheSnapshot, after: CacheSnapshot) -> str:
    """Describe cache reuse without claiming an unobservable runtime hit."""
    if before == after and before.file_count > 0:
        return "populated_tree_unchanged"
    if before.file_count == 0 and after.file_count > 0:
        return "cache_populated"
    if before != after:
        return "cache_tree_changed"
    return "empty_tree_unchanged"


def resolve_data_path(data_directory: Path, value: typing.Any) -> Path:
    """Resolve a data path relative to the configured data directory."""
    return tooling_paths.resolve_data_path(data_directory, Path(str(value)))


def default_output_directory() -> Path:
    """Return a timestamped ignored benchmark output directory."""
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return DEFAULT_OUTPUT_PARENT / f"regenie2_binary_hot_{timestamp}_{os.getpid()}"


def toml_string(value: str | Path) -> str:
    """Encode a string as a TOML-compatible quoted value."""
    return json.dumps(str(value))


def write_native_config(
    arguments: BenchmarkArguments,
    *,
    output_root: Path,
    telemetry: str,
) -> Path:
    """Write one native CLI config with a distinct production output root."""
    phenotype_columns = ", ".join(toml_string(value) for value in (arguments.phenotype_column,))
    covariate_columns = ", ".join(toml_string(value) for value in arguments.covariate_columns)
    lines = [
        "[input]",
        f"bgen = {toml_string(arguments.bgen_path)}",
        f"sample = {toml_string(arguments.sample_path)}",
        f"pheno_file = {toml_string(arguments.phenotype_path)}",
        f"pheno_columns = [{phenotype_columns}]",
        f"covar_file = {toml_string(arguments.covariate_path)}",
        f"covar_columns = [{covariate_columns}]",
        f"pred = {toml_string(arguments.prediction_list_path)}",
        "",
        "[trait]",
        'trait_type = "binary"',
        f"bsize = {arguments.chunk_size}",
        "",
        "[binary]",
        'fallback_method = "firth_approximate"',
        f"p_threshold = {arguments.p_threshold}",
        "firth_se = false",
        "",
        "[compute]",
        f"device = {toml_string(arguments.device)}",
        f"firth_batch_size = {arguments.firth_batch_size}",
        f"firth_candidate_capacity = {arguments.firth_candidate_capacity}",
        f"jax_cache_dir = {toml_string(arguments.jax_cache_directory)}",
        "",
        "[output]",
        f"out = {toml_string(output_root)}",
        f"output_run_directory = {toml_string(output_root)}",
        f"writer_threads = {arguments.writer_thread_count}",
        "resume = false",
        "",
        "[diagnostics]",
        f"telemetry = {toml_string(telemetry)}",
        "",
    ]
    config_path = Path(f"{output_root}.toml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("\n".join(lines), encoding="utf-8")
    return config_path


def native_result_from_binding(result: typing.Any, elapsed_seconds: float) -> NativeRunResult:
    """Convert the extension result to an immutable benchmark record."""
    return NativeRunResult(
        elapsed_seconds=elapsed_seconds,
        process_elapsed_seconds=None,
        exit_code=int(result.exit_code),
        stdout_chunks=tuple(str(value) for value in result.stdout_chunks),
        stderr_chunks=tuple(str(value) for value in result.stderr_chunks),
    )


def run_same_process(config_path: Path) -> NativeRunResult:
    """Run one lifecycle through the supported in-process native boundary."""
    started_at = time.perf_counter()
    result = g._core.cli.run(["regenie", "--config", str(config_path)])
    elapsed_seconds = time.perf_counter() - started_at
    return native_result_from_binding(result, elapsed_seconds)


def run_fresh_process(arguments: BenchmarkArguments, config_path: Path) -> NativeRunResult:
    """Run one lifecycle inside a newly started Python process."""
    started_at = time.perf_counter()
    completed = subprocess.run(
        [arguments.python_executable, "-c", CHILD_RUN_SOURCE, str(config_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    process_elapsed_seconds = time.perf_counter() - started_at
    payload = json.loads(completed.stdout)
    return NativeRunResult(
        elapsed_seconds=float(payload["elapsed_seconds"]),
        process_elapsed_seconds=process_elapsed_seconds,
        exit_code=int(payload["exit_code"]),
        stdout_chunks=tuple(str(value) for value in payload["stdout_chunks"]),
        stderr_chunks=tuple(str(value) for value in payload["stderr_chunks"]),
    )


def decode_metadata(metadata: dict[bytes, bytes] | None) -> dict[str, str]:
    """Decode Arrow or Parquet key-value metadata for JSON output."""
    if metadata is None:
        return {}
    return {
        key.decode("utf-8", errors="replace"): value.decode("utf-8", errors="replace")
        for key, value in sorted(metadata.items())
    }


def hash_paths(paths: list[Path], root: Path) -> str:
    """Hash relative paths and file contents in a stable order."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def collect_output_evidence(output_root: Path, expected_variant_count: int | None) -> OutputEvidence:
    """Validate direct Parquet parts and collect deterministic evidence."""
    run_directories = sorted(path for path in output_root.rglob("*.run") if path.is_dir())
    if len(run_directories) != 1:
        message = f"Expected one phenotype run below {output_root}, found {len(run_directories)}."
        raise RuntimeError(message)
    run_directory = run_directories[0]
    parquet_paths = sorted((run_directory / "parts").glob("*.parquet"))
    if not parquet_paths:
        message = f"No direct Parquet parts found below {run_directory}."
        raise RuntimeError(message)
    row_count = 0
    schema: pa.Schema | None = None
    parquet_metadata: list[dict[str, str]] = []
    for parquet_path in parquet_paths:
        parquet_file = pq.ParquetFile(parquet_path)
        row_count += parquet_file.metadata.num_rows
        candidate_schema = parquet_file.schema_arrow
        if schema is None:
            schema = candidate_schema
        elif not schema.equals(candidate_schema, check_metadata=True):
            message = f"Parquet schema changed within {run_directory}."
            raise RuntimeError(message)
        parquet_metadata.append(decode_metadata(parquet_file.metadata.metadata))
    if expected_variant_count is not None and row_count != expected_variant_count:
        message = f"Expected {expected_variant_count} output rows, observed {row_count}."
        raise RuntimeError(message)
    if schema is None:
        raise RuntimeError("Output schema was not observed.")
    manifest_path = run_directory / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return OutputEvidence(
        output_root=str(output_root),
        run_directory=str(run_directory),
        parquet_file_count=len(parquet_paths),
        parquet_size_bytes=sum(path.stat().st_size for path in parquet_paths),
        parquet_sha256=hash_paths(parquet_paths, run_directory),
        row_count=row_count,
        schema=str(schema),
        schema_metadata=decode_metadata(schema.metadata),
        parquet_metadata=tuple(parquet_metadata),
        manifest_sha256=sha256_file(manifest_path),
        manifest=typing.cast("dict[str, typing.Any]", manifest),
    )


def run_trial(
    arguments: BenchmarkArguments,
    *,
    name: str,
    role: str,
    headline: bool,
    telemetry: str,
    fresh_process: bool,
) -> TrialResult:
    """Run and validate one lifecycle."""
    output_root = arguments.output_directory / "runs" / name
    config_path = write_native_config(arguments, output_root=output_root, telemetry=telemetry)
    before = snapshot_tree(arguments.jax_cache_directory)
    native = run_fresh_process(arguments, config_path) if fresh_process else run_same_process(config_path)
    if native.exit_code != 0:
        message = "".join((*native.stderr_chunks, *native.stdout_chunks))
        raise RuntimeError(f"Native CLI failed for {name}: {message}")
    after = snapshot_tree(arguments.jax_cache_directory)
    return TrialResult(
        name=name,
        role=role,
        headline=headline,
        telemetry=telemetry,
        native=native,
        output=collect_output_evidence(output_root, arguments.expected_variant_count),
        cache_before=before,
        cache_after=after,
        cache_state=cache_state(before, after),
    )


def command_output(command: list[str]) -> str | None:
    """Return bounded diagnostic command output when the command is available."""
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=15)
    except OSError, subprocess.TimeoutExpired:
        return None
    output = completed.stdout.strip() or completed.stderr.strip()
    return output[:20_000] if output else None


def distribution_version(name: str) -> str | None:
    """Return an installed distribution version without importing it."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_environment(arguments: BenchmarkArguments) -> dict[str, typing.Any]:
    """Collect the reproducibility envelope outside measured lifecycles."""
    native_library_path = Path(g._core.__file__)
    input_paths = {
        "bgen": arguments.bgen_path,
        "sample": arguments.sample_path,
        "phenotype": arguments.phenotype_path,
        "covariate": arguments.covariate_path,
        "prediction_list": arguments.prediction_list_path,
    }
    return {
        "baseline_commit": command_output(["git", "rev-parse", "HEAD"]),
        "dependency_lock_sha256": sha256_file(REPOSITORY_ROOT / "uv.lock"),
        "cargo_lock_sha256": sha256_file(REPOSITORY_ROOT / "Cargo.lock"),
        "native_library_path": str(native_library_path),
        "native_library_sha256": sha256_file(native_library_path),
        "rustflags": os.environ.get("RUSTFLAGS"),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": command_output(["lscpu"]),
        "numa": command_output(["numactl", "--show"]),
        "affinity": command_output(["taskset", "-pc", str(os.getpid())]),
        "gpu": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,pstate,clocks.current.graphics,clocks.current.memory",
                "--format=csv,noheader",
            ]
        ),
        "cuda": command_output(["nvcc", "--version"]),
        "jax_version": distribution_version("jax"),
        "jaxlib_version": distribution_version("jaxlib"),
        "nvidia_cuda_runtime_version": distribution_version("nvidia-cuda-runtime-cu12"),
        "nvcomp_linkage": command_output(["ldd", str(native_library_path)]),
        "input_sha256": {name: sha256_file(path) for name, path in input_paths.items()},
        "configuration": dataclasses.asdict(arguments),
    }


def verify_hot_contract(trials: list[TrialResult]) -> None:
    """Require stable cache and output contracts for all headline trials."""
    headline_trials = [trial for trial in trials if trial.headline]
    if not headline_trials:
        raise RuntimeError("At least one headline hot trial is required.")
    reference = headline_trials[0].output
    for trial in headline_trials:
        if trial.cache_before != trial.cache_after:
            message = f"JAX cache changed during headline trial {trial.name}."
            raise RuntimeError(message)
        output = trial.output
        if (
            output.parquet_sha256 != reference.parquet_sha256
            or output.row_count != reference.row_count
            or output.schema != reference.schema
            or output.schema_metadata != reference.schema_metadata
            or output.parquet_metadata != reference.parquet_metadata
        ):
            message = f"Output contract differs for headline trial {trial.name}."
            raise RuntimeError(message)


def run_benchmark(arguments: BenchmarkArguments) -> dict[str, typing.Any]:
    """Run fresh, warm, hot, and isolated diagnostic lifecycles."""
    if arguments.hot_run_count <= 0:
        raise ValueError("hot_run_count must be positive.")
    arguments.output_directory.mkdir(parents=True, exist_ok=False)
    arguments.jax_cache_directory.mkdir(parents=True, exist_ok=True)
    environment = collect_environment(arguments)
    trials: list[TrialResult] = []
    trials.append(
        run_trial(
            arguments,
            name="discarded_warm",
            role="discarded_compile_warmup",
            headline=False,
            telemetry="off",
            fresh_process=False,
        )
    )
    for run_index in range(arguments.hot_run_count):
        trials.append(
            run_trial(
                arguments,
                name=f"hot_{run_index + 1:02d}",
                role="same_process_hot_production",
                headline=True,
                telemetry="off",
                fresh_process=False,
            )
        )
    if arguments.include_fresh_process:
        trials.append(
            run_trial(
                arguments,
                name="fresh_process",
                role="fresh_process_diagnostic",
                headline=False,
                telemetry="off",
                fresh_process=True,
            )
        )
    for run_index in range(arguments.diagnostic_run_count):
        trials.append(
            run_trial(
                arguments,
                name=f"stage_timing_diagnostic_{run_index + 1:02d}",
                role="instrumented_diagnostic",
                headline=False,
                telemetry="profile",
                fresh_process=False,
            )
        )
    verify_hot_contract(trials)
    headline_seconds = [trial.native.elapsed_seconds for trial in trials if trial.headline]
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "environment": environment,
        "headline": {
            "metric": "same_process_hot_production_elapsed_seconds",
            "telemetry": "off",
            "run_count": len(headline_seconds),
            "elapsed_seconds": headline_seconds,
        },
        "trials": [dataclasses.asdict(trial) for trial in trials],
    }


def build_arguments_from_config(config: omegaconf.DictConfig) -> BenchmarkArguments:
    """Adapt Hydra configuration into the fixed benchmark contract."""
    values = tooling_hydra_arguments.tool_config_to_dictionary(config)
    data_directory = tooling_paths.resolve_repo_relative_path(Path(str(values["data_dir"])), REPOSITORY_ROOT)
    output_directory = tooling_hydra_arguments.path_or_none(values.get("output_dir"))
    if output_directory is not None:
        output_directory = tooling_paths.resolve_repo_relative_path(output_directory, REPOSITORY_ROOT)
    cache_directory = tooling_paths.resolve_repo_relative_path(Path(str(values["jax_cache_dir"])), REPOSITORY_ROOT)
    python_executable = values.get("python_executable")
    configured_summary_path = tooling_hydra_arguments.path_or_none(values.get("summary_path"))
    return BenchmarkArguments(
        data_directory=data_directory,
        bgen_path=resolve_data_path(data_directory, values["bgen"]),
        sample_path=resolve_data_path(data_directory, values["sample"]),
        phenotype_path=resolve_data_path(data_directory, values["phenotype_file"]),
        covariate_path=resolve_data_path(data_directory, values["covariate_file"]),
        prediction_list_path=resolve_data_path(data_directory, values["prediction_list"]),
        phenotype_column=str(values["phenotype_column"]),
        covariate_columns=tuple(str(value) for value in values["covariate_columns"]),
        output_directory=output_directory or default_output_directory(),
        device=str(values["device"]),
        chunk_size=int(values["chunk_size"]),
        firth_batch_size=int(values["firth_batch_size"]),
        firth_candidate_capacity=int(values["firth_candidate_capacity"]),
        writer_thread_count=int(values["writer_thread_count"]),
        p_threshold=float(values["p_threshold"]),
        expected_variant_count=tooling_hydra_arguments.integer_or_none(values.get("expected_variant_count")),
        jax_cache_directory=cache_directory,
        include_fresh_process=bool(values["include_fresh_process"]),
        hot_run_count=int(values["hot_run_count"]),
        diagnostic_run_count=int(values["diagnostic_run_count"]),
        python_executable=sys.executable if python_executable is None else str(python_executable),
        summary_path=(
            None
            if configured_summary_path is None
            else tooling_paths.resolve_repo_relative_path(configured_summary_path, REPOSITORY_ROOT)
        ),
    )


def build_arguments_from_overrides(overrides: typing.Sequence[str] | None = None) -> BenchmarkArguments:
    """Compose benchmark configuration and return resolved arguments."""
    config = tooling_configuration.compose_config(config_name="benchmark_regenie2_binary_hot", overrides=overrides)
    return build_arguments_from_config(config)


def run_tool(arguments: BenchmarkArguments) -> Path:
    """Run the benchmark and write its evidence summary."""
    summary = run_benchmark(arguments)
    summary_path = arguments.summary_path or arguments.output_directory / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"Wrote benchmark evidence: {summary_path}")
    return summary_path


@hydra.main(version_base=None, config_path="../configs", config_name="benchmark_regenie2_binary_hot")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Run the benchmark through Hydra."""
    run_tool(build_arguments_from_config(config))


def main() -> None:
    """Run the native binary hot benchmark."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()


if __name__ == "__main__":
    main()
