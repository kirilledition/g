"""Shared evidence helpers for native production lifecycle benchmarks."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
import typing
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet

from tooling.common import g_regenie as tooling_g_regenie

CHILD_RUN_SOURCE = """
import json
import logging
import sys
import time

logging.disable(logging.DEBUG)
logging.getLogger().setLevel(logging.INFO)
logging.getLogger("jax").setLevel(logging.INFO)

import g._core

started_at = time.perf_counter()
result = g._core.cli.run(["regenie", "--config", sys.argv[1]])
elapsed_seconds = time.perf_counter() - started_at
logging.getLogger().setLevel(logging.INFO)
print(json.dumps({
    "elapsed_seconds": elapsed_seconds,
    "exit_code": result.exit_code,
    "stdout_chunks": result.stdout_chunks,
    "stderr_chunks": result.stderr_chunks,
}))
"""

NATIVE_PROFILE_STAGE_NAMES = (
    "jax_runtime_configuration",
    "jax_backend_initialization",
    "native_run_preparation",
    "native_run_execution",
    "runner_total",
)


@dataclass(frozen=True)
class CacheSnapshot:
    """Content snapshot of a persistent JAX cache tree."""

    file_count: int
    total_size_bytes: int
    sha256: str


@dataclass(frozen=True)
class NativeRunResult:
    """Result returned by one native CLI lifecycle."""

    elapsed_seconds: float
    process_elapsed_seconds: float | None
    exit_code: int
    stdout_chunks: tuple[str, ...]
    stderr_chunks: tuple[str, ...]


@dataclass(frozen=True)
class TrialPlan:
    """Execution policy for one benchmark lifecycle."""

    name: str
    role: str
    headline: bool
    telemetry: tooling_g_regenie.RegenieTelemetry
    fresh_process: bool


@dataclass(frozen=True)
class StageTimingEvidence:
    """Evidence from one per-phenotype output stage-timing artifact."""

    path: str
    sha256: str
    stage_totals_seconds: dict[str, float]


@dataclass(frozen=True)
class DiagnosticEvidence:
    """Artifacts required from an instrumented production lifecycle."""

    profile_summary_path: str | None
    profile_summary_sha256: str | None
    profile_stage_totals_seconds: dict[str, float]
    events_path: str | None
    events_sha256: str | None
    output_stage_timings: tuple[StageTimingEvidence, ...]


@dataclass(frozen=True)
class CompletedOutputEvidence:
    """Validated schema, metadata, manifest, and Parquet evidence for one output run."""

    run_directory: str
    row_count: int
    committed_chunk_count: int
    parquet_file_count: int
    parquet_total_bytes: int
    parquet_sha256: str
    parquet_paths: tuple[str, ...]
    schema: str
    schema_metadata: dict[str, str]
    parquet_metadata: tuple[dict[str, str], ...]
    manifest_path: str
    manifest_sha256: str
    manifest: dict[str, typing.Any]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    with path.open("rb") as file_handle:
        return hashlib.file_digest(file_handle, "sha256").hexdigest()


def hash_paths(paths: list[Path], root: Path) -> str:
    """Hash relative paths and file contents in stable order."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def decode_metadata(metadata: dict[bytes, bytes] | None) -> dict[str, str]:
    """Decode Arrow or Parquet metadata for JSON evidence."""
    if metadata is None:
        return {}
    return {
        key.decode("utf-8", errors="replace"): value.decode("utf-8", errors="replace")
        for key, value in sorted(metadata.items())
    }


def snapshot_tree(path: Path) -> CacheSnapshot:
    """Hash regular files below a directory in stable relative-path order."""
    digest = hashlib.sha256()
    file_count = 0
    total_size_bytes = 0
    if path.exists():
        for file_path in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
            digest.update(file_path.relative_to(path).as_posix().encode())
            digest.update(b"\0")
            digest.update(bytes.fromhex(sha256_file(file_path)))
            file_count += 1
            total_size_bytes += file_path.stat().st_size
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


def run_same_process(config_path: Path) -> NativeRunResult:
    """Run one lifecycle through the supported in-process native boundary."""
    import g._core

    previous_disable_level = logging.root.manager.disable
    logging.disable(max(previous_disable_level, logging.DEBUG))
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger("jax").setLevel(logging.INFO)
    try:
        started_at = time.perf_counter()
        result = g._core.cli.run(tooling_g_regenie.render_native_cli_arguments(config_path))
        elapsed_seconds = time.perf_counter() - started_at
    finally:
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger("jax").setLevel(logging.INFO)
        logging.disable(previous_disable_level)
    return NativeRunResult(
        elapsed_seconds=elapsed_seconds,
        process_elapsed_seconds=None,
        exit_code=int(result.exit_code),
        stdout_chunks=tuple(str(value) for value in result.stdout_chunks),
        stderr_chunks=tuple(str(value) for value in result.stderr_chunks),
    )


def run_fresh_process(python_executable: str, config_path: Path) -> NativeRunResult:
    """Run one lifecycle inside a newly started Python process."""
    resolved_executable = shutil.which(python_executable)
    if resolved_executable is None:
        raise ValueError(f"Python executable was not found: {python_executable}")
    if Path(resolved_executable).resolve() != Path(sys.executable).resolve():
        raise ValueError(
            "Fresh-process benchmarks must use the current Python environment so recorded "
            f"native/JAX evidence remains attributable: {resolved_executable} != {sys.executable}"
        )
    started_at = time.perf_counter()
    completed = subprocess.run(
        [resolved_executable, "-c", CHILD_RUN_SOURCE, str(config_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    process_elapsed_seconds = time.perf_counter() - started_at
    payload: dict[str, typing.Any] | None = None
    for stdout_line in reversed(completed.stdout.splitlines()):
        try:
            raw_payload: object = json.loads(stdout_line)
        except json.JSONDecodeError:
            continue
        if isinstance(raw_payload, dict) and "elapsed_seconds" in raw_payload:
            payload = typing.cast("dict[str, typing.Any]", raw_payload)
            break
    if payload is None:
        raise RuntimeError("Fresh native lifecycle did not emit its JSON result payload.")
    return NativeRunResult(
        elapsed_seconds=float(payload["elapsed_seconds"]),
        process_elapsed_seconds=process_elapsed_seconds,
        exit_code=int(payload["exit_code"]),
        stdout_chunks=tuple(str(value) for value in payload["stdout_chunks"]),
        stderr_chunks=tuple(str(value) for value in payload["stderr_chunks"]),
    )


def command_output(command: list[str]) -> str | None:
    """Return bounded diagnostic command output when available."""
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


def prediction_dependency_paths(prediction_list_path: Path) -> dict[str, Path]:
    """Resolve phenotype-specific LOCO files named by a prediction list."""
    dependencies: dict[str, Path] = {}
    for line_number, line in enumerate(prediction_list_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"Prediction-list row {line_number} must contain a phenotype and LOCO path.")
        phenotype_name, raw_path = fields
        dependency_name = f"loco:{phenotype_name}"
        if dependency_name in dependencies:
            raise ValueError(f"Prediction list contains duplicate phenotype {phenotype_name!r}.")
        candidate_path = Path(raw_path)
        dependencies[dependency_name] = (
            candidate_path if candidate_path.is_absolute() else (prediction_list_path.parent / candidate_path).resolve()
        )
    if not dependencies:
        raise ValueError(f"Prediction list is empty: {prediction_list_path}")
    return dependencies


def collect_environment(
    *,
    repository_root: Path,
    input_paths: dict[str, Path],
    configuration: dict[str, typing.Any],
    jax_cache_directory: Path,
) -> dict[str, typing.Any]:
    """Collect the reproducibility envelope outside measured lifecycles."""
    import g._core

    native_library_path = Path(g._core.__file__)
    resolved_input_paths = dict(input_paths)
    prediction_list_path = resolved_input_paths.get("prediction_list")
    if prediction_list_path is not None:
        resolved_input_paths.update(prediction_dependency_paths(prediction_list_path))
    return {
        "baseline_commit": command_output(["git", "-C", str(repository_root), "rev-parse", "HEAD"]),
        "dependency_lock_sha256": sha256_file(repository_root / "uv.lock"),
        "cargo_lock_sha256": sha256_file(repository_root / "Cargo.lock"),
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
        "nvidia_libnvcomp_version": distribution_version("nvidia-libnvcomp-cu12"),
        "nvcomp_linkage": command_output(["ldd", str(native_library_path)]),
        "input_sha256": {name: sha256_file(path) for name, path in resolved_input_paths.items()},
        "jax_cache_before_warm": dataclasses.asdict(snapshot_tree(jax_cache_directory)),
        "configuration": configuration,
    }


def read_stage_totals(path: Path) -> dict[str, float]:
    """Read required stage totals from a JSON diagnostic artifact."""
    payload = typing.cast("dict[str, typing.Any]", json.loads(path.read_text(encoding="utf-8")))
    raw_stage_totals = payload.get("stage_totals_seconds")
    if not isinstance(raw_stage_totals, dict) or not raw_stage_totals:
        raise RuntimeError(f"Diagnostic artifact has no stage totals: {path}")
    return {str(name): float(value) for name, value in raw_stage_totals.items()}


def read_native_profile_stage_totals(path: Path) -> dict[str, float]:
    """Read a profile summary and require every current native runtime stage."""
    stage_totals = read_stage_totals(path)
    missing_stage_names = [name for name in NATIVE_PROFILE_STAGE_NAMES if name not in stage_totals]
    if missing_stage_names:
        raise RuntimeError(f"Native profile summary is missing current stages {missing_stage_names}: {path}")
    return stage_totals


def collect_diagnostic_evidence(
    *,
    telemetry: tooling_g_regenie.RegenieTelemetry,
    telemetry_root: Path,
    run_directories: tuple[Path, ...],
) -> DiagnosticEvidence:
    """Require and record artifacts promised by the telemetry mode."""
    if telemetry == tooling_g_regenie.RegenieTelemetry.OFF:
        return DiagnosticEvidence(
            profile_summary_path=None,
            profile_summary_sha256=None,
            profile_stage_totals_seconds={},
            events_path=None,
            events_sha256=None,
            output_stage_timings=(),
        )
    events_path = telemetry_root / "logs" / "events.jsonl"
    if not events_path.is_file() or events_path.stat().st_size == 0:
        raise RuntimeError(f"Telemetry lifecycle has no events: {events_path}")
    if telemetry == tooling_g_regenie.RegenieTelemetry.PROGRESS:
        return DiagnosticEvidence(
            profile_summary_path=None,
            profile_summary_sha256=None,
            profile_stage_totals_seconds={},
            events_path=str(events_path),
            events_sha256=sha256_file(events_path),
            output_stage_timings=(),
        )
    profile_summary_path = telemetry_root / "logs" / "profile.summary.json"
    if not profile_summary_path.is_file():
        raise RuntimeError(f"Profile lifecycle has no profile summary: {profile_summary_path}")
    profile_stage_totals = read_native_profile_stage_totals(profile_summary_path)
    output_stage_timings: list[StageTimingEvidence] = []
    for run_directory in run_directories:
        timing_path = run_directory / "output_stage_timings.json"
        if not timing_path.is_file():
            raise RuntimeError(f"Profile lifecycle has no output stage timings: {timing_path}")
        output_stage_timings.append(
            StageTimingEvidence(
                path=str(timing_path),
                sha256=sha256_file(timing_path),
                stage_totals_seconds=read_stage_totals(timing_path),
            )
        )
    return DiagnosticEvidence(
        profile_summary_path=str(profile_summary_path),
        profile_summary_sha256=sha256_file(profile_summary_path),
        profile_stage_totals_seconds=profile_stage_totals,
        events_path=str(events_path),
        events_sha256=sha256_file(events_path),
        output_stage_timings=tuple(output_stage_timings),
    )


def discover_completed_run_directory(
    *,
    expected_run_directory: Path | None,
    output_root: Path,
    glob_pattern: str,
    run_label: str,
) -> Path:
    """Require exactly one manifest-backed production output run."""
    if (
        expected_run_directory is not None
        and expected_run_directory.is_dir()
        and (expected_run_directory / "run_manifest.json").is_file()
    ):
        return expected_run_directory
    discovered_directories = sorted(
        path for path in output_root.glob(glob_pattern) if path.is_dir() and (path / "run_manifest.json").is_file()
    )
    if len(discovered_directories) == 1:
        return discovered_directories[0]
    raise RuntimeError(
        f"Expected exactly one completed output run below {output_root} for {run_label}; "
        f"found {len(discovered_directories)}."
    )


def measure_completed_output_run(run_directory: Path) -> CompletedOutputEvidence:
    """Validate complete chunk coverage and collect direct-Parquet evidence."""
    manifest_path = run_directory / "run_manifest.json"
    payload = typing.cast("dict[str, typing.Any]", json.loads(manifest_path.read_text(encoding="utf-8")))
    if payload.get("status") != "completed":
        raise RuntimeError(f"Output run manifest is not completed: {run_directory}")
    execution_plan = payload.get("execution_plan")
    if (
        not isinstance(execution_plan, dict)
        or not isinstance(execution_plan.get("variant_count"), int)
        or isinstance(execution_plan.get("variant_count"), bool)
    ):
        raise RuntimeError(f"Output run manifest has no expected variant count: {run_directory}")
    expected_variant_count = int(execution_plan["variant_count"])
    if expected_variant_count <= 0:
        raise RuntimeError(f"Output run manifest has a non-positive variant count: {run_directory}")
    raw_committed_chunks = payload.get("committed_chunks")
    if not isinstance(raw_committed_chunks, list) or not raw_committed_chunks:
        raise RuntimeError(f"Completed run has no committed chunks: {run_directory}")
    chunk_intervals: list[tuple[int, int, int]] = []
    chunk_identifiers: set[int] = set()
    expected_parquet_names: set[str] = set()
    for chunk_payload in raw_committed_chunks:
        if not isinstance(chunk_payload, dict) or not isinstance(chunk_payload.get("chunk_file_name"), str):
            raise RuntimeError(f"Completed run has malformed committed chunks: {run_directory}")
        integer_fields = ("chunk_identifier", "variant_start_index", "variant_stop_index", "row_count")
        if any(
            not isinstance(chunk_payload.get(field_name), int) or isinstance(chunk_payload.get(field_name), bool)
            for field_name in integer_fields
        ):
            raise RuntimeError(f"Completed run has malformed committed chunks: {run_directory}")
        chunk_identifier = int(chunk_payload["chunk_identifier"])
        variant_start_index = int(chunk_payload["variant_start_index"])
        variant_stop_index = int(chunk_payload["variant_stop_index"])
        chunk_row_count = int(chunk_payload["row_count"])
        if chunk_row_count <= 0:
            raise RuntimeError(f"Completed run has a non-positive committed chunk: {run_directory}")
        if chunk_identifier in chunk_identifiers:
            raise RuntimeError(f"Completed run has a duplicate chunk identifier: {run_directory}")
        if chunk_identifier != variant_start_index:
            raise RuntimeError(f"Completed run has a chunk identifier that differs from its start: {run_directory}")
        if variant_stop_index - variant_start_index != chunk_row_count:
            raise RuntimeError(f"Completed run has a chunk interval that differs from its row count: {run_directory}")
        chunk_identifiers.add(chunk_identifier)
        chunk_intervals.append((variant_start_index, variant_stop_index, chunk_row_count))
        chunk_file_name = str(chunk_payload["chunk_file_name"])
        if Path(chunk_file_name).name != chunk_file_name or not chunk_file_name.endswith(".parquet"):
            raise RuntimeError(f"Completed run has an invalid Parquet part name: {run_directory}")
        expected_parquet_names.add(chunk_file_name)
    next_variant_index = 0
    row_count = 0
    for variant_start_index, variant_stop_index, chunk_row_count in sorted(chunk_intervals):
        if variant_start_index != next_variant_index:
            raise RuntimeError(
                f"Completed run chunks do not exactly cover the variant range for {run_directory}: "
                f"expected next index {next_variant_index}, observed {variant_start_index}."
            )
        next_variant_index = variant_stop_index
        row_count += chunk_row_count
    if next_variant_index != expected_variant_count or row_count != expected_variant_count:
        raise RuntimeError(
            f"Completed run chunks do not cover its manifest variant count for {run_directory}: "
            f"expected {expected_variant_count}, observed stop {next_variant_index} and {row_count} rows."
        )
    parquet_paths = sorted(path for path in (run_directory / "parts").glob("*.parquet") if path.is_file())
    observed_parquet_names = {path.name for path in parquet_paths}
    if observed_parquet_names != expected_parquet_names:
        raise RuntimeError(
            f"Completed run Parquet parts differ from its manifest for {run_directory}: "
            f"expected {sorted(expected_parquet_names)}, observed {sorted(observed_parquet_names)}."
        )
    parquet_row_count = 0
    schema: pa.Schema | None = None
    parquet_metadata: list[dict[str, str]] = []
    for parquet_path in parquet_paths:
        parquet_file = pyarrow.parquet.ParquetFile(parquet_path)
        parquet_row_count += parquet_file.metadata.num_rows
        candidate_schema = parquet_file.schema_arrow
        if schema is None:
            schema = candidate_schema
        elif not schema.equals(candidate_schema, check_metadata=True):
            raise RuntimeError(f"Parquet schema changed within {run_directory}.")
        parquet_metadata.append(decode_metadata(parquet_file.metadata.metadata))
    if parquet_row_count != expected_variant_count:
        raise RuntimeError(
            f"Completed run Parquet rows differ from its manifest for {run_directory}: "
            f"expected {expected_variant_count}, observed {parquet_row_count}."
        )
    if schema is None:
        raise RuntimeError(f"Completed run has no Parquet schema: {run_directory}")
    return CompletedOutputEvidence(
        run_directory=str(run_directory),
        row_count=row_count,
        committed_chunk_count=len(raw_committed_chunks),
        parquet_file_count=len(parquet_paths),
        parquet_total_bytes=sum(path.stat().st_size for path in parquet_paths),
        parquet_sha256=hash_paths(parquet_paths, run_directory),
        parquet_paths=tuple(str(path) for path in parquet_paths),
        schema=str(schema),
        schema_metadata=decode_metadata(schema.metadata),
        parquet_metadata=tuple(parquet_metadata),
        manifest_path=str(manifest_path),
        manifest_sha256=sha256_file(manifest_path),
        manifest=payload,
    )
