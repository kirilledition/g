"""Shared evidence helpers for native production lifecycle benchmarks."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import shutil
import stat
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
PARQUET_DIRECTORY_LINE_PREFIX = "Parquet dataset saved to "
PART_BINDING_METADATA_KEY = b"g.output.part_binding"
IDENTITY_METADATA_KEYS = frozenset({"g.output.part_binding"})
MAXIMUM_OWNER_TRANSITION_COUNT = 4_096
CANONICAL_OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("CHROM", pa.string(), nullable=False),
        pa.field("GENPOS", pa.int64(), nullable=False),
        pa.field("ID", pa.string(), nullable=False),
        pa.field("ALLELE0", pa.string(), nullable=False),
        pa.field("ALLELE1", pa.string(), nullable=False),
        pa.field("A1FREQ", pa.float32(), nullable=False),
        pa.field("INFO", pa.float32(), nullable=True),
        pa.field("N", pa.int32(), nullable=False),
        pa.field("BETA", pa.float32(), nullable=False),
        pa.field("SE", pa.float32(), nullable=False),
        pa.field("CHISQ", pa.float32(), nullable=False),
        pa.field("LOG10P", pa.float32(), nullable=False),
        pa.field("CORRECTION_METHOD", pa.string(), nullable=False),
        pa.field("CORRECTION_STATUS", pa.string(), nullable=False),
    ]
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
class CompletedRunArtifact:
    """One CLI-reported phenotype output.

    Attributes:
        run_directory: Concrete immutable-attempt phenotype directory.
        parts_directory: Direct Parquet dataset directory.
        attempt_id: Attempt identifier named by the output path.
        output_directory_name: Phenotype directory name within the attempt.

    """

    run_directory: Path
    parts_directory: Path
    attempt_id: str
    output_directory_name: str


@dataclass(frozen=True)
class CompletedRunArtifacts:
    """Ordered CLI-reported phenotype outputs."""

    artifacts: tuple[CompletedRunArtifact, ...]


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


@dataclass(frozen=True)
class ImmutableFileEvidence:
    """Raw-byte identity for one absolute immutable record path."""

    absolute_path: str
    raw_sha256: str


@dataclass(frozen=True)
class OwnerAuthorityEvidence:
    """Exact ordered evidence for one released owner-authority leaf."""

    files: tuple[ImmutableFileEvidence, ...]
    aggregate_sha256: str
    released_state_id: str


@dataclass(frozen=True)
class ImmutableAuthorityEvidence:
    """Complete ordered raw-byte proof for immutable output authority."""

    files: tuple[ImmutableFileEvidence, ...]
    aggregate_sha256: str


@dataclass(frozen=True)
class CompletedOutputEvidenceSet:
    """Verified output and owner-authority evidence in CLI order."""

    runs: tuple[CompletedOutputEvidence, ...]
    owner_authority: OwnerAuthorityEvidence
    immutable_authority: ImmutableAuthorityEvidence


@dataclass(frozen=True)
class LineagePhenotypeBinding:
    """Completed-lineage binding for one phenotype."""

    phenotype_name: str
    output_directory_name: str
    execution_plan_sha256: str
    run_manifest_sha256: str


@dataclass(frozen=True)
class CompletedLineage:
    """Verified immutable lineage for one completed attempt."""

    run_set_id: str
    attempt_id: str
    chunk_plan_sha256: str
    producer_attempt_ids: frozenset[str]
    phenotypes: tuple[LineagePhenotypeBinding, ...]
    immutable_files: tuple[ImmutableFileEvidence, ...]


@dataclass(frozen=True)
class VerifiedParts:
    """Verified direct Parquet parts and their receipt evidence."""

    chunks: list[dict[str, typing.Any]]
    files: list[Path]
    row_count: int
    schema: pa.Schema
    schema_metadata: dict[str, str]
    parquet_metadata: tuple[dict[str, str], ...]
    receipt_files: tuple[ImmutableFileEvidence, ...]


@dataclass(frozen=True)
class MeasuredCompletedOutput:
    """One verified output plus immutable files read during validation."""

    output: CompletedOutputEvidence
    manifest_file: ImmutableFileEvidence
    receipt_files: tuple[ImmutableFileEvidence, ...]


@dataclass(frozen=True)
class LineageOutcome:
    """One validated attempt outcome record."""

    outcome_kind: str
    record: dict[str, typing.Any]
    path: Path
    sha256: str


@dataclass(frozen=True)
class JsonNumber:
    """JSON numeric token retained without float reserialization."""

    text: str


def parse_strict_json(
    text: str,
    *,
    role: str,
    preserve_numeric_tokens: bool = False,
) -> object:
    """Parse standards-compliant JSON while rejecting duplicate object fields."""
    try:
        if preserve_numeric_tokens:
            value = json.loads(
                text,
                object_pairs_hook=reject_duplicate_json_object_pairs,
                parse_constant=reject_nonstandard_json_constant,
                parse_float=JsonNumber,
                parse_int=JsonNumber,
            )
        else:
            value = json.loads(
                text,
                object_pairs_hook=reject_duplicate_json_object_pairs,
                parse_constant=reject_nonstandard_json_constant,
                parse_float=parse_finite_json_float,
                parse_int=parse_bounded_json_integer,
            )
        validate_json_unicode(value)
        return value
    except ValueError as error:
        raise RuntimeError(f"{role.capitalize()} is not strict JSON: {error}") from error


def reject_duplicate_json_object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build one JSON object while rejecting repeated field names."""
    payload: dict[str, object] = {}
    for field_name, value in pairs:
        if field_name in payload:
            raise ValueError(f"duplicate object field {field_name!r}")
        payload[field_name] = value
    return payload


def reject_nonstandard_json_constant(constant: str) -> typing.NoReturn:
    """Reject non-standard NaN and infinity tokens accepted by Python."""
    raise ValueError(f"non-standard numeric constant {constant!r}")


def parse_finite_json_float(token: str) -> float:
    """Parse one JSON float while rejecting overflow to infinity."""
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"JSON number is outside the finite float range: {token!r}")
    return value


def parse_bounded_json_integer(token: str) -> int:
    """Parse one integer representable by serde_json's default number type."""
    value = int(token)
    if value < -(2**63) or value > 2**64 - 1:
        raise ValueError(f"JSON integer is outside the serde_json number range: {token!r}")
    return value


def validate_json_unicode(value: object) -> None:
    """Reject lone surrogate code points that cannot occur in a Rust string."""
    if isinstance(value, str):
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise ValueError("JSON string contains a lone surrogate code point")
        return
    if isinstance(value, list):
        for item in value:
            validate_json_unicode(item)
        return
    if isinstance(value, dict):
        for field_name, item in value.items():
            validate_json_unicode(field_name)
            validate_json_unicode(item)


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


def decode_comparable_metadata(metadata: dict[bytes, bytes] | None) -> dict[str, str]:
    """Decode metadata after removing per-attempt transaction bindings."""
    return {key: value for key, value in decode_metadata(metadata).items() if key not in IDENTITY_METADATA_KEYS}


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


def parse_completed_run_artifacts(
    stdout_chunks: typing.Sequence[str],
    *,
    output_root: Path,
    expected_phenotype_count: int,
    run_label: str,
) -> CompletedRunArtifacts:
    """Parse and validate ordered CLI Parquet dataset artifacts.

    Args:
        stdout_chunks: Native or subprocess output containing completion lines.
        output_root: Requested output transaction root.
        expected_phenotype_count: Required number of completed phenotype outputs.
        run_label: Human-readable run identifier for failures.

    Returns:
        Ordered, resolved output artifacts.

    Raises:
        RuntimeError: If lines are malformed, paths escape the root, or cardinality differs.

    """
    if expected_phenotype_count <= 0:
        raise ValueError("Expected phenotype count must be positive.")
    try:
        resolved_output_root = output_root.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"Requested output root does not exist for {run_label}: {output_root}.") from error
    if not resolved_output_root.is_dir():
        raise RuntimeError(f"Requested output root is not a directory for {run_label}: {resolved_output_root}.")
    artifacts: list[CompletedRunArtifact] = []
    for stdout_chunk in stdout_chunks:
        for output_line in stdout_chunk.split("\n"):
            if output_line.startswith(PARQUET_DIRECTORY_LINE_PREFIX):
                raw_parts_path = output_line.removeprefix(PARQUET_DIRECTORY_LINE_PREFIX)
                if not raw_parts_path:
                    raise RuntimeError(f"CLI artifact dataset path is empty for {run_label}.")
                artifact = resolve_completed_run_artifact(
                    raw_parts_path,
                    resolved_output_root=resolved_output_root,
                    run_label=run_label,
                )
                artifacts.append(artifact)
                continue
            if output_line.startswith("Parquet dataset saved to"):
                raise RuntimeError(f"CLI artifact line has an unsupported format for {run_label}: {output_line!r}.")
    if len(artifacts) != expected_phenotype_count:
        raise RuntimeError(
            f"Expected {expected_phenotype_count} CLI Parquet dataset artifacts for {run_label}; "
            f"found {len(artifacts)}."
        )
    run_directories = [artifact.run_directory for artifact in artifacts]
    parts_directories = [artifact.parts_directory for artifact in artifacts]
    if len(set(run_directories)) != len(run_directories) or len(set(parts_directories)) != len(parts_directories):
        raise RuntimeError(f"CLI artifact lines for {run_label} contain duplicate output paths.")
    attempt_ids = {artifact.attempt_id for artifact in artifacts}
    if len(attempt_ids) != 1:
        raise RuntimeError(f"CLI artifact lines for {run_label} span multiple output attempts.")
    return CompletedRunArtifacts(artifacts=tuple(artifacts))


def resolve_completed_run_artifact(
    raw_parts_path: str,
    *,
    resolved_output_root: Path,
    run_label: str,
) -> CompletedRunArtifact:
    """Resolve one reported dataset and its parent run under its transaction root."""
    try:
        raw_parts_path.encode("utf-8")
    except UnicodeEncodeError as error:
        raise RuntimeError(f"CLI artifact dataset path is not valid UTF-8 for {run_label}.") from error
    raw_parts_directory = Path(raw_parts_path)
    if not raw_parts_directory.is_absolute():
        raise RuntimeError(f"CLI artifact dataset path is not absolute for {run_label}: {raw_parts_path!r}.")
    try:
        parts_directory = raw_parts_directory.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"CLI artifact path does not exist for {run_label}: {error}") from error
    if not parts_directory.is_dir():
        raise RuntimeError(f"CLI artifact dataset path is not a directory for {run_label}.")
    if parts_directory.name != "parts":
        raise RuntimeError(f"CLI artifact dataset is not a direct parts directory for {run_label}.")
    run_directory = parts_directory.parent
    try:
        relative_run_directory = run_directory.relative_to(resolved_output_root)
    except ValueError as error:
        raise RuntimeError(f"CLI artifact path escapes the requested output root for {run_label}.") from error
    relative_components = relative_run_directory.parts
    if len(relative_components) != 3 or relative_components[0] != "attempts":
        raise RuntimeError(
            f"CLI artifact path for {run_label} must have shape attempts/<attempt>/<phenotype>: {run_directory}."
        )
    attempt_id, output_directory_name = relative_components[1:]
    validate_path_identifier(attempt_id, role="attempt identifier", maximum_length=128)
    validate_safe_component(output_directory_name, role="phenotype output directory")
    return CompletedRunArtifact(
        run_directory=run_directory,
        parts_directory=parts_directory,
        attempt_id=attempt_id,
        output_directory_name=output_directory_name,
    )


def collect_completed_output_evidence(
    stdout_chunks: typing.Sequence[str],
    *,
    output_root: Path,
    expected_phenotype_count: int,
    run_label: str,
) -> CompletedOutputEvidenceSet:
    """Parse CLI artifacts and verify immutable output authority.

    The native Rust verifier is not exported through the Python binding. This
    schema-v0 reader therefore mirrors its lineage, receipt, embedded-footer,
    and raw-byte checks without mutating output state.
    """
    artifacts = parse_completed_run_artifacts(
        stdout_chunks,
        output_root=output_root,
        expected_phenotype_count=expected_phenotype_count,
        run_label=run_label,
    )
    resolved_output_root = output_root.resolve(strict=True)
    owner_authority = verify_released_owner_authority(
        resolved_output_root / ".g-output",
        run_label=run_label,
    )
    lineage = verify_completed_lineage(resolved_output_root, artifacts, run_label=run_label)
    measured_outputs = tuple(
        measure_completed_output_artifact(artifact, phenotype, lineage)
        for artifact, phenotype in zip(artifacts.artifacts, lineage.phenotypes, strict=True)
    )
    runs = tuple(measured_output.output for measured_output in measured_outputs)
    immutable_authority = build_immutable_authority(
        owner_authority,
        lineage,
        measured_outputs,
    )
    observed_owner_authority = verify_released_owner_authority(
        resolved_output_root / ".g-output",
        run_label=run_label,
    )
    observed_lineage = verify_completed_lineage(
        resolved_output_root,
        artifacts,
        run_label=run_label,
    )
    observed_immutable_authority = recheck_immutable_authority(
        observed_owner_authority,
        observed_lineage,
        measured_outputs,
    )
    if observed_owner_authority != owner_authority or observed_immutable_authority != immutable_authority:
        raise RuntimeError(f"Immutable output authority changed while collecting evidence for {run_label}.")
    return CompletedOutputEvidenceSet(
        runs=runs,
        owner_authority=owner_authority,
        immutable_authority=immutable_authority,
    )


def build_immutable_authority(
    owner_authority: OwnerAuthorityEvidence,
    lineage: CompletedLineage,
    measured_outputs: tuple[MeasuredCompletedOutput, ...],
) -> ImmutableAuthorityEvidence:
    """Build the category-ordered immutable authority proof."""
    files = (
        owner_authority.files
        + lineage.immutable_files
        + tuple(measured_output.manifest_file for measured_output in measured_outputs)
        + tuple(receipt_file for measured_output in measured_outputs for receipt_file in measured_output.receipt_files)
    )
    return ImmutableAuthorityEvidence(
        files=files,
        aggregate_sha256=aggregate_immutable_files_sha256(files),
    )


def recheck_immutable_authority(
    owner_authority: OwnerAuthorityEvidence,
    lineage: CompletedLineage,
    measured_outputs: tuple[MeasuredCompletedOutput, ...],
) -> ImmutableAuthorityEvidence:
    """Rehash exact manifests/receipt sets and combine a fresh lineage traversal."""
    manifest_files = tuple(rehash_immutable_file(measured_output.manifest_file) for measured_output in measured_outputs)
    receipt_files: list[ImmutableFileEvidence] = []
    for measured_output in measured_outputs:
        expected_receipt_files = measured_output.receipt_files
        if not expected_receipt_files:
            raise RuntimeError(f"Completed run has no immutable receipts: {measured_output.output.run_directory}.")
        commits_directory = Path(expected_receipt_files[0].absolute_path).parent
        observed_receipt_paths = direct_regular_files(
            commits_directory,
            required_suffix=".json",
            role="receipt",
        )
        expected_receipt_names = {Path(file_evidence.absolute_path).name for file_evidence in expected_receipt_files}
        if set(observed_receipt_paths) != expected_receipt_names:
            raise RuntimeError(
                f"Completed run receipt files changed while collecting evidence: "
                f"{measured_output.output.run_directory}."
            )
        receipt_files.extend(rehash_immutable_file(file_evidence) for file_evidence in expected_receipt_files)
    files = owner_authority.files + lineage.immutable_files + manifest_files + tuple(receipt_files)
    return ImmutableAuthorityEvidence(
        files=files,
        aggregate_sha256=aggregate_immutable_files_sha256(files),
    )


def rehash_immutable_file(file_evidence: ImmutableFileEvidence) -> ImmutableFileEvidence:
    """Re-read one exact immutable path for the final authority comparison."""
    path = Path(file_evidence.absolute_path)
    raw_bytes = read_required_file_bytes(path, role="immutable authority record")
    return immutable_file_evidence(path, raw_bytes)


def verify_completed_lineage(
    output_root: Path,
    artifacts: CompletedRunArtifacts,
    *,
    run_label: str,
) -> CompletedLineage:
    """Verify that CLI artifacts name the finalized completed lineage leaf."""
    control_directory = output_root / ".g-output"
    genesis_path = control_directory / "genesis.json"
    genesis_bytes = read_required_file_bytes(genesis_path, role="lineage genesis")
    genesis = parse_json_mapping_bytes(
        genesis_bytes,
        path=genesis_path,
        role="lineage genesis",
    )
    immutable_files = [immutable_file_evidence(genesis_path, genesis_bytes)]
    require_exact_fields(
        genesis,
        frozenset(
            {
                "record_kind",
                "schema_version",
                "run_set_id",
                "attempt_id",
                "chunk_plan_sha256",
                "phenotypes",
            }
        ),
        role="lineage genesis",
    )
    require_record_header(genesis, expected_kind="genesis", role="lineage genesis")
    run_set_id = require_nonempty_string(genesis, "run_set_id", role="lineage genesis")
    validate_path_identifier(run_set_id, role="run-set identifier", maximum_length=128)
    initial_attempt_id = require_nonempty_string(genesis, "attempt_id", role="lineage genesis")
    validate_path_identifier(initial_attempt_id, role="attempt identifier", maximum_length=128)
    chunk_plan_sha256 = require_nonempty_string(genesis, "chunk_plan_sha256", role="lineage genesis")
    validate_sha256(chunk_plan_sha256, role="chunk plan")
    genesis_phenotypes = read_genesis_phenotypes(genesis)
    target_attempt_id = artifacts.artifacts[0].attempt_id
    producer_attempt_ids: set[str] = set()
    visited_attempt_ids: set[str] = set()
    unmaterialized_exact_recovery_parent_ids: set[str] = set()
    current_attempt_id = initial_attempt_id
    terminal_record: dict[str, typing.Any] | None = None
    while True:
        if current_attempt_id in visited_attempt_ids:
            raise RuntimeError(f"Output lineage contains a cycle for {run_label}.")
        visited_attempt_ids.add(current_attempt_id)
        producer_attempt_ids.add(current_attempt_id)
        reject_legacy_terminal(control_directory, current_attempt_id)
        outcome = read_lineage_outcome(control_directory, current_attempt_id)
        immutable_files.append(
            ImmutableFileEvidence(
                absolute_path=str(outcome.path),
                raw_sha256=outcome.sha256,
            )
        )
        successor_path = control_directory / "successors" / f"{current_attempt_id}.json"
        successor_file = read_optional_json_mapping_bytes(
            successor_path,
            role="terminal-resume successor",
        )
        successor_exists = successor_file is not None
        successor = None if successor_file is None else successor_file[0]
        if successor_file is not None:
            immutable_files.append(immutable_file_evidence(successor_path, successor_file[1]))
        if current_attempt_id == target_attempt_id:
            if outcome.outcome_kind != "terminal_claim" or successor_exists:
                raise RuntimeError(f"CLI artifacts do not name the terminal lineage leaf for {run_label}.")
            validate_terminal_record(
                outcome.record,
                expected_attempt_id=current_attempt_id,
                expected_run_set_id=run_set_id,
                role="completed terminal",
            )
            if outcome.record["status"] != "completed":
                raise RuntimeError(f"CLI artifacts name a non-completed attempt for {run_label}.")
            immutable_files.append(verify_terminal_finalization(control_directory, outcome))
            terminal_record = outcome.record
            break
        if outcome.outcome_kind == "exact_recovery_claim":
            if successor_exists:
                raise RuntimeError(f"Exact-recovery outcome also has a normal successor for {run_label}.")
            unmaterialized_exact_recovery_parent_ids.add(current_attempt_id)
            current_attempt_id = validate_successor_record(
                outcome.record,
                expected_parent_attempt_id=current_attempt_id,
                expected_run_set_id=run_set_id,
                expected_recovery_kind="exact_nonterminal_recovery",
                expected_parent_terminal_sha256=None,
                role="exact-recovery successor",
            )
            continue
        validate_terminal_record(
            outcome.record,
            expected_attempt_id=current_attempt_id,
            expected_run_set_id=run_set_id,
            role="ancestor terminal",
        )
        if outcome.record["status"] == "completed":
            raise RuntimeError(f"Completed output attempt has a successor path before {run_label}.")
        immutable_files.append(verify_terminal_finalization(control_directory, outcome))
        if not successor_exists:
            raise RuntimeError(f"Output lineage ends before the CLI-reported attempt for {run_label}.")
        if successor is None:
            raise RuntimeError(f"Output lineage successor disappeared for {run_label}: {successor_path}.")
        current_attempt_id = validate_successor_record(
            successor,
            expected_parent_attempt_id=current_attempt_id,
            expected_run_set_id=run_set_id,
            expected_recovery_kind="terminal_resume",
            expected_parent_terminal_sha256=outcome.sha256,
            role="terminal-resume successor",
        )
    for visited_attempt_id in visited_attempt_ids - unmaterialized_exact_recovery_parent_ids:
        attempt_directory = output_root / "attempts" / visited_attempt_id
        attempt_directory_metadata = strict_path_metadata_or_none(
            attempt_directory,
            role="output attempt directory",
            follow_symlinks=False,
        )
        if attempt_directory_metadata is None or not stat.S_ISDIR(attempt_directory_metadata.st_mode):
            raise RuntimeError(
                f"Output lineage references a missing attempt directory for {run_label}: {attempt_directory}."
            )
    for exact_recovery_parent_id in unmaterialized_exact_recovery_parent_ids:
        attempt_directory = output_root / "attempts" / exact_recovery_parent_id
        attempt_directory_metadata = strict_path_metadata_or_none(
            attempt_directory,
            role="exact-recovery parent attempt directory",
            follow_symlinks=False,
        )
        if attempt_directory_metadata is not None and not stat.S_ISDIR(attempt_directory_metadata.st_mode):
            raise RuntimeError(f"Output exact-recovery parent is not a directory for {run_label}: {attempt_directory}.")
    if terminal_record is None:
        raise RuntimeError(f"Output lineage has no completed terminal for {run_label}.")
    terminal_phenotypes = read_terminal_phenotypes(terminal_record)
    artifact_output_names = tuple(artifact.output_directory_name for artifact in artifacts.artifacts)
    terminal_output_names = tuple(phenotype["output_directory_name"] for phenotype in terminal_phenotypes)
    if artifact_output_names != terminal_output_names:
        raise RuntimeError(f"CLI artifact order does not match completed terminal phenotype order for {run_label}.")
    if set(genesis_phenotypes) != set(terminal_output_names):
        raise RuntimeError(f"Completed terminal phenotype coverage differs from genesis for {run_label}.")
    phenotype_bindings: list[LineagePhenotypeBinding] = []
    for terminal_phenotype in terminal_phenotypes:
        output_directory_name = terminal_phenotype["output_directory_name"]
        genesis_phenotype = genesis_phenotypes[output_directory_name]
        phenotype_name = terminal_phenotype["phenotype_name"]
        if genesis_phenotype["phenotype_name"] != phenotype_name:
            raise RuntimeError(f"Completed terminal phenotype identity differs from genesis for {run_label}.")
        phenotype_bindings.append(
            LineagePhenotypeBinding(
                phenotype_name=phenotype_name,
                output_directory_name=output_directory_name,
                execution_plan_sha256=genesis_phenotype["execution_plan_sha256"],
                run_manifest_sha256=terminal_phenotype["run_manifest_sha256"],
            )
        )
    return CompletedLineage(
        run_set_id=run_set_id,
        attempt_id=target_attempt_id,
        chunk_plan_sha256=chunk_plan_sha256,
        producer_attempt_ids=frozenset(producer_attempt_ids),
        phenotypes=tuple(phenotype_bindings),
        immutable_files=tuple(immutable_files),
    )


def verify_released_owner_authority(
    control_directory: Path,
    *,
    run_label: str,
) -> OwnerAuthorityEvidence:
    """Require released owner authority and retain its exact raw-byte proof."""
    owner_claim_path = control_directory / "session.claim.json"
    owner_claim_bytes = read_required_file_bytes(owner_claim_path, role="output owner claim")
    root_claim = parse_json_mapping_bytes(
        owner_claim_bytes,
        path=owner_claim_path,
        role="output owner claim",
    )
    authority_files = [
        immutable_file_evidence(owner_claim_path, owner_claim_bytes),
    ]
    current_state = "active"
    current_state_id = validate_owner_claim(root_claim, role="output owner claim")
    visited_state_identifiers: set[str] = set()
    transitions_directory = control_directory / "owner-transitions"
    for _ in range(MAXIMUM_OWNER_TRANSITION_COUNT):
        if current_state_id in visited_state_identifiers:
            raise RuntimeError(f"Output owner authority contains a cycle for {run_label}.")
        visited_state_identifiers.add(current_state_id)
        transition_path = transitions_directory / f"{current_state_id}.json"
        transition_file = read_optional_json_mapping_bytes(
            transition_path,
            role="output owner transition",
        )
        if transition_file is None:
            if current_state == "active":
                raise RuntimeError(f"Completed CLI output still has active owner authority for {run_label}.")
            files = tuple(authority_files)
            return OwnerAuthorityEvidence(
                files=files,
                aggregate_sha256=aggregate_immutable_files_sha256(files),
                released_state_id=current_state_id,
            )
        transition, transition_bytes = transition_file
        authority_files.append(
            immutable_file_evidence(transition_path, transition_bytes),
        )
        transition_kind = require_nonempty_string(
            transition,
            "transition_kind",
            role="output owner transition",
        )
        if transition_kind == "graceful_release":
            require_exact_fields(
                transition,
                frozenset(
                    {
                        "transition_kind",
                        "schema_version",
                        "predecessor_claim_id",
                        "released_state_id",
                    }
                ),
                role="output owner transition",
            )
            require_schema_version_zero(transition, role="output owner transition")
            predecessor_claim_id = require_nonempty_string(
                transition,
                "predecessor_claim_id",
                role="output owner transition",
            )
            released_state_id = require_nonempty_string(
                transition,
                "released_state_id",
                role="output owner transition",
            )
            validate_path_identifier(predecessor_claim_id, role="owner claim identifier", maximum_length=128)
            validate_path_identifier(released_state_id, role="released owner state identifier", maximum_length=128)
            if (
                current_state != "active"
                or predecessor_claim_id != current_state_id
                or released_state_id == current_state_id
            ):
                raise RuntimeError(f"Output owner release transition is invalid for {run_label}.")
            current_state = "released"
            current_state_id = released_state_id
            continue
        if transition_kind == "fenced_takeover":
            require_exact_fields(
                transition,
                frozenset({"transition_kind", "schema_version", "predecessor_claim_id", "claim"}),
                role="output owner transition",
            )
            require_schema_version_zero(transition, role="output owner transition")
            predecessor_claim_id = require_nonempty_string(
                transition,
                "predecessor_claim_id",
                role="output owner transition",
            )
            validate_path_identifier(predecessor_claim_id, role="owner claim identifier", maximum_length=128)
            claim = require_mapping(transition.get("claim"), role="replacement owner claim")
            replacement_claim_id = validate_owner_claim(claim, role="replacement owner claim")
            if (
                current_state != "active"
                or predecessor_claim_id != current_state_id
                or replacement_claim_id == current_state_id
            ):
                raise RuntimeError(f"Output owner takeover transition is invalid for {run_label}.")
            current_state_id = replacement_claim_id
            continue
        if transition_kind == "acquire_after_release":
            require_exact_fields(
                transition,
                frozenset(
                    {
                        "transition_kind",
                        "schema_version",
                        "predecessor_released_state_id",
                        "claim",
                    }
                ),
                role="output owner transition",
            )
            require_schema_version_zero(transition, role="output owner transition")
            predecessor_released_state_id = require_nonempty_string(
                transition,
                "predecessor_released_state_id",
                role="output owner transition",
            )
            validate_path_identifier(
                predecessor_released_state_id,
                role="released owner state identifier",
                maximum_length=128,
            )
            claim = require_mapping(transition.get("claim"), role="replacement owner claim")
            replacement_claim_id = validate_owner_claim(claim, role="replacement owner claim")
            if (
                current_state != "released"
                or predecessor_released_state_id != current_state_id
                or replacement_claim_id == current_state_id
            ):
                raise RuntimeError(f"Output owner reacquisition transition is invalid for {run_label}.")
            current_state = "active"
            current_state_id = replacement_claim_id
            continue
        raise RuntimeError(f"Output owner transition has unsupported kind {transition_kind!r} for {run_label}.")
    raise RuntimeError(
        f"Output owner authority exceeds the supported {MAXIMUM_OWNER_TRANSITION_COUNT} transitions for {run_label}."
    )


def immutable_file_evidence(path: Path, raw_bytes: bytes) -> ImmutableFileEvidence:
    """Bind one absolute immutable path to the bytes used for validation."""
    if not path.is_absolute():
        raise RuntimeError(f"Immutable-authority evidence path is not absolute: {path}.")
    return ImmutableFileEvidence(
        absolute_path=str(path),
        raw_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def aggregate_immutable_files_sha256(
    files: tuple[ImmutableFileEvidence, ...],
) -> str:
    """Hash ordered length-prefixed absolute paths and raw record digests."""
    digest = hashlib.sha256()
    for file_evidence in files:
        try:
            encoded_path = file_evidence.absolute_path.encode("utf-8")
        except UnicodeEncodeError as error:
            raise RuntimeError(f"Immutable-authority evidence path is not valid UTF-8: {error}") from error
        if (
            not Path(file_evidence.absolute_path).is_absolute()
            or b"\0" in encoded_path
            or len(encoded_path) > 2**64 - 1
        ):
            raise RuntimeError("Immutable-authority evidence contains an invalid absolute path.")
        validate_sha256(file_evidence.raw_sha256, role="immutable-authority record")
        raw_sha256 = bytes.fromhex(file_evidence.raw_sha256)
        digest.update(len(encoded_path).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded_path)
        digest.update(raw_sha256)
    return digest.hexdigest()


def validate_owner_claim(claim: dict[str, typing.Any], *, role: str) -> str:
    """Validate one schema-v0 owner claim and return its identifier."""
    require_exact_fields(
        claim,
        frozenset({"schema_version", "claim_id", "host_name", "process_id"}),
        role=role,
    )
    require_schema_version_zero(claim, role=role)
    claim_id = require_nonempty_string(claim, "claim_id", role=role)
    validate_path_identifier(claim_id, role="owner claim identifier", maximum_length=128)
    host_name = require_nonempty_string(claim, "host_name", role=role)
    if (
        not host_name.strip()
        or len(host_name.encode()) > 255
        or any(ord(character) < 32 or 127 <= ord(character) <= 159 for character in host_name)
    ):
        raise RuntimeError(f"{role.capitalize()} host name is invalid.")
    process_id = require_integer(claim, "process_id", role=role)
    if process_id <= 0 or process_id > 2**32 - 1:
        raise RuntimeError(f"{role.capitalize()} process identifier is outside the unsigned 32-bit range.")
    return claim_id


def read_genesis_phenotypes(genesis: dict[str, typing.Any]) -> dict[str, dict[str, str]]:
    """Read strict genesis phenotype contracts keyed by output name."""
    raw_phenotypes = require_mapping_list(genesis.get("phenotypes"), role="lineage genesis phenotypes")
    if not raw_phenotypes:
        raise RuntimeError("Lineage genesis contains no phenotypes.")
    phenotypes: dict[str, dict[str, str]] = {}
    phenotype_names: set[str] = set()
    for raw_phenotype in raw_phenotypes:
        require_exact_fields(
            raw_phenotype,
            frozenset({"phenotype_name", "output_directory_name", "execution_plan_sha256"}),
            role="lineage genesis phenotype",
        )
        phenotype_name = require_nonempty_string(raw_phenotype, "phenotype_name", role="lineage genesis phenotype")
        output_directory_name = require_nonempty_string(
            raw_phenotype,
            "output_directory_name",
            role="lineage genesis phenotype",
        )
        execution_plan_sha256 = require_nonempty_string(
            raw_phenotype,
            "execution_plan_sha256",
            role="lineage genesis phenotype",
        )
        validate_safe_component(output_directory_name, role="phenotype output directory")
        validate_sha256(execution_plan_sha256, role="execution plan")
        if phenotype_name in phenotype_names or output_directory_name in phenotypes:
            raise RuntimeError("Lineage genesis contains duplicate phenotype bindings.")
        phenotype_names.add(phenotype_name)
        phenotypes[output_directory_name] = {
            "phenotype_name": phenotype_name,
            "execution_plan_sha256": execution_plan_sha256,
        }
    return phenotypes


def read_terminal_phenotypes(terminal: dict[str, typing.Any]) -> tuple[dict[str, str], ...]:
    """Read strict terminal phenotype bindings in production order."""
    raw_phenotypes = require_mapping_list(terminal.get("phenotypes"), role="terminal phenotypes")
    if not raw_phenotypes:
        raise RuntimeError("Output terminal contains no phenotype bindings.")
    phenotypes: list[dict[str, str]] = []
    phenotype_names: set[str] = set()
    output_names: set[str] = set()
    for raw_phenotype in raw_phenotypes:
        require_exact_fields(
            raw_phenotype,
            frozenset({"phenotype_name", "output_directory_name", "run_manifest_sha256"}),
            role="terminal phenotype",
        )
        phenotype_name = require_nonempty_string(raw_phenotype, "phenotype_name", role="terminal phenotype")
        output_directory_name = require_nonempty_string(
            raw_phenotype,
            "output_directory_name",
            role="terminal phenotype",
        )
        manifest_sha256 = require_nonempty_string(
            raw_phenotype,
            "run_manifest_sha256",
            role="terminal phenotype",
        )
        validate_safe_component(output_directory_name, role="phenotype output directory")
        validate_sha256(manifest_sha256, role="run manifest")
        if phenotype_name in phenotype_names or output_directory_name in output_names:
            raise RuntimeError("Output terminal contains duplicate phenotype bindings.")
        phenotype_names.add(phenotype_name)
        output_names.add(output_directory_name)
        phenotypes.append(
            {
                "phenotype_name": phenotype_name,
                "output_directory_name": output_directory_name,
                "run_manifest_sha256": manifest_sha256,
            }
        )
    return tuple(phenotypes)


def read_lineage_outcome(control_directory: Path, attempt_id: str) -> LineageOutcome:
    """Read one strict immutable attempt outcome."""
    outcome_path = control_directory / "outcomes" / f"{attempt_id}.json"
    outcome_bytes = read_required_file_bytes(outcome_path, role="attempt outcome")
    outcome = parse_json_mapping_bytes(outcome_bytes, path=outcome_path, role="attempt outcome")
    require_exact_fields(outcome, frozenset({"outcome_kind", "record"}), role="attempt outcome")
    outcome_kind = require_nonempty_string(outcome, "outcome_kind", role="attempt outcome")
    if outcome_kind not in {"terminal_claim", "exact_recovery_claim"}:
        raise RuntimeError(f"Attempt outcome has unsupported kind {outcome_kind!r}: {outcome_path}.")
    record = require_mapping(outcome.get("record"), role="attempt outcome record")
    return LineageOutcome(
        outcome_kind=outcome_kind,
        record=record,
        path=outcome_path,
        sha256=hashlib.sha256(outcome_bytes).hexdigest(),
    )


def validate_terminal_record(
    terminal: dict[str, typing.Any],
    *,
    expected_attempt_id: str,
    expected_run_set_id: str,
    role: str,
) -> None:
    """Validate one strict terminal record and its identity."""
    require_exact_fields(
        terminal,
        frozenset(
            {
                "record_kind",
                "schema_version",
                "run_set_id",
                "attempt_id",
                "status",
                "interrupted_signal",
                "failure_reason",
                "phenotypes",
            }
        ),
        role=role,
    )
    require_record_header(terminal, expected_kind="terminal", role=role)
    if terminal.get("run_set_id") != expected_run_set_id or terminal.get("attempt_id") != expected_attempt_id:
        raise RuntimeError(f"{role.capitalize()} is not bound to its traversed attempt and run set.")
    status = terminal.get("status")
    terminal_details = (terminal.get("interrupted_signal"), terminal.get("failure_reason"))
    valid_details = (
        (status == "completed" and terminal_details == (None, None))
        or (
            status == "interrupted"
            and isinstance(terminal_details[0], str)
            and bool(terminal_details[0].strip())
            and terminal_details[1] is None
        )
        or (
            status == "failed"
            and terminal_details[0] is None
            and isinstance(terminal_details[1], str)
            and bool(terminal_details[1].strip())
        )
    )
    if not valid_details:
        raise RuntimeError(f"{role.capitalize()} has inconsistent terminal details.")
    read_terminal_phenotypes(terminal)


def validate_successor_record(
    successor: dict[str, typing.Any],
    *,
    expected_parent_attempt_id: str,
    expected_run_set_id: str,
    expected_recovery_kind: str,
    expected_parent_terminal_sha256: str | None,
    role: str,
) -> str:
    """Validate one strict successor and return its child attempt."""
    require_exact_fields(
        successor,
        frozenset(
            {
                "record_kind",
                "schema_version",
                "run_set_id",
                "parent_attempt_id",
                "attempt_id",
                "recovery_kind",
                "parent_terminal_sha256",
            }
        ),
        role=role,
    )
    require_record_header(successor, expected_kind="successor", role=role)
    if (
        successor.get("run_set_id") != expected_run_set_id
        or successor.get("parent_attempt_id") != expected_parent_attempt_id
        or successor.get("recovery_kind") != expected_recovery_kind
        or successor.get("parent_terminal_sha256") != expected_parent_terminal_sha256
    ):
        raise RuntimeError(f"{role.capitalize()} has a stale or mismatched lineage binding.")
    attempt_id = require_nonempty_string(successor, "attempt_id", role=role)
    validate_path_identifier(attempt_id, role="attempt identifier", maximum_length=128)
    if attempt_id == expected_parent_attempt_id:
        raise RuntimeError(f"{role.capitalize()} repeats its parent attempt.")
    return attempt_id


def verify_terminal_finalization(
    control_directory: Path,
    outcome: LineageOutcome,
) -> ImmutableFileEvidence:
    """Verify and retain a finalization against raw terminal-outcome bytes."""
    attempt_id = require_nonempty_string(outcome.record, "attempt_id", role="terminal")
    finalization_path = control_directory / "terminal-finalizations" / f"{attempt_id}.json"
    finalization_bytes = read_required_file_bytes(
        finalization_path,
        role="terminal finalization",
    )
    finalization = parse_json_mapping_bytes(
        finalization_bytes,
        path=finalization_path,
        role="terminal finalization",
    )
    require_exact_fields(
        finalization,
        frozenset({"record_kind", "schema_version", "run_set_id", "attempt_id", "terminal_claim_sha256"}),
        role="terminal finalization",
    )
    require_record_header(finalization, expected_kind="terminal_finalization", role="terminal finalization")
    if (
        finalization.get("run_set_id") != outcome.record.get("run_set_id")
        or finalization.get("attempt_id") != attempt_id
        or finalization.get("terminal_claim_sha256") != outcome.sha256
    ):
        raise RuntimeError(f"Terminal finalization has a stale terminal binding: {finalization_path}.")
    return immutable_file_evidence(finalization_path, finalization_bytes)


def reject_legacy_terminal(control_directory: Path, attempt_id: str) -> None:
    """Reject removed mutable-terminal layout artifacts."""
    legacy_terminal_path = control_directory / "terminals" / f"{attempt_id}.json"
    if (
        strict_path_metadata_or_none(
            legacy_terminal_path,
            role="legacy output terminal",
            follow_symlinks=True,
        )
        is not None
    ):
        raise RuntimeError(f"Output lineage contains unsupported legacy terminal: {legacy_terminal_path}.")


def measure_completed_output_artifact(
    artifact: CompletedRunArtifact,
    phenotype: LineagePhenotypeBinding,
    lineage: CompletedLineage,
) -> MeasuredCompletedOutput:
    """Verify one completed phenotype manifest, receipts, and Parquet parts."""
    run_directory = artifact.run_directory
    manifest_path = run_directory / "run_manifest.json"
    manifest_bytes = read_required_file_bytes(manifest_path, role="run manifest")
    try:
        manifest_text = manifest_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"Run manifest at {manifest_path} is not valid UTF-8: {error}") from error
    payload = require_mapping(
        parse_strict_json(manifest_text, role=f"run manifest at {manifest_path}"),
        role="run manifest",
    )
    require_exact_fields(
        payload,
        frozenset(
            {
                "schema_version",
                "output_schema_version",
                "execution_plan",
                "execution_plan_hash",
                "attempt_manifest_schema_version",
                "run_set_id",
                "attempt_id",
                "phenotype_name",
                "output_directory_name",
                "chunk_plan_hash",
                "status",
                "committed_parts",
                "committed_chunks",
                "command",
                "runtime",
            }
        ),
        role="completed run manifest",
    )
    for version_field in ("schema_version", "output_schema_version", "attempt_manifest_schema_version"):
        if require_integer(payload, version_field, role="completed run manifest") != 0:
            raise RuntimeError(f"Completed run manifest has unsupported {version_field}: {manifest_path}.")
    expected_identity = {
        "run_set_id": lineage.run_set_id,
        "attempt_id": lineage.attempt_id,
        "phenotype_name": phenotype.phenotype_name,
        "output_directory_name": phenotype.output_directory_name,
        "execution_plan_hash": phenotype.execution_plan_sha256,
        "chunk_plan_hash": lineage.chunk_plan_sha256,
        "status": "completed",
    }
    for field_name, expected_value in expected_identity.items():
        if payload.get(field_name) != expected_value:
            raise RuntimeError(f"Completed run manifest field {field_name!r} has a stale lineage binding.")
    observed_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if observed_manifest_sha256 != phenotype.run_manifest_sha256:
        raise RuntimeError(f"Completed run manifest does not match its terminal hash: {manifest_path}.")
    verify_execution_plan_hash(manifest_text, payload, phenotype.execution_plan_sha256, manifest_path)
    execution_plan = require_mapping(payload.get("execution_plan"), role="manifest execution plan")
    expected_variant_count = require_integer(execution_plan, "variant_count", role="manifest execution plan")
    if expected_variant_count <= 0:
        raise RuntimeError(f"Completed run manifest has a non-positive variant count: {manifest_path}.")
    manifest_receipts = require_mapping_list(payload.get("committed_parts"), role="manifest committed parts")
    manifest_chunks = require_mapping_list(payload.get("committed_chunks"), role="manifest committed chunks")
    if not manifest_receipts or not manifest_chunks:
        raise RuntimeError(f"Completed run manifest contains no committed output: {manifest_path}.")
    verified_parts = verify_receipts_and_parts(
        artifact,
        phenotype,
        lineage,
        manifest_receipts,
    )
    if manifest_chunks != verified_parts.chunks:
        raise RuntimeError(f"Completed run manifest chunks differ from immutable receipts: {manifest_path}.")
    row_count = validate_chunk_coverage(
        manifest_chunks,
        expected_variant_count=expected_variant_count,
        expected_chunk_plan_sha256=lineage.chunk_plan_sha256,
        run_directory=run_directory,
    )
    if verified_parts.row_count != expected_variant_count:
        raise RuntimeError(
            f"Completed run Parquet rows differ from its manifest for {run_directory}: "
            f"expected {expected_variant_count}, observed {verified_parts.row_count}."
        )
    return MeasuredCompletedOutput(
        output=CompletedOutputEvidence(
            run_directory=str(run_directory),
            row_count=row_count,
            committed_chunk_count=len(manifest_chunks),
            parquet_file_count=len(verified_parts.files),
            parquet_total_bytes=sum(path.stat().st_size for path in verified_parts.files),
            parquet_sha256=hash_paths(verified_parts.files, run_directory),
            parquet_paths=tuple(str(path) for path in verified_parts.files),
            schema=str(verified_parts.schema),
            schema_metadata=verified_parts.schema_metadata,
            parquet_metadata=verified_parts.parquet_metadata,
            manifest_path=str(manifest_path),
            manifest_sha256=observed_manifest_sha256,
            manifest=payload,
        ),
        manifest_file=immutable_file_evidence(manifest_path, manifest_bytes),
        receipt_files=verified_parts.receipt_files,
    )


def verify_receipts_and_parts(
    artifact: CompletedRunArtifact,
    phenotype: LineagePhenotypeBinding,
    lineage: CompletedLineage,
    manifest_receipts: list[dict[str, typing.Any]],
) -> VerifiedParts:
    """Verify receipt files, embedded footers, raw bytes, and direct part set."""
    receipts_by_name: dict[str, dict[str, typing.Any]] = {}
    expected_part_names: set[str] = set()
    receipt_identifiers: list[str] = []
    for receipt in manifest_receipts:
        footer = validate_receipt(receipt, phenotype=phenotype, lineage=lineage)
        receipt_file_name = typing.cast("str", footer["receipt_file_name"])
        part_file_name = typing.cast("str", footer["part_file_name"])
        if receipt_file_name in receipts_by_name or part_file_name in expected_part_names:
            raise RuntimeError(
                f"Completed run contains duplicate receipt or part identifiers: {artifact.run_directory}."
            )
        receipts_by_name[receipt_file_name] = receipt
        expected_part_names.add(part_file_name)
        receipt_identifiers.append(typing.cast("str", footer["receipt_id"]))
    if receipt_identifiers != sorted(receipt_identifiers):
        raise RuntimeError(f"Completed run receipts are not sorted by identifier: {artifact.run_directory}.")
    commit_files = direct_regular_files(
        artifact.run_directory / "commits",
        required_suffix=".json",
        role="receipt",
    )
    if set(commit_files) != set(receipts_by_name):
        raise RuntimeError(f"Completed run receipt files differ from its manifest: {artifact.run_directory}.")
    receipt_files: list[ImmutableFileEvidence] = []
    for receipt_file_name, receipt in receipts_by_name.items():
        receipt_path = commit_files[receipt_file_name]
        receipt_bytes = read_required_file_bytes(receipt_path, role="part receipt")
        observed_receipt = parse_json_mapping_bytes(
            receipt_bytes,
            path=receipt_path,
            role="part receipt",
        )
        validate_receipt(observed_receipt, phenotype=phenotype, lineage=lineage)
        if observed_receipt != receipt:
            raise RuntimeError(f"Immutable receipt differs from the completed manifest: {receipt_path}.")
        receipt_files.append(immutable_file_evidence(receipt_path, receipt_bytes))
    part_files = direct_regular_files(artifact.parts_directory, required_suffix=".parquet", role="Parquet part")
    if set(part_files) != expected_part_names:
        raise RuntimeError(f"Completed run Parquet parts differ from its receipts: {artifact.run_directory}.")
    flattened_chunks: list[dict[str, typing.Any]] = []
    parquet_row_count = 0
    logical_schema: pa.Schema | None = None
    schema_metadata: dict[str, str] | None = None
    parquet_metadata: list[dict[str, str]] = []
    for receipt_file_name in sorted(receipts_by_name):
        receipt = receipts_by_name[receipt_file_name]
        footer = require_mapping(receipt.get("footer"), role="part receipt footer")
        part_path = part_files[typing.cast("str", footer["part_file_name"])]
        if part_path.stat().st_size != receipt["part_size_bytes"] or sha256_file(part_path) != receipt["part_sha256"]:
            raise RuntimeError(f"Parquet part raw bytes do not match its immutable receipt: {part_path}.")
        parquet_file = pyarrow.parquet.ParquetFile(part_path)
        raw_metadata = parquet_file.metadata.metadata
        encoded_footer = None if raw_metadata is None else raw_metadata.get(PART_BINDING_METADATA_KEY)
        if encoded_footer is None:
            raise RuntimeError(f"Parquet part has no bound schema-v0 footer: {part_path}.")
        try:
            footer_text = encoded_footer.decode("utf-8")
        except UnicodeDecodeError as error:
            raise RuntimeError(f"Parquet part has invalid bound footer metadata: {part_path}.") from error
        embedded_footer = require_mapping(
            parse_strict_json(footer_text, role=f"embedded part footer at {part_path}"),
            role="embedded part footer",
        )
        validate_receipt_footer(embedded_footer, phenotype=phenotype, lineage=lineage)
        if embedded_footer != footer:
            raise RuntimeError(f"Parquet part footer differs from its immutable receipt: {part_path}.")
        footer_chunks = require_mapping_list(footer.get("chunks"), role="part footer chunks")
        expected_part_rows = sum(
            require_integer(chunk, "row_count", role="part footer chunk") for chunk in footer_chunks
        )
        if parquet_file.metadata.num_rows != expected_part_rows:
            raise RuntimeError(f"Parquet part row count differs from its bound chunks: {part_path}.")
        parquet_row_count += parquet_file.metadata.num_rows
        flattened_chunks.extend(footer_chunks)
        candidate_schema = parquet_file.schema_arrow
        candidate_logical_schema = candidate_schema.remove_metadata()
        if not CANONICAL_OUTPUT_SCHEMA.equals(candidate_logical_schema, check_metadata=True):
            raise RuntimeError(f"Parquet part does not match the canonical output schema: {part_path}.")
        if logical_schema is None:
            logical_schema = candidate_logical_schema
            schema_metadata = decode_comparable_metadata(candidate_schema.metadata)
        elif not logical_schema.equals(candidate_logical_schema, check_metadata=True):
            raise RuntimeError(f"Parquet schema changed within {artifact.run_directory}.")
        parquet_metadata.append(decode_comparable_metadata(raw_metadata))
    if logical_schema is None or schema_metadata is None:
        raise RuntimeError(f"Completed run has no Parquet schema: {artifact.run_directory}.")
    flattened_chunks.sort(key=lambda chunk: require_integer(chunk, "chunk_identifier", role="part footer chunk"))
    return VerifiedParts(
        chunks=flattened_chunks,
        files=[part_files[name] for name in sorted(part_files)],
        row_count=parquet_row_count,
        schema=logical_schema,
        schema_metadata=schema_metadata,
        parquet_metadata=tuple(parquet_metadata),
        receipt_files=tuple(receipt_files),
    )


def validate_receipt(
    receipt: dict[str, typing.Any],
    *,
    phenotype: LineagePhenotypeBinding,
    lineage: CompletedLineage,
) -> dict[str, typing.Any]:
    """Validate one schema-v0 immutable part receipt."""
    require_exact_fields(
        receipt,
        frozenset({"footer", "part_size_bytes", "part_sha256"}),
        role="part receipt",
    )
    part_size_bytes = require_integer(receipt, "part_size_bytes", role="part receipt")
    if part_size_bytes <= 0:
        raise RuntimeError("Part receipt byte size must be positive.")
    part_sha256 = require_nonempty_string(receipt, "part_sha256", role="part receipt")
    validate_sha256(part_sha256, role="Parquet part")
    footer = require_mapping(receipt.get("footer"), role="part receipt footer")
    validate_receipt_footer(footer, phenotype=phenotype, lineage=lineage)
    return footer


def validate_receipt_footer(
    footer: dict[str, typing.Any],
    *,
    phenotype: LineagePhenotypeBinding,
    lineage: CompletedLineage,
) -> None:
    """Validate one schema-v0 immutable part footer."""
    require_exact_fields(
        footer,
        frozenset(
            {
                "schema_version",
                "run_set_id",
                "attempt_id",
                "phenotype_name",
                "execution_plan_sha256",
                "chunk_plan_sha256",
                "part_id",
                "part_file_name",
                "receipt_id",
                "receipt_file_name",
                "chunks",
            }
        ),
        role="part receipt footer",
    )
    if require_integer(footer, "schema_version", role="part receipt footer") != 0:
        raise RuntimeError("Part receipt footer has an unsupported schema version.")
    footer_phenotype_name = require_nonempty_string(footer, "phenotype_name", role="part receipt footer")
    if not footer_phenotype_name.strip():
        raise RuntimeError("Part receipt footer phenotype name must not be whitespace-only.")
    expected_binding = {
        "run_set_id": lineage.run_set_id,
        "phenotype_name": phenotype.phenotype_name,
        "execution_plan_sha256": phenotype.execution_plan_sha256,
        "chunk_plan_sha256": lineage.chunk_plan_sha256,
    }
    for field_name, expected_value in expected_binding.items():
        if footer.get(field_name) != expected_value:
            raise RuntimeError(f"Part receipt footer has a stale {field_name} binding.")
    producer_attempt_id = require_nonempty_string(footer, "attempt_id", role="part receipt footer")
    if producer_attempt_id not in lineage.producer_attempt_ids:
        raise RuntimeError("Part receipt footer names an attempt outside the completed lineage ancestry.")
    part_id = require_nonempty_string(footer, "part_id", role="part receipt footer")
    receipt_id = require_nonempty_string(footer, "receipt_id", role="part receipt footer")
    validate_path_identifier(part_id, role="part identifier", maximum_length=128)
    validate_path_identifier(receipt_id, role="receipt identifier", maximum_length=128)
    if part_id != receipt_id:
        raise RuntimeError("Schema-v0 part and receipt identifiers must match.")
    if footer.get("part_file_name") != f"{part_id}.parquet":
        raise RuntimeError("Part receipt footer has an invalid Parquet file name.")
    if footer.get("receipt_file_name") != f"{receipt_id}.json":
        raise RuntimeError("Part receipt footer has an invalid receipt file name.")
    chunks = require_mapping_list(footer.get("chunks"), role="part receipt footer chunks")
    if not chunks:
        raise RuntimeError("Part receipt footer contains no chunks.")
    previous_chunk_identifier: int | None = None
    for chunk in chunks:
        validate_chunk(chunk, expected_part_file_name=f"{part_id}.parquet")
        chunk_identifier = require_integer(chunk, "chunk_identifier", role="part footer chunk")
        if previous_chunk_identifier is not None and chunk_identifier <= previous_chunk_identifier:
            raise RuntimeError("Part receipt footer chunks are not strictly ordered.")
        previous_chunk_identifier = chunk_identifier


def validate_chunk(chunk: dict[str, typing.Any], *, expected_part_file_name: str) -> None:
    """Validate one exact output chunk record."""
    require_exact_fields(
        chunk,
        frozenset(
            {
                "chunk_identifier",
                "variant_start_index",
                "variant_stop_index",
                "row_count",
                "chunk_file_name",
            }
        ),
        role="output chunk",
    )
    chunk_identifier = require_integer(chunk, "chunk_identifier", role="output chunk")
    variant_start_index = require_integer(chunk, "variant_start_index", role="output chunk")
    variant_stop_index = require_integer(chunk, "variant_stop_index", role="output chunk")
    row_count = require_integer(chunk, "row_count", role="output chunk")
    if (
        chunk_identifier < 0
        or chunk_identifier != variant_start_index
        or variant_stop_index <= variant_start_index
        or row_count != variant_stop_index - variant_start_index
    ):
        raise RuntimeError("Output chunk has invalid identifier, geometry, or row count.")
    if chunk.get("chunk_file_name") != expected_part_file_name:
        raise RuntimeError("Output chunk names a different Parquet part.")


def validate_chunk_coverage(
    chunks: list[dict[str, typing.Any]],
    *,
    expected_variant_count: int,
    expected_chunk_plan_sha256: str,
    run_directory: Path,
) -> int:
    """Require exact contiguous coverage and canonical chunk-plan binding."""
    next_variant_index = 0
    chunk_identifiers: set[int] = set()
    chunk_geometries: list[dict[str, int]] = []
    for chunk in chunks:
        chunk_identifier = require_integer(chunk, "chunk_identifier", role="manifest chunk")
        variant_start_index = require_integer(chunk, "variant_start_index", role="manifest chunk")
        variant_stop_index = require_integer(chunk, "variant_stop_index", role="manifest chunk")
        row_count = require_integer(chunk, "row_count", role="manifest chunk")
        if chunk_identifier in chunk_identifiers or variant_start_index != next_variant_index:
            raise RuntimeError(f"Completed run chunks do not exactly cover the variant range: {run_directory}.")
        chunk_identifiers.add(chunk_identifier)
        if (
            chunk_identifier != variant_start_index
            or variant_stop_index <= variant_start_index
            or row_count != variant_stop_index - variant_start_index
        ):
            raise RuntimeError(f"Completed run contains invalid chunk geometry: {run_directory}.")
        chunk_geometries.append(
            {
                "chunk_identifier": chunk_identifier,
                "variant_start_index": variant_start_index,
                "variant_stop_index": variant_stop_index,
                "row_count": row_count,
            }
        )
        next_variant_index = variant_stop_index
    if next_variant_index != expected_variant_count:
        raise RuntimeError(f"Completed run chunks do not cover its manifest variant count: {run_directory}.")
    chunk_plan = {"algorithm": "sha256", "chunks": chunk_geometries}
    if sha256_json_value(chunk_plan) != expected_chunk_plan_sha256:
        raise RuntimeError(f"Completed run chunks do not match the lineage chunk-plan hash: {run_directory}.")
    return next_variant_index


def verify_execution_plan_hash(
    manifest_text: str,
    manifest: dict[str, typing.Any],
    expected_sha256: str,
    manifest_path: Path,
) -> None:
    """Recompute an execution-plan hash without normalizing numeric lexemes."""
    lexical_manifest = require_mapping(
        parse_strict_json(
            manifest_text,
            role=f"lexical run manifest at {manifest_path}",
            preserve_numeric_tokens=True,
        ),
        role="lexical run manifest",
    )
    lexical_execution_plan = lexical_manifest.get("execution_plan")
    if lexical_execution_plan is None or sha256_canonical_json(lexical_execution_plan) != expected_sha256:
        raise RuntimeError(f"Run manifest execution plan does not match its declared hash: {manifest_path}.")
    if manifest.get("execution_plan_hash") != expected_sha256:
        raise RuntimeError(f"Run manifest execution-plan hash differs from lineage: {manifest_path}.")


def sha256_json_value(value: typing.Any) -> str:
    """Hash a JSON value using compact sorted UTF-8 serialization."""
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise RuntimeError(f"Cannot encode canonical JSON value: {error}") from error
    return hashlib.sha256(encoded).hexdigest()


def sha256_canonical_json(value: typing.Any) -> str:
    """Hash a parsed JSON value while preserving numeric token spelling."""
    return hashlib.sha256(encode_canonical_json(value)).hexdigest()


def encode_canonical_json(value: typing.Any) -> bytes:
    """Encode JSON with sorted keys, compact separators, and raw numeric tokens."""
    if isinstance(value, JsonNumber):
        return value.text.encode()
    if value is None:
        return b"null"
    if value is True:
        return b"true"
    if value is False:
        return b"false"
    if isinstance(value, str):
        try:
            return json.dumps(value, ensure_ascii=False).encode()
        except UnicodeEncodeError as error:
            raise RuntimeError(f"Canonical JSON string is not valid UTF-8: {error}") from error
    if isinstance(value, list):
        return b"[" + b",".join(encode_canonical_json(item) for item in value) + b"]"
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise RuntimeError("Canonical JSON object contains a non-string key.")
        encoded_items = (
            json.dumps(key, ensure_ascii=False).encode() + b":" + encode_canonical_json(value[key])
            for key in sorted(value)
        )
        return b"{" + b",".join(encoded_items) + b"}"
    raise RuntimeError(f"Canonical JSON contains unsupported value {type(value).__name__}.")


def direct_regular_files(directory: Path, *, required_suffix: str, role: str) -> dict[str, Path]:
    """Read the exact direct regular-file set for a completed directory."""
    if not directory.is_dir():
        raise RuntimeError(f"Completed output has no {role} directory: {directory}.")
    files: dict[str, Path] = {}
    for path in directory.iterdir():
        if path.name.startswith(".") and path.name.endswith(".tmp"):
            continue
        if path.is_symlink() or not path.is_file() or path.suffix != required_suffix:
            raise RuntimeError(f"Completed {role} directory contains an unexpected entry: {path}.")
        files[path.name] = path
    return files


def strict_path_metadata_or_none(
    path: Path,
    *,
    role: str,
    follow_symlinks: bool,
) -> os.stat_result | None:
    """Inspect a path while treating only genuine absence as optional."""
    try:
        return path.stat() if follow_symlinks else path.lstat()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise RuntimeError(f"Failed to inspect {role} at {path}: {error}") from error


def read_required_file_bytes(path: Path, *, role: str) -> bytes:
    """Read one required file without converting non-absence errors to missing."""
    try:
        return path.read_bytes()
    except OSError as error:
        raise RuntimeError(f"Failed to read {role} at {path}: {error}") from error


def parse_json_mapping_bytes(raw_bytes: bytes, *, path: Path, role: str) -> dict[str, typing.Any]:
    """Decode and parse one strict JSON object from an already-read byte buffer."""
    try:
        json_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"Failed to read valid {role} JSON at {path}: {error}") from error
    raw_value = parse_strict_json(json_text, role=f"{role} at {path}")
    return require_mapping(raw_value, role=role)


def read_json_mapping(path: Path, *, role: str) -> dict[str, typing.Any]:
    """Read a required JSON object."""
    raw_bytes = read_required_file_bytes(path, role=role)
    return parse_json_mapping_bytes(raw_bytes, path=path, role=role)


def read_optional_json_mapping(path: Path, *, role: str) -> dict[str, typing.Any] | None:
    """Read an optional JSON object while failing closed on non-absence errors."""
    raw_mapping = read_optional_json_mapping_bytes(path, role=role)
    return None if raw_mapping is None else raw_mapping[0]


def read_optional_json_mapping_bytes(
    path: Path,
    *,
    role: str,
) -> tuple[dict[str, typing.Any], bytes] | None:
    """Read optional raw JSON bytes and parse the same byte buffer."""
    try:
        raw_bytes = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise RuntimeError(f"Failed to read optional {role} at {path}: {error}") from error
    return parse_json_mapping_bytes(raw_bytes, path=path, role=role), raw_bytes


def require_mapping(value: object, *, role: str) -> dict[str, typing.Any]:
    """Require a JSON object with string keys."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise RuntimeError(f"{role.capitalize()} must contain a JSON object.")
    return typing.cast("dict[str, typing.Any]", value)


def require_mapping_list(value: object, *, role: str) -> list[dict[str, typing.Any]]:
    """Require a JSON array containing only objects."""
    if not isinstance(value, list):
        raise RuntimeError(f"{role.capitalize()} must contain a JSON array.")
    return [require_mapping(item, role=role) for item in value]


def require_exact_fields(payload: dict[str, typing.Any], expected_fields: frozenset[str], *, role: str) -> None:
    """Require the exact field set for an immutable schema-v0 record."""
    observed_fields = frozenset(payload)
    if observed_fields != expected_fields:
        raise RuntimeError(
            f"{role.capitalize()} fields differ from schema v0; "
            f"missing {sorted(expected_fields - observed_fields)}, "
            f"unexpected {sorted(observed_fields - expected_fields)}."
        )


def require_record_header(payload: dict[str, typing.Any], *, expected_kind: str, role: str) -> None:
    """Require a schema-v0 lineage record header."""
    if payload.get("record_kind") != expected_kind:
        raise RuntimeError(f"{role.capitalize()} has an unsupported record kind or schema version.")
    require_schema_version_zero(payload, role=role)


def require_schema_version_zero(payload: dict[str, typing.Any], *, role: str) -> None:
    """Require a schema-v0 integer version field."""
    if require_integer(payload, "schema_version", role=role) != 0:
        raise RuntimeError(f"{role.capitalize()} has an unsupported schema version.")


def require_nonempty_string(payload: dict[str, typing.Any], field_name: str, *, role: str) -> str:
    """Read one required nonempty string field."""
    value = payload.get(field_name)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{role.capitalize()} field {field_name!r} must be a nonempty string.")
    return value


def require_integer(payload: dict[str, typing.Any], field_name: str, *, role: str) -> int:
    """Read one required JSON integer without accepting booleans."""
    value = payload.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{role.capitalize()} field {field_name!r} must be an integer.")
    return value


def validate_sha256(digest: str, *, role: str) -> None:
    """Require one lowercase hexadecimal SHA-256 digest."""
    if len(digest) != 64 or not digest.isascii() or any(character not in "0123456789abcdef" for character in digest):
        raise RuntimeError(f"{role.capitalize()} SHA-256 must contain 64 lowercase hexadecimal characters.")


def validate_path_identifier(identifier: str, *, role: str, maximum_length: int) -> None:
    """Require one bounded ASCII identifier accepted by native output."""
    if (
        not identifier
        or len(identifier) > maximum_length
        or not identifier.isascii()
        or any(not (character.isalnum() or character in "-_") for character in identifier)
    ):
        raise RuntimeError(f"{role.capitalize()} is not a valid path-safe identifier.")


def validate_safe_component(component: str, *, role: str) -> None:
    """Require one nonempty bounded path component."""
    try:
        component_size_bytes = len(component.encode())
    except UnicodeEncodeError as error:
        raise RuntimeError(f"{role.capitalize()} is not valid UTF-8.") from error
    if not component or component_size_bytes > 255 or component in {".", ".."} or "/" in component or "\\" in component:
        raise RuntimeError(f"{role.capitalize()} is not one safe path component.")
