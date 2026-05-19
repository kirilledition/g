"""Output persistence orchestration backed by the native Rust writer."""

from __future__ import annotations

import json
import logging
import re
import typing
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from g import _core, types

logger = logging.getLogger(__name__)


OUTPUT_COMPRESSION_CODEC = "zstd"
CHUNK_FILENAME_PATTERN = re.compile(r"^chunk_(\d+)(?:_(\d+))?\.arrow$")
RUN_MANIFEST_FILENAME = "run_manifest.json"
RUN_MANIFEST_SCHEMA_VERSION = 1
DEFAULT_WRITER_QUEUE_DEPTH = 4
DEFAULT_WRITER_THREAD_COUNT = 8


@dataclass(frozen=True)
class OutputRunPaths:
    """Filesystem paths for one chunked output run."""

    run_directory: Path
    chunks_directory: Path


@dataclass(frozen=True)
class PreparedOutputRun:
    """Prepared output run state for chunk persistence."""

    output_run_paths: OutputRunPaths
    committed_chunk_identifiers: frozenset[int]


def get_run_manifest_path(output_run_paths: OutputRunPaths) -> Path:
    """Return the run manifest path for an output run."""
    return output_run_paths.run_directory / RUN_MANIFEST_FILENAME


def resolve_output_run_paths(output_root: Path, association_mode: types.AssociationMode) -> OutputRunPaths:
    """Derive run paths from an output root and association mode."""
    run_directory = output_root if output_root.suffix == ".run" else output_root.with_suffix(f".{association_mode}.run")
    return OutputRunPaths(run_directory=run_directory, chunks_directory=run_directory / "chunks")


def build_chunk_file_name(chunk_identifier: int) -> str:
    """Build a deterministic chunk file name from a chunk identifier."""
    return f"chunk_{chunk_identifier:09d}.arrow"


def read_chunk_file(chunk_file_path: Path) -> pl.DataFrame:
    """Read one persisted chunk file into memory."""
    if chunk_file_path.suffix != ".arrow":
        message = f"Unsupported chunk file suffix: {chunk_file_path.suffix}"
        raise ValueError(message)
    return pl.read_ipc(chunk_file_path)


def scan_chunk_file(chunk_file_path: Path) -> pl.LazyFrame:
    """Open one persisted chunk file as a lazy frame."""
    if chunk_file_path.suffix != ".arrow":
        message = f"Unsupported chunk file suffix: {chunk_file_path.suffix}"
        raise ValueError(message)
    return pl.scan_ipc(chunk_file_path, rechunk=False)


def load_committed_chunk_identifiers_from_chunk_file(chunk_file_path: Path) -> frozenset[int]:
    """Load committed chunk identifiers from one chunk file."""
    chunk_identifier_values = read_chunk_file(chunk_file_path).get_column("chunk_identifier").unique().to_list()
    return frozenset(int(chunk_identifier_value) for chunk_identifier_value in chunk_identifier_values)


def scan_committed_chunk_identifiers(chunks_directory: Path) -> frozenset[int]:
    """Scan a chunks directory and return identifiers of completed chunks."""
    chunk_identifiers = _core.scan_committed_chunk_identifiers(str(chunks_directory))
    return frozenset(int(chunk_identifier) for chunk_identifier in chunk_identifiers)


def load_run_manifest(output_run_paths: OutputRunPaths) -> dict[str, typing.Any] | None:
    """Load a run manifest when present."""
    manifest_path = get_run_manifest_path(output_run_paths)
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if not isinstance(manifest, dict):
        message = f"Run manifest '{manifest_path}' must contain a JSON object."
        raise ValueError(message)
    return manifest


def write_run_manifest(output_run_paths: OutputRunPaths, manifest: dict[str, typing.Any]) -> None:
    """Atomically write a run manifest."""
    manifest_path = get_run_manifest_path(output_run_paths)
    temporary_manifest_path = manifest_path.with_suffix(".json.tmp")
    with temporary_manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")
    temporary_manifest_path.replace(manifest_path)


def validate_manifest_compatibility(
    manifest: dict[str, typing.Any],
    association_mode: types.AssociationMode,
) -> None:
    """Validate manifest fields that are known before engine setup."""
    if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
        message = "Run manifest schema version is incompatible."
        raise ValueError(message)
    if manifest.get("association_mode") != str(association_mode):
        message = "Run manifest association mode is incompatible with the requested run."
        raise ValueError(message)


def build_initial_run_manifest(association_mode: types.AssociationMode) -> dict[str, typing.Any]:
    """Build the initial manifest written before output starts."""
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "association_mode": str(association_mode),
        "bgen": None,
        "sample_count": None,
        "variant_count": None,
        "chunk_size": None,
        "binary_correction_plan": None,
        "trusted_no_missing_diploid": None,
        "committed_chunks": [],
        "finalized": False,
    }


def read_manifest_committed_chunk_identifiers(manifest: dict[str, typing.Any]) -> frozenset[int]:
    """Read committed chunk identifiers from a run manifest."""
    committed_chunks = manifest.get("committed_chunks", [])
    if not isinstance(committed_chunks, list):
        message = "Run manifest committed_chunks field must be a list."
        raise ValueError(message)
    chunk_identifiers = set[int]()
    for committed_chunk in committed_chunks:
        if not isinstance(committed_chunk, dict):
            message = "Run manifest committed chunk entries must be objects."
            raise ValueError(message)
        chunk_identifiers.add(int(committed_chunk["chunk_identifier"]))
    return frozenset(chunk_identifiers)


def require_integer_scalar(value: object, column_name: str) -> int:
    """Return a Polars scalar as an integer when it has integer semantics."""
    if isinstance(value, bool) or not isinstance(value, typing.SupportsIndex):
        message = f"Expected integer scalar for column {column_name}."
        raise ValueError(message)
    return int(value)


def validate_strict_manifest_chunks(
    output_run_paths: OutputRunPaths,
    manifest: dict[str, typing.Any],
) -> frozenset[int]:
    """Validate committed manifest chunks against Arrow files."""
    committed_chunks = manifest.get("committed_chunks", [])
    if not isinstance(committed_chunks, list):
        message = "Run manifest committed_chunks field must be a list."
        raise ValueError(message)
    committed_identifiers = set[int]()
    expected_columns = None
    for committed_chunk in committed_chunks:
        if not isinstance(committed_chunk, dict):
            message = "Run manifest committed chunk entries must be objects."
            raise ValueError(message)
        chunk_identifier = int(committed_chunk["chunk_identifier"])
        variant_start_index = int(committed_chunk["variant_start_index"])
        variant_stop_index = int(committed_chunk["variant_stop_index"])
        row_count = int(committed_chunk["row_count"])
        chunk_file_name = str(committed_chunk["chunk_file_name"])
        chunk_file_path = output_run_paths.chunks_directory / chunk_file_name
        if not chunk_file_path.exists():
            message = f"Strict resume manifest references missing chunk file: {chunk_file_path}"
            raise ValueError(message)
        chunk_frame = read_chunk_file(chunk_file_path)
        if expected_columns is None:
            expected_columns = chunk_frame.columns
        elif chunk_frame.columns != expected_columns:
            message = f"Strict resume found incompatible Arrow schema in {chunk_file_path}."
            raise ValueError(message)
        chunk_rows = chunk_frame.filter(pl.col("chunk_identifier") == chunk_identifier)
        if chunk_rows.height != row_count:
            message = f"Strict resume row count mismatch for chunk {chunk_identifier}."
            raise ValueError(message)
        observed_start = require_integer_scalar(
            chunk_rows.get_column("variant_start_index").min(),
            "variant_start_index",
        )
        observed_stop = require_integer_scalar(
            chunk_rows.get_column("variant_stop_index").max(),
            "variant_stop_index",
        )
        if observed_start != variant_start_index or observed_stop != variant_stop_index:
            message = f"Strict resume variant range mismatch for chunk {chunk_identifier}."
            raise ValueError(message)
        committed_identifiers.add(chunk_identifier)
    return frozenset(committed_identifiers)


def write_run_manifest_header(
    *,
    output_run_paths: OutputRunPaths,
    association_mode: types.AssociationMode,
    bgen_path: Path,
    sample_count: int,
    variant_count: int,
    chunk_size: int,
    binary_correction_plan: types.BinaryCorrectionPlan,
    trusted_no_missing_diploid: bool,
) -> None:
    """Write run-level manifest details once native input metadata is known."""
    manifest = load_run_manifest(output_run_paths) or build_initial_run_manifest(association_mode)
    validate_manifest_compatibility(manifest, association_mode)
    bgen_stat = bgen_path.stat()
    manifest.update(
        {
            "bgen": {
                "path": str(bgen_path.resolve()),
                "size": bgen_stat.st_size,
                "mtime_ns": bgen_stat.st_mtime_ns,
            },
            "sample_count": sample_count,
            "variant_count": variant_count,
            "chunk_size": chunk_size,
            "binary_correction_plan": {
                "method": str(binary_correction_plan.method),
                "p_threshold": binary_correction_plan.p_threshold,
                "firth_se": binary_correction_plan.firth_se,
            },
            "trusted_no_missing_diploid": trusted_no_missing_diploid,
        }
    )
    write_run_manifest(output_run_paths, manifest)


def prepare_output_run(
    *,
    output_root: Path,
    association_mode: types.AssociationMode,
    resume: bool,
    resume_mode: types.ResumeMode = types.ResumeMode.FAST,
) -> PreparedOutputRun:
    """Prepare a chunked output run directory and discover resumable state."""
    output_run_paths = resolve_output_run_paths(output_root, association_mode)
    if not resume and output_run_paths.run_directory.exists() and any(output_run_paths.run_directory.iterdir()):
        message = (
            f"Output run directory '{output_run_paths.run_directory}' already exists and is not empty. "
            "Use --resume or choose a new output path."
        )
        raise ValueError(message)
    output_run_paths.chunks_directory.mkdir(parents=True, exist_ok=True)
    manifest = load_run_manifest(output_run_paths)
    if manifest is not None:
        validate_manifest_compatibility(manifest, association_mode)
    committed_chunk_identifiers = frozenset[int]()
    if resume:
        if resume_mode == types.ResumeMode.STRICT:
            if manifest is None:
                message = "Strict resume requires run_manifest.json."
                raise ValueError(message)
            committed_chunk_identifiers = validate_strict_manifest_chunks(output_run_paths, manifest)
        elif manifest is not None:
            committed_chunk_identifiers = read_manifest_committed_chunk_identifiers(manifest)
        else:
            committed_chunk_identifiers = scan_committed_chunk_identifiers(output_run_paths.chunks_directory)
        logger.info("Resuming run with %d previously committed chunks.", len(committed_chunk_identifiers))
    elif manifest is None:
        write_run_manifest(output_run_paths, build_initial_run_manifest(association_mode))
    return PreparedOutputRun(
        output_run_paths=output_run_paths,
        committed_chunk_identifiers=committed_chunk_identifiers,
    )


def create_output_writer_session(
    output_run_paths: OutputRunPaths,
    association_mode: types.AssociationMode,
    *,
    writer_thread_count: int,
    writer_queue_depth: int,
    finalize_parquet: bool,
) -> typing.Any:
    """Create one native Rust output writer session."""
    return _core.OutputWriterSession(
        run_directory=str(output_run_paths.run_directory),
        chunks_directory=str(output_run_paths.chunks_directory),
        association_mode=str(association_mode),
        writer_thread_count=writer_thread_count,
        writer_queue_depth=writer_queue_depth,
        finalize_parquet=finalize_parquet,
    )


def iter_sorted_chunk_file_paths(chunks_directory: Path) -> tuple[Path, ...]:
    """Return all persisted chunk files in deterministic filename order."""
    if not chunks_directory.exists():
        return ()
    return tuple(
        sorted(
            child_path
            for child_path in chunks_directory.iterdir()
            if CHUNK_FILENAME_PATTERN.match(child_path.name) is not None
        )
    )


def finalize_chunks_to_parquet(
    output_run_paths: OutputRunPaths,
    association_mode: types.AssociationMode,
) -> Path:
    """Compact committed chunk files into one compressed Parquet file in Rust."""
    final_parquet_path = _core.finalize_output_run_chunks(
        run_directory=str(output_run_paths.run_directory),
        chunks_directory=str(output_run_paths.chunks_directory),
        association_mode=str(association_mode),
    )
    return Path(final_parquet_path)
