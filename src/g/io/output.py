"""Output persistence orchestration backed by the native Rust writer."""

from __future__ import annotations

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


def prepare_output_run(
    *,
    output_root: Path,
    association_mode: types.AssociationMode,
    resume: bool,
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
    committed_chunk_identifiers = frozenset[int]()
    if resume:
        committed_chunk_identifiers = scan_committed_chunk_identifiers(output_run_paths.chunks_directory)
        logger.info("Resuming run with %d previously committed chunks.", len(committed_chunk_identifiers))
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
