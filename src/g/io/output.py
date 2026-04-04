"""Output persistence and schema helpers."""

from __future__ import annotations

import logging
import queue
import re
import threading
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final

import polars as pl

from g.engine import (
    LinearChunkAccumulator,
    LinearChunkPayload,
    LogisticChunkAccumulator,
    build_chunk_payload,
)
from g.types import AssociationMode

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from g.engine import ChunkPayload

logger = logging.getLogger(__name__)


class OutputCompressionCodec(StrEnum):
    """Compression codecs used for persisted output artifacts."""

    ZSTD = "zstd"


OUTPUT_COMPRESSION_CODEC: Final[OutputCompressionCodec] = OutputCompressionCodec.ZSTD
CHUNK_FILENAME_PATTERN = re.compile(r"^chunk_(\d+)\.arrow$")
DEFAULT_WRITER_QUEUE_DEPTH = 4
DEFAULT_WRITER_TIMEOUT_SECONDS = 120.0

LINEAR_OUTPUT_SCHEMA: Final[dict[str, pl.DataType]] = {
    "chunk_identifier": pl.Int64(),
    "variant_start_index": pl.Int64(),
    "variant_stop_index": pl.Int64(),
    "chromosome": pl.String(),
    "position": pl.Int64(),
    "variant_identifier": pl.String(),
    "allele_one": pl.String(),
    "allele_two": pl.String(),
    "allele_one_frequency": pl.Float32(),
    "observation_count": pl.Int32(),
    "beta": pl.Float32(),
    "standard_error": pl.Float32(),
    "t_statistic": pl.Float32(),
    "p_value": pl.Float32(),
    "is_valid": pl.Boolean(),
}

LOGISTIC_OUTPUT_SCHEMA: Final[dict[str, pl.DataType]] = {
    "chunk_identifier": pl.Int64(),
    "variant_start_index": pl.Int64(),
    "variant_stop_index": pl.Int64(),
    "chromosome": pl.String(),
    "position": pl.Int64(),
    "variant_identifier": pl.String(),
    "allele_one": pl.String(),
    "allele_two": pl.String(),
    "allele_one_frequency": pl.Float32(),
    "observation_count": pl.Int32(),
    "beta": pl.Float32(),
    "standard_error": pl.Float32(),
    "z_statistic": pl.Float32(),
    "p_value": pl.Float32(),
    "firth_flag": pl.String(),
    "error_code": pl.String(),
    "converged": pl.Boolean(),
    "iteration_count": pl.Int32(),
    "is_valid": pl.Boolean(),
}


@dataclass(frozen=True)
class OutputRunPaths:
    """Filesystem paths for one chunked output run.

    Attributes:
        run_directory: Root directory for the run.
        chunks_directory: Directory containing Arrow IPC chunk files.

    """

    run_directory: Path
    chunks_directory: Path


@dataclass(frozen=True)
class PreparedOutputRun:
    """Prepared output run state for chunk persistence.

    Attributes:
        output_run_paths: Resolved output run paths.
        committed_chunk_identifiers: Identifiers of already-committed chunks.

    """

    output_run_paths: OutputRunPaths
    committed_chunk_identifiers: frozenset[int]


def get_output_schema(association_mode: AssociationMode) -> dict[str, pl.DataType]:
    """Return the fixed output schema for the requested mode."""
    if association_mode == AssociationMode.LINEAR:
        return LINEAR_OUTPUT_SCHEMA
    return LOGISTIC_OUTPUT_SCHEMA


def cast_frame_to_schema(data_frame: pl.DataFrame, association_mode: AssociationMode) -> pl.DataFrame:
    """Cast an output frame to the fixed mode-specific schema."""
    output_schema = get_output_schema(association_mode)
    return data_frame.select(
        [pl.col(column_name).cast(column_type).alias(column_name) for column_name, column_type in output_schema.items()]
    )


def resolve_output_run_paths(output_root: Path, association_mode: AssociationMode) -> OutputRunPaths:
    """Derive run paths from an output root and association mode.

    Args:
        output_root: User-specified output directory or prefix.
        association_mode: Association mode enum value.

    Returns:
        Resolved run and chunk directory paths.

    """
    run_directory = output_root if output_root.suffix == ".run" else output_root.with_suffix(f".{association_mode}.run")
    return OutputRunPaths(
        run_directory=run_directory,
        chunks_directory=run_directory / "chunks",
    )


def build_chunk_file_name(chunk_identifier: int) -> str:
    """Build a deterministic chunk file name from a chunk identifier.

    Args:
        chunk_identifier: Variant start index for this chunk.

    """
    return f"chunk_{chunk_identifier:09d}.arrow"


def scan_committed_chunk_identifiers(chunks_directory: Path) -> frozenset[int]:
    """Scan a chunks directory and return identifiers of completed chunks.

    Only files matching the ``chunk_NNNNNNNNN.arrow`` pattern are considered.
    Temporary ``.arrow.tmp`` files left by interrupted writes are ignored.

    Args:
        chunks_directory: Directory to scan.

    """
    if not chunks_directory.exists():
        return frozenset()
    committed_identifiers: set[int] = set()
    for child_path in chunks_directory.iterdir():
        filename_match = CHUNK_FILENAME_PATTERN.match(child_path.name)
        if filename_match is not None:
            committed_identifiers.add(int(filename_match.group(1)))
    return frozenset(committed_identifiers)


def prepare_output_run(
    *,
    output_root: Path,
    association_mode: AssociationMode,
    resume: bool,
) -> PreparedOutputRun:
    """Prepare a chunked output run directory and discover resumable state.

    When ``resume`` is ``True``, existing chunk files are scanned and their
    identifiers returned so the compute pipeline can skip them.

    Args:
        output_root: User-specified output directory or prefix.
        association_mode: Association mode enum value.
        resume: Whether to resume from previously written chunks.

    Raises:
        ValueError: If the run directory is non-empty and resume is not set.

    Returns:
        Prepared output paths and a frozenset of already-committed chunk
        identifiers.

    """
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
        logger.info(
            "Resuming run with %d previously committed chunks.",
            len(committed_chunk_identifiers),
        )
    return PreparedOutputRun(
        output_run_paths=output_run_paths,
        committed_chunk_identifiers=committed_chunk_identifiers,
    )


def build_output_frame_from_payload(chunk_payload: ChunkPayload) -> pl.DataFrame:
    """Build a Polars DataFrame from a host-side chunk payload.

    Args:
        chunk_payload: Payload containing numpy arrays ready for persistence.

    """
    row_count = len(chunk_payload.position)
    shared_columns = {
        "chunk_identifier": [chunk_payload.chunk_identifier] * row_count,
        "variant_start_index": [chunk_payload.variant_start_index] * row_count,
        "variant_stop_index": [chunk_payload.variant_stop_index] * row_count,
        "chromosome": chunk_payload.chromosome,
        "position": chunk_payload.position,
        "variant_identifier": chunk_payload.variant_identifier,
        "allele_one": chunk_payload.allele_one,
        "allele_two": chunk_payload.allele_two,
        "allele_one_frequency": chunk_payload.allele_one_frequency,
        "observation_count": chunk_payload.observation_count,
        "beta": chunk_payload.beta,
        "standard_error": chunk_payload.standard_error,
        "p_value": chunk_payload.p_value,
        "is_valid": chunk_payload.is_valid,
    }
    if isinstance(chunk_payload, LinearChunkPayload):
        output_frame = pl.DataFrame({**shared_columns, "t_statistic": chunk_payload.t_statistic})
    else:
        output_frame = pl.DataFrame(
            {
                **shared_columns,
                "z_statistic": chunk_payload.z_statistic,
                "firth_flag": chunk_payload.firth_flag,
                "error_code": chunk_payload.error_code,
                "converged": chunk_payload.converged,
                "iteration_count": chunk_payload.iteration_count,
            }
        )
    return output_frame


def write_chunk_to_disk(
    chunk_payload: ChunkPayload,
    chunks_directory: Path,
    association_mode: AssociationMode,
) -> None:
    """Atomically persist one chunk as an Arrow IPC file.

    Writes to a temporary file first, then renames. This guarantees that
    the chunk file is either fully written or absent, never partial.

    Args:
        chunk_payload: Host-side payload to persist.
        chunks_directory: Directory to write the chunk into.
        association_mode: Association mode enum value.

    """
    chunk_file_name = build_chunk_file_name(chunk_payload.chunk_identifier)
    chunk_file_path = chunks_directory / chunk_file_name
    temporary_path = chunk_file_path.with_suffix(".arrow.tmp")
    output_frame = build_output_frame_from_payload(chunk_payload)
    cast_output_frame = cast_frame_to_schema(output_frame, association_mode)
    cast_output_frame.write_ipc(temporary_path, compression=OUTPUT_COMPRESSION_CODEC.value)
    temporary_path.replace(chunk_file_path)


def run_background_writer(
    work_queue: queue.Queue[ChunkPayload | None],
    chunks_directory: Path,
    association_mode: AssociationMode,
    error_container: list[BaseException],
) -> None:
    """Background thread loop that drains the work queue and writes chunks.

    Reads payloads from ``work_queue`` until a ``None`` sentinel is received.
    If an exception occurs, it is stored in ``error_container`` so the main
    thread can re-raise it.

    Args:
        work_queue: Bounded queue of chunk payloads (``None`` = shutdown).
        chunks_directory: Target directory for Arrow IPC files.
        association_mode: Association mode enum value.
        error_container: Mutable list where a caught exception is appended.

    """
    try:
        while True:
            chunk_payload = work_queue.get()
            if chunk_payload is None:
                return
            write_chunk_to_disk(chunk_payload, chunks_directory, association_mode)
    except Exception as error:  # noqa: BLE001
        error_container.append(error)


def persist_chunked_results(
    frame_iterator: Iterator[LinearChunkAccumulator] | Iterator[LogisticChunkAccumulator],
    output_run_paths: OutputRunPaths,
    association_mode: AssociationMode,
    *,
    writer_queue_depth: int = DEFAULT_WRITER_QUEUE_DEPTH,
    writer_timeout_seconds: float = DEFAULT_WRITER_TIMEOUT_SECONDS,
) -> None:
    """Persist chunked results through a non-blocking background writer.

    A bounded queue feeds a single background thread. When the queue is full,
    the compute thread blocks until the writer drains a slot, providing
    natural backpressure without unbounded memory growth.

    Args:
        frame_iterator: Iterator yielding chunk accumulators from the engine.
        output_run_paths: Resolved output run paths.
        association_mode: Association mode enum value.
        writer_queue_depth: Maximum queued chunks before backpressure.
        writer_timeout_seconds: Maximum wait when placing a chunk on the queue.

    Raises:
        RuntimeError: If the background writer thread dies.
        TimeoutError: If the queue remains full beyond the timeout.

    """
    work_queue: queue.Queue[ChunkPayload | None] = queue.Queue(maxsize=writer_queue_depth)
    writer_errors: list[BaseException] = []
    writer_thread = threading.Thread(
        target=run_background_writer,
        args=(work_queue, output_run_paths.chunks_directory, association_mode, writer_errors),
        daemon=True,
    )
    writer_thread.start()
    try:
        for chunk_accumulator in frame_iterator:
            if writer_errors:
                message = f"Background writer failed: {writer_errors[0]}"
                raise RuntimeError(message) from writer_errors[0]
            chunk_payload = build_chunk_payload(chunk_accumulator)
            try:
                work_queue.put(chunk_payload, timeout=writer_timeout_seconds)
            except queue.Full as error:
                message = (
                    "Background writer queue remained full for too long. Storage throughput is bottlenecking compute."
                )
                raise TimeoutError(message) from error
        work_queue.put(None, timeout=writer_timeout_seconds)
        writer_thread.join(timeout=writer_timeout_seconds)
        if writer_errors:
            message = f"Background writer failed: {writer_errors[0]}"
            raise RuntimeError(message) from writer_errors[0]
    except Exception:
        while not work_queue.empty():
            try:
                work_queue.get_nowait()
            except queue.Empty:
                break
        work_queue.put(None)
        writer_thread.join(timeout=5.0)
        raise


def finalize_chunks_to_parquet(
    output_run_paths: OutputRunPaths,
    association_mode: AssociationMode,
) -> Path:
    """Compact committed Arrow chunk files into a single compressed Parquet file.

    Args:
        output_run_paths: Run paths containing the chunks directory.
        association_mode: Association mode enum value.

    Returns:
        Path to the finalized Parquet file.

    """
    chunk_file_paths = sorted(output_run_paths.chunks_directory.glob("chunk_*.arrow"))
    final_parquet_path = output_run_paths.run_directory / "final.parquet"
    temporary_parquet_path = final_parquet_path.with_suffix(".parquet.tmp")
    if chunk_file_paths:
        pl.scan_ipc(chunk_file_paths, cache=False, rechunk=False).sink_parquet(
            temporary_parquet_path,
            compression=OUTPUT_COMPRESSION_CODEC.value,
        )
    else:
        pl.DataFrame(schema=get_output_schema(association_mode)).write_parquet(
            temporary_parquet_path,
            compression=OUTPUT_COMPRESSION_CODEC.value,
        )
    temporary_parquet_path.replace(final_parquet_path)
    return final_parquet_path
