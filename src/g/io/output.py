"""Output persistence and schema helpers."""

from __future__ import annotations

import logging
import queue
import re
import threading
import typing
from dataclasses import dataclass

import numpy as np
import polars as pl

from g import engine, types

if typing.TYPE_CHECKING:
    import collections.abc
    from pathlib import Path

logger = logging.getLogger(__name__)


OUTPUT_COMPRESSION_CODEC = "zstd"
CHUNK_FILENAME_PATTERN = re.compile(r"^chunk_(\d+)(?:_(\d+))?\.arrow$")
DEFAULT_WRITER_QUEUE_DEPTH = 4
DEFAULT_WRITER_TIMEOUT_SECONDS = 120.0
DEFAULT_PAYLOAD_BATCH_SIZE = 1
DEFAULT_WRITER_THREAD_COUNT = 1

ChunkPayloadBatch = tuple[engine.ChunkPayload, ...]

REGENIE2_LINEAR_OUTPUT_SCHEMA: typing.Final[dict[str, pl.DataType]] = {
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
    "chi_squared": pl.Float32(),
    "log10_p_value": pl.Float32(),
    "is_valid": pl.Boolean(),
}


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


def get_output_schema(association_mode: types.AssociationMode) -> dict[str, pl.DataType]:
    """Return the fixed output schema for the requested mode."""
    if association_mode != types.AssociationMode.REGENIE2_LINEAR:
        message = f"Unsupported association mode for active output schema: {association_mode}"
        raise ValueError(message)
    return REGENIE2_LINEAR_OUTPUT_SCHEMA


def cast_frame_to_schema(data_frame: pl.DataFrame, association_mode: types.AssociationMode) -> pl.DataFrame:
    """Cast an output frame to the fixed mode-specific schema."""
    output_schema = get_output_schema(association_mode)
    return data_frame.select(
        [pl.col(column_name).cast(column_type).alias(column_name) for column_name, column_type in output_schema.items()]
    )


def resolve_output_run_paths(output_root: Path, association_mode: types.AssociationMode) -> OutputRunPaths:
    """Derive run paths from an output root and association mode."""
    run_directory = output_root if output_root.suffix == ".run" else output_root.with_suffix(f".{association_mode}.run")
    return OutputRunPaths(
        run_directory=run_directory,
        chunks_directory=run_directory / "chunks",
    )


def build_chunk_file_name(chunk_identifier: int) -> str:
    """Build a deterministic chunk file name from a chunk identifier."""
    return f"chunk_{chunk_identifier:09d}.arrow"


def build_chunk_batch_file_name(chunk_payload_batch: ChunkPayloadBatch) -> str:
    """Build a deterministic chunk file name for one payload batch."""
    if len(chunk_payload_batch) == 1:
        return build_chunk_file_name(chunk_payload_batch[0].chunk_identifier)
    first_chunk_identifier = chunk_payload_batch[0].chunk_identifier
    last_chunk_identifier = chunk_payload_batch[-1].chunk_identifier
    return f"chunk_{first_chunk_identifier:09d}_{last_chunk_identifier:09d}.arrow"


def load_committed_chunk_identifiers_from_chunk_file(chunk_file_path: Path) -> frozenset[int]:
    """Load committed chunk identifiers from one Arrow IPC chunk file."""
    chunk_identifier_values = pl.read_ipc(chunk_file_path).get_column("chunk_identifier").unique().to_list()
    return frozenset(int(chunk_identifier_value) for chunk_identifier_value in chunk_identifier_values)


def scan_committed_chunk_identifiers(chunks_directory: Path) -> frozenset[int]:
    """Scan a chunks directory and return identifiers of completed chunks."""
    if not chunks_directory.exists():
        return frozenset()
    committed_identifiers: set[int] = set()
    for child_path in chunks_directory.iterdir():
        filename_match = CHUNK_FILENAME_PATTERN.match(child_path.name)
        if filename_match is not None:
            if filename_match.group(2) is None:
                committed_identifiers.add(int(filename_match.group(1)))
            else:
                committed_identifiers.update(load_committed_chunk_identifiers_from_chunk_file(child_path))
    return frozenset(committed_identifiers)


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
        logger.info(
            "Resuming run with %d previously committed chunks.",
            len(committed_chunk_identifiers),
        )
    return PreparedOutputRun(
        output_run_paths=output_run_paths,
        committed_chunk_identifiers=committed_chunk_identifiers,
    )


def build_output_frame_from_payload(chunk_payload: engine.ChunkPayload) -> pl.DataFrame:
    """Build a Polars DataFrame from a host-side chunk payload."""
    row_count = len(chunk_payload.position)

    def materialize_host_array(value: object) -> object:
        if isinstance(value, np.ndarray):
            if value.flags.c_contiguous:
                return value
            return np.ascontiguousarray(value)
        return value

    return pl.DataFrame(
        {
            "chunk_identifier": np.full(row_count, chunk_payload.chunk_identifier, dtype=np.int64),
            "variant_start_index": np.full(row_count, chunk_payload.variant_start_index, dtype=np.int64),
            "variant_stop_index": np.full(row_count, chunk_payload.variant_stop_index, dtype=np.int64),
            "chromosome": materialize_host_array(chunk_payload.chromosome),
            "position": materialize_host_array(chunk_payload.position),
            "variant_identifier": materialize_host_array(chunk_payload.variant_identifier),
            "allele_one": materialize_host_array(chunk_payload.allele_one),
            "allele_two": materialize_host_array(chunk_payload.allele_two),
            "allele_one_frequency": materialize_host_array(chunk_payload.allele_one_frequency),
            "observation_count": materialize_host_array(chunk_payload.observation_count),
            "beta": materialize_host_array(chunk_payload.beta),
            "standard_error": materialize_host_array(chunk_payload.standard_error),
            "chi_squared": materialize_host_array(chunk_payload.chi_squared),
            "log10_p_value": materialize_host_array(chunk_payload.log10_p_value),
            "is_valid": materialize_host_array(chunk_payload.is_valid),
        },
        schema=REGENIE2_LINEAR_OUTPUT_SCHEMA,
    )


def build_output_frame_from_payload_batch(chunk_payload_batch: ChunkPayloadBatch) -> pl.DataFrame:
    """Build one Polars DataFrame from a batch of host-side chunk payloads."""
    if len(chunk_payload_batch) == 1:
        return build_output_frame_from_payload(chunk_payload_batch[0])

    return pl.DataFrame(
        {
            "chunk_identifier": np.concatenate(
                [
                    np.full(len(chunk_payload.position), chunk_payload.chunk_identifier, dtype=np.int64)
                    for chunk_payload in chunk_payload_batch
                ]
            ),
            "variant_start_index": np.concatenate(
                [
                    np.full(len(chunk_payload.position), chunk_payload.variant_start_index, dtype=np.int64)
                    for chunk_payload in chunk_payload_batch
                ]
            ),
            "variant_stop_index": np.concatenate(
                [
                    np.full(len(chunk_payload.position), chunk_payload.variant_stop_index, dtype=np.int64)
                    for chunk_payload in chunk_payload_batch
                ]
            ),
            "chromosome": np.concatenate([chunk_payload.chromosome for chunk_payload in chunk_payload_batch]),
            "position": np.concatenate([chunk_payload.position for chunk_payload in chunk_payload_batch]),
            "variant_identifier": np.concatenate(
                [chunk_payload.variant_identifier for chunk_payload in chunk_payload_batch]
            ),
            "allele_one": np.concatenate([chunk_payload.allele_one for chunk_payload in chunk_payload_batch]),
            "allele_two": np.concatenate([chunk_payload.allele_two for chunk_payload in chunk_payload_batch]),
            "allele_one_frequency": np.concatenate(
                [chunk_payload.allele_one_frequency for chunk_payload in chunk_payload_batch]
            ),
            "observation_count": np.concatenate(
                [chunk_payload.observation_count for chunk_payload in chunk_payload_batch]
            ),
            "beta": np.concatenate([chunk_payload.beta for chunk_payload in chunk_payload_batch]),
            "standard_error": np.concatenate([chunk_payload.standard_error for chunk_payload in chunk_payload_batch]),
            "chi_squared": np.concatenate([chunk_payload.chi_squared for chunk_payload in chunk_payload_batch]),
            "log10_p_value": np.concatenate([chunk_payload.log10_p_value for chunk_payload in chunk_payload_batch]),
            "is_valid": np.concatenate([chunk_payload.is_valid for chunk_payload in chunk_payload_batch]),
        },
        schema=REGENIE2_LINEAR_OUTPUT_SCHEMA,
    )


def write_chunk_batch_to_disk(
    chunk_payload_batch: ChunkPayloadBatch,
    chunks_directory: Path,
    association_mode: types.AssociationMode,
) -> None:
    """Atomically persist one chunk payload batch as an Arrow IPC file."""
    chunk_file_name = build_chunk_batch_file_name(chunk_payload_batch)
    chunk_file_path = chunks_directory / chunk_file_name
    temporary_path = chunk_file_path.with_suffix(".arrow.tmp")
    get_output_schema(association_mode)
    output_frame = build_output_frame_from_payload_batch(chunk_payload_batch)
    output_frame.write_ipc(temporary_path, compression=OUTPUT_COMPRESSION_CODEC)
    temporary_path.replace(chunk_file_path)


def write_chunk_to_disk(
    chunk_payload: engine.ChunkPayload,
    chunks_directory: Path,
    association_mode: types.AssociationMode,
) -> None:
    """Atomically persist one chunk as an Arrow IPC file."""
    write_chunk_batch_to_disk((chunk_payload,), chunks_directory, association_mode)


def run_background_writer(
    work_queue: queue.Queue[ChunkPayloadBatch | None],
    chunks_directory: Path,
    association_mode: types.AssociationMode,
    error_container: list[BaseException],
) -> None:
    """Background thread loop that drains the work queue and writes chunks."""
    try:
        while True:
            chunk_payload_batch = work_queue.get()
            if chunk_payload_batch is None:
                return
            write_chunk_batch_to_disk(chunk_payload_batch, chunks_directory, association_mode)
    except Exception as error:  # noqa: BLE001
        error_container.append(error)


def persist_chunked_results(
    frame_iterator: collections.abc.Iterator[engine.Regenie2LinearChunkAccumulator],
    output_run_paths: OutputRunPaths,
    association_mode: types.AssociationMode,
    *,
    writer_thread_count: int = DEFAULT_WRITER_THREAD_COUNT,
    writer_queue_depth: int = DEFAULT_WRITER_QUEUE_DEPTH,
    writer_timeout_seconds: float = DEFAULT_WRITER_TIMEOUT_SECONDS,
    payload_batch_size: int = DEFAULT_PAYLOAD_BATCH_SIZE,
) -> None:
    """Persist chunked results through a non-blocking background writer."""
    if writer_thread_count < 1:
        message = "Writer thread count must be at least 1."
        raise ValueError(message)
    work_queue: queue.Queue[ChunkPayloadBatch | None] = queue.Queue(maxsize=writer_queue_depth)
    writer_errors: list[BaseException] = []
    writer_threads = [
        threading.Thread(
            target=run_background_writer,
            args=(work_queue, output_run_paths.chunks_directory, association_mode, writer_errors),
            daemon=True,
            name=f"chunk-writer-{writer_thread_index}",
        )
        for writer_thread_index in range(writer_thread_count)
    ]
    for writer_thread in writer_threads:
        writer_thread.start()

    def stop_writer_threads(timeout_seconds: float) -> None:
        for _writer_thread in writer_threads:
            work_queue.put(None, timeout=timeout_seconds)
        for writer_thread in writer_threads:
            writer_thread.join(timeout=timeout_seconds)

    def clear_work_queue() -> None:
        while not work_queue.empty():
            try:
                work_queue.get_nowait()
            except queue.Empty:
                break

    def enqueue_chunk_payload_batch(chunk_payloads: list[engine.ChunkPayload]) -> None:
        try:
            work_queue.put(tuple(chunk_payloads), timeout=writer_timeout_seconds)
        except queue.Full as error:
            message = "Background writer queue remained full for too long. Storage throughput is bottlenecking compute."
            raise TimeoutError(message) from error

    try:
        chunk_accumulator_batch: list[engine.Regenie2LinearChunkAccumulator] = []
        for chunk_accumulator in frame_iterator:
            if writer_errors:
                message = f"Background writer failed: {writer_errors[0]}"
                raise RuntimeError(message) from writer_errors[0]
            chunk_accumulator_batch.append(chunk_accumulator)
            if len(chunk_accumulator_batch) >= payload_batch_size:
                enqueue_chunk_payload_batch(engine.build_chunk_payload_batch(chunk_accumulator_batch))
                chunk_accumulator_batch.clear()
        if chunk_accumulator_batch:
            enqueue_chunk_payload_batch(engine.build_chunk_payload_batch(chunk_accumulator_batch))
        stop_writer_threads(writer_timeout_seconds)
        if writer_errors:
            message = f"Background writer failed: {writer_errors[0]}"
            raise RuntimeError(message) from writer_errors[0]
    except Exception:
        clear_work_queue()
        stop_writer_threads(5.0)
        raise


def finalize_chunks_to_parquet(
    output_run_paths: OutputRunPaths,
    association_mode: types.AssociationMode,
) -> Path:
    """Compact committed Arrow chunk files into a single compressed Parquet file."""
    chunk_file_paths = sorted(output_run_paths.chunks_directory.glob("chunk_*.arrow"))
    final_parquet_path = output_run_paths.run_directory / "final.parquet"
    temporary_parquet_path = final_parquet_path.with_suffix(".parquet.tmp")
    if chunk_file_paths:
        pl.scan_ipc(chunk_file_paths, rechunk=False).sink_parquet(
            temporary_parquet_path,
            compression=OUTPUT_COMPRESSION_CODEC,
        )
    else:
        pl.DataFrame(schema=get_output_schema(association_mode)).write_parquet(
            temporary_parquet_path,
            compression=OUTPUT_COMPRESSION_CODEC,
        )
    temporary_parquet_path.replace(final_parquet_path)
    return final_parquet_path
