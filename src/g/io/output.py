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
import pyarrow as pa
import pyarrow.ipc as pa_ipc

from g import engine, types

if typing.TYPE_CHECKING:
    import collections.abc
    from pathlib import Path

    import numpy.typing as npt

logger = logging.getLogger(__name__)


OUTPUT_COMPRESSION_CODEC = "zstd"
CHUNK_FILENAME_PATTERN = re.compile(r"^chunk_(\d+)(?:_(\d+))?\.arrow$")
DEFAULT_WRITER_QUEUE_DEPTH = 4
DEFAULT_WRITER_TIMEOUT_SECONDS = 120.0
DEFAULT_PAYLOAD_BATCH_SIZE = 4
DEFAULT_WRITER_THREAD_COUNT = 8
ChunkWritePayload = engine.ChunkWritePayload
HostArray = np.ndarray

LEGACY_REGENIE2_LINEAR_OUTPUT_SCHEMA: typing.Final[dict[str, pl.DataType]] = {
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
LOW_CARDINALITY_STRING_COLUMN_NAMES: typing.Final[frozenset[str]] = frozenset(
    {"chromosome", "allele_one", "allele_two"}
)
REGENIE2_BINARY_OUTPUT_SCHEMA: typing.Final[dict[str, pl.DataType]] = {
    "chunk_identifier": pl.Int64(),
    "variant_start_index": pl.Int64(),
    "variant_stop_index": pl.Int64(),
    "CHROM": pl.Categorical(),
    "GENPOS": pl.Int64(),
    "ID": pl.String(),
    "ALLELE0": pl.Categorical(),
    "ALLELE1": pl.Categorical(),
    "A1FREQ": pl.Float32(),
    "INFO": pl.Float32(),
    "N": pl.Int32(),
    "TEST": pl.Categorical(),
    "BETA": pl.Float32(),
    "SE": pl.Float32(),
    "CHISQ": pl.Float32(),
    "LOG10P": pl.Float32(),
    "EXTRA": pl.Categorical(),
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


def build_output_schema() -> dict[str, pl.DataType]:
    """Build the fixed output schema."""
    output_schema = dict(LEGACY_REGENIE2_LINEAR_OUTPUT_SCHEMA)
    for column_name in LOW_CARDINALITY_STRING_COLUMN_NAMES:
        output_schema[column_name] = pl.Categorical()
    return output_schema


def get_output_schema(association_mode: types.AssociationMode) -> dict[str, pl.DataType]:
    """Return the fixed output schema for the requested mode."""
    if association_mode == types.AssociationMode.REGENIE2_LINEAR:
        return build_output_schema()
    if association_mode == types.AssociationMode.REGENIE2_BINARY:
        return dict(REGENIE2_BINARY_OUTPUT_SCHEMA)
    message = f"Unsupported association mode for active output schema: {association_mode}"
    raise ValueError(message)


def cast_frame_to_schema(data_frame: pl.DataFrame, association_mode: types.AssociationMode) -> pl.DataFrame:
    """Cast an output frame to the fixed mode-specific schema."""
    output_schema = get_output_schema(association_mode)
    with pl.StringCache():
        return data_frame.select(
            [
                pl.col(column_name).cast(column_type).alias(column_name)
                for column_name, column_type in output_schema.items()
            ]
        )


def cast_lazy_frame_to_schema(lazy_frame: pl.LazyFrame, association_mode: types.AssociationMode) -> pl.LazyFrame:
    """Cast a lazy output frame to the fixed mode-specific schema."""
    output_schema = get_output_schema(association_mode)
    return lazy_frame.select(
        [pl.col(column_name).cast(column_type).alias(column_name) for column_name, column_type in output_schema.items()]
    )


def resolve_output_run_paths(output_root: Path, association_mode: types.AssociationMode) -> OutputRunPaths:
    """Derive run paths from an output root and association mode."""
    run_directory = output_root if output_root.suffix == ".run" else output_root.with_suffix(f".{association_mode}.run")
    return OutputRunPaths(run_directory=run_directory, chunks_directory=run_directory / "chunks")


def build_chunk_file_name(chunk_identifier: int) -> str:
    """Build a deterministic chunk file name from a chunk identifier."""
    return f"chunk_{chunk_identifier:09d}.arrow"


def build_chunk_batch_file_name(chunk_write_payload: ChunkWritePayload) -> str:
    """Build a deterministic chunk file name for one payload batch."""
    if isinstance(chunk_write_payload, engine.Regenie2BinaryChunkPayloadBatch):
        first_chunk_identifier = chunk_write_payload.first_chunk_identifier
        last_chunk_identifier = chunk_write_payload.last_chunk_identifier
    else:
        if len(chunk_write_payload) == 1:
            return build_chunk_file_name(chunk_write_payload[0].chunk_identifier)
        first_chunk_identifier = chunk_write_payload[0].chunk_identifier
        last_chunk_identifier = chunk_write_payload[-1].chunk_identifier
    if first_chunk_identifier == last_chunk_identifier:
        return build_chunk_file_name(first_chunk_identifier)
    return f"chunk_{first_chunk_identifier:09d}_{last_chunk_identifier:09d}.arrow"


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
    if not chunks_directory.exists():
        return frozenset()
    committed_identifiers: set[int] = set()
    for child_path in chunks_directory.iterdir():
        filename_match = CHUNK_FILENAME_PATTERN.match(child_path.name)
        if filename_match is None:
            continue
        if filename_match.group(2) is None:
            committed_identifiers.add(int(filename_match.group(1)))
            continue
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
        logger.info("Resuming run with %d previously committed chunks.", len(committed_chunk_identifiers))
    return PreparedOutputRun(
        output_run_paths=output_run_paths,
        committed_chunk_identifiers=committed_chunk_identifiers,
    )


def materialize_host_array(values: HostArray) -> HostArray:
    """Ensure NumPy arrays are materialized in a layout Polars can ingest efficiently."""
    if values.flags.c_contiguous:
        return values
    return np.ascontiguousarray(values)


def build_low_cardinality_output_series(
    column_name: str,
    values: HostArray,
    association_mode: types.AssociationMode,
) -> pl.Series:
    """Build one low-cardinality string output series."""
    output_schema = get_output_schema(association_mode)
    return pl.Series(
        column_name,
        materialize_host_array(values),
        dtype=output_schema[column_name],
    )


def build_high_cardinality_output_series(
    column_name: str,
    values: HostArray,
    association_mode: types.AssociationMode,
) -> pl.Series:
    """Build one high-cardinality string output series."""
    output_schema = get_output_schema(association_mode)
    return pl.Series(
        column_name,
        materialize_host_array(values),
        dtype=output_schema[column_name],
    )


def build_numeric_or_boolean_output_series(
    column_name: str,
    values: HostArray,
    association_mode: types.AssociationMode,
) -> pl.Series:
    """Build one numeric or boolean output series."""
    output_schema = get_output_schema(association_mode)
    return pl.Series(
        column_name,
        materialize_host_array(values),
        dtype=output_schema[column_name],
    )


def build_constant_numeric_array(value: int | float, row_count: int, data_type: npt.DTypeLike) -> HostArray:
    """Build a repeated numeric NumPy array."""
    return np.full(row_count, value, dtype=data_type)


def build_regenie2_linear_output_frame_from_payload(chunk_payload: engine.Regenie2LinearChunkPayload) -> pl.DataFrame:
    """Build a Polars DataFrame from a host-side chunk payload."""
    row_count = len(chunk_payload.position)
    with pl.StringCache():
        return pl.DataFrame(
            [
                build_numeric_or_boolean_output_series(
                    "chunk_identifier",
                    np.full(row_count, chunk_payload.chunk_identifier, dtype=np.int64),
                    types.AssociationMode.REGENIE2_LINEAR,
                ),
                build_numeric_or_boolean_output_series(
                    "variant_start_index",
                    np.full(row_count, chunk_payload.variant_start_index, dtype=np.int64),
                    types.AssociationMode.REGENIE2_LINEAR,
                ),
                build_numeric_or_boolean_output_series(
                    "variant_stop_index",
                    np.full(row_count, chunk_payload.variant_stop_index, dtype=np.int64),
                    types.AssociationMode.REGENIE2_LINEAR,
                ),
                build_low_cardinality_output_series(
                    "chromosome", chunk_payload.chromosome, types.AssociationMode.REGENIE2_LINEAR
                ),
                build_numeric_or_boolean_output_series(
                    "position", chunk_payload.position, types.AssociationMode.REGENIE2_LINEAR
                ),
                build_high_cardinality_output_series(
                    "variant_identifier", chunk_payload.variant_identifier, types.AssociationMode.REGENIE2_LINEAR
                ),
                build_low_cardinality_output_series(
                    "allele_one", chunk_payload.allele_one, types.AssociationMode.REGENIE2_LINEAR
                ),
                build_low_cardinality_output_series(
                    "allele_two", chunk_payload.allele_two, types.AssociationMode.REGENIE2_LINEAR
                ),
                build_numeric_or_boolean_output_series(
                    "allele_one_frequency", chunk_payload.allele_one_frequency, types.AssociationMode.REGENIE2_LINEAR
                ),
                build_numeric_or_boolean_output_series(
                    "observation_count", chunk_payload.observation_count, types.AssociationMode.REGENIE2_LINEAR
                ),
                build_numeric_or_boolean_output_series(
                    "beta", chunk_payload.beta, types.AssociationMode.REGENIE2_LINEAR
                ),
                build_numeric_or_boolean_output_series(
                    "standard_error", chunk_payload.standard_error, types.AssociationMode.REGENIE2_LINEAR
                ),
                build_numeric_or_boolean_output_series(
                    "chi_squared", chunk_payload.chi_squared, types.AssociationMode.REGENIE2_LINEAR
                ),
                build_numeric_or_boolean_output_series(
                    "log10_p_value", chunk_payload.log10_p_value, types.AssociationMode.REGENIE2_LINEAR
                ),
                build_numeric_or_boolean_output_series(
                    "is_valid", chunk_payload.is_valid, types.AssociationMode.REGENIE2_LINEAR
                ),
            ],
            schema=get_output_schema(types.AssociationMode.REGENIE2_LINEAR),
        )


def build_regenie2_binary_output_frame_from_payload(chunk_payload: engine.Regenie2BinaryChunkPayload) -> pl.DataFrame:
    """Build a Polars DataFrame from a host-side binary chunk payload."""
    row_count = len(chunk_payload.position)
    association_mode = types.AssociationMode.REGENIE2_BINARY
    with pl.StringCache():
        return pl.DataFrame(
            [
                build_numeric_or_boolean_output_series(
                    "chunk_identifier",
                    np.full(row_count, chunk_payload.chunk_identifier, dtype=np.int64),
                    association_mode,
                ),
                build_numeric_or_boolean_output_series(
                    "variant_start_index",
                    np.full(row_count, chunk_payload.variant_start_index, dtype=np.int64),
                    association_mode,
                ),
                build_numeric_or_boolean_output_series(
                    "variant_stop_index",
                    np.full(row_count, chunk_payload.variant_stop_index, dtype=np.int64),
                    association_mode,
                ),
                build_low_cardinality_output_series("CHROM", chunk_payload.chromosome, association_mode),
                build_numeric_or_boolean_output_series("GENPOS", chunk_payload.position, association_mode),
                build_high_cardinality_output_series("ID", chunk_payload.variant_identifier, association_mode),
                build_low_cardinality_output_series("ALLELE0", chunk_payload.allele_two, association_mode),
                build_low_cardinality_output_series("ALLELE1", chunk_payload.allele_one, association_mode),
                build_numeric_or_boolean_output_series("A1FREQ", chunk_payload.allele_one_frequency, association_mode),
                build_numeric_or_boolean_output_series("INFO", np.ones(row_count, dtype=np.float32), association_mode),
                build_numeric_or_boolean_output_series("N", chunk_payload.observation_count, association_mode),
                build_low_cardinality_output_series("TEST", np.full(row_count, "ADD", dtype="<U3"), association_mode),
                build_numeric_or_boolean_output_series("BETA", chunk_payload.beta, association_mode),
                build_numeric_or_boolean_output_series("SE", chunk_payload.standard_error, association_mode),
                build_numeric_or_boolean_output_series("CHISQ", chunk_payload.chi_squared, association_mode),
                build_numeric_or_boolean_output_series("LOG10P", chunk_payload.log10_p_value, association_mode),
                build_low_cardinality_output_series(
                    "EXTRA",
                    np.asarray(engine.REGENIE2_BINARY_EXTRA_LABELS, dtype="<U9")[chunk_payload.extra_code],
                    association_mode,
                ),
            ],
            schema=get_output_schema(association_mode),
        )


def build_output_frame_from_payload(chunk_payload: engine.ChunkPayload) -> pl.DataFrame:
    """Build a Polars DataFrame from a host-side chunk payload."""
    if isinstance(chunk_payload, engine.Regenie2BinaryChunkPayload):
        return build_regenie2_binary_output_frame_from_payload(chunk_payload)
    return build_regenie2_linear_output_frame_from_payload(chunk_payload)


def build_output_frame_from_payload_batch(
    chunk_payload_batch: tuple[engine.ChunkPayload, ...],
) -> pl.DataFrame:
    """Build one Polars DataFrame from a batch of host-side chunk payloads."""
    if len(chunk_payload_batch) == 1:
        return build_output_frame_from_payload(chunk_payload_batch[0])
    with pl.StringCache():
        return pl.concat(
            [build_output_frame_from_payload(chunk_payload) for chunk_payload in chunk_payload_batch],
            how="vertical",
        )


def write_output_frame_to_chunk_file(output_frame: pl.DataFrame, chunk_file_path: Path) -> None:
    """Write one prepared output frame to an Arrow IPC chunk file."""
    output_frame.write_ipc(chunk_file_path, compression=OUTPUT_COMPRESSION_CODEC)


def write_output_record_batch_to_chunk_file(output_batch: pa.RecordBatch, chunk_file_path: Path) -> None:
    """Write one Arrow RecordBatch to an Arrow IPC chunk file."""
    write_options = pa_ipc.IpcWriteOptions(compression=OUTPUT_COMPRESSION_CODEC)
    with pa.OSFile(str(chunk_file_path), "wb") as sink, pa_ipc.new_file(
        sink,
        output_batch.schema,
        options=write_options,
    ) as writer:
        writer.write_batch(output_batch)


def write_chunk_batch_to_disk(
    chunk_payload_batch: ChunkWritePayload,
    chunks_directory: Path,
    association_mode: types.AssociationMode,
) -> None:
    """Atomically persist one chunk payload batch."""
    del association_mode
    chunk_file_name = build_chunk_batch_file_name(chunk_payload_batch)
    chunk_file_path = chunks_directory / chunk_file_name
    temporary_path = chunk_file_path.with_suffix(".arrow.tmp")
    if isinstance(chunk_payload_batch, engine.Regenie2BinaryChunkPayloadBatch):
        write_output_record_batch_to_chunk_file(chunk_payload_batch.output_batch, temporary_path)
    elif chunk_payload_batch and isinstance(chunk_payload_batch[0], engine.Regenie2BinaryChunkPayload):
        write_output_record_batch_to_chunk_file(
            engine.build_regenie2_binary_output_record_batch_from_payloads(
                typing.cast("tuple[engine.Regenie2BinaryChunkPayload, ...]", chunk_payload_batch)
            ),
            temporary_path,
        )
    else:
        output_frame = build_output_frame_from_payload_batch(chunk_payload_batch)
        write_output_frame_to_chunk_file(output_frame, temporary_path)
    temporary_path.replace(chunk_file_path)


def write_chunk_to_disk(
    chunk_payload: engine.ChunkPayload,
    chunks_directory: Path,
    association_mode: types.AssociationMode,
) -> None:
    """Atomically persist one chunk."""
    write_chunk_batch_to_disk((chunk_payload,), chunks_directory, association_mode)


def put_chunk_payload_batch_into_queue(
    work_queue: queue.Queue[ChunkWritePayload | None],
    chunk_payload_batch: ChunkWritePayload,
    writer_timeout_seconds: float,
) -> None:
    """Enqueue one payload batch for asynchronous writing."""
    try:
        work_queue.put(chunk_payload_batch, timeout=writer_timeout_seconds)
    except queue.Full as error:
        message = "Background writer queue remained full for too long. Storage throughput is bottlenecking compute."
        raise TimeoutError(message) from error


def join_writer_threads(
    work_queue: queue.Queue[ChunkWritePayload | None],
    writer_threads: list[threading.Thread],
    timeout_seconds: float,
) -> None:
    """Stop and join all writer threads."""
    for _writer_thread in writer_threads:
        work_queue.put(None, timeout=timeout_seconds)
    for writer_thread in writer_threads:
        writer_thread.join(timeout=timeout_seconds)


def run_background_writer(
    work_queue: queue.Queue[ChunkWritePayload | None],
    chunks_directory: Path,
    association_mode: types.AssociationMode,
    error_container: list[BaseException],
) -> None:
    """Background thread loop that drains the work queue and writes chunks."""
    try:
        with pl.StringCache():
            while True:
                chunk_payload_batch = work_queue.get()
                if chunk_payload_batch is None:
                    return
                write_chunk_batch_to_disk(chunk_payload_batch, chunks_directory, association_mode)
    except Exception as error:  # noqa: BLE001
        error_container.append(error)


def persist_chunked_results(
    frame_iterator: collections.abc.Iterator[engine.ChunkAccumulator],
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
    work_queue: queue.Queue[ChunkWritePayload | None] = queue.Queue(maxsize=writer_queue_depth)
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

    def clear_work_queue() -> None:
        while not work_queue.empty():
            try:
                work_queue.get_nowait()
            except queue.Empty:
                break

    def enqueue_chunk_payload_batch(chunk_payloads: ChunkWritePayload) -> None:
        put_chunk_payload_batch_into_queue(work_queue, chunk_payloads, writer_timeout_seconds)

    try:
        chunk_accumulator_batch: list[engine.ChunkAccumulator] = []
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
        join_writer_threads(work_queue, writer_threads, writer_timeout_seconds)
        if writer_errors:
            message = f"Background writer failed: {writer_errors[0]}"
            raise RuntimeError(message) from writer_errors[0]
    except Exception:
        clear_work_queue()
        join_writer_threads(work_queue, writer_threads, 5.0)
        raise


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
    """Compact committed chunk files into one compressed Parquet file."""
    chunk_file_paths = iter_sorted_chunk_file_paths(output_run_paths.chunks_directory)
    final_parquet_path = output_run_paths.run_directory / "final.parquet"
    temporary_parquet_path = final_parquet_path.with_suffix(".parquet.tmp")
    output_schema = get_output_schema(association_mode)
    if chunk_file_paths:
        with pl.StringCache():
            chunk_lazy_frames = [
                cast_lazy_frame_to_schema(scan_chunk_file(chunk_file_path), association_mode)
                for chunk_file_path in chunk_file_paths
            ]
            pl.concat(chunk_lazy_frames, how="vertical").sink_parquet(
                temporary_parquet_path,
                compression=OUTPUT_COMPRESSION_CODEC,
            )
    else:
        pl.DataFrame(schema=output_schema).write_parquet(
            temporary_parquet_path,
            compression=OUTPUT_COMPRESSION_CODEC,
        )
    temporary_parquet_path.replace(final_parquet_path)
    return final_parquet_path
