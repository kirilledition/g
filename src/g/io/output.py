"""Output persistence orchestration backed by the native Rust writer."""

from __future__ import annotations

import importlib
import logging
import re
import typing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from g import engine, types

if typing.TYPE_CHECKING:
    import collections.abc


logger = logging.getLogger(__name__)


OUTPUT_COMPRESSION_CODEC = "zstd"
CHUNK_FILENAME_PATTERN = re.compile(r"^chunk_(\d+)(?:_(\d+))?\.arrow$")
DEFAULT_WRITER_QUEUE_DEPTH = 4
DEFAULT_PAYLOAD_BATCH_SIZE = 4
DEFAULT_WRITER_THREAD_COUNT = 8
ChunkWritePayload = engine.ChunkWritePayload


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


def load_backend_core() -> typing.Any:
    """Load the native extension module."""
    return importlib.import_module("g._core")


def resolve_output_run_paths(output_root: Path, association_mode: types.AssociationMode) -> OutputRunPaths:
    """Derive run paths from an output root and association mode."""
    run_directory = output_root if output_root.suffix == ".run" else output_root.with_suffix(f".{association_mode}.run")
    return OutputRunPaths(run_directory=run_directory, chunks_directory=run_directory / "chunks")


def build_chunk_file_name(chunk_identifier: int) -> str:
    """Build a deterministic chunk file name from a chunk identifier."""
    return f"chunk_{chunk_identifier:09d}.arrow"


def build_chunk_batch_file_name(chunk_write_payload: ChunkWritePayload) -> str:
    """Build a deterministic chunk file name for one payload batch."""
    first_chunk_identifier = chunk_write_payload.first_chunk_identifier
    last_chunk_identifier = chunk_write_payload.last_chunk_identifier
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


def enqueue_chunk_payload_batch(
    writer_session: typing.Any,
    chunk_payload_batch: ChunkWritePayload,
    chunk_file_name: str,
    association_mode: types.AssociationMode,
) -> None:
    """Enqueue one chunk batch into the Rust output writer session."""
    if association_mode not in {types.AssociationMode.REGENIE2_LINEAR, types.AssociationMode.REGENIE2_BINARY}:
        message = f"Unsupported association mode for chunk enqueue: {association_mode}"
        raise ValueError(message)
    writer_session.enqueue_regenie_step2_chunk_batch(
        chunk_file_name=chunk_file_name,
        chunk_identifier=chunk_payload_batch.chunk_identifier,
        variant_start_index=chunk_payload_batch.variant_start_index,
        variant_stop_index=chunk_payload_batch.variant_stop_index,
        chrom=list(chunk_payload_batch.chromosome),
        genpos=chunk_payload_batch.position,
        variant_id=list(chunk_payload_batch.variant_identifier),
        allele0=list(chunk_payload_batch.allele_zero),
        allele1=list(chunk_payload_batch.allele_one),
        a1freq=chunk_payload_batch.allele_one_frequency,
        n=chunk_payload_batch.observation_count,
        beta=chunk_payload_batch.beta,
        se=chunk_payload_batch.standard_error,
        chisq=chunk_payload_batch.chi_squared,
        log10p=chunk_payload_batch.log10_p_value,
        extra_code=chunk_payload_batch.extra_code,
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
    core_module = load_backend_core()
    return core_module.PyOutputWriterSession(
        run_directory=str(output_run_paths.run_directory),
        chunks_directory=str(output_run_paths.chunks_directory),
        association_mode=str(association_mode),
        writer_thread_count=writer_thread_count,
        writer_queue_depth=writer_queue_depth,
        finalize_parquet=finalize_parquet,
    )


def write_chunk_batch_to_disk(
    chunk_payload_batch: ChunkWritePayload,
    chunks_directory: Path,
    association_mode: types.AssociationMode,
) -> None:
    """Persist one payload batch through the Rust writer."""
    output_run_paths = OutputRunPaths(run_directory=chunks_directory.parent, chunks_directory=chunks_directory)
    writer_session = create_output_writer_session(
        output_run_paths,
        association_mode,
        writer_thread_count=1,
        writer_queue_depth=1,
        finalize_parquet=False,
    )
    try:
        enqueue_chunk_payload_batch(
            writer_session,
            chunk_payload_batch,
            build_chunk_batch_file_name(chunk_payload_batch),
            association_mode,
        )
        writer_session.finish()
    except Exception:
        writer_session.abort()
        raise


def write_chunk_to_disk(
    chunk_payload: engine.ChunkPayload,
    chunks_directory: Path,
    association_mode: types.AssociationMode,
) -> None:
    """Persist one chunk through the Rust writer."""
    write_chunk_batch_to_disk(
        engine.Regenie2ChunkPayloadBatch(
            first_chunk_identifier=chunk_payload.chunk_identifier,
            last_chunk_identifier=chunk_payload.chunk_identifier,
            chunk_identifier=np.full(len(chunk_payload.position), chunk_payload.chunk_identifier, dtype=np.int64),
            variant_start_index=np.full(len(chunk_payload.position), chunk_payload.variant_start_index, dtype=np.int64),
            variant_stop_index=np.full(len(chunk_payload.position), chunk_payload.variant_stop_index, dtype=np.int64),
            chromosome=tuple(chunk_payload.chromosome.tolist()),
            position=chunk_payload.position,
            variant_identifier=tuple(chunk_payload.variant_identifier.tolist()),
            allele_zero=tuple(chunk_payload.allele_zero.tolist()),
            allele_one=tuple(chunk_payload.allele_one.tolist()),
            allele_one_frequency=chunk_payload.allele_one_frequency,
            observation_count=chunk_payload.observation_count,
            beta=chunk_payload.beta,
            standard_error=chunk_payload.standard_error,
            chi_squared=chunk_payload.chi_squared,
            log10_p_value=chunk_payload.log10_p_value,
            extra_code=chunk_payload.extra_code,
        ),
        chunks_directory,
        association_mode,
    )


def persist_chunked_results(
    frame_iterator: collections.abc.Iterator[engine.ChunkAccumulator],
    output_run_paths: OutputRunPaths,
    association_mode: types.AssociationMode,
    *,
    finalize_parquet: bool = False,
    writer_thread_count: int = DEFAULT_WRITER_THREAD_COUNT,
    writer_queue_depth: int = DEFAULT_WRITER_QUEUE_DEPTH,
    payload_batch_size: int = DEFAULT_PAYLOAD_BATCH_SIZE,
) -> Path | None:
    """Persist chunked results through the native Rust writer."""
    if writer_thread_count < 1:
        message = "Writer thread count must be at least 1."
        raise ValueError(message)
    if payload_batch_size <= 0:
        message = "Payload batch size must be at least 1."
        raise ValueError(message)
    writer_session = create_output_writer_session(
        output_run_paths,
        association_mode,
        writer_thread_count=writer_thread_count,
        writer_queue_depth=writer_queue_depth,
        finalize_parquet=finalize_parquet,
    )
    try:
        chunk_accumulator_batch: list[engine.ChunkAccumulator] = []
        for chunk_accumulator in frame_iterator:
            chunk_accumulator_batch.append(chunk_accumulator)
            if len(chunk_accumulator_batch) >= payload_batch_size:
                chunk_payload_batch = engine.build_chunk_write_payload_batch(chunk_accumulator_batch)
                enqueue_chunk_payload_batch(
                    writer_session,
                    chunk_payload_batch,
                    build_chunk_batch_file_name(chunk_payload_batch),
                    association_mode,
                )
                chunk_accumulator_batch.clear()
        if chunk_accumulator_batch:
            chunk_payload_batch = engine.build_chunk_write_payload_batch(chunk_accumulator_batch)
            enqueue_chunk_payload_batch(
                writer_session,
                chunk_payload_batch,
                build_chunk_batch_file_name(chunk_payload_batch),
                association_mode,
            )
        final_parquet_path = writer_session.finish()
    except Exception:
        writer_session.abort()
        raise
    if final_parquet_path is None:
        return None
    return Path(final_parquet_path)


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
    core_module = load_backend_core()
    final_parquet_path = core_module.finalize_output_run_chunks(
        run_directory=str(output_run_paths.run_directory),
        chunks_directory=str(output_run_paths.chunks_directory),
        association_mode=str(association_mode),
    )
    return Path(final_parquet_path)
