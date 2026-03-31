"""Chunked Arrow output writer with manifest-backed durability."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from g.engine import (
    LinearChunkAccumulator,
    LogisticChunkAccumulator,
    build_linear_output_frame,
    build_logistic_output_frame,
)
from g.output.manifest import ManifestChunkRecord, OutputManifest
from g.output.schema import SCHEMA_VERSION, cast_frame_to_schema, write_schema_file

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@dataclass(frozen=True)
class OutputRunPaths:
    """Paths associated with one chunked output run."""

    run_directory: Path
    chunks_directory: Path
    manifest_path: Path
    schema_path: Path


def resolve_output_run_paths(output_root: Path, association_mode: str) -> OutputRunPaths:
    """Derive run file-system paths from the user output root."""
    run_directory = output_root if output_root.suffix == ".run" else output_root.with_suffix(f".{association_mode}.run")
    return OutputRunPaths(
        run_directory=run_directory,
        chunks_directory=run_directory / "chunks",
        manifest_path=run_directory / "manifest.sqlite",
        schema_path=run_directory / "schema.arrow.json",
    )


def calculate_file_checksum(file_path: Path) -> str:
    """Calculate the SHA-256 checksum for a file."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as opened_file:
        for file_block in iter(lambda: opened_file.read(1024 * 1024), b""):
            hasher.update(file_block)
    return hasher.hexdigest()


def build_chunk_label(association_mode: str, chunk_identifier: int) -> str:
    """Build a deterministic chunk label from mode and index."""
    return f"{association_mode}_chunk_{chunk_identifier:06d}"


def write_chunk_file(data_frame: pl.DataFrame, chunk_file_path: Path) -> str:
    """Persist one chunk as Arrow IPC and return the checksum."""
    temporary_path = chunk_file_path.with_suffix(".arrow.tmp")
    data_frame.write_ipc(temporary_path)
    temporary_path.replace(chunk_file_path)
    return calculate_file_checksum(chunk_file_path)


def persist_chunked_results(
    frame_iterator: Iterator[LinearChunkAccumulator] | Iterator[LogisticChunkAccumulator],
    association_mode: str,
    output_root: Path,
    *,
    resume: bool,
) -> OutputRunPaths:
    """Write result chunks to Arrow files and track commits in a manifest."""
    output_run_paths = resolve_output_run_paths(output_root, association_mode)
    output_run_paths.chunks_directory.mkdir(parents=True, exist_ok=True)
    write_schema_file(output_run_paths.schema_path, association_mode)

    manifest = OutputManifest(output_run_paths.manifest_path)
    manifest.register_run(
        run_identifier=output_run_paths.run_directory.name,
        association_mode=association_mode,
        schema_version=SCHEMA_VERSION,
    )
    committed_chunk_identifiers = manifest.load_committed_chunk_identifiers() if resume else set()

    for chunk_identifier, chunk_accumulator in enumerate(frame_iterator):
        if chunk_identifier in committed_chunk_identifiers:
            continue
        if isinstance(chunk_accumulator, LinearChunkAccumulator):
            output_frame = build_linear_output_frame(
                metadata=chunk_accumulator.metadata,
                allele_one_frequency=chunk_accumulator.allele_one_frequency,
                observation_count=chunk_accumulator.observation_count,
                linear_result=chunk_accumulator.linear_result,
            )
        else:
            output_frame = build_logistic_output_frame(
                metadata=chunk_accumulator.metadata,
                allele_one_frequency=chunk_accumulator.allele_one_frequency,
                observation_count=chunk_accumulator.observation_count,
                logistic_result=chunk_accumulator.logistic_result,
            )
        cast_output_frame = cast_frame_to_schema(output_frame, association_mode)
        chunk_file_name = f"{chunk_identifier:06d}.arrow"
        chunk_file_path = output_run_paths.chunks_directory / chunk_file_name
        chunk_checksum = write_chunk_file(cast_output_frame, chunk_file_path)
        chunk_record = ManifestChunkRecord(
            chunk_identifier=chunk_identifier,
            chunk_label=build_chunk_label(association_mode, chunk_identifier),
            row_count=cast_output_frame.height,
            file_path=f"chunks/{chunk_file_name}",
            checksum_sha256=chunk_checksum,
            status="committed",
        )
        manifest.insert_committed_chunk(chunk_record)

    manifest.close()
    return output_run_paths


def finalize_chunks_to_parquet(output_run_paths: OutputRunPaths) -> Path:
    """Compact committed Arrow chunk files into one compressed Parquet file."""
    manifest = OutputManifest(output_run_paths.manifest_path)
    chunk_records = manifest.load_committed_chunk_records()
    manifest.close()

    chunk_frames: list[pl.DataFrame] = []
    for chunk_record in chunk_records:
        chunk_file_path = output_run_paths.run_directory / chunk_record.file_path
        if not chunk_file_path.exists():
            continue
        chunk_frames.append(pl.read_ipc(chunk_file_path))

    final_parquet_path = output_run_paths.run_directory / "final.parquet"
    if chunk_frames:
        pl.concat(chunk_frames, how="vertical").write_parquet(final_parquet_path, compression="zstd")
    else:
        pl.DataFrame().write_parquet(final_parquet_path, compression="zstd")
    return final_parquet_path
