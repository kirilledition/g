"""Chunked output orchestration with manifest-backed durability."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import multiprocessing
import queue as queue_module
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from g.engine import (
    ChunkPayload,
    LinearChunkAccumulator,
    LinearChunkPayload,
    LogisticChunkAccumulator,
    build_chunk_payload,
)
from g.output.manifest import ManifestChunkRecord, OutputManifest, RunManifestRecord
from g.output.schema import SCHEMA_VERSION, cast_frame_to_schema, get_output_schema, write_schema_file

if TYPE_CHECKING:
    from collections.abc import Iterator
    from multiprocessing.context import SpawnProcess
    from multiprocessing.queues import Queue
    from pathlib import Path


DEFAULT_SPOOLER_QUEUE_SIZE = 4
DEFAULT_SPOOLER_QUEUE_TIMEOUT_SECONDS = 60.0
DEFAULT_SPOOLER_SHUTDOWN_TIMEOUT_SECONDS = 15.0
CHUNK_IPC_COMPRESSION = "lz4"


@dataclass(frozen=True)
class OutputRunPaths:
    """Paths associated with one chunked output run."""

    run_directory: Path
    chunks_directory: Path
    manifest_path: Path
    schema_path: Path


@dataclass(frozen=True)
class OutputRunConfiguration:
    """Configuration that must remain stable for deterministic resume."""

    association_mode: str
    bed_file_signatures: tuple[str, str, str]
    phenotype_file_signature: str
    phenotype_name: str
    covariate_file_signature: str | None
    covariate_names: tuple[str, ...] | None
    chunk_size: int
    variant_limit: int | None
    max_iterations: int | None = None
    tolerance: float | None = None
    firth_fallback: bool | None = None


@dataclass(frozen=True)
class PreparedOutputRun:
    """Prepared run metadata used by the compute and persistence paths."""

    output_run_paths: OutputRunPaths
    run_manifest_record: RunManifestRecord
    committed_chunk_identifiers: frozenset[int]


@dataclass(frozen=True)
class ChunkValidationResult:
    """Validation outcome for manifest-tracked chunk files."""

    valid_chunk_records: list[ManifestChunkRecord]
    missing_chunk_records: list[ManifestChunkRecord]
    checksum_mismatch_chunk_records: list[ManifestChunkRecord]


@dataclass(frozen=True)
class SpoolerOutcome:
    """Final outcome reported by the background spooler."""

    processed_chunk_count: int
    error_message: str | None = None
    traceback_text: str | None = None


def resolve_output_run_paths(output_root: Path, association_mode: str) -> OutputRunPaths:
    """Derive run file-system paths from the user output root."""
    run_directory = output_root if output_root.suffix == ".run" else output_root.with_suffix(f".{association_mode}.run")
    return OutputRunPaths(
        run_directory=run_directory,
        chunks_directory=run_directory / "chunks",
        manifest_path=run_directory / "manifest.sqlite",
        schema_path=run_directory / "schema.arrow.json",
    )


def build_path_signature(path: Path) -> str:
    """Build a reproducibility signature for one input file."""
    resolved_path = path.resolve(strict=True)
    path_status = resolved_path.stat()
    serialized_signature = {
        "path": str(resolved_path),
        "size": path_status.st_size,
        "modified_time_nanoseconds": path_status.st_mtime_ns,
    }
    return json.dumps(serialized_signature, sort_keys=True)


def build_output_run_configuration(
    *,
    association_mode: str,
    bed_prefix: Path,
    phenotype_path: Path,
    phenotype_name: str,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    chunk_size: int,
    variant_limit: int | None,
    max_iterations: int | None = None,
    tolerance: float | None = None,
    firth_fallback: bool | None = None,
) -> OutputRunConfiguration:
    """Build the reproducibility configuration for one output run."""
    return OutputRunConfiguration(
        association_mode=association_mode,
        bed_file_signatures=(
            build_path_signature(bed_prefix.with_suffix(".bed")),
            build_path_signature(bed_prefix.with_suffix(".bim")),
            build_path_signature(bed_prefix.with_suffix(".fam")),
        ),
        phenotype_file_signature=build_path_signature(phenotype_path),
        phenotype_name=phenotype_name,
        covariate_file_signature=None if covariate_path is None else build_path_signature(covariate_path),
        covariate_names=covariate_names,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        max_iterations=max_iterations,
        tolerance=tolerance,
        firth_fallback=firth_fallback,
    )


def build_run_manifest_record(
    output_run_paths: OutputRunPaths,
    output_run_configuration: OutputRunConfiguration,
) -> RunManifestRecord:
    """Build the manifest record used to validate run compatibility."""
    configuration_json = json.dumps(dataclasses.asdict(output_run_configuration), sort_keys=True)
    configuration_fingerprint = hashlib.sha256(configuration_json.encode("utf-8")).hexdigest()
    return RunManifestRecord(
        run_identifier=output_run_paths.run_directory.name,
        association_mode=output_run_configuration.association_mode,
        schema_version=SCHEMA_VERSION,
        chunk_size=output_run_configuration.chunk_size,
        variant_limit=output_run_configuration.variant_limit,
        configuration_fingerprint=configuration_fingerprint,
        configuration_json=configuration_json,
    )


def validate_chunk_records(
    output_run_paths: OutputRunPaths,
    chunk_records: list[ManifestChunkRecord],
) -> ChunkValidationResult:
    """Validate that committed chunk files exist and match manifest checksums."""
    valid_chunk_records: list[ManifestChunkRecord] = []
    missing_chunk_records: list[ManifestChunkRecord] = []
    checksum_mismatch_chunk_records: list[ManifestChunkRecord] = []
    for chunk_record in chunk_records:
        chunk_file_path = output_run_paths.run_directory / chunk_record.file_path
        if not chunk_file_path.exists():
            missing_chunk_records.append(chunk_record)
            continue
        if calculate_file_checksum(chunk_file_path) != chunk_record.checksum_sha256:
            checksum_mismatch_chunk_records.append(chunk_record)
            continue
        valid_chunk_records.append(chunk_record)
    return ChunkValidationResult(
        valid_chunk_records=valid_chunk_records,
        missing_chunk_records=missing_chunk_records,
        checksum_mismatch_chunk_records=checksum_mismatch_chunk_records,
    )


def build_chunk_validation_error_message(chunk_validation_result: ChunkValidationResult) -> str:
    """Build a concise validation error for missing or corrupted chunk files."""
    message_parts: list[str] = []
    if chunk_validation_result.missing_chunk_records:
        missing_labels = ", ".join(record.chunk_label for record in chunk_validation_result.missing_chunk_records)
        message_parts.append(f"missing chunk files: {missing_labels}")
    if chunk_validation_result.checksum_mismatch_chunk_records:
        checksum_labels = ", ".join(
            record.chunk_label for record in chunk_validation_result.checksum_mismatch_chunk_records
        )
        message_parts.append(f"checksum mismatch: {checksum_labels}")
    return "; ".join(message_parts)


def prepare_output_run(
    *,
    output_root: Path,
    output_run_configuration: OutputRunConfiguration,
    resume: bool,
) -> PreparedOutputRun:
    """Prepare and validate a chunked output run before compute starts."""
    output_run_paths = resolve_output_run_paths(output_root, output_run_configuration.association_mode)
    run_manifest_record = build_run_manifest_record(output_run_paths, output_run_configuration)
    if not resume and output_run_paths.run_directory.exists() and any(output_run_paths.run_directory.iterdir()):
        message = (
            f"Output run directory '{output_run_paths.run_directory}' already exists and is not empty. "
            "Use resume mode or choose a new output path."
        )
        raise ValueError(message)
    if not output_run_paths.manifest_path.exists():
        if resume:
            message = f"Cannot resume run '{output_run_paths.run_directory}' because no manifest exists yet."
            raise ValueError(message)
        return PreparedOutputRun(
            output_run_paths=output_run_paths,
            run_manifest_record=run_manifest_record,
            committed_chunk_identifiers=frozenset(),
        )

    manifest = OutputManifest(output_run_paths.manifest_path)
    try:
        manifest.register_or_validate_run(run_manifest_record, resume=resume)
        committed_chunk_records = manifest.load_committed_chunk_records()
    finally:
        manifest.close()
    if not resume:
        message = (
            f"Output run '{output_run_paths.run_directory.name}' already exists. "
            "Use resume mode or choose a new output path."
        )
        raise ValueError(message)
    chunk_validation_result = validate_chunk_records(output_run_paths, committed_chunk_records)
    committed_chunk_identifiers = frozenset(
        chunk_record.chunk_identifier for chunk_record in chunk_validation_result.valid_chunk_records
    )
    return PreparedOutputRun(
        output_run_paths=output_run_paths,
        run_manifest_record=run_manifest_record,
        committed_chunk_identifiers=committed_chunk_identifiers,
    )


def build_output_frame_from_payload(chunk_payload: ChunkPayload) -> pl.DataFrame:
    """Build a Polars frame from one host-side payload."""
    shared_columns = {
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
    return output_frame.with_columns(
        pl.lit(chunk_payload.chunk_identifier, dtype=pl.Int64).alias("chunk_identifier"),
        pl.lit(chunk_payload.variant_start_index, dtype=pl.Int64).alias("variant_start_index"),
        pl.lit(chunk_payload.variant_stop_index, dtype=pl.Int64).alias("variant_stop_index"),
    )


def build_chunk_file_name(chunk_payload: ChunkPayload) -> str:
    """Build a deterministic chunk file name from genomic bounds."""
    return f"{chunk_payload.chunk_identifier:06d}_{chunk_payload.variant_stop_index:06d}.arrow"


def build_chunk_label(association_mode: str, chunk_payload: ChunkPayload) -> str:
    """Build a deterministic chunk label from mode and genomic bounds."""
    return f"{association_mode}_chunk_{chunk_payload.chunk_identifier:06d}_{chunk_payload.variant_stop_index:06d}"


def calculate_file_checksum(file_path: Path) -> str:
    """Calculate the SHA-256 checksum for a file."""
    file_hasher = hashlib.sha256()
    with file_path.open("rb") as opened_file:
        for file_block in iter(lambda: opened_file.read(1024 * 1024), b""):
            file_hasher.update(file_block)
    return file_hasher.hexdigest()


def write_chunk_file(
    chunk_payload: ChunkPayload,
    chunk_file_path: Path,
    association_mode: str,
) -> str:
    """Persist one chunk as Arrow IPC and return the checksum."""
    output_frame = build_output_frame_from_payload(chunk_payload)
    cast_output_frame = cast_frame_to_schema(output_frame, association_mode)
    temporary_path = chunk_file_path.with_suffix(".arrow.tmp")
    cast_output_frame.write_ipc(temporary_path, compression=CHUNK_IPC_COMPRESSION)
    temporary_path.replace(chunk_file_path)
    return calculate_file_checksum(chunk_file_path)


def persist_chunk_payload(
    manifest: OutputManifest,
    output_run_paths: OutputRunPaths,
    association_mode: str,
    chunk_payload: ChunkPayload,
) -> None:
    """Persist one chunk payload and commit it in the manifest."""
    chunk_file_name = build_chunk_file_name(chunk_payload)
    chunk_file_path = output_run_paths.chunks_directory / chunk_file_name
    chunk_checksum = write_chunk_file(chunk_payload, chunk_file_path, association_mode)
    manifest.insert_committed_chunk(
        ManifestChunkRecord(
            chunk_identifier=chunk_payload.chunk_identifier,
            chunk_label=build_chunk_label(association_mode, chunk_payload),
            variant_start_index=chunk_payload.variant_start_index,
            variant_stop_index=chunk_payload.variant_stop_index,
            row_count=int(chunk_payload.position.shape[0]),
            file_path=f"chunks/{chunk_file_name}",
            checksum_sha256=chunk_checksum,
            status="committed",
        )
    )


def run_output_spooler(
    message_queue: Queue,
    result_queue: Queue,
    output_run_paths: OutputRunPaths,
    run_manifest_record: RunManifestRecord,
    *,
    resume: bool,
) -> None:
    """Run the background output spooler loop."""
    manifest: OutputManifest | None = None
    processed_chunk_count = 0
    try:
        output_run_paths.chunks_directory.mkdir(parents=True, exist_ok=True)
        write_schema_file(output_run_paths.schema_path, run_manifest_record.association_mode)
        manifest = OutputManifest(output_run_paths.manifest_path)
        manifest.register_or_validate_run(run_manifest_record, resume=resume)
        while True:
            chunk_payload = message_queue.get()
            if chunk_payload is None:
                result_queue.put(
                    SpoolerOutcome(
                        processed_chunk_count=processed_chunk_count,
                    )
                )
                return
            persist_chunk_payload(
                manifest=manifest,
                output_run_paths=output_run_paths,
                association_mode=run_manifest_record.association_mode,
                chunk_payload=chunk_payload,
            )
            processed_chunk_count += 1
    except Exception as error:
        result_queue.put(
            SpoolerOutcome(
                processed_chunk_count=processed_chunk_count,
                error_message=str(error),
                traceback_text=traceback.format_exc(),
            )
        )
        raise
    finally:
        if manifest is not None:
            manifest.close()


def raise_for_spooler_outcome(spooler_outcome: SpoolerOutcome) -> None:
    """Raise a Python exception for a failed spooler outcome."""
    if spooler_outcome.error_message is None:
        return
    message = f"Output spooler failed: {spooler_outcome.error_message}"
    if spooler_outcome.traceback_text is not None:
        message = f"{message}\n{spooler_outcome.traceback_text}"
    raise RuntimeError(message)


def get_pending_spooler_outcome(result_queue: Queue) -> SpoolerOutcome | None:
    """Return a pending spooler outcome without blocking."""
    try:
        return result_queue.get_nowait()
    except queue_module.Empty:
        return None


def ensure_spooler_is_healthy(
    spooler_process: SpawnProcess,
    result_queue: Queue,
) -> None:
    """Fail fast when the spooler exits or reports an error."""
    pending_spooler_outcome = get_pending_spooler_outcome(result_queue)
    if pending_spooler_outcome is not None:
        raise_for_spooler_outcome(pending_spooler_outcome)
    if not spooler_process.is_alive() and spooler_process.exitcode not in {None, 0}:
        message = f"Output spooler exited unexpectedly with code {spooler_process.exitcode}."
        raise RuntimeError(message)


def wait_for_spooler_outcome(
    spooler_process: SpawnProcess,
    result_queue: Queue,
    timeout_seconds: float,
) -> SpoolerOutcome:
    """Wait for the spooler to finish and report its final status."""
    try:
        spooler_outcome = result_queue.get(timeout=timeout_seconds)
    except queue_module.Empty as error:
        if spooler_process.is_alive():
            message = (
                "Output spooler did not finish within the configured timeout. "
                "This usually means storage could not keep up with chunk persistence."
            )
            raise TimeoutError(message) from error
        message = "Output spooler exited without reporting a completion status."
        raise RuntimeError(message) from error
    spooler_process.join(timeout=timeout_seconds)
    if spooler_process.is_alive():
        message = "Output spooler reported completion but did not exit cleanly."
        raise RuntimeError(message)
    raise_for_spooler_outcome(spooler_outcome)
    return spooler_outcome


def terminate_spooler_process(spooler_process: SpawnProcess) -> None:
    """Terminate the background spooler when the compute path aborts."""
    if not spooler_process.is_alive():
        return
    spooler_process.terminate()
    spooler_process.join(timeout=DEFAULT_SPOOLER_SHUTDOWN_TIMEOUT_SECONDS)


def close_queue(queue_object: Queue) -> None:
    """Close a multiprocessing queue cleanly when possible."""
    queue_object.close()
    queue_object.join_thread()


def persist_chunked_results(
    frame_iterator: Iterator[LinearChunkAccumulator] | Iterator[LogisticChunkAccumulator],
    prepared_output_run: PreparedOutputRun,
    *,
    resume: bool,
    maximum_queued_chunks: int = DEFAULT_SPOOLER_QUEUE_SIZE,
    queue_timeout_seconds: float = DEFAULT_SPOOLER_QUEUE_TIMEOUT_SECONDS,
) -> OutputRunPaths:
    """Persist result chunks through a bounded background spooler."""
    spawn_context = multiprocessing.get_context("spawn")
    message_queue: Queue = spawn_context.Queue(maxsize=maximum_queued_chunks)
    result_queue: Queue = spawn_context.Queue(maxsize=1)
    spooler_process = spawn_context.Process(
        target=run_output_spooler,
        args=(
            message_queue,
            result_queue,
            prepared_output_run.output_run_paths,
            prepared_output_run.run_manifest_record,
        ),
        kwargs={"resume": resume},
        daemon=True,
    )
    spooler_process.start()
    try:
        for chunk_accumulator in frame_iterator:
            ensure_spooler_is_healthy(spooler_process, result_queue)
            chunk_payload = build_chunk_payload(chunk_accumulator)
            try:
                message_queue.put(chunk_payload, timeout=queue_timeout_seconds)
            except queue_module.Full as error:
                message = (
                    "Output spooler queue remained full for too long. "
                    "Storage throughput is bottlenecking compute, so the run was stopped explicitly."
                )
                raise TimeoutError(message) from error
        try:
            message_queue.put(None, timeout=queue_timeout_seconds)
        except queue_module.Full as error:
            message = (
                "Output spooler queue remained full while shutting down. "
                "Storage throughput is bottlenecking compute, so the run was stopped explicitly."
            )
            raise TimeoutError(message) from error
        wait_for_spooler_outcome(spooler_process, result_queue, queue_timeout_seconds)
        return prepared_output_run.output_run_paths
    except Exception:
        terminate_spooler_process(spooler_process)
        raise
    finally:
        close_queue(message_queue)
        close_queue(result_queue)


def finalize_chunks_to_parquet(output_run_paths: OutputRunPaths) -> Path:
    """Compact committed Arrow chunk files into one compressed Parquet file."""
    manifest = OutputManifest(output_run_paths.manifest_path)
    try:
        run_manifest_record = manifest.load_run_record()
        if run_manifest_record is None:
            message = f"No run metadata was found in '{output_run_paths.manifest_path}'."
            raise ValueError(message)
        committed_chunk_records = manifest.load_committed_chunk_records()
    finally:
        manifest.close()

    chunk_validation_result = validate_chunk_records(output_run_paths, committed_chunk_records)
    if chunk_validation_result.missing_chunk_records or chunk_validation_result.checksum_mismatch_chunk_records:
        message = (
            "Cannot finalize chunked output because committed chunks are incomplete or corrupted: "
            f"{build_chunk_validation_error_message(chunk_validation_result)}."
        )
        raise ValueError(message)

    final_parquet_path = output_run_paths.run_directory / "final.parquet"
    temporary_parquet_path = final_parquet_path.with_suffix(".parquet.tmp")
    if chunk_validation_result.valid_chunk_records:
        chunk_file_paths = [
            output_run_paths.run_directory / chunk_record.file_path
            for chunk_record in chunk_validation_result.valid_chunk_records
        ]
        pl.scan_ipc(chunk_file_paths, cache=False, rechunk=False, glob=False).sink_parquet(
            temporary_parquet_path,
            compression="zstd",
        )
    else:
        pl.DataFrame(schema=get_output_schema(run_manifest_record.association_mode)).write_parquet(
            temporary_parquet_path,
            compression="zstd",
        )
    temporary_parquet_path.replace(final_parquet_path)
    return final_parquet_path
