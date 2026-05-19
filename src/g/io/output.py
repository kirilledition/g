"""Output persistence orchestration backed by the native Rust writer."""

from __future__ import annotations

import json
import logging
import re
import typing
from dataclasses import dataclass
from pathlib import Path

from g import _core, types

logger = logging.getLogger(__name__)


OUTPUT_COMPRESSION_CODEC = "zstd"
CHUNK_FILENAME_PATTERN = re.compile(r"^chunk_(\d+)(?:_(\d+))?\.arrow$")
RUN_MANIFEST_FILENAME = "run_manifest.json"
RUN_MANIFEST_SCHEMA_VERSION = 3
DEFAULT_WRITER_QUEUE_DEPTH = 4
DEFAULT_WRITER_THREAD_COUNT = 8
DEFAULT_CHUNKS_PER_ARROW_FILE = 4


@dataclass(frozen=True)
class OutputRunPaths:
    """Filesystem paths for one chunked output run."""

    run_directory: Path
    chunks_directory: Path


@dataclass(frozen=True)
class PreparedOutputRun:
    """Prepared output run state for chunk persistence."""

    output_run_paths: OutputRunPaths
    existing_manifest: dict[str, typing.Any] | None


@dataclass(frozen=True)
class InitializedOutputRun:
    """Validated output run state for chunk persistence."""

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


def build_file_fingerprint(path: Path | None) -> dict[str, typing.Any] | None:
    """Build a lightweight immutable fingerprint for an input file."""
    if path is None:
        return None
    path_stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": path_stat.st_size,
        "mtime_ns": path_stat.st_mtime_ns,
    }


def build_binary_correction_plan_manifest(binary_correction_plan: types.BinaryCorrectionPlan) -> dict[str, typing.Any]:
    """Build the manifest representation of a binary correction plan."""
    return {
        "method": str(binary_correction_plan.method),
        "p_threshold": binary_correction_plan.p_threshold,
        "firth_se": binary_correction_plan.firth_se,
    }


def build_current_run_manifest_header(
    *,
    association_mode: types.AssociationMode,
    bgen_path: Path,
    sample_path: Path | None,
    phenotype_path: Path,
    phenotype_name: str,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...],
    prediction_list_path: Path,
    sample_count: int,
    variant_count: int,
    chunk_size: int,
    variant_limit: int | None,
    binary_correction_plan: types.BinaryCorrectionPlan,
    trusted_no_missing_diploid: bool,
    sample_key_mode: types.SampleKeyMode,
) -> dict[str, typing.Any]:
    """Build immutable run manifest fields from the current execution plan."""
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "association_mode": str(association_mode),
        "bgen": build_file_fingerprint(bgen_path),
        "sample": build_file_fingerprint(sample_path),
        "phenotype_file": build_file_fingerprint(phenotype_path),
        "phenotype_name": phenotype_name,
        "covariate_file": build_file_fingerprint(covariate_path),
        "covariate_names": list(covariate_names),
        "prediction_list": build_file_fingerprint(prediction_list_path),
        "sample_count": sample_count,
        "variant_count": variant_count,
        "chunk_size": chunk_size,
        "variant_limit": variant_limit,
        "binary_correction_plan": build_binary_correction_plan_manifest(binary_correction_plan),
        "trusted_no_missing_diploid": trusted_no_missing_diploid,
        "sample_key_mode": str(sample_key_mode),
    }


def validate_manifest_compatibility(
    manifest: dict[str, typing.Any],
    current_header: dict[str, typing.Any],
) -> None:
    """Validate immutable manifest fields against the current run header."""
    for field_name, current_value in current_header.items():
        if field_name not in manifest:
            message = f"Run manifest field '{field_name}' is missing."
            raise ValueError(message)
        if manifest[field_name] != current_value:
            message = f"Run manifest field '{field_name}' is incompatible with the requested run."
            raise ValueError(message)


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


def validate_strict_manifest_chunks(
    output_run_paths: OutputRunPaths,
    manifest: dict[str, typing.Any],
) -> frozenset[int]:
    """Validate committed manifest chunks against Arrow files."""
    chunk_identifiers = _core.validate_strict_manifest_chunks(
        str(output_run_paths.chunks_directory),
        json.dumps(manifest),
    )
    return frozenset(int(chunk_identifier) for chunk_identifier in chunk_identifiers)


def initialize_output_run(
    *,
    output_run_paths: OutputRunPaths,
    existing_manifest: dict[str, typing.Any] | None,
    current_header: dict[str, typing.Any],
    resume: bool,
    resume_mode: types.ResumeMode,
) -> InitializedOutputRun:
    """Validate/write the manifest header and return accepted committed chunks."""
    committed_chunk_identifiers = frozenset[int]()
    committed_chunks: list[typing.Any] = []
    manifest = dict(existing_manifest or {})
    if existing_manifest is not None:
        validate_manifest_compatibility(existing_manifest, current_header)
        committed_chunks_value = existing_manifest.get("committed_chunks", [])
        if not isinstance(committed_chunks_value, list):
            message = "Run manifest committed_chunks field must be a list."
            raise ValueError(message)
        committed_chunks = committed_chunks_value
        if resume:
            if resume_mode == types.ResumeMode.STRICT:
                committed_chunk_identifiers = validate_strict_manifest_chunks(output_run_paths, existing_manifest)
            else:
                committed_chunk_identifiers = read_manifest_committed_chunk_identifiers(existing_manifest)
            logger.info("Resuming run with %d previously committed chunks.", len(committed_chunk_identifiers))
    elif resume:
        message = "Resume requires run_manifest.json."
        raise ValueError(message)
    manifest.update(current_header)
    manifest["committed_chunks"] = committed_chunks
    manifest.setdefault("finalized", False)
    write_run_manifest(output_run_paths, manifest)
    return InitializedOutputRun(committed_chunk_identifiers=committed_chunk_identifiers)


def prepare_output_run(
    *,
    output_root: Path,
    association_mode: types.AssociationMode,
    resume: bool,
    resume_mode: types.ResumeMode = types.ResumeMode.FAST,
) -> PreparedOutputRun:
    """Prepare a chunked output run directory and load existing manifest state."""
    output_run_paths = resolve_output_run_paths(output_root, association_mode)
    if not resume and output_run_paths.run_directory.exists() and any(output_run_paths.run_directory.iterdir()):
        message = (
            f"Output run directory '{output_run_paths.run_directory}' already exists and is not empty. "
            "Use --resume or choose a new output path."
        )
        raise ValueError(message)
    output_run_paths.chunks_directory.mkdir(parents=True, exist_ok=True)
    manifest = load_run_manifest(output_run_paths)
    if resume and manifest is None:
        message = "Resume requires run_manifest.json."
        raise ValueError(message)
    return PreparedOutputRun(
        output_run_paths=output_run_paths,
        existing_manifest=manifest,
    )


def create_output_writer_session(
    output_run_paths: OutputRunPaths,
    association_mode: types.AssociationMode,
    *,
    writer_thread_count: int,
    writer_queue_depth: int,
    finalize_parquet: bool,
    chunks_per_arrow_file: int = DEFAULT_CHUNKS_PER_ARROW_FILE,
    arrow_compression: types.ArrowCompression = types.ArrowCompression.ZSTD,
) -> typing.Any:
    """Create one native Rust output writer session."""
    return _core.OutputWriterSession(
        run_directory=str(output_run_paths.run_directory),
        chunks_directory=str(output_run_paths.chunks_directory),
        association_mode=str(association_mode),
        writer_thread_count=writer_thread_count,
        writer_queue_depth=writer_queue_depth,
        finalize_parquet=finalize_parquet,
        chunks_per_arrow_file=chunks_per_arrow_file,
        arrow_compression=arrow_compression.value,
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


def finalize_chunks_to_regenie_text(
    output_run_paths: OutputRunPaths,
    regenie_text_path: Path,
) -> None:
    """Materialize committed chunk files as REGENIE-compatible text."""
    _core.finalize_output_run_chunks_to_regenie_text(
        chunks_directory=str(output_run_paths.chunks_directory),
        regenie_text_path=str(regenie_text_path),
    )
