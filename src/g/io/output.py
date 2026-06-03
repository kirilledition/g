"""Output persistence orchestration backed by the native Rust writer."""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import logging
import re
import typing
from dataclasses import dataclass
from pathlib import Path

from g import _core, types
from g.interface import config

logger = logging.getLogger(__name__)


OUTPUT_COMPRESSION_CODEC = "zstd"
CHUNK_FILENAME_PATTERN = re.compile(r"^chunk_(\d+)(?:_(\d+))?\.arrow$")
RUN_MANIFEST_FILENAME = "run_manifest.json"
RUN_MANIFEST_SCHEMA_VERSION = 4
OUTPUT_SCHEMA_VERSION = 1
DEFAULT_BGEN_DECODE_TILE_VARIANT_COUNT = 64
DEFAULT_JAX_MATMUL_PRECISION = "float32"
RESUME_POLICY = "manifest_committed_chunks"
DEFAULT_WRITER_QUEUE_DEPTH = config.DEFAULT_OUTPUT_WRITER_QUEUE_DEPTH
DEFAULT_WRITER_THREAD_COUNT = config.DEFAULT_OUTPUT_WRITER_THREADS
DEFAULT_CHUNKS_PER_ARROW_FILE = config.DEFAULT_OUTPUT_CHUNKS_PER_ARROW_FILE
RESULT_STATISTIC_OUTPUT_DTYPE = "float32"


class MultiPhenotypeSampleMode(enum.StrEnum):
    """Sample inclusion policy for one output run."""

    SINGLE_PHENOTYPE = "single_phenotype"
    COMPLETE_CASE_INTERSECTION = "complete_case_intersection"


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


def normalize_execution_plan_value(value: typing.Any) -> typing.Any:
    """Normalize execution-plan values for stable JSON hashing."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return normalize_execution_plan_value(dataclasses.asdict(value))
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): normalize_execution_plan_value(item_value)
            for key, item_value in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list | tuple):
        return [normalize_execution_plan_value(item_value) for item_value in value]
    return value


def build_execution_plan_hash(execution_plan: dict[str, typing.Any]) -> str:
    """Build a stable SHA-256 hash for compute/output-affecting run state."""
    normalized_execution_plan = normalize_execution_plan_value(execution_plan)
    execution_plan_bytes = json.dumps(
        normalized_execution_plan,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(execution_plan_bytes).hexdigest()


def build_jax_policy_manifest(
    *,
    device: types.Device = types.Device.CPU,
    matmul_precision: types.JaxMatmulPrecision | None = None,
    enable_x64: bool = config.DEFAULT_JAX_ENABLE_X64,
) -> dict[str, typing.Any]:
    """Build manifest fields for JAX precision and backend policy."""
    return {
        "device": device.value,
        "enable_x64": enable_x64,
        "matmul_precision": DEFAULT_JAX_MATMUL_PRECISION if matmul_precision is None else matmul_precision.value,
    }


def build_output_writer_manifest(
    *,
    output_format: types.OutputFormat = types.OutputFormat.PARQUET,
    finalize_parquet: bool = False,
    writer_thread_count: int = DEFAULT_WRITER_THREAD_COUNT,
    writer_queue_depth: int = DEFAULT_WRITER_QUEUE_DEPTH,
    chunks_per_arrow_file: int = DEFAULT_CHUNKS_PER_ARROW_FILE,
    arrow_compression: types.ArrowCompression = types.ArrowCompression.ZSTD,
) -> dict[str, typing.Any]:
    """Build manifest fields for output materialization and writer settings."""
    return {
        "output_format": output_format.value,
        "finalize_parquet": finalize_parquet,
        "writer_thread_count": writer_thread_count,
        "writer_queue_depth": writer_queue_depth,
        "chunks_per_arrow_file": chunks_per_arrow_file,
        "arrow_compression": arrow_compression.value,
        "result_statistic_dtype": RESULT_STATISTIC_OUTPUT_DTYPE,
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
    binary_kernel_config: typing.Any | None = None,
    bgen_decode_tile_variant_count: int = DEFAULT_BGEN_DECODE_TILE_VARIANT_COUNT,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode = types.TrustedBgenValidationMode.CACHE_ON_MISS,
    jax_device: types.Device = types.Device.CPU,
    jax_matmul_precision: types.JaxMatmulPrecision | None = None,
    jax_enable_x64: bool = config.DEFAULT_JAX_ENABLE_X64,
    score_dtype: types.FloatingPointDtype = config.DEFAULT_SCORE_DTYPE,
    firth_dtype: types.FloatingPointDtype = config.DEFAULT_FIRTH_DTYPE,
    multi_phenotype_sample_mode: MultiPhenotypeSampleMode = MultiPhenotypeSampleMode.SINGLE_PHENOTYPE,
    output_format: types.OutputFormat = types.OutputFormat.PARQUET,
    finalize_parquet: bool = False,
    writer_thread_count: int = DEFAULT_WRITER_THREAD_COUNT,
    writer_queue_depth: int = DEFAULT_WRITER_QUEUE_DEPTH,
    chunks_per_arrow_file: int = DEFAULT_CHUNKS_PER_ARROW_FILE,
    arrow_compression: types.ArrowCompression = types.ArrowCompression.ZSTD,
) -> dict[str, typing.Any]:
    """Build immutable run manifest fields from the current execution plan."""
    bgen_fingerprint = build_file_fingerprint(bgen_path)
    sample_fingerprint = build_file_fingerprint(sample_path)
    phenotype_file_fingerprint = build_file_fingerprint(phenotype_path)
    covariate_file_fingerprint = build_file_fingerprint(covariate_path)
    prediction_list_fingerprint = build_file_fingerprint(prediction_list_path)
    binary_correction_plan_manifest = build_binary_correction_plan_manifest(binary_correction_plan)
    output_writer_manifest = build_output_writer_manifest(
        output_format=output_format,
        finalize_parquet=finalize_parquet,
        writer_thread_count=writer_thread_count,
        writer_queue_depth=writer_queue_depth,
        chunks_per_arrow_file=chunks_per_arrow_file,
        arrow_compression=arrow_compression,
    )
    jax_policy_manifest = build_jax_policy_manifest(
        device=jax_device,
        matmul_precision=jax_matmul_precision,
        enable_x64=jax_enable_x64,
    )
    execution_plan = normalize_execution_plan_value(
        {
            "manifest_schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
            "association_mode": association_mode,
            "bgen": bgen_fingerprint,
            "sample": sample_fingerprint,
            "phenotype_file": phenotype_file_fingerprint,
            "phenotype_name": phenotype_name,
            "covariate_file": covariate_file_fingerprint,
            "covariate_names": covariate_names,
            "prediction_list": prediction_list_fingerprint,
            "sample_count": sample_count,
            "variant_count": variant_count,
            "chunk_size": chunk_size,
            "variant_limit": variant_limit,
            "binary_correction_plan": binary_correction_plan_manifest,
            "binary_kernel_config": binary_kernel_config,
            "trusted_no_missing_diploid": trusted_no_missing_diploid,
            "trusted_bgen_validation_mode": trusted_bgen_validation_mode,
            "sample_key_mode": sample_key_mode,
            "bgen_decode_tile_variant_count": bgen_decode_tile_variant_count,
            "jax_policy": jax_policy_manifest,
            "score_dtype": score_dtype,
            "firth_dtype": firth_dtype,
            "multi_phenotype_sample_mode": multi_phenotype_sample_mode,
            "output_writer": output_writer_manifest,
            "resume_policy": RESUME_POLICY,
        }
    )
    header = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "association_mode": str(association_mode),
        "bgen": bgen_fingerprint,
        "sample": sample_fingerprint,
        "phenotype_file": phenotype_file_fingerprint,
        "phenotype_name": phenotype_name,
        "covariate_file": covariate_file_fingerprint,
        "covariate_names": list(covariate_names),
        "prediction_list": prediction_list_fingerprint,
        "sample_count": sample_count,
        "variant_count": variant_count,
        "chunk_size": chunk_size,
        "variant_limit": variant_limit,
        "binary_correction_plan": binary_correction_plan_manifest,
        "binary_kernel_config": normalize_execution_plan_value(binary_kernel_config),
        "trusted_no_missing_diploid": trusted_no_missing_diploid,
        "trusted_bgen_validation_mode": str(trusted_bgen_validation_mode),
        "sample_key_mode": str(sample_key_mode),
        "bgen_decode_tile_variant_count": bgen_decode_tile_variant_count,
        "jax_policy": jax_policy_manifest,
        "score_dtype": score_dtype.value,
        "firth_dtype": firth_dtype.value,
        "multi_phenotype_sample_mode": multi_phenotype_sample_mode.value,
        "output_writer": output_writer_manifest,
        "resume_policy": RESUME_POLICY,
        "execution_plan": execution_plan,
    }
    header["execution_plan_hash"] = build_execution_plan_hash(execution_plan)
    return header


def find_first_manifest_mismatch_path(
    manifest_value: typing.Any,
    current_value: typing.Any,
    field_path: str,
) -> str | None:
    """Return the first nested field path that differs between two manifest values."""
    if isinstance(manifest_value, dict) and isinstance(current_value, dict):
        for key in sorted(set(manifest_value) | set(current_value)):
            nested_path = f"{field_path}.{key}"
            if key not in manifest_value or key not in current_value:
                return nested_path
            mismatch_path = find_first_manifest_mismatch_path(manifest_value[key], current_value[key], nested_path)
            if mismatch_path is not None:
                return mismatch_path
        return None
    if isinstance(manifest_value, list) and isinstance(current_value, list):
        for index, (manifest_item, current_item) in enumerate(zip(manifest_value, current_value, strict=False)):
            nested_path = f"{field_path}[{index}]"
            mismatch_path = find_first_manifest_mismatch_path(manifest_item, current_item, nested_path)
            if mismatch_path is not None:
                return mismatch_path
        if len(manifest_value) != len(current_value):
            return field_path
        return None
    if manifest_value != current_value:
        return field_path
    return None


def validate_manifest_compatibility(
    manifest: dict[str, typing.Any],
    current_header: dict[str, typing.Any],
) -> None:
    """Validate immutable manifest fields against the current run header."""
    for field_name, current_value in current_header.items():
        if field_name not in manifest:
            message = f"Run manifest field '{field_name}' is missing."
            raise ValueError(message)
        mismatch_path = find_first_manifest_mismatch_path(manifest[field_name], current_value, field_name)
        if mismatch_path is not None:
            message = f"Run manifest field '{mismatch_path}' is incompatible with the requested run."
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


def repair_strict_manifest_chunk_commits(
    output_run_paths: OutputRunPaths,
    manifest: dict[str, typing.Any],
) -> list[typing.Any]:
    """Recover committed chunk manifest records from Arrow metadata."""
    repaired_commits = json.loads(
        _core.repair_strict_manifest_chunk_commits(
            str(output_run_paths.chunks_directory),
            json.dumps(manifest),
        )
    )
    if not isinstance(repaired_commits, list):
        message = "Strict resume repaired committed chunks must be a list."
        raise ValueError(message)
    return repaired_commits


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
    manifest = dict(load_run_manifest(output_run_paths) or {})
    if existing_manifest is not None:
        validate_manifest_compatibility(existing_manifest, current_header)
        if not manifest:
            manifest = dict(existing_manifest)
        committed_chunks_value = existing_manifest.get("committed_chunks", [])
        if not isinstance(committed_chunks_value, list):
            message = "Run manifest committed_chunks field must be a list."
            raise ValueError(message)
        committed_chunks = committed_chunks_value
        if resume:
            if resume_mode == types.ResumeMode.STRICT:
                committed_chunks = repair_strict_manifest_chunk_commits(output_run_paths, existing_manifest)
                committed_chunk_identifiers = read_manifest_committed_chunk_identifiers(
                    {"committed_chunks": committed_chunks}
                )
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
    collect_stage_timings: bool = False,
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
        collect_stage_timings=collect_stage_timings,
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
