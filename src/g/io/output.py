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

from g import _core, runtime_policy, types

logger = logging.getLogger(__name__)


OUTPUT_COMPRESSION_CODEC = "zstd"
CHUNK_FILENAME_PATTERN = re.compile(r"^chunk_(\d+)(?:_(\d+))?\.arrow$")
PART_FILENAME_PATTERN = re.compile(r"^part_(\d+)(?:_(\d+))?\.parquet$")
REGENIE_PART_FILENAME_PATTERN = re.compile(r"^part_(\d+)(?:_(\d+))?\.regenie$")
RUN_MANIFEST_FILENAME = "run_manifest.json"
RUN_MANIFEST_SCHEMA_VERSION = 8
OUTPUT_SCHEMA_VERSION = 1
JAX_MATMUL_PRECISION_WHEN_UNSET = "float32"
RESUME_POLICY = "manifest_committed_chunks"
RESULT_STATISTIC_OUTPUT_DTYPE = "float32"


class MultiPhenotypeSampleMode(enum.StrEnum):
    """Sample inclusion policy for one output run."""

    SINGLE_PHENOTYPE = "single-phenotype"
    PER_PHENOTYPE = "per-phenotype"
    COMPLETE_CASE = "complete-case"


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


def parse_run_manifest_json(manifest_json: str, manifest_path: Path | None = None) -> dict[str, typing.Any]:
    """Parse a native run manifest JSON payload for Python callers."""
    manifest: typing.Any = json.loads(manifest_json)
    if not isinstance(manifest, dict):
        message = (
            "Run manifest must contain a JSON object."
            if manifest_path is None
            else f"Run manifest '{manifest_path}' must contain a JSON object."
        )
        raise ValueError(message)
    return manifest


def resolve_output_run_paths(
    output_root: Path,
    association_mode: types.AssociationMode,
    output_format: types.OutputFormat = types.OutputFormat.PARQUET,
) -> OutputRunPaths:
    """Derive run paths from an output root and association mode."""
    native_run_paths = _core.resolve_output_run_paths(str(output_root), association_mode.value, output_format.value)
    return OutputRunPaths(
        run_directory=Path(native_run_paths.run_directory),
        chunks_directory=Path(native_run_paths.chunks_directory),
    )


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
    manifest_json = _core.load_run_manifest_json(str(output_run_paths.run_directory))
    if manifest_json is None:
        return None
    return parse_run_manifest_json(manifest_json, manifest_path)


def write_run_manifest(output_run_paths: OutputRunPaths, manifest: dict[str, typing.Any]) -> None:
    """Atomically write a run manifest."""
    _core.write_run_manifest_json(
        str(output_run_paths.run_directory),
        json.dumps(manifest, sort_keys=True),
    )


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
) -> dict[str, typing.Any]:
    """Build manifest fields for JAX precision and backend policy."""
    return {
        "device": device.value,
        "enable_x64": runtime_policy.JAX_ENABLE_X64,
        "matmul_precision": JAX_MATMUL_PRECISION_WHEN_UNSET if matmul_precision is None else matmul_precision.value,
    }


def build_association_backend_manifest(
    *,
    association_backend_kind: types.AssociationBackendKind,
    association_mode: types.AssociationMode,
    jax_device: types.Device,
    gpu_genotype_format: types.GpuGenotypeFormat,
) -> dict[str, typing.Any]:
    """Build manifest fields for the selected association backend."""
    return {
        "kind": association_backend_kind.value,
        "association_mode": association_mode.value,
        "device": jax_device.value,
        "genotype_format": gpu_genotype_format.value,
    }


def build_output_writer_manifest(
    *,
    output_format: types.OutputFormat,
    finalize_parquet: bool,
    writer_thread_count: int,
    writer_queue_depth: int,
    chunks_per_arrow_file: int,
    arrow_compression: types.ArrowCompression,
    parquet_compression: types.ParquetCompression,
) -> dict[str, typing.Any]:
    """Build manifest fields for output materialization and writer settings."""
    return {
        "output_format": output_format.value,
        "finalize_parquet": finalize_parquet,
        "writer_thread_count": writer_thread_count,
        "writer_queue_depth": writer_queue_depth,
        "chunks_per_arrow_file": chunks_per_arrow_file,
        "arrow_compression": arrow_compression.value,
        "parquet_compression": parquet_compression.value,
        "result_statistic_dtype": RESULT_STATISTIC_OUTPUT_DTYPE,
    }


def build_current_run_manifest_header(
    *,
    association_mode: types.AssociationMode,
    association_backend_kind: types.AssociationBackendKind,
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
    bgen_decode_tile_variant_count: int,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode,
    jax_device: types.Device,
    jax_matmul_precision: types.JaxMatmulPrecision | None = None,
    gpu_genotype_format: types.GpuGenotypeFormat,
    score_dtype: types.FloatingPointDtype,
    firth_dtype: types.FloatingPointDtype,
    multi_phenotype_sample_mode: MultiPhenotypeSampleMode = MultiPhenotypeSampleMode.SINGLE_PHENOTYPE,
    phenotype_compute_group_id: str | None = None,
    sample_set_fingerprint: str | None = None,
    covariate_design_fingerprint: str | None = None,
    prediction_alignment_fingerprint: str | None = None,
    output_format: types.OutputFormat,
    finalize_parquet: bool,
    writer_thread_count: int,
    writer_queue_depth: int,
    chunks_per_arrow_file: int,
    arrow_compression: types.ArrowCompression,
    parquet_compression: types.ParquetCompression,
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
        parquet_compression=parquet_compression,
    )
    jax_policy_manifest = build_jax_policy_manifest(
        device=jax_device,
        matmul_precision=jax_matmul_precision,
    )
    association_backend_manifest = build_association_backend_manifest(
        association_backend_kind=association_backend_kind,
        association_mode=association_mode,
        jax_device=jax_device,
        gpu_genotype_format=gpu_genotype_format,
    )
    execution_plan = normalize_execution_plan_value(
        {
            "manifest_schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
            "association_mode": association_mode,
            "association_backend": association_backend_manifest,
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
            "gpu_genotype_format": gpu_genotype_format,
            "score_dtype": score_dtype,
            "firth_dtype": firth_dtype,
            "multi_phenotype_sample_mode": multi_phenotype_sample_mode,
            "phenotype_compute_group_id": phenotype_compute_group_id,
            "sample_set_fingerprint": sample_set_fingerprint,
            "covariate_design_fingerprint": covariate_design_fingerprint,
            "prediction_alignment_fingerprint": prediction_alignment_fingerprint,
            "output_writer": output_writer_manifest,
            "resume_policy": RESUME_POLICY,
        }
    )
    header = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "association_mode": str(association_mode),
        "association_backend": association_backend_manifest,
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
        "gpu_genotype_format": gpu_genotype_format.value,
        "score_dtype": score_dtype.value,
        "firth_dtype": firth_dtype.value,
        "multi_phenotype_sample_mode": multi_phenotype_sample_mode.value,
        "phenotype_compute_group_id": phenotype_compute_group_id,
        "sample_set_fingerprint": sample_set_fingerprint,
        "covariate_design_fingerprint": covariate_design_fingerprint,
        "prediction_alignment_fingerprint": prediction_alignment_fingerprint,
        "output_writer": output_writer_manifest,
        "resume_policy": RESUME_POLICY,
        "execution_plan": execution_plan,
    }
    header["execution_plan_hash"] = build_execution_plan_hash(execution_plan)
    return header


def validate_manifest_compatibility(
    manifest: dict[str, typing.Any],
    current_header: dict[str, typing.Any],
) -> None:
    """Validate immutable manifest fields against the current run header."""
    _core.validate_run_manifest_compatibility(
        json.dumps(manifest, sort_keys=True),
        json.dumps(current_header, sort_keys=True),
    )


def read_manifest_committed_chunk_identifiers(manifest: dict[str, typing.Any]) -> frozenset[int]:
    """Read committed chunk identifiers from a run manifest."""
    chunk_identifiers = _core.read_manifest_committed_chunk_identifiers(json.dumps(manifest, sort_keys=True))
    return frozenset(int(chunk_identifier) for chunk_identifier in chunk_identifiers)


def validate_strict_manifest_chunks(
    output_run_paths: OutputRunPaths,
    manifest: dict[str, typing.Any],
) -> frozenset[int]:
    """Validate committed manifest chunks against output files."""
    chunk_identifiers = _core.validate_strict_manifest_chunks(
        str(output_run_paths.chunks_directory),
        json.dumps(manifest),
    )
    return frozenset(int(chunk_identifier) for chunk_identifier in chunk_identifiers)


def repair_strict_manifest_chunk_commits(
    output_run_paths: OutputRunPaths,
    manifest: dict[str, typing.Any],
) -> list[typing.Any]:
    """Recover committed chunk manifest records from output metadata."""
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
    native_initialized_output_run = _core.initialize_output_run(
        str(output_run_paths.run_directory),
        str(output_run_paths.chunks_directory),
        None if existing_manifest is None else json.dumps(existing_manifest, sort_keys=True),
        json.dumps(current_header, sort_keys=True),
        resume,
        resume_mode.value,
    )
    committed_chunk_identifiers = frozenset(
        int(chunk_identifier) for chunk_identifier in native_initialized_output_run.committed_chunk_identifiers
    )
    if resume:
        logger.info("Resuming run with %d previously committed chunks.", len(committed_chunk_identifiers))
    return InitializedOutputRun(committed_chunk_identifiers=committed_chunk_identifiers)


def prepare_output_run(
    *,
    output_root: Path,
    association_mode: types.AssociationMode,
    output_format: types.OutputFormat = types.OutputFormat.PARQUET,
    resume: bool,
    resume_mode: types.ResumeMode = types.ResumeMode.FAST,
) -> PreparedOutputRun:
    """Prepare a chunked output run directory and load existing manifest state."""
    del resume_mode
    native_prepared_output_run = _core.prepare_output_run(
        str(output_root),
        association_mode.value,
        output_format.value,
        resume,
    )
    output_run_paths = OutputRunPaths(
        run_directory=Path(native_prepared_output_run.run_directory),
        chunks_directory=Path(native_prepared_output_run.chunks_directory),
    )
    manifest = (
        None
        if native_prepared_output_run.existing_manifest_json is None
        else parse_run_manifest_json(
            native_prepared_output_run.existing_manifest_json,
            get_run_manifest_path(output_run_paths),
        )
    )
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
    output_format: types.OutputFormat,
    chunks_per_arrow_file: int,
    arrow_compression: types.ArrowCompression,
    parquet_compression: types.ParquetCompression,
    collect_stage_timings: bool = False,
) -> typing.Any:
    """Create one native Rust output writer session."""
    return _core.OutputWriterSession(
        run_directory=str(output_run_paths.run_directory),
        chunks_directory=str(output_run_paths.chunks_directory),
        association_mode=str(association_mode),
        writer_thread_count=writer_thread_count,
        writer_queue_depth=writer_queue_depth,
        output_format=output_format.value,
        finalize_parquet=finalize_parquet,
        chunks_per_arrow_file=chunks_per_arrow_file,
        arrow_compression=arrow_compression.value,
        parquet_compression=parquet_compression.value,
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
            or PART_FILENAME_PATTERN.match(child_path.name) is not None
            or REGENIE_PART_FILENAME_PATTERN.match(child_path.name) is not None
        )
    )


def finalize_chunks_to_parquet(
    output_run_paths: OutputRunPaths,
    association_mode: types.AssociationMode,
    output_format: types.OutputFormat = types.OutputFormat.ARROW,
) -> Path:
    """Compact committed chunk files into one compressed Parquet file in Rust."""
    final_parquet_path = _core.finalize_output_run_chunks(
        run_directory=str(output_run_paths.run_directory),
        chunks_directory=str(output_run_paths.chunks_directory),
        association_mode=str(association_mode),
        output_format=output_format.value,
    )
    return Path(final_parquet_path)
