"""Runner-owned output persistence backed by the native Rust writer."""

from __future__ import annotations

import dataclasses
import enum
import re
import typing
from dataclasses import dataclass
from pathlib import Path

from g import _core, types

if typing.TYPE_CHECKING:
    from g import execution_plan

OUTPUT_COMPRESSION_CODEC = "zstd"
CHUNK_FILENAME_PATTERN = re.compile(r"^chunk_(\d+)(?:_(\d+))?\.arrow$")
PART_FILENAME_PATTERN = re.compile(r"^part_(\d+)(?:_(\d+))?\.parquet$")
REGENIE_PART_FILENAME_PATTERN = re.compile(r"^part_(\d+)(?:_(\d+))?\.regenie$")
RUN_MANIFEST_FILENAME = "run_manifest.json"
RUN_MANIFEST_SCHEMA_VERSION = 9
OUTPUT_SCHEMA_VERSION = 2
RESUME_POLICY = "manifest_committed_chunks"
DEFAULT_RESULT_STATISTIC_OUTPUT_DTYPE = types.FloatingPointDtype.FLOAT32


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
class OutputWriterSettings:
    """Output writer and finalization settings for a run.

    Attributes:
        finalize_parquet: Whether chunk output should be finalized to one Parquet file.
        writer_thread_count: Number of writer worker threads.
        writer_queue_depth: Maximum queued chunk writes.
        chunks_per_arrow_file: Number of chunks per Arrow output file.
        arrow_compression: Arrow IPC compression codec.
        parquet_compression: Parquet finalization compression codec.
        output_format: Chunk output format.
        output_statistic_dtype: Persisted dtype for public statistic columns.

    """

    finalize_parquet: bool
    writer_thread_count: int
    writer_queue_depth: int
    chunks_per_arrow_file: int
    arrow_compression: types.ArrowCompression
    parquet_compression: types.ParquetCompression
    output_format: types.OutputFormat
    output_statistic_dtype: types.FloatingPointDtype


@dataclass(frozen=True)
class PreparedOutputRun:
    """Prepared output run state for chunk persistence."""

    output_run_paths: OutputRunPaths
    existing_manifest: dict[str, typing.Any] | None


@dataclass(frozen=True)
class InitializedOutputRun:
    """Validated output run state for chunk persistence."""

    committed_chunk_identifiers: frozenset[int]


@dataclass(frozen=True)
class ManifestFileFingerprint:
    """Stable fingerprint for an input file recorded in a run manifest.

    Attributes:
        path: Absolute input file path.
        size: File size in bytes.
        mtime_ns: File modification timestamp in nanoseconds.
        content_hash_algorithm: Content hash algorithm or metadata-only marker.
        content_sha256: SHA-256 content hash when content hashing is enabled.

    """

    path: str
    size: int
    mtime_ns: int
    content_hash_algorithm: str
    content_sha256: str | None


class ManifestFileFingerprintCache:
    """Native run-scoped cache for immutable input file fingerprints."""

    def __init__(self) -> None:
        """Initialize an empty native fingerprint cache."""
        self.native_cache: _core.NativeManifestFileFingerprintCache = _core.NativeManifestFileFingerprintCache()

    def build_file_fingerprint(
        self,
        path: Path | None,
        *,
        include_content_hash: bool,
    ) -> ManifestFileFingerprint | None:
        """Build or reuse a fingerprint for the observed input file state."""
        if path is None:
            return None
        return manifest_file_fingerprint_from_native_payload(
            self.native_cache.build_file_fingerprint_payload(str(path), include_content_hash)
        )


@dataclass(frozen=True)
class PredictionLocoFileFingerprint:
    """Manifest identity for one phenotype's LOCO prediction file.

    Attributes:
        phenotype: Phenotype name resolved from the prediction list.
        path: Absolute LOCO prediction file path.
        size: File size in bytes.
        mtime_ns: File modification timestamp in nanoseconds.
        content_hash_algorithm: Content hash algorithm.
        content_sha256: SHA-256 content hash.

    """

    phenotype: str
    path: str
    size: int
    mtime_ns: int
    content_hash_algorithm: str
    content_sha256: str


RunManifestHeaderInput = dict[str, typing.Any]


def native_output_lifecycle_policy() -> _core.NativeOutputLifecyclePolicy:
    """Build the native output lifecycle policy handle."""
    return _core.NativeOutputLifecyclePolicy()


def get_run_manifest_path(output_run_paths: OutputRunPaths) -> Path:
    """Return the run manifest path for an output run."""
    return output_run_paths.run_directory / RUN_MANIFEST_FILENAME


def resolve_output_run_paths(
    output_root: Path,
    association_mode: types.AssociationMode,
    output_format: types.OutputFormat,
) -> OutputRunPaths:
    """Derive run paths from an output root and association mode."""
    native_run_paths = native_output_lifecycle_policy().resolve_output_run_paths(
        str(output_root),
        association_mode.value,
        output_format.value,
    )
    return OutputRunPaths(
        run_directory=Path(native_run_paths.run_directory),
        chunks_directory=Path(native_run_paths.chunks_directory),
    )


def build_chunk_file_name(chunk_identifier: int) -> str:
    """Build a deterministic chunk file name from a chunk identifier."""
    return f"chunk_{chunk_identifier:09d}.arrow"


def scan_committed_chunk_identifiers(chunks_directory: Path) -> frozenset[int]:
    """Scan a chunks directory and return identifiers of completed chunks."""
    chunk_identifiers = native_output_lifecycle_policy().scan_committed_chunk_identifiers(str(chunks_directory))
    return frozenset(int(chunk_identifier) for chunk_identifier in chunk_identifiers)


def load_run_manifest(output_run_paths: OutputRunPaths) -> dict[str, typing.Any] | None:
    """Load a run manifest when present."""
    manifest_payload = native_output_lifecycle_policy().load_run_manifest_payload(str(output_run_paths.run_directory))
    if manifest_payload is None:
        return None
    return require_native_mapping_payload(
        manifest_payload,
        f"Run manifest '{get_run_manifest_path(output_run_paths)}' must contain a JSON object.",
    )


def write_run_manifest(output_run_paths: OutputRunPaths, manifest: dict[str, typing.Any]) -> None:
    """Atomically write a run manifest."""
    native_output_lifecycle_policy().write_run_manifest(
        str(output_run_paths.run_directory),
        manifest,
    )


def build_file_fingerprint(path: Path | None, *, include_content_hash: bool) -> ManifestFileFingerprint | None:
    """Build a lightweight immutable fingerprint for an input file."""
    return ManifestFileFingerprintCache().build_file_fingerprint(path, include_content_hash=include_content_hash)


def manifest_file_fingerprint_from_native_payload(payload: object) -> ManifestFileFingerprint:
    """Adapt a native file-fingerprint payload to the public Python dataclass."""
    fingerprint_payload = native_mapping_payload(payload)
    return ManifestFileFingerprint(
        path=typing.cast("str", fingerprint_payload["path"]),
        size=typing.cast("int", fingerprint_payload["size"]),
        mtime_ns=typing.cast("int", fingerprint_payload["mtime_ns"]),
        content_hash_algorithm=typing.cast("str", fingerprint_payload["content_hash_algorithm"]),
        content_sha256=typing.cast("str | None", fingerprint_payload["content_sha256"]),
    )


def native_mapping_payload(payload: object) -> dict[str, typing.Any]:
    """Adapt a native mapping payload to a mutable Python dictionary."""
    return dict(typing.cast("typing.Mapping[str, typing.Any]", payload))


def native_json_payload(payload: object) -> typing.Any:
    """Normalize native JSON payload containers to mutable Python containers."""
    if isinstance(payload, dict):
        mapping_payload = typing.cast("dict[object, object]", payload)
        return {
            str(payload_key): native_json_payload(payload_value)
            for payload_key, payload_value in mapping_payload.items()
        }
    if isinstance(payload, tuple | list):
        sequence_payload = typing.cast("typing.Iterable[object]", payload)
        return [native_json_payload(payload_value) for payload_value in sequence_payload]
    return payload


def require_native_mapping_payload(payload: object, message: str) -> dict[str, typing.Any]:
    """Adapt a native mapping payload and reject non-object JSON payloads."""
    if not isinstance(payload, dict):
        raise ValueError(message)
    return typing.cast("dict[str, typing.Any]", native_json_payload(payload))


def build_prediction_loco_file_fingerprints(
    *,
    prediction_list_path: Path,
    phenotype_names: tuple[str, ...],
    fingerprint_cache: ManifestFileFingerprintCache | None,
) -> tuple[PredictionLocoFileFingerprint, ...]:
    """Build content fingerprints for LOCO files selected from a prediction list."""
    resolved_fingerprint_cache = fingerprint_cache if fingerprint_cache is not None else ManifestFileFingerprintCache()
    loco_file_payloads = resolved_fingerprint_cache.native_cache.build_prediction_loco_file_fingerprints_payload(
        str(prediction_list_path),
        list(phenotype_names),
    )
    if not isinstance(loco_file_payloads, tuple):
        message = "Native LOCO fingerprint payload must contain a JSON array."
        raise ValueError(message)
    return tuple(prediction_loco_file_fingerprint_from_native_payload(payload) for payload in loco_file_payloads)


def prediction_loco_file_fingerprint_from_native_payload(payload: object) -> PredictionLocoFileFingerprint:
    """Adapt a native LOCO file fingerprint payload."""
    loco_file_payload = native_mapping_payload(payload)
    content_sha256 = typing.cast("str | None", loco_file_payload["content_sha256"])
    if content_sha256 is None:
        message = "LOCO prediction file fingerprint must include a content hash."
        raise ValueError(message)
    return PredictionLocoFileFingerprint(
        phenotype=typing.cast("str", loco_file_payload["phenotype"]),
        path=typing.cast("str", loco_file_payload["path"]),
        size=typing.cast("int", loco_file_payload["size"]),
        mtime_ns=typing.cast("int", loco_file_payload["mtime_ns"]),
        content_hash_algorithm=typing.cast("str", loco_file_payload["content_hash_algorithm"]),
        content_sha256=content_sha256,
    )


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


def build_execution_plan_hash(execution_plan: typing.Any) -> str:
    """Build a stable SHA-256 hash for compute/output-affecting run state."""
    normalized_execution_plan = normalize_execution_plan_value(execution_plan)
    return native_output_lifecycle_policy().build_manifest_json_sha256_from_value(normalized_execution_plan)


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
    prediction_input_phenotype_names: tuple[str, ...],
    fingerprint_cache: ManifestFileFingerprintCache | None,
    sample_count: int,
    variant_count: int,
    chunk_size: int,
    variant_limit: int | None,
    binary_correction_plan: types.BinaryCorrectionPlan,
    trusted_no_missing_diploid: bool,
    sample_key_mode: types.SampleKeyMode,
    binary_kernel_config: typing.Any | None,
    bgen_decode_tile_variant_count: int,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode,
    jax_device: types.Device,
    jax_enable_x64: bool,
    jax_matmul_precision: types.JaxMatmulPrecision | None,
    requested_gpu_genotype_format: types.GpuGenotypeFormat,
    gpu_genotype_format: types.GpuGenotypeFormat,
    score_dtype: types.FloatingPointDtype,
    firth_dtype: types.FloatingPointDtype,
    multi_phenotype_sample_mode: MultiPhenotypeSampleMode,
    phenotype_compute_group_id: str | None,
    sample_set_fingerprint: str | None,
    covariate_design_fingerprint: str | None,
    prediction_alignment_fingerprint: str | None,
    output_format: types.OutputFormat,
    finalize_parquet: bool,
    writer_thread_count: int,
    writer_queue_depth: int,
    chunks_per_arrow_file: int,
    arrow_compression: types.ArrowCompression,
    parquet_compression: types.ParquetCompression,
    output_statistic_dtype: types.FloatingPointDtype,
) -> dict[str, typing.Any]:
    """Build immutable run manifest fields from the current execution plan."""
    prediction_loco_files = build_prediction_loco_file_fingerprints(
        prediction_list_path=prediction_list_path,
        phenotype_names=prediction_input_phenotype_names,
        fingerprint_cache=fingerprint_cache,
    )
    current_header_input = {
        "association_mode": association_mode.value,
        "association_backend_kind": association_backend_kind.value,
        "bgen_path": str(bgen_path),
        "sample_path": None if sample_path is None else str(sample_path),
        "phenotype_path": str(phenotype_path),
        "phenotype_name": phenotype_name,
        "covariate_path": None if covariate_path is None else str(covariate_path),
        "covariate_names": list(covariate_names),
        "prediction_list_path": str(prediction_list_path),
        "prediction_loco_files": normalize_execution_plan_value(prediction_loco_files),
        "sample_count": sample_count,
        "variant_count": variant_count,
        "chunk_size": chunk_size,
        "variant_limit": variant_limit,
        "binary_correction_plan_method": binary_correction_plan.method.value,
        "binary_correction_plan_p_threshold": binary_correction_plan.p_threshold,
        "binary_correction_plan_firth_se": binary_correction_plan.firth_se,
        "trusted_no_missing_diploid": trusted_no_missing_diploid,
        "sample_key_mode": sample_key_mode.value,
        "binary_kernel_config": None
        if binary_kernel_config is None
        else normalize_execution_plan_value(binary_kernel_config),
        "bgen_decode_tile_variant_count": bgen_decode_tile_variant_count,
        "trusted_bgen_validation_mode": trusted_bgen_validation_mode.value,
        "jax_device": jax_device.value,
        "jax_enable_x64": jax_enable_x64,
        "jax_matmul_precision": None if jax_matmul_precision is None else jax_matmul_precision.value,
        "requested_gpu_genotype_format": requested_gpu_genotype_format.value,
        "gpu_genotype_format": gpu_genotype_format.value,
        "score_dtype": score_dtype.value,
        "firth_dtype": firth_dtype.value,
        "multi_phenotype_sample_mode": multi_phenotype_sample_mode.value,
        "phenotype_compute_group_id": phenotype_compute_group_id,
        "sample_set_fingerprint": sample_set_fingerprint,
        "covariate_design_fingerprint": covariate_design_fingerprint,
        "prediction_alignment_fingerprint": prediction_alignment_fingerprint,
        "output_format": output_format.value,
        "finalize_parquet": finalize_parquet,
        "writer_thread_count": writer_thread_count,
        "writer_queue_depth": writer_queue_depth,
        "chunks_per_arrow_file": chunks_per_arrow_file,
        "arrow_compression": arrow_compression.value,
        "parquet_compression": parquet_compression.value,
        "output_statistic_dtype": output_statistic_dtype.value,
    }
    resolved_fingerprint_cache = fingerprint_cache if fingerprint_cache is not None else ManifestFileFingerprintCache()
    prepared_header = resolved_fingerprint_cache.native_cache.build_current_run_manifest_header_payload_from_input(
        current_header_input
    )
    return native_mapping_payload(prepared_header)


def build_native_prepared_run_plan_json(current_header: RunManifestHeaderInput) -> str:
    """Build the native prepared-run contract from the transitional header."""
    return native_output_lifecycle_policy().build_prepared_run_plan_json_from_current_header(current_header)


def validate_manifest_compatibility(
    manifest: dict[str, typing.Any],
    current_header: RunManifestHeaderInput,
) -> None:
    """Validate immutable manifest fields against the current run header."""
    native_output_lifecycle_policy().validate_run_manifest_compatibility_from_values(manifest, current_header)


def read_manifest_committed_chunk_identifiers(manifest: dict[str, typing.Any]) -> frozenset[int]:
    """Read committed chunk identifiers from a run manifest."""
    chunk_identifiers = native_output_lifecycle_policy().read_manifest_committed_chunk_identifiers_from_value(manifest)
    return frozenset(int(chunk_identifier) for chunk_identifier in chunk_identifiers)


def validate_strict_manifest_chunks(
    output_run_paths: OutputRunPaths,
    manifest: dict[str, typing.Any],
) -> frozenset[int]:
    """Validate committed manifest chunks against output files."""
    chunk_identifiers = native_output_lifecycle_policy().validate_strict_manifest_chunks_from_value(
        str(output_run_paths.chunks_directory),
        manifest,
    )
    return frozenset(int(chunk_identifier) for chunk_identifier in chunk_identifiers)


def repair_strict_manifest_chunk_commits(
    output_run_paths: OutputRunPaths,
    manifest: dict[str, typing.Any],
) -> list[typing.Any]:
    """Recover committed chunk manifest records from output metadata."""
    repaired_commits = native_output_lifecycle_policy().repair_strict_manifest_chunk_commits_from_value(
        str(output_run_paths.chunks_directory),
        manifest,
    )
    if not isinstance(repaired_commits, tuple):
        message = "Strict resume repaired committed chunks must be a list."
        raise ValueError(message)
    return list(repaired_commits)


def native_pipeline_output_preparation_policy() -> _core.NativePipelineOutputPreparationPolicy:
    """Build the native pipeline output-preparation policy handle."""
    return _core.NativePipelineOutputPreparationPolicy()


def build_native_pipeline_output_preparation_batch(
    *,
    output_run_paths_by_trait: tuple[OutputRunPaths, ...],
    existing_manifests_by_trait: tuple[dict[str, typing.Any] | None, ...],
    current_headers_by_trait: tuple[RunManifestHeaderInput, ...],
    resume: bool,
    resume_mode: types.ResumeMode,
) -> _core.NativePipelineOutputPreparationBatch:
    """Build a native output-preparation batch from output adapter inputs."""
    return native_pipeline_output_preparation_policy().build_pipeline_output_preparation_batch_from_values(
        tuple(str(output_run_paths.run_directory) for output_run_paths in output_run_paths_by_trait),
        tuple(str(output_run_paths.chunks_directory) for output_run_paths in output_run_paths_by_trait),
        tuple(existing_manifest for existing_manifest in existing_manifests_by_trait),
        tuple(current_header for current_header in current_headers_by_trait),
        resume,
        resume_mode.value,
    )


def initialize_output_run(
    *,
    output_run_paths: OutputRunPaths,
    existing_manifest: dict[str, typing.Any] | None,
    current_header: RunManifestHeaderInput,
    resume: bool,
    resume_mode: types.ResumeMode,
    runtime_compatibility_token: _core.NativeRuntimeCompatibilityToken,
) -> InitializedOutputRun:
    """Validate/write the manifest header and return accepted committed chunks."""
    native_initialized_output_run = native_output_lifecycle_policy().initialize_output_run_from_values(
        str(output_run_paths.run_directory),
        str(output_run_paths.chunks_directory),
        existing_manifest,
        current_header,
        resume,
        resume_mode.value,
        runtime_compatibility_token,
    )
    committed_chunk_identifiers = frozenset(
        int(chunk_identifier) for chunk_identifier in native_initialized_output_run.committed_chunk_identifiers
    )
    return InitializedOutputRun(committed_chunk_identifiers=committed_chunk_identifiers)


def prepare_output_run(
    *,
    output_root: Path,
    association_mode: types.AssociationMode,
    output_format: types.OutputFormat,
    resume: bool,
    resume_mode: types.ResumeMode,
    runtime_compatibility_token: _core.NativeRuntimeCompatibilityToken,
) -> PreparedOutputRun:
    """Prepare a chunked output run directory and load existing manifest state."""
    del resume_mode
    native_prepared_output_run = native_output_lifecycle_policy().prepare_output_run(
        str(output_root),
        association_mode.value,
        output_format.value,
        resume,
        runtime_compatibility_token,
    )
    output_run_paths = OutputRunPaths(
        run_directory=Path(native_prepared_output_run.run_directory),
        chunks_directory=Path(native_prepared_output_run.chunks_directory),
    )
    existing_manifest_payload = native_prepared_output_run.existing_manifest_payload()
    manifest = None
    if existing_manifest_payload is not None:
        manifest = require_native_mapping_payload(
            existing_manifest_payload,
            f"Run manifest '{get_run_manifest_path(output_run_paths)}' must contain a JSON object.",
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
    output_statistic_dtype: types.FloatingPointDtype,
    collect_stage_timings: bool,
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
        output_statistic_dtype=output_statistic_dtype.value,
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
    output_format: types.OutputFormat,
) -> Path:
    """Compact committed chunk files into one compressed Parquet file in Rust."""
    final_parquet_path = native_output_lifecycle_policy().finalize_output_run_chunks(
        str(output_run_paths.run_directory),
        str(output_run_paths.chunks_directory),
        str(association_mode),
        output_format.value,
    )
    return Path(final_parquet_path)


@dataclass(frozen=True)
class PreparedPhenotypeRunPlan:
    """Prepared output state for one phenotype run.

    Attributes:
        phenotype_name: Phenotype column name.
        output_run_paths: Chunked output paths for the phenotype.
        existing_manifest: Existing manifest loaded for resume, if present.
        effective_config_path: Path where the effective TOML config is written.

    """

    phenotype_name: str
    output_run_paths: OutputRunPaths
    existing_manifest: dict[str, typing.Any] | None
    effective_config_path: Path


def prepare_execution_plan_outputs(
    *,
    plan: execution_plan.RegenieExecutionPlan,
    runtime_compatibility_token: _core.NativeRuntimeCompatibilityToken,
) -> tuple[PreparedPhenotypeRunPlan, ...]:
    """Prepare output paths and resume state for a requested execution plan."""
    return tuple(
        prepare_phenotype_run_plan(
            phenotype_run_plan=phenotype_run_plan,
            association_mode=plan.association_mode,
            output_plan=plan.output_plan,
            runtime_compatibility_token=runtime_compatibility_token,
        )
        for phenotype_run_plan in plan.phenotype_run_plans
    )


def prepare_phenotype_run_plan(
    *,
    phenotype_run_plan: execution_plan.PhenotypeRunPlan,
    association_mode: types.AssociationMode,
    output_plan: execution_plan.OutputPlan,
    runtime_compatibility_token: _core.NativeRuntimeCompatibilityToken,
) -> PreparedPhenotypeRunPlan:
    """Prepare output paths and resume manifest state for one phenotype."""
    prepared_output_run = prepare_output_run(
        output_root=output_plan.output_run_root / phenotype_run_plan.output_directory_name,
        association_mode=association_mode,
        output_format=output_plan.writer_settings.output_format,
        resume=output_plan.resume,
        resume_mode=output_plan.resume_mode,
        runtime_compatibility_token=runtime_compatibility_token,
    )
    return PreparedPhenotypeRunPlan(
        phenotype_name=phenotype_run_plan.phenotype_name,
        output_run_paths=prepared_output_run.output_run_paths,
        existing_manifest=prepared_output_run.existing_manifest,
        effective_config_path=prepared_output_run.output_run_paths.run_directory / "effective_config.toml",
    )


def output_writer_settings_from_plan(writer_plan: execution_plan.OutputWriterPlan) -> OutputWriterSettings:
    """Adapt requested output writer settings to the output adapter dataclass."""
    return OutputWriterSettings(
        finalize_parquet=writer_plan.finalize_parquet,
        writer_thread_count=writer_plan.writer_thread_count,
        writer_queue_depth=writer_plan.writer_queue_depth,
        chunks_per_arrow_file=writer_plan.chunks_per_arrow_file,
        arrow_compression=writer_plan.arrow_compression,
        parquet_compression=writer_plan.parquet_compression,
        output_format=writer_plan.output_format,
        output_statistic_dtype=writer_plan.output_statistic_dtype,
    )
