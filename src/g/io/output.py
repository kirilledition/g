"""Output persistence orchestration backed by the native Rust writer."""

from __future__ import annotations

import dataclasses
import enum
import json
import re
import typing
from dataclasses import dataclass
from pathlib import Path

from g import _core, types
from g.jax_runtime import models as jax_runtime_models

OUTPUT_COMPRESSION_CODEC = "zstd"
CHUNK_FILENAME_PATTERN = re.compile(r"^chunk_(\d+)(?:_(\d+))?\.arrow$")
PART_FILENAME_PATTERN = re.compile(r"^part_(\d+)(?:_(\d+))?\.parquet$")
REGENIE_PART_FILENAME_PATTERN = re.compile(r"^part_(\d+)(?:_(\d+))?\.regenie$")
RUN_MANIFEST_FILENAME = "run_manifest.json"
RUN_MANIFEST_SCHEMA_VERSION = 9
OUTPUT_SCHEMA_VERSION = 2
JAX_MATMUL_PRECISION_WHEN_UNSET = "float32"
RESUME_POLICY = "manifest_committed_chunks"
DEFAULT_RESULT_STATISTIC_OUTPUT_DTYPE = types.FloatingPointDtype.FLOAT32


class MultiPhenotypeSampleMode(enum.StrEnum):
    """Sample inclusion policy for one output run."""

    SINGLE_PHENOTYPE = "single-phenotype"
    PER_PHENOTYPE = "per-phenotype"
    COMPLETE_CASE = "complete-case"


def emit_output_diagnostic_event_payload(payload: typing.Mapping[str, object]) -> None:
    """Emit one native output diagnostic payload through native tracing."""
    _core.emit_diagnostic_event_fields(
        str(payload["level"]),
        str(payload["event_name"]),
        str(payload["message"]),
        typing.cast("typing.Mapping[str, object]", payload["fields"]),
    )


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


@dataclass(frozen=True)
class ManifestFileFingerprintCacheKey:
    """Cache key for one observed file fingerprint request.

    Attributes:
        path: Canonical input file path.
        include_content_hash: Whether the request includes a content hash.
        size: File size observed before hashing.
        mtime_ns: File modification timestamp observed before hashing.

    """

    path: str
    include_content_hash: bool
    size: int
    mtime_ns: int


class ManifestFileFingerprintCache:
    """Run-scoped cache for immutable input file fingerprints."""

    def __init__(self) -> None:
        """Initialize an empty fingerprint cache."""
        self._fingerprints_by_key: dict[ManifestFileFingerprintCacheKey, ManifestFileFingerprint] = {}

    def build_file_fingerprint(
        self,
        path: Path | None,
        *,
        include_content_hash: bool,
    ) -> ManifestFileFingerprint | None:
        """Build or reuse a fingerprint for the observed input file state."""
        if path is None:
            return None
        canonical_path = path.resolve(strict=True)
        metadata = canonical_path.stat()
        cache_key = ManifestFileFingerprintCacheKey(
            path=str(canonical_path),
            include_content_hash=include_content_hash,
            size=metadata.st_size,
            mtime_ns=metadata.st_mtime_ns,
        )
        cached_fingerprint = self._fingerprints_by_key.get(cache_key)
        if cached_fingerprint is not None:
            return cached_fingerprint
        file_fingerprint = require_manifest_file_fingerprint(
            build_file_fingerprint(canonical_path, include_content_hash=include_content_hash),
            "input file",
        )
        self._fingerprints_by_key[cache_key] = file_fingerprint
        return file_fingerprint


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


@dataclass(frozen=True)
class PredictionInputsManifest:
    """Manifest identity for REGENIE Step 1 prediction inputs.

    Attributes:
        prediction_list: Prediction-list file fingerprint.
        loco_files: LOCO prediction files selected for this output run.

    """

    prediction_list: ManifestFileFingerprint
    loco_files: tuple[PredictionLocoFileFingerprint, ...]


@dataclass(frozen=True)
class BinaryCorrectionPlanManifest:
    """Manifest representation of binary fallback policy.

    Attributes:
        method: Binary fallback method name.
        p_threshold: Score-test p-value threshold for correction.
        firth_se: Whether Firth standard errors are requested.

    """

    method: str
    p_threshold: float
    firth_se: bool


@dataclass(frozen=True)
class AssociationBackendManifest:
    """Manifest representation of the selected association backend.

    Attributes:
        kind: Concrete backend implementation.
        association_mode: Statistical association mode.
        device: JAX device requested for the backend.
        genotype_format: Native genotype delivery format.

    """

    kind: str
    association_mode: str
    device: str
    genotype_format: str


@dataclass(frozen=True)
class JaxPolicyManifest:
    """Manifest representation of JAX numerical and backend policy.

    Attributes:
        device: Requested JAX device.
        enable_x64: Whether x64 is enabled for JAX.
        matmul_precision: Default JAX matmul precision.

    """

    device: str
    enable_x64: bool
    matmul_precision: str


@dataclass(frozen=True)
class OutputWriterManifest:
    """Manifest representation of output writer settings.

    Attributes:
        output_format: Persisted chunk output format.
        finalize_parquet: Whether chunks are finalized into Parquet.
        writer_thread_count: Number of writer threads.
        writer_queue_depth: Maximum queued chunk writes.
        chunks_per_arrow_file: Number of chunks per Arrow file.
        arrow_compression: Arrow IPC compression codec.
        parquet_compression: Parquet compression codec.
        result_statistic_dtype: Persisted public statistic dtype.

    """

    output_format: str
    finalize_parquet: bool
    writer_thread_count: int
    writer_queue_depth: int
    chunks_per_arrow_file: int
    arrow_compression: str
    parquet_compression: str
    result_statistic_dtype: str


@dataclass(frozen=True)
class CurrentRunManifestHeader:
    """Typed current-run manifest header before JSON serialization.

    Attributes:
        association_mode: Statistical association mode.
        association_backend: Selected association backend metadata.
        bgen: BGEN file fingerprint.
        sample: Optional sample file fingerprint.
        phenotype_file: Phenotype file fingerprint.
        phenotype_name: Phenotype column name.
        covariate_file: Optional covariate file fingerprint.
        covariate_names: Covariate column names.
        prediction_list: REGENIE prediction-list fingerprint.
        prediction_inputs: REGENIE prediction-list and selected LOCO file fingerprints.
        sample_count: Number of aligned samples.
        variant_count: Number of variants in the source.
        chunk_size: Native variant chunk size.
        variant_limit: Optional variant processing cap.
        binary_correction_plan: Binary correction policy.
        binary_kernel_config: Binary kernel configuration when binary.
        trusted_no_missing_diploid: Trusted BGEN fast-path policy.
        trusted_bgen_validation_mode: Trusted BGEN validation policy.
        sample_key_mode: Sample identity matching mode.
        bgen_decode_tile_variant_count: Native BGEN decode tile size.
        jax_policy: JAX backend and precision policy.
        requested_gpu_genotype_format: User-requested genotype format before resolution.
        gpu_genotype_format: Native genotype format delivered to GPU kernels.
        score_dtype: Score-test compute dtype.
        firth_dtype: Firth compute dtype.
        multi_phenotype_sample_mode: Multi-phenotype sample inclusion mode.
        phenotype_compute_group_id: Stable compute-group identifier.
        sample_set_fingerprint: Aligned sample-set fingerprint.
        covariate_design_fingerprint: Aligned covariate-design fingerprint.
        prediction_alignment_fingerprint: Prediction alignment fingerprint.
        output_writer: Output writer policy.

    """

    association_mode: types.AssociationMode
    association_backend: AssociationBackendManifest
    bgen: ManifestFileFingerprint
    sample: ManifestFileFingerprint | None
    phenotype_file: ManifestFileFingerprint
    phenotype_name: str
    covariate_file: ManifestFileFingerprint | None
    covariate_names: tuple[str, ...]
    prediction_list: ManifestFileFingerprint
    prediction_inputs: PredictionInputsManifest
    sample_count: int
    variant_count: int
    chunk_size: int
    variant_limit: int | None
    binary_correction_plan: BinaryCorrectionPlanManifest
    binary_kernel_config: typing.Any | None
    trusted_no_missing_diploid: bool
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode
    sample_key_mode: types.SampleKeyMode
    bgen_decode_tile_variant_count: int
    jax_policy: JaxPolicyManifest
    requested_gpu_genotype_format: types.GpuGenotypeFormat
    gpu_genotype_format: types.GpuGenotypeFormat
    score_dtype: types.FloatingPointDtype
    firth_dtype: types.FloatingPointDtype
    multi_phenotype_sample_mode: MultiPhenotypeSampleMode
    phenotype_compute_group_id: str | None
    sample_set_fingerprint: str | None
    covariate_design_fingerprint: str | None
    prediction_alignment_fingerprint: str | None
    output_writer: OutputWriterManifest


RunManifestHeaderInput = CurrentRunManifestHeader | dict[str, typing.Any]


def get_run_manifest_path(output_run_paths: OutputRunPaths) -> Path:
    """Return the run manifest path for an output run."""
    return output_run_paths.run_directory / RUN_MANIFEST_FILENAME


def parse_run_manifest_json(manifest_json: str, manifest_path: Path | None) -> dict[str, typing.Any]:
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
    output_format: types.OutputFormat,
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


def build_file_content_sha256(path: Path) -> str:
    """Build a streaming SHA-256 content hash for a local input file."""
    return _core.build_file_content_sha256_value(str(path))


def build_file_fingerprint(path: Path | None, *, include_content_hash: bool) -> ManifestFileFingerprint | None:
    """Build a lightweight immutable fingerprint for an input file."""
    if path is None:
        return None
    return manifest_file_fingerprint_from_native_payload(
        _core.build_manifest_file_fingerprint_payload(str(path), include_content_hash)
    )


def build_file_fingerprint_with_cache(
    path: Path | None,
    *,
    include_content_hash: bool,
    fingerprint_cache: ManifestFileFingerprintCache | None,
) -> ManifestFileFingerprint | None:
    """Build a file fingerprint through a run-scoped cache when available."""
    if fingerprint_cache is None:
        return build_file_fingerprint(path, include_content_hash=include_content_hash)
    return fingerprint_cache.build_file_fingerprint(path, include_content_hash=include_content_hash)


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


def file_fingerprint_to_mapping(
    file_fingerprint: ManifestFileFingerprint | None,
) -> dict[str, typing.Any] | None:
    """Serialize a file fingerprint to manifest JSON fields."""
    if file_fingerprint is None:
        return None
    return native_mapping_payload(
        _core.build_manifest_file_fingerprint_mapping_payload(
            file_fingerprint.path,
            file_fingerprint.size,
            file_fingerprint.mtime_ns,
            file_fingerprint.content_hash_algorithm,
            file_fingerprint.content_sha256,
        )
    )


def require_manifest_file_fingerprint(
    file_fingerprint: ManifestFileFingerprint | None,
    role_name: str,
) -> ManifestFileFingerprint:
    """Return a required file fingerprint or fail at an internal boundary."""
    if file_fingerprint is None:
        message = f"{role_name} fingerprint is required."
        raise ValueError(message)
    return file_fingerprint


def prediction_loco_file_fingerprint_to_mapping(
    loco_file_fingerprint: PredictionLocoFileFingerprint,
) -> dict[str, typing.Any]:
    """Serialize a LOCO file fingerprint to manifest JSON fields."""
    return {
        "phenotype": loco_file_fingerprint.phenotype,
        "path": loco_file_fingerprint.path,
        "size": loco_file_fingerprint.size,
        "mtime_ns": loco_file_fingerprint.mtime_ns,
        "content_hash_algorithm": loco_file_fingerprint.content_hash_algorithm,
        "content_sha256": loco_file_fingerprint.content_sha256,
    }


def build_prediction_loco_file_fingerprints(
    *,
    prediction_list_path: Path,
    phenotype_names: tuple[str, ...],
    fingerprint_cache: ManifestFileFingerprintCache | None,
) -> tuple[PredictionLocoFileFingerprint, ...]:
    """Build content fingerprints for LOCO files selected from a prediction list."""
    resolved_loco_paths = _core.resolve_prediction_loco_paths(str(prediction_list_path), list(phenotype_names))
    loco_file_fingerprints: list[PredictionLocoFileFingerprint] = []
    for resolved_loco_path in resolved_loco_paths:
        resolved_loco_path_payload = native_mapping_payload(resolved_loco_path)
        phenotype = typing.cast("str", resolved_loco_path_payload["phenotype"])
        loco_path = Path(typing.cast("str", resolved_loco_path_payload["path"]))
        loco_file_fingerprint = require_manifest_file_fingerprint(
            build_file_fingerprint_with_cache(
                loco_path,
                include_content_hash=True,
                fingerprint_cache=fingerprint_cache,
            ),
            "LOCO prediction file",
        )
        content_sha256 = loco_file_fingerprint.content_sha256
        if content_sha256 is None:
            message = "LOCO prediction file fingerprint must include a content hash."
            raise ValueError(message)
        loco_file_fingerprints.append(
            PredictionLocoFileFingerprint(
                phenotype=phenotype,
                path=loco_file_fingerprint.path,
                size=loco_file_fingerprint.size,
                mtime_ns=loco_file_fingerprint.mtime_ns,
                content_hash_algorithm=loco_file_fingerprint.content_hash_algorithm,
                content_sha256=content_sha256,
            )
        )
    return tuple(loco_file_fingerprints)


def build_binary_correction_plan_manifest(
    binary_correction_plan: types.BinaryCorrectionPlan,
) -> BinaryCorrectionPlanManifest:
    """Build the manifest representation of a binary correction plan."""
    return BinaryCorrectionPlanManifest(
        method=str(binary_correction_plan.method),
        p_threshold=binary_correction_plan.p_threshold,
        firth_se=binary_correction_plan.firth_se,
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
    execution_plan_json = json.dumps(
        normalized_execution_plan,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _core.build_manifest_json_sha256(execution_plan_json)


def build_jax_policy_manifest(
    *,
    device: types.Device,
    matmul_precision: types.JaxMatmulPrecision | None,
) -> JaxPolicyManifest:
    """Build manifest fields for JAX precision and backend policy."""
    return JaxPolicyManifest(
        device=device.value,
        enable_x64=jax_runtime_models.JAX_ENABLE_X64,
        matmul_precision=JAX_MATMUL_PRECISION_WHEN_UNSET if matmul_precision is None else matmul_precision.value,
    )


def build_association_backend_manifest(
    *,
    association_backend_kind: types.AssociationBackendKind,
    association_mode: types.AssociationMode,
    jax_device: types.Device,
    gpu_genotype_format: types.GpuGenotypeFormat,
) -> AssociationBackendManifest:
    """Build manifest fields for the selected association backend."""
    return AssociationBackendManifest(
        kind=association_backend_kind.value,
        association_mode=association_mode.value,
        device=jax_device.value,
        genotype_format=gpu_genotype_format.value,
    )


def build_output_writer_manifest(
    *,
    output_format: types.OutputFormat,
    finalize_parquet: bool,
    writer_thread_count: int,
    writer_queue_depth: int,
    chunks_per_arrow_file: int,
    arrow_compression: types.ArrowCompression,
    parquet_compression: types.ParquetCompression,
    output_statistic_dtype: types.FloatingPointDtype,
) -> OutputWriterManifest:
    """Build manifest fields for output materialization and writer settings."""
    return OutputWriterManifest(
        output_format=output_format.value,
        finalize_parquet=finalize_parquet,
        writer_thread_count=writer_thread_count,
        writer_queue_depth=writer_queue_depth,
        chunks_per_arrow_file=chunks_per_arrow_file,
        arrow_compression=arrow_compression.value,
        parquet_compression=parquet_compression.value,
        result_statistic_dtype=output_statistic_dtype.value,
    )


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
) -> CurrentRunManifestHeader:
    """Build immutable run manifest fields from the current execution plan."""
    bgen_fingerprint = require_manifest_file_fingerprint(
        build_file_fingerprint_with_cache(
            bgen_path,
            include_content_hash=False,
            fingerprint_cache=fingerprint_cache,
        ),
        "BGEN",
    )
    sample_fingerprint = build_file_fingerprint_with_cache(
        sample_path,
        include_content_hash=True,
        fingerprint_cache=fingerprint_cache,
    )
    phenotype_file_fingerprint = require_manifest_file_fingerprint(
        build_file_fingerprint_with_cache(
            phenotype_path,
            include_content_hash=True,
            fingerprint_cache=fingerprint_cache,
        ),
        "phenotype file",
    )
    covariate_file_fingerprint = build_file_fingerprint_with_cache(
        covariate_path,
        include_content_hash=True,
        fingerprint_cache=fingerprint_cache,
    )
    prediction_list_fingerprint = require_manifest_file_fingerprint(
        build_file_fingerprint_with_cache(
            prediction_list_path,
            include_content_hash=True,
            fingerprint_cache=fingerprint_cache,
        ),
        "prediction list",
    )
    prediction_loco_files = build_prediction_loco_file_fingerprints(
        prediction_list_path=prediction_list_path,
        phenotype_names=prediction_input_phenotype_names,
        fingerprint_cache=fingerprint_cache,
    )
    prediction_inputs_manifest = PredictionInputsManifest(
        prediction_list=prediction_list_fingerprint,
        loco_files=prediction_loco_files,
    )
    binary_correction_plan_manifest = build_binary_correction_plan_manifest(binary_correction_plan)
    output_writer_manifest = build_output_writer_manifest(
        output_format=output_format,
        finalize_parquet=finalize_parquet,
        writer_thread_count=writer_thread_count,
        writer_queue_depth=writer_queue_depth,
        chunks_per_arrow_file=chunks_per_arrow_file,
        arrow_compression=arrow_compression,
        parquet_compression=parquet_compression,
        output_statistic_dtype=output_statistic_dtype,
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
    return CurrentRunManifestHeader(
        association_mode=association_mode,
        association_backend=association_backend_manifest,
        bgen=bgen_fingerprint,
        sample=sample_fingerprint,
        phenotype_file=phenotype_file_fingerprint,
        phenotype_name=phenotype_name,
        covariate_file=covariate_file_fingerprint,
        covariate_names=covariate_names,
        prediction_list=prediction_list_fingerprint,
        prediction_inputs=prediction_inputs_manifest,
        sample_count=sample_count,
        variant_count=variant_count,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        binary_correction_plan=binary_correction_plan_manifest,
        binary_kernel_config=binary_kernel_config,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
        trusted_bgen_validation_mode=trusted_bgen_validation_mode,
        sample_key_mode=sample_key_mode,
        bgen_decode_tile_variant_count=bgen_decode_tile_variant_count,
        jax_policy=jax_policy_manifest,
        requested_gpu_genotype_format=requested_gpu_genotype_format,
        gpu_genotype_format=gpu_genotype_format,
        score_dtype=score_dtype,
        firth_dtype=firth_dtype,
        multi_phenotype_sample_mode=multi_phenotype_sample_mode,
        phenotype_compute_group_id=phenotype_compute_group_id,
        sample_set_fingerprint=sample_set_fingerprint,
        covariate_design_fingerprint=covariate_design_fingerprint,
        prediction_alignment_fingerprint=prediction_alignment_fingerprint,
        output_writer=output_writer_manifest,
    )


def build_native_prepared_run_manifest_header_mapping(
    current_header: CurrentRunManifestHeader,
) -> dict[str, typing.Any]:
    """Build the manifest header from a native prepared-run plan payload."""
    manifest_json = _core.build_prepared_run_manifest_header_json(build_native_prepared_run_plan_json(current_header))
    manifest_header = json.loads(manifest_json)
    if not isinstance(manifest_header, dict):
        message = "Native prepared run manifest header must contain a JSON object."
        raise ValueError(message)
    return manifest_header


def build_native_prepared_run_plan_json(current_header: CurrentRunManifestHeader) -> str:
    """Build the native prepared-run contract from the transitional header."""
    return _core.build_prepared_run_plan_json(
        json.dumps(build_native_prepared_run_plan_input_mapping(current_header), sort_keys=True)
    )


def build_native_prepared_run_plan_input_mapping(
    current_header: CurrentRunManifestHeader,
) -> dict[str, typing.Any]:
    """Build the transitional input payload consumed by the native prepared-plan builder."""
    matmul_precision: str | None
    if current_header.jax_policy.matmul_precision == JAX_MATMUL_PRECISION_WHEN_UNSET:
        matmul_precision = None
    else:
        matmul_precision = current_header.jax_policy.matmul_precision
    binary_kernel_config = (
        None
        if current_header.binary_kernel_config is None
        else normalize_execution_plan_value(current_header.binary_kernel_config)
    )
    prediction_list = typing.cast("dict[str, typing.Any]", file_fingerprint_to_mapping(current_header.prediction_list))
    phenotype_compute_group = (
        None
        if current_header.phenotype_compute_group_id is None
        else {
            "group_id": current_header.phenotype_compute_group_id,
            "sample_set_fingerprint": current_header.sample_set_fingerprint,
            "covariate_design_fingerprint": current_header.covariate_design_fingerprint,
            "prediction_alignment_fingerprint": current_header.prediction_alignment_fingerprint,
        }
    )
    return {
        "association_mode": current_header.association_mode.value,
        "input_identity": {
            "bgen": file_fingerprint_to_mapping(current_header.bgen),
            "sample": file_fingerprint_to_mapping(current_header.sample),
            "phenotype_file": file_fingerprint_to_mapping(current_header.phenotype_file),
            "covariate_file": file_fingerprint_to_mapping(current_header.covariate_file),
            "prediction_list": prediction_list,
            "prediction_inputs": {
                "prediction_list": prediction_list,
                "loco_files": [
                    prediction_loco_file_fingerprint_to_mapping(loco_file)
                    for loco_file in current_header.prediction_inputs.loco_files
                ],
            },
        },
        "phenotype_name": current_header.phenotype_name,
        "covariate_names": list(current_header.covariate_names),
        "sample_count": current_header.sample_count,
        "variant_count": current_header.variant_count,
        "chunk_size": current_header.chunk_size,
        "variant_limit": current_header.variant_limit,
        "correction": {
            "method": current_header.binary_correction_plan.method,
            "p_threshold": current_header.binary_correction_plan.p_threshold,
            "firth_se": current_header.binary_correction_plan.firth_se,
        },
        "binary_kernel_config": binary_kernel_config,
        "compute": {
            "trusted_no_missing_diploid": current_header.trusted_no_missing_diploid,
            "trusted_bgen_validation_mode": current_header.trusted_bgen_validation_mode.value,
            "sample_key_mode": current_header.sample_key_mode.value,
            "bgen_decode_tile_variant_count": current_header.bgen_decode_tile_variant_count,
            "jax_policy": {
                "device": current_header.jax_policy.device,
                "enable_x64": current_header.jax_policy.enable_x64,
                "matmul_precision": matmul_precision,
            },
            "requested_gpu_genotype_format": current_header.requested_gpu_genotype_format.value,
            "resolved_gpu_genotype_format": current_header.gpu_genotype_format.value,
            "score_dtype": current_header.score_dtype.value,
            "firth_dtype": current_header.firth_dtype.value,
            "sample_mode": current_header.multi_phenotype_sample_mode.value,
        },
        "phenotype_compute_group": phenotype_compute_group,
        "output_writer": {
            "output_format": current_header.output_writer.output_format,
            "finalize_parquet": current_header.output_writer.finalize_parquet,
            "writer_thread_count": current_header.output_writer.writer_thread_count,
            "writer_queue_depth": current_header.output_writer.writer_queue_depth,
            "chunks_per_arrow_file": current_header.output_writer.chunks_per_arrow_file,
            "arrow_compression": current_header.output_writer.arrow_compression,
            "parquet_compression": current_header.output_writer.parquet_compression,
            "output_statistic_dtype": current_header.output_writer.result_statistic_dtype,
        },
    }


def current_run_manifest_header_to_mapping(current_header: CurrentRunManifestHeader) -> dict[str, typing.Any]:
    """Serialize a typed current-run manifest header to native JSON fields."""
    return build_native_prepared_run_manifest_header_mapping(current_header)


def run_manifest_header_input_to_mapping(current_header: RunManifestHeaderInput) -> dict[str, typing.Any]:
    """Serialize typed headers while accepting raw test/native boundary mappings."""
    if isinstance(current_header, CurrentRunManifestHeader):
        return current_run_manifest_header_to_mapping(current_header)
    return current_header


def validate_manifest_compatibility(
    manifest: dict[str, typing.Any],
    current_header: RunManifestHeaderInput,
) -> None:
    """Validate immutable manifest fields against the current run header."""
    current_header_mapping = run_manifest_header_input_to_mapping(current_header)
    _core.validate_run_manifest_compatibility(
        json.dumps(manifest, sort_keys=True),
        json.dumps(current_header_mapping, sort_keys=True),
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
    current_header: RunManifestHeaderInput,
    resume: bool,
    resume_mode: types.ResumeMode,
    runtime_compatibility_token: _core.NativeRuntimeCompatibilityToken,
) -> InitializedOutputRun:
    """Validate/write the manifest header and return accepted committed chunks."""
    current_header_mapping = run_manifest_header_input_to_mapping(current_header)
    native_initialized_output_run = _core.initialize_output_run(
        str(output_run_paths.run_directory),
        str(output_run_paths.chunks_directory),
        None if existing_manifest is None else json.dumps(existing_manifest, sort_keys=True),
        json.dumps(current_header_mapping, sort_keys=True),
        resume,
        resume_mode.value,
        runtime_compatibility_token,
    )
    committed_chunk_identifiers = frozenset(
        int(chunk_identifier) for chunk_identifier in native_initialized_output_run.committed_chunk_identifiers
    )
    if resume:
        committed_chunk_count = len(committed_chunk_identifiers)
        emit_output_diagnostic_event_payload(
            _core.build_io_output_resume_committed_chunks_diagnostic_payload(
                str(output_run_paths.chunks_directory),
                committed_chunk_count,
                str(output_run_paths.run_directory),
            )
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
    native_prepared_output_run = _core.prepare_output_run(
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
    final_parquet_path = _core.finalize_output_run_chunks(
        run_directory=str(output_run_paths.run_directory),
        chunks_directory=str(output_run_paths.chunks_directory),
        association_mode=str(association_mode),
        output_format=output_format.value,
    )
    return Path(final_parquet_path)
