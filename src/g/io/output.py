"""Output persistence orchestration backed by the native Rust writer."""

from __future__ import annotations

import dataclasses
import enum
import typing
from pathlib import Path

from g import _core, types


class MultiPhenotypeSampleMode(enum.StrEnum):
    """Sample inclusion policy for one output run."""

    SINGLE_PHENOTYPE = "single-phenotype"
    PER_PHENOTYPE = "per-phenotype"
    COMPLETE_CASE = "complete-case"


class ManifestFileFingerprintCache:
    """Native run-scoped cache for immutable input file fingerprints."""

    def __init__(self) -> None:
        """Initialize an empty native fingerprint cache."""
        self.native_cache: _core.NativeManifestFileFingerprintCache = _core.NativeManifestFileFingerprintCache()


RunManifestHeaderInput = dict[str, typing.Any]


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
        "prediction_input_phenotype_names": list(prediction_input_phenotype_names),
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
    return typing.cast("dict[str, typing.Any]", prepared_header)
