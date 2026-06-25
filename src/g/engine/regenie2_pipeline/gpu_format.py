"""GPU genotype format resolution for REGENIE step 2 pipelines."""

from __future__ import annotations

import collections.abc
import logging
import time
import typing
from dataclasses import dataclass

from g import _core, types
from g.engine import timing
from g.engine.native_dispatch import engine as native_dispatch_engine

if typing.TYPE_CHECKING:
    from g.engine import telemetry
    from g.io import source

logger = logging.getLogger(__name__)

MANIFEST_GPU_GENOTYPE_FORMAT_FIELD = "gpu_genotype_format"
MANIFEST_ASSOCIATION_BACKEND_FIELD = "association_backend"
MANIFEST_ASSOCIATION_BACKEND_GENOTYPE_FORMAT_FIELD = "genotype_format"


@dataclass(frozen=True)
class GpuGenotypeFormatResolution:
    """Resolved GPU genotype format and optional prevalidated BGEN engine.

    Attributes:
        requested_gpu_genotype_format: User-requested GPU genotype format.
        resolved_gpu_genotype_format: Concrete GPU genotype format for native delivery.
        resolution_reason: Stable reason for the concrete selection.
        prepared_engine: Trusted prevalidated BGEN engine when auto selected packed8.

    """

    requested_gpu_genotype_format: types.GpuGenotypeFormat
    resolved_gpu_genotype_format: types.GpuGenotypeFormat
    resolution_reason: str
    prepared_engine: _core.Regenie2RunEngine | None


def log_auto_resolution(
    *,
    telemetry_session: telemetry.TelemetrySession | None,
    requested_gpu_genotype_format: types.GpuGenotypeFormat,
    resolved_gpu_genotype_format: types.GpuGenotypeFormat,
    resolution_reason: str,
    fallback_error: str | None,
) -> None:
    """Emit logging and telemetry for an auto GPU genotype format decision."""
    logger.info(
        "Resolved gpu_genotype_format=%s to %s: %s.",
        requested_gpu_genotype_format.value,
        resolved_gpu_genotype_format.value,
        resolution_reason,
    )
    if telemetry_session is None:
        return
    event_fields: dict[str, typing.Any] = {
        "requested_gpu_genotype_format": requested_gpu_genotype_format.value,
        "resolved_gpu_genotype_format": resolved_gpu_genotype_format.value,
        "resolution_reason": resolution_reason,
    }
    if fallback_error is not None:
        event_fields["fallback_error"] = fallback_error
    telemetry_session.log_event("gpu_genotype_format_resolved", level="info", **event_fields)


def resolve_auto_to_dosage(
    *,
    requested_gpu_genotype_format: types.GpuGenotypeFormat,
    telemetry_session: telemetry.TelemetrySession | None,
    resolution_reason: str,
) -> types.GpuGenotypeFormat:
    """Resolve non-profiled auto requests to dosage."""
    if requested_gpu_genotype_format != types.GpuGenotypeFormat.AUTO:
        return requested_gpu_genotype_format
    log_auto_resolution(
        telemetry_session=telemetry_session,
        requested_gpu_genotype_format=requested_gpu_genotype_format,
        resolved_gpu_genotype_format=types.GpuGenotypeFormat.DOSAGE,
        resolution_reason=resolution_reason,
        fallback_error=None,
    )
    return types.GpuGenotypeFormat.DOSAGE


def read_manifest_gpu_genotype_format(
    existing_manifest: collections.abc.Mapping[str, typing.Any],
) -> types.GpuGenotypeFormat | None:
    """Read a concrete GPU genotype format from an existing manifest."""
    raw_gpu_genotype_format = existing_manifest.get(MANIFEST_GPU_GENOTYPE_FORMAT_FIELD)
    if not isinstance(raw_gpu_genotype_format, str):
        association_backend = existing_manifest.get(MANIFEST_ASSOCIATION_BACKEND_FIELD)
        if isinstance(association_backend, collections.abc.Mapping):
            raw_gpu_genotype_format = association_backend.get(MANIFEST_ASSOCIATION_BACKEND_GENOTYPE_FORMAT_FIELD)
    if raw_gpu_genotype_format not in (
        types.GpuGenotypeFormat.DOSAGE.value,
        types.GpuGenotypeFormat.PACKED8.value,
    ):
        return None
    return types.GpuGenotypeFormat(raw_gpu_genotype_format)


def resolve_manifest_gpu_genotype_format(
    *,
    existing_manifest: dict[str, typing.Any] | None,
    resume: bool,
) -> types.GpuGenotypeFormat | None:
    """Return the manifest's concrete GPU genotype format when resume can reuse it."""
    if not resume or existing_manifest is None:
        return None
    return read_manifest_gpu_genotype_format(existing_manifest)


def build_explicit_resolution(
    requested_gpu_genotype_format: types.GpuGenotypeFormat,
) -> GpuGenotypeFormatResolution:
    """Build a pass-through resolution for explicit concrete requests."""
    return GpuGenotypeFormatResolution(
        requested_gpu_genotype_format=requested_gpu_genotype_format,
        resolved_gpu_genotype_format=requested_gpu_genotype_format,
        resolution_reason="explicit",
        prepared_engine=None,
    )


def validate_auto_packed8_bgen_engine(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    chunk_size: int,
    variant_limit: int | None,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode,
    stage_timing_recorder: timing.StageTimingRecorder | None,
) -> _core.Regenie2RunEngine:
    """Open and validate the trusted BGEN engine required for packed8 delivery."""
    engine_start_time = time.perf_counter()
    engine = native_dispatch_engine.open_bgen_run_engine(
        genotype_source_config=genotype_source_config,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        trusted_no_missing_diploid=True,
    )
    native_dispatch_engine.validate_trusted_bgen_run_engine(
        engine=engine,
        genotype_source_config=genotype_source_config,
        trusted_bgen_validation_mode=trusted_bgen_validation_mode,
        trusted_bgen_validator=None,
    )
    timing.record_stage_duration(stage_timing_recorder, "bgen_engine_open_index_setup", engine_start_time)
    return engine


def resolve_single_trait_binary_gpu_genotype_format(
    *,
    requested_gpu_genotype_format: types.GpuGenotypeFormat,
    existing_manifest: dict[str, typing.Any] | None,
    resume: bool,
    jax_device: types.Device,
    genotype_source_config: source.GenotypeSourceConfig,
    chunk_size: int,
    variant_limit: int | None,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    telemetry_session: telemetry.TelemetrySession | None,
) -> GpuGenotypeFormatResolution:
    """Resolve the single-trait binary GPU genotype format before output initialization."""
    if requested_gpu_genotype_format != types.GpuGenotypeFormat.AUTO:
        return build_explicit_resolution(requested_gpu_genotype_format)

    manifest_gpu_genotype_format = resolve_manifest_gpu_genotype_format(
        existing_manifest=existing_manifest,
        resume=resume,
    )
    if manifest_gpu_genotype_format is not None:
        log_auto_resolution(
            telemetry_session=telemetry_session,
            requested_gpu_genotype_format=requested_gpu_genotype_format,
            resolved_gpu_genotype_format=manifest_gpu_genotype_format,
            resolution_reason="resume_manifest",
            fallback_error=None,
        )
        return GpuGenotypeFormatResolution(
            requested_gpu_genotype_format=requested_gpu_genotype_format,
            resolved_gpu_genotype_format=manifest_gpu_genotype_format,
            resolution_reason="resume_manifest",
            prepared_engine=None,
        )

    if jax_device != types.Device.GPU:
        log_auto_resolution(
            telemetry_session=telemetry_session,
            requested_gpu_genotype_format=requested_gpu_genotype_format,
            resolved_gpu_genotype_format=types.GpuGenotypeFormat.DOSAGE,
            resolution_reason="non_gpu_device",
            fallback_error=None,
        )
        return GpuGenotypeFormatResolution(
            requested_gpu_genotype_format=requested_gpu_genotype_format,
            resolved_gpu_genotype_format=types.GpuGenotypeFormat.DOSAGE,
            resolution_reason="non_gpu_device",
            prepared_engine=None,
        )

    try:
        prepared_engine = validate_auto_packed8_bgen_engine(
            genotype_source_config=genotype_source_config,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
            trusted_bgen_validation_mode=trusted_bgen_validation_mode,
            stage_timing_recorder=stage_timing_recorder,
        )
    except ValueError as error:
        fallback_error = str(error)
        log_auto_resolution(
            telemetry_session=telemetry_session,
            requested_gpu_genotype_format=requested_gpu_genotype_format,
            resolved_gpu_genotype_format=types.GpuGenotypeFormat.DOSAGE,
            resolution_reason="trusted_validation_failed",
            fallback_error=fallback_error,
        )
        return GpuGenotypeFormatResolution(
            requested_gpu_genotype_format=requested_gpu_genotype_format,
            resolved_gpu_genotype_format=types.GpuGenotypeFormat.DOSAGE,
            resolution_reason="trusted_validation_failed",
            prepared_engine=None,
        )

    log_auto_resolution(
        telemetry_session=telemetry_session,
        requested_gpu_genotype_format=requested_gpu_genotype_format,
        resolved_gpu_genotype_format=types.GpuGenotypeFormat.PACKED8,
        resolution_reason="trusted_validation_passed",
        fallback_error=None,
    )
    return GpuGenotypeFormatResolution(
        requested_gpu_genotype_format=requested_gpu_genotype_format,
        resolved_gpu_genotype_format=types.GpuGenotypeFormat.PACKED8,
        resolution_reason="trusted_validation_passed",
        prepared_engine=prepared_engine,
    )
