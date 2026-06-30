"""GPU genotype format resolution for REGENIE step 2 pipelines."""

from __future__ import annotations

import collections.abc
import time
import typing
from dataclasses import dataclass

from g import _core, types
from g.engine import timing
from g.engine.native_dispatch import engine as native_dispatch_engine

if typing.TYPE_CHECKING:
    from g.engine import telemetry
    from g.io import source

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


@dataclass(frozen=True)
class ManifestGpuGenotypeFormatFields:
    """Manifest GPU genotype-format fields read by the Python adapter.

    Attributes:
        manifest_gpu_genotype_format: Top-level manifest GPU genotype format.
        association_backend_genotype_format: Legacy association-backend genotype format.

    """

    manifest_gpu_genotype_format: str | None
    association_backend_genotype_format: str | None


def emit_gpu_format_diagnostic_event(
    level: str,
    event: str,
    message: str,
    fields: typing.Mapping[str, object],
) -> None:
    """Emit one structured GPU genotype-format diagnostic through native tracing."""
    _core.emit_diagnostic_event_fields(level, event, message, fields)


def log_auto_resolution(
    *,
    telemetry_session: telemetry.TelemetrySession | None,
    requested_gpu_genotype_format: types.GpuGenotypeFormat,
    resolved_gpu_genotype_format: types.GpuGenotypeFormat,
    resolution_reason: str,
    fallback_error: str | None,
) -> None:
    """Emit logging and telemetry for an auto GPU genotype format decision."""
    emit_gpu_format_diagnostic_event(
        "info",
        "pipeline_gpu_genotype_format_resolved",
        (
            f"Resolved gpu_genotype_format={requested_gpu_genotype_format.value} "
            f"to {resolved_gpu_genotype_format.value}: {resolution_reason}."
        ),
        {
            "fallback_error": fallback_error,
            "requested_gpu_genotype_format": requested_gpu_genotype_format.value,
            "resolution_reason": resolution_reason,
            "resolved_gpu_genotype_format": resolved_gpu_genotype_format.value,
        },
    )
    if telemetry_session is None:
        return
    telemetry_session.log_gpu_genotype_format_resolved(
        requested_gpu_genotype_format=requested_gpu_genotype_format,
        resolved_gpu_genotype_format=resolved_gpu_genotype_format,
        resolution_reason=resolution_reason,
        fallback_error=fallback_error,
    )


def resolve_auto_to_dosage(
    *,
    requested_gpu_genotype_format: types.GpuGenotypeFormat,
    telemetry_session: telemetry.TelemetrySession | None,
    resolution_reason: str,
) -> types.GpuGenotypeFormat:
    """Resolve non-profiled auto requests to dosage."""
    native_resolution_plan = _core.plan_gpu_genotype_format_auto_to_dosage(
        requested_gpu_genotype_format=requested_gpu_genotype_format.value,
        resolution_reason=resolution_reason,
    )
    log_native_auto_resolution(
        telemetry_session=telemetry_session,
        native_resolution_plan=native_resolution_plan,
    )
    return concrete_gpu_genotype_format_from_native_plan(native_resolution_plan)


def read_manifest_gpu_genotype_format_fields(
    existing_manifest: collections.abc.Mapping[str, typing.Any],
) -> ManifestGpuGenotypeFormatFields:
    """Read manifest GPU genotype-format fields for native policy planning."""
    raw_gpu_genotype_format = existing_manifest.get(MANIFEST_GPU_GENOTYPE_FORMAT_FIELD)
    manifest_gpu_genotype_format = raw_gpu_genotype_format if isinstance(raw_gpu_genotype_format, str) else None
    association_backend_genotype_format: str | None = None
    if manifest_gpu_genotype_format is None:
        association_backend = existing_manifest.get(MANIFEST_ASSOCIATION_BACKEND_FIELD)
        if isinstance(association_backend, collections.abc.Mapping):
            raw_gpu_genotype_format = association_backend.get(MANIFEST_ASSOCIATION_BACKEND_GENOTYPE_FORMAT_FIELD)
            if isinstance(raw_gpu_genotype_format, str):
                association_backend_genotype_format = raw_gpu_genotype_format
    return ManifestGpuGenotypeFormatFields(
        manifest_gpu_genotype_format=manifest_gpu_genotype_format,
        association_backend_genotype_format=association_backend_genotype_format,
    )


def read_manifest_gpu_genotype_format(
    existing_manifest: collections.abc.Mapping[str, typing.Any],
) -> types.GpuGenotypeFormat | None:
    """Read a concrete GPU genotype format from an existing manifest."""
    manifest_fields = read_manifest_gpu_genotype_format_fields(existing_manifest)
    native_gpu_genotype_format = _core.resolve_manifest_gpu_genotype_format(
        resume=True,
        manifest_gpu_genotype_format=manifest_fields.manifest_gpu_genotype_format,
        association_backend_genotype_format=manifest_fields.association_backend_genotype_format,
    )
    if native_gpu_genotype_format is None:
        return None
    return types.GpuGenotypeFormat(native_gpu_genotype_format)


def resolve_manifest_gpu_genotype_format(
    *,
    existing_manifest: dict[str, typing.Any] | None,
    resume: bool,
) -> types.GpuGenotypeFormat | None:
    """Return the manifest's concrete GPU genotype format when resume can reuse it."""
    if not resume or existing_manifest is None:
        return None
    return read_manifest_gpu_genotype_format(existing_manifest)


def concrete_gpu_genotype_format_from_native_plan(
    native_resolution_plan: _core.NativeGpuGenotypeFormatResolutionPlan,
) -> types.GpuGenotypeFormat:
    """Return the concrete GPU genotype format from a resolved native plan."""
    resolved_gpu_genotype_format = native_resolution_plan.resolved_gpu_genotype_format
    if resolved_gpu_genotype_format is None:
        raise RuntimeError("Native GPU genotype-format resolution plan is not resolved.")
    return types.GpuGenotypeFormat(resolved_gpu_genotype_format)


def log_native_auto_resolution(
    *,
    telemetry_session: telemetry.TelemetrySession | None,
    native_resolution_plan: _core.NativeGpuGenotypeFormatResolutionPlan,
) -> None:
    """Emit logging and telemetry for a resolved native auto decision."""
    if not native_resolution_plan.should_log_auto_resolution:
        return
    resolution_reason = native_resolution_plan.resolution_reason
    if resolution_reason is None:
        raise RuntimeError("Native GPU genotype-format resolution plan has no resolution reason.")
    log_auto_resolution(
        telemetry_session=telemetry_session,
        requested_gpu_genotype_format=types.GpuGenotypeFormat(native_resolution_plan.requested_gpu_genotype_format),
        resolved_gpu_genotype_format=concrete_gpu_genotype_format_from_native_plan(native_resolution_plan),
        resolution_reason=resolution_reason,
        fallback_error=native_resolution_plan.fallback_error,
    )


def build_resolution_from_native_plan(
    *,
    native_resolution_plan: _core.NativeGpuGenotypeFormatResolutionPlan,
    prepared_engine: _core.Regenie2RunEngine | None,
) -> GpuGenotypeFormatResolution:
    """Build the public Python resolution dataclass from native policy."""
    resolution_reason = native_resolution_plan.resolution_reason
    if resolution_reason is None:
        raise RuntimeError("Native GPU genotype-format resolution plan has no resolution reason.")
    return GpuGenotypeFormatResolution(
        requested_gpu_genotype_format=types.GpuGenotypeFormat(native_resolution_plan.requested_gpu_genotype_format),
        resolved_gpu_genotype_format=concrete_gpu_genotype_format_from_native_plan(native_resolution_plan),
        resolution_reason=resolution_reason,
        prepared_engine=prepared_engine,
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
    manifest_fields = (
        read_manifest_gpu_genotype_format_fields(existing_manifest)
        if existing_manifest is not None
        else ManifestGpuGenotypeFormatFields(
            manifest_gpu_genotype_format=None,
            association_backend_genotype_format=None,
        )
    )
    native_resolution_plan = _core.plan_single_trait_binary_gpu_genotype_format_resolution(
        requested_gpu_genotype_format=requested_gpu_genotype_format.value,
        manifest_gpu_genotype_format=manifest_fields.manifest_gpu_genotype_format,
        association_backend_genotype_format=manifest_fields.association_backend_genotype_format,
        resume=resume,
        jax_device=jax_device.value,
    )
    prepared_engine: _core.Regenie2RunEngine | None = None
    if not native_resolution_plan.requires_trusted_validation:
        log_native_auto_resolution(
            telemetry_session=telemetry_session,
            native_resolution_plan=native_resolution_plan,
        )
        return build_resolution_from_native_plan(
            native_resolution_plan=native_resolution_plan,
            prepared_engine=prepared_engine,
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
        native_resolution_plan = _core.plan_auto_gpu_genotype_format_after_trusted_validation(
            fallback_error=str(error),
        )
        prepared_engine = None
    else:
        native_resolution_plan = _core.plan_auto_gpu_genotype_format_after_trusted_validation(
            fallback_error=None,
        )

    log_native_auto_resolution(
        telemetry_session=telemetry_session,
        native_resolution_plan=native_resolution_plan,
    )
    return build_resolution_from_native_plan(
        native_resolution_plan=native_resolution_plan,
        prepared_engine=prepared_engine,
    )
