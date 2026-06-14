"""JAX runtime setup diagnostic event conversion and emission."""

from __future__ import annotations

from g.jax_runtime import models


def diagnostic_fields(**fields: object) -> tuple[models.JaxRuntimeDiagnosticField, ...]:
    """Build immutable diagnostic fields without `None` values.

    Args:
        fields: Candidate event fields.

    Returns:
        Event field tuple.

    """
    return tuple(
        models.JaxRuntimeDiagnosticField(name=key, value=value) for key, value in fields.items() if value is not None
    )


def diagnostic_events_from_setup_report(
    setup_report: models.JaxRuntimeSetupReport,
) -> tuple[models.JaxRuntimeDiagnosticEvent, ...]:
    """Convert a setup report into ordered structured diagnostic events.

    Args:
        setup_report: Resolved setup report to describe.

    Returns:
        Ordered diagnostic events.

    """
    gpu_validation_level = models.JaxRuntimeDiagnosticLevel.INFO
    if setup_report.gpu_validation_status == models.GpuValidationStatus.FAILED:
        gpu_validation_level = models.JaxRuntimeDiagnosticLevel.ERROR
    xla_auxiliary_cache_enabled = setup_report.xla_auxiliary_cache_mode != models.XlaAuxiliaryCacheMode.DISABLED
    return (
        models.JaxRuntimeDiagnosticEvent(
            event_name="jax_platform_selected",
            level=models.JaxRuntimeDiagnosticLevel.INFO,
            message=f"Selected JAX platform {setup_report.platform_name}.",
            fields=diagnostic_fields(
                requested_device=setup_report.requested_device.value,
                platform=setup_report.platform_name,
            ),
        ),
        models.JaxRuntimeDiagnosticEvent(
            event_name="jax_persistent_cache_configured",
            level=models.JaxRuntimeDiagnosticLevel.INFO,
            message=(
                "JAX persistent compilation cache enabled."
                if setup_report.persistent_cache_enabled
                else "JAX persistent compilation cache disabled."
            ),
            fields=diagnostic_fields(
                enabled=setup_report.persistent_cache_enabled,
                cache_directory=str(setup_report.cache_directory),
                min_entry_size_bytes=setup_report.persistent_cache_min_entry_size_bytes,
                min_compile_time_seconds=setup_report.persistent_cache_min_compile_time_seconds,
            ),
        ),
        models.JaxRuntimeDiagnosticEvent(
            event_name="jax_xla_auxiliary_cache_configured",
            level=models.JaxRuntimeDiagnosticLevel.INFO,
            message=(
                "XLA auxiliary persistent cache enabled."
                if xla_auxiliary_cache_enabled
                else "XLA auxiliary persistent cache disabled."
            ),
            fields=diagnostic_fields(
                enabled=xla_auxiliary_cache_enabled,
                mode=setup_report.xla_auxiliary_cache_mode.value,
                reason=setup_report.xla_auxiliary_cache_reason,
            ),
        ),
        models.JaxRuntimeDiagnosticEvent(
            event_name="jax_transfer_guard_configured",
            level=models.JaxRuntimeDiagnosticLevel.INFO,
            message=(
                "JAX transfer guard diagnostics enabled."
                if setup_report.transfer_guard_enabled
                else "JAX transfer guard diagnostics disabled."
            ),
            fields=diagnostic_fields(enabled=setup_report.transfer_guard_enabled),
        ),
        models.JaxRuntimeDiagnosticEvent(
            event_name="jax_gpu_validation",
            level=gpu_validation_level,
            message=f"JAX GPU validation {setup_report.gpu_validation_status.value}.",
            fields=diagnostic_fields(
                status=setup_report.gpu_validation_status.value,
                message=setup_report.gpu_validation_message,
            ),
        ),
    )
