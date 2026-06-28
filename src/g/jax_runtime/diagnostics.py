"""JAX runtime setup diagnostic event conversion and emission."""

from __future__ import annotations

import typing

from g import _core
from g.jax_runtime import models


def diagnostic_event_from_native_payload(payload: object) -> models.JaxRuntimeDiagnosticEvent:
    """Adapt a native JAX runtime diagnostic event payload.

    Args:
        payload: Native event payload.

    Returns:
        Structured JAX runtime diagnostic event.

    """
    event_payload = native_mapping_payload(payload)
    field_payloads = tuple(typing.cast("typing.Iterable[object]", event_payload["fields"]))
    return models.JaxRuntimeDiagnosticEvent(
        event_name=str(event_payload["event_name"]),
        level=models.JaxRuntimeDiagnosticLevel(str(event_payload["level"])),
        message=str(event_payload["message"]),
        fields=tuple(diagnostic_field_from_native_payload(field_payload) for field_payload in field_payloads),
    )


def diagnostic_field_from_native_payload(payload: object) -> models.JaxRuntimeDiagnosticField:
    """Adapt a native JAX runtime diagnostic field payload."""
    field_payload = native_mapping_payload(payload)
    return models.JaxRuntimeDiagnosticField(name=str(field_payload["name"]), value=field_payload["value"])


def native_mapping_payload(payload: object) -> dict[str, typing.Any]:
    """Adapt a native mapping payload to a mutable Python dictionary."""
    return dict(typing.cast("typing.Mapping[str, typing.Any]", payload))


def diagnostic_events_from_setup_report(
    setup_report: models.JaxRuntimeSetupReport,
) -> tuple[models.JaxRuntimeDiagnosticEvent, ...]:
    """Convert a setup report into ordered structured diagnostic events.

    Args:
        setup_report: Resolved setup report to describe.

    Returns:
        Ordered diagnostic events.

    """
    native_payloads = _core.build_jax_runtime_setup_diagnostic_payloads(
        requested_device=setup_report.requested_device.value,
        platform_name=setup_report.platform_name,
        cache_directory=str(setup_report.cache_directory),
        matmul_precision=setup_report.matmul_precision.value,
        persistent_cache_enabled=setup_report.persistent_cache_enabled,
        persistent_cache_min_entry_size_bytes=setup_report.persistent_cache_min_entry_size_bytes,
        persistent_cache_min_compile_time_seconds=setup_report.persistent_cache_min_compile_time_seconds,
        xla_auxiliary_cache_mode=setup_report.xla_auxiliary_cache_mode.value,
        xla_auxiliary_cache_reason=setup_report.xla_auxiliary_cache_reason,
        transfer_guard_enabled=setup_report.transfer_guard_enabled,
        gpu_validation_status=setup_report.gpu_validation_status.value,
        gpu_validation_message=setup_report.gpu_validation_message,
    )
    return tuple(
        diagnostic_event_from_native_payload(native_payload)
        for native_payload in typing.cast("typing.Iterable[object]", native_payloads)
    )


def diagnostic_events_from_native_setup_session(
    native_setup_session: _core.NativeJaxRuntimeSetupSession,
) -> tuple[models.JaxRuntimeDiagnosticEvent, ...]:
    """Convert a native setup session into ordered structured diagnostic events.

    Args:
        native_setup_session: Native JAX runtime setup session.

    Returns:
        Ordered diagnostic events.

    """
    native_payloads = native_setup_session.diagnostic_event_payloads()
    return tuple(
        diagnostic_event_from_native_payload(native_payload)
        for native_payload in typing.cast("typing.Iterable[object]", native_payloads)
    )
