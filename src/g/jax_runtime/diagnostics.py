"""JAX runtime setup diagnostic event conversion and emission."""

from __future__ import annotations

import typing

from g.jax_runtime import models

if typing.TYPE_CHECKING:
    from g import _core


def diagnostic_event_from_native_event(
    native_event: _core.NativeJaxRuntimeDiagnosticEvent,
) -> models.JaxRuntimeDiagnosticEvent:
    """Adapt a native JAX runtime diagnostic event.

    Args:
        native_event: Native event.

    Returns:
        Structured JAX runtime diagnostic event.

    """
    return models.JaxRuntimeDiagnosticEvent(
        event_name=native_event.event_name,
        level=models.JaxRuntimeDiagnosticLevel(native_event.level),
        message=native_event.message,
        fields=tuple(diagnostic_field_from_native_field(native_field) for native_field in native_event.fields),
    )


def diagnostic_field_from_native_field(
    native_field: _core.NativeJaxRuntimeDiagnosticField,
) -> models.JaxRuntimeDiagnosticField:
    """Adapt a native JAX runtime diagnostic field."""
    return models.JaxRuntimeDiagnosticField(name=native_field.name, value=native_field.value)


def diagnostic_events_from_native_setup_session(
    native_setup_session: _core.NativeJaxRuntimeSetupSession,
) -> tuple[models.JaxRuntimeDiagnosticEvent, ...]:
    """Convert a native setup session into ordered structured diagnostic events.

    Args:
        native_setup_session: Native JAX runtime setup session.

    Returns:
        Ordered diagnostic events.

    """
    return tuple(
        diagnostic_event_from_native_event(native_event)
        for native_event in native_setup_session.diagnostic_events()
    )
