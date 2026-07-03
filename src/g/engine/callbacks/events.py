"""Callback-local diagnostic event and telemetry helpers."""

from __future__ import annotations

import typing

from g.engine import run_events, telemetry

if typing.TYPE_CHECKING:
    from g import _core
    from g.engine.telemetry import TelemetrySession
else:
    TelemetrySession = telemetry.TelemetrySession


def native_pipeline_diagnostic_policy() -> _core.NativePipelineDiagnosticPolicy:
    """Build the native pipeline diagnostic policy handle for callback diagnostics."""
    return run_events.native_pipeline_diagnostic_policy()
