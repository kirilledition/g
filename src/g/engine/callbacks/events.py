"""Callback-local diagnostic event and telemetry helpers."""

from __future__ import annotations

from g import _core

type TelemetrySession = object


def native_pipeline_diagnostic_policy() -> _core.NativePipelineDiagnosticPolicy:
    """Build the native pipeline diagnostic policy handle for callback diagnostics."""
    return _core.NativePipelineDiagnosticPolicy()
