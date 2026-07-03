"""Preflight diagnostic event helpers."""

from __future__ import annotations

import typing

from g.engine import run_events

if typing.TYPE_CHECKING:
    from g import _core


def native_output_preflight_diagnostic_policy() -> _core.NativeOutputPreflightDiagnosticPolicy:
    """Build the native output/preflight diagnostic policy handle."""
    return run_events.native_output_preflight_diagnostic_policy()
