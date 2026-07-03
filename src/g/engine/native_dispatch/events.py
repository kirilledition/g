"""Native-dispatch diagnostic event helpers."""

from __future__ import annotations

import typing

from g.engine import run_events

if typing.TYPE_CHECKING:
    from g import _core


def native_dispatch_diagnostic_policy() -> _core.NativeDispatchDiagnosticPolicy:
    """Build the native-dispatch diagnostic policy handle."""
    return run_events.native_dispatch_diagnostic_policy()
