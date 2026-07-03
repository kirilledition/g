"""Native-dispatch diagnostic event helpers."""

from __future__ import annotations

from g import _core


def native_dispatch_diagnostic_policy() -> _core.NativeDispatchDiagnosticPolicy:
    """Build the native-dispatch diagnostic policy handle."""
    return _core.NativeDispatchDiagnosticPolicy()
