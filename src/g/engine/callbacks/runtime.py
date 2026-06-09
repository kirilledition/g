"""Core callback lifecycle and bounded-queue runtime used by REGENIE callbacks."""

from __future__ import annotations

from g.engine.callbacks import _legacy

require_current_chromosome_state = _legacy.require_current_chromosome_state
NativeBgenCallbackRunner = _legacy.NativeBgenCallbackRunner

__all__ = [
    "require_current_chromosome_state",
    "NativeBgenCallbackRunner",
]
