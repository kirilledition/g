"""Binary-trait callback implementations for REGENIE step 2."""

from __future__ import annotations

from g.engine.callbacks import _legacy

BinaryRegenie2PipelineCallback = _legacy.BinaryRegenie2PipelineCallback
MultiBinaryRegenie2PipelineCallback = _legacy.MultiBinaryRegenie2PipelineCallback

__all__ = [
    "BinaryRegenie2PipelineCallback",
    "MultiBinaryRegenie2PipelineCallback",
]
