"""Linear-trait callback implementations for REGENIE step 2."""

from __future__ import annotations

from g.engine.callbacks import _legacy

LinearRegenie2PipelineCallback = _legacy.LinearRegenie2PipelineCallback
MultiLinearRegenie2PipelineCallback = _legacy.MultiLinearRegenie2PipelineCallback

__all__ = [
    "LinearRegenie2PipelineCallback",
    "MultiLinearRegenie2PipelineCallback",
]
