"""Grouped multi-phenotype fanout callback compatibility helpers."""

from __future__ import annotations

from g.engine.callbacks import _legacy

GroupedMultiPhenotypeFanoutCallback = _legacy.GroupedMultiPhenotypeFanoutCallback

__all__ = [
    "GroupedMultiPhenotypeFanoutCallback",
]
