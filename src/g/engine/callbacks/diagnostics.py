"""Timing and diagnostics helpers used by REGENIE callback flows."""

from __future__ import annotations

from g.engine.callbacks import _legacy

block_until_ready = _legacy.block_until_ready
enforce_null_logistic_nonconvergence_policy = _legacy.enforce_null_logistic_nonconvergence_policy
record_binary_chunk_diagnostics = _legacy.record_binary_chunk_diagnostics
record_binary_chunk_diagnostics_from_count = _legacy.record_binary_chunk_diagnostics_from_count

__all__ = [
    "block_until_ready",
    "enforce_null_logistic_nonconvergence_policy",
    "record_binary_chunk_diagnostics",
    "record_binary_chunk_diagnostics_from_count",
]
