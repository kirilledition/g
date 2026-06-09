"""Materialization and output-write helpers for callback workers."""

from __future__ import annotations

from g.engine.callbacks import _legacy

write_regenie2_native_chunk_with_optional_timing = _legacy.write_regenie2_native_chunk_with_optional_timing
write_regenie2_multi_native_chunk_with_optional_timing = _legacy.write_regenie2_multi_native_chunk_with_optional_timing
get_metadata_chromosome = _legacy.get_metadata_chromosome

__all__ = [
    "write_regenie2_native_chunk_with_optional_timing",
    "write_regenie2_multi_native_chunk_with_optional_timing",
    "get_metadata_chromosome",
]
