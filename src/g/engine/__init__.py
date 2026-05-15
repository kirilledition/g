"""High-level orchestration helpers for REGENIE step 2 association runs."""

from g.engine.regenie2_pipeline import (
    BinaryRegenie2PipelineCallback,
    LinearRegenie2PipelineCallback,
    run_regenie2_binary_bgen_pipeline,
    run_regenie2_linear_bgen_pipeline,
)
from g.engine.types import Regenie2ChunkAccumulator, Regenie2ChunkPayload

ChunkAccumulator = Regenie2ChunkAccumulator
ChunkPayload = Regenie2ChunkPayload

__all__ = [
    "BinaryRegenie2PipelineCallback",
    "ChunkAccumulator",
    "ChunkPayload",
    "LinearRegenie2PipelineCallback",
    "Regenie2ChunkAccumulator",
    "Regenie2ChunkPayload",
    "run_regenie2_binary_bgen_pipeline",
    "run_regenie2_linear_bgen_pipeline",
]
