"""High-level orchestration helpers for REGENIE step 2 association runs."""

from g.engine.regenie2_pipeline import (
    BinaryRegenie2PipelineCallback,
    LinearRegenie2PipelineCallback,
    StageTimingRecorder,
    WarmCacheReport,
    WarmCacheShape,
    build_stage_timing_recorder_from_environment,
    record_stage_duration,
    run_regenie2_binary_bgen_pipeline,
    run_regenie2_linear_bgen_pipeline,
    warm_regenie2_binary_bgen_cache,
    warm_regenie2_linear_bgen_cache,
    write_stage_timing_snapshot_from_environment,
)
from g.engine.types import Regenie2ChunkPayload

ChunkPayload = Regenie2ChunkPayload

__all__ = [
    "BinaryRegenie2PipelineCallback",
    "ChunkPayload",
    "LinearRegenie2PipelineCallback",
    "Regenie2ChunkPayload",
    "StageTimingRecorder",
    "WarmCacheReport",
    "WarmCacheShape",
    "build_stage_timing_recorder_from_environment",
    "record_stage_duration",
    "run_regenie2_binary_bgen_pipeline",
    "run_regenie2_linear_bgen_pipeline",
    "warm_regenie2_binary_bgen_cache",
    "warm_regenie2_linear_bgen_cache",
    "write_stage_timing_snapshot_from_environment",
]
