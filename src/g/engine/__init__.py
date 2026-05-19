"""High-level orchestration helpers for REGENIE step 2 association runs."""

from g.engine.callbacks import (
    BinaryRegenie2PipelineCallback,
    LinearRegenie2PipelineCallback,
)
from g.engine.regenie2_pipeline import (
    run_regenie2_binary_bgen_pipeline,
    run_regenie2_linear_bgen_pipeline,
)
from g.engine.timing import (
    StageTimingRecorder,
    build_stage_timing_recorder,
    record_stage_duration,
    write_stage_timing_snapshot,
)
from g.engine.warm_cache import (
    WarmCacheReport,
    WarmCacheShape,
    warm_regenie2_binary_bgen_cache,
    warm_regenie2_linear_bgen_cache,
)

__all__ = [
    "BinaryRegenie2PipelineCallback",
    "LinearRegenie2PipelineCallback",
    "StageTimingRecorder",
    "WarmCacheReport",
    "WarmCacheShape",
    "build_stage_timing_recorder",
    "record_stage_duration",
    "run_regenie2_binary_bgen_pipeline",
    "run_regenie2_linear_bgen_pipeline",
    "warm_regenie2_binary_bgen_cache",
    "warm_regenie2_linear_bgen_cache",
    "write_stage_timing_snapshot",
]
