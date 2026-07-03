"""Native-dispatch timing helpers."""

from __future__ import annotations

from g.engine import timing as engine_timing

type StageTimingRecorder = engine_timing.StageTimingRecorder


def record_stage_duration(
    stage_timing_recorder: StageTimingRecorder | None,
    stage_name: str,
    start_time: float,
) -> None:
    """Record elapsed wall time for a native-dispatch stage when diagnostics are active."""
    engine_timing.record_stage_duration(stage_timing_recorder, stage_name, start_time)
