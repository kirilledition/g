"""Callback-local timing helpers."""

from __future__ import annotations

import typing

from g.engine import timing as engine_timing

if typing.TYPE_CHECKING:
    from g.engine.timing import ChunkTimingIdentity, StageTimingRecorder
else:
    ChunkTimingIdentity = engine_timing.ChunkTimingIdentity
    StageTimingRecorder = engine_timing.StageTimingRecorder


def should_collect_exact_stage_timings(stage_timing_recorder: StageTimingRecorder | None) -> bool:
    """Return whether timing should force synchronized exact stage measurements."""
    return engine_timing.should_collect_exact_stage_timings(stage_timing_recorder)


def record_stage_duration(
    stage_timing_recorder: StageTimingRecorder | None,
    stage_name: str,
    start_time: float,
) -> None:
    """Record elapsed wall time for a callback stage when diagnostics are active."""
    engine_timing.record_stage_duration(stage_timing_recorder, stage_name, start_time)


def record_chunk_stage_duration(
    stage_timing_recorder: StageTimingRecorder | None,
    *,
    chunk_identity: ChunkTimingIdentity,
    stage_name: str,
    start_time: float,
) -> None:
    """Record elapsed wall time for one callback chunk stage."""
    engine_timing.record_chunk_stage_duration(
        stage_timing_recorder,
        chunk_identity=chunk_identity,
        stage_name=stage_name,
        start_time=start_time,
    )
