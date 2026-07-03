"""Timing helpers for REGENIE step 2 pipelines."""

from __future__ import annotations

import typing

from g.engine import timing as engine_timing

if typing.TYPE_CHECKING:
    from pathlib import Path

type StageTimingRecorder = engine_timing.StageTimingRecorder


def build_stage_timing_recorder(
    stage_timing_path: Path | None,
    *,
    force: bool,
) -> StageTimingRecorder | None:
    """Create a diagnostic stage recorder when requested."""
    return engine_timing.build_stage_timing_recorder(stage_timing_path, force=force)


def record_stage_duration(
    stage_timing_recorder: StageTimingRecorder | None,
    stage_name: str,
    start_time: float,
) -> None:
    """Record elapsed wall time for a stage when diagnostics are active."""
    engine_timing.record_stage_duration(stage_timing_recorder, stage_name, start_time)


def should_collect_exact_stage_timings(stage_timing_recorder: StageTimingRecorder | None) -> bool:
    """Return whether timing should force synchronized exact stage measurements."""
    return engine_timing.should_collect_exact_stage_timings(stage_timing_recorder)
