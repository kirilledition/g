"""Runner-local timing helpers."""

from __future__ import annotations

import typing

from g.engine import timing as engine_timing

if typing.TYPE_CHECKING:
    from pathlib import Path

    from g.runner import events

type StageTimingRecorder = engine_timing.StageTimingRecorder
type FinalTimingOutputContext = engine_timing.FinalTimingOutputContext


def resolve_final_timing_output_context(
    diagnostics_stage_timing_path: Path | None,
    telemetry_session: events.TelemetrySession | None,
) -> FinalTimingOutputContext:
    """Resolve final timing output paths through runner timing helpers."""
    return engine_timing.resolve_final_timing_output_context(diagnostics_stage_timing_path, telemetry_session)


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


def record_final_timing_outputs_write_started_diagnostic_event(
    stage_timing_path: Path | None,
    profile_summary_path: Path | None,
    run_id: str | None,
) -> None:
    """Record that final timing output writes are starting."""
    engine_timing.record_final_timing_outputs_write_started_diagnostic_event(
        stage_timing_path,
        profile_summary_path,
        run_id,
    )


def write_final_timing_outputs(
    stage_timing_recorder: StageTimingRecorder | None,
    *,
    stage_timing_path: Path | None,
    profile_summary_path: Path | None,
    run_id: str | None,
) -> dict[str, bool]:
    """Persist all configured final timing outputs."""
    return engine_timing.write_final_timing_outputs(
        stage_timing_recorder,
        stage_timing_path=stage_timing_path,
        profile_summary_path=profile_summary_path,
        run_id=run_id,
    )
