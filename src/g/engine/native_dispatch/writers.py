"""Callback and writer lifecycle helpers for native dispatch."""

from __future__ import annotations

import concurrent.futures
import time
import typing
from pathlib import Path

from g import _core
from g.engine import timing as engine_timing


def finish_writer_session_to_path(writer_session: typing.Any) -> Path | None:
    """Finish one writer session and normalize its optional final Parquet path."""
    final_parquet_path = typing.cast("str | None", writer_session.finish())
    return None if final_parquet_path is None else Path(final_parquet_path)


def finish_writer_sessions(
    *,
    writer_sessions: tuple[typing.Any, ...],
    stage_timing_recorder: engine_timing.StageTimingRecorder | None,
    writer_finish_thread_count: int,
) -> tuple[Path | None, ...]:
    """Finish writer sessions and optionally finalize Parquet output."""
    writer_finish_start_time = time.perf_counter()
    _core.record_native_dispatch_writer_sessions_finish_started_diagnostic_event(
        requested_thread_count=writer_finish_thread_count,
        writer_session_count=len(writer_sessions),
    )
    finish_plan = _core.plan_writer_finish_execution(len(writer_sessions), writer_finish_thread_count)
    if not finish_plan.uses_parallel_finish:
        final_parquet_paths = tuple(finish_writer_session_to_path(writer_session) for writer_session in writer_sessions)
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=finish_plan.thread_count,
            thread_name_prefix="g-writer-finish",
        ) as executor:
            futures = tuple(
                executor.submit(finish_writer_session_to_path, writer_session) for writer_session in writer_sessions
            )
            final_parquet_paths = tuple(future.result() for future in futures)
    engine_timing.record_stage_duration(
        stage_timing_recorder, "writer_finish_and_parquet_finalization", writer_finish_start_time
    )
    return final_parquet_paths
