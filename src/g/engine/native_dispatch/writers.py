"""Callback and writer lifecycle helpers for native dispatch."""

from __future__ import annotations

import concurrent.futures
import contextlib
import logging
import time
import typing
from pathlib import Path

from g import _core
from g.engine import shutdown, timing

logger = logging.getLogger(__name__)


def finish_callback_drain(
    *,
    callback: object,
    stage_timing_recorder: timing.StageTimingRecorder | None,
) -> None:
    """Wait for queued callback work to drain."""
    callback_finish_start_time = time.perf_counter()
    logger.debug("Draining native callback worker queues.")
    typing.cast("typing.Any", callback).finish()
    timing.record_stage_duration(stage_timing_recorder, "callback_drain", callback_finish_start_time)


def start_callback(callback: object) -> None:
    """Start callback workers when the callback exposes an explicit lifecycle hook."""
    start_callback_method = getattr(callback, "start", None)
    if callable(start_callback_method):
        start_callback_method()


def finish_writer_session(
    *,
    writer_session: typing.Any,
    stage_timing_recorder: timing.StageTimingRecorder | None,
) -> str | None:
    """Finish the writer session and optionally finalize Parquet output."""
    writer_finish_start_time = time.perf_counter()
    logger.debug("Finishing output writer and optional Parquet finalization.")
    final_parquet_path = writer_session.finish()
    timing.record_stage_duration(
        stage_timing_recorder, "writer_finish_and_parquet_finalization", writer_finish_start_time
    )
    return typing.cast("str | None", final_parquet_path)


def resolve_writer_finish_thread_count(writer_session_count: int, requested_thread_count: int) -> int:
    """Return the bounded number of threads used to finish writer sessions."""
    return int(_core.resolve_writer_finish_thread_count(writer_session_count, requested_thread_count))


def plan_writer_finish_execution(
    writer_session_count: int,
    requested_thread_count: int,
) -> _core.NativeWriterFinishExecutionPlan:
    """Return the native writer-finish execution plan."""
    return _core.plan_writer_finish_execution(writer_session_count, requested_thread_count)


def finish_writer_session_to_path(writer_session: typing.Any) -> Path | None:
    """Finish one writer session and normalize its optional final Parquet path."""
    final_parquet_path = typing.cast("str | None", writer_session.finish())
    return None if final_parquet_path is None else Path(final_parquet_path)


def finish_writer_sessions(
    *,
    writer_sessions: tuple[typing.Any, ...],
    stage_timing_recorder: timing.StageTimingRecorder | None,
    writer_finish_thread_count: int,
) -> tuple[Path | None, ...]:
    """Finish writer sessions and optionally finalize Parquet output."""
    writer_finish_start_time = time.perf_counter()
    logger.debug("Finishing output writer(s) and optional Parquet finalization.")
    finish_plan = plan_writer_finish_execution(len(writer_sessions), writer_finish_thread_count)
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
    timing.record_stage_duration(
        stage_timing_recorder, "writer_finish_and_parquet_finalization", writer_finish_start_time
    )
    return final_parquet_paths


def finish_writer_session_interrupted(
    *,
    writer_session: typing.Any,
    shutdown_request: shutdown.GracefulShutdownRequested,
    stage_timing_recorder: timing.StageTimingRecorder | None,
) -> None:
    """Flush writer output for an interrupted run without final Parquet."""
    writer_finish_start_time = time.perf_counter()
    logger.info("Flushing interrupted output writer after %s.", shutdown_request.signal_name)
    writer_session.finish_interrupted(shutdown_request.signal_name)
    timing.record_stage_duration(stage_timing_recorder, "writer_finish_interrupted", writer_finish_start_time)


def finish_writer_sessions_interrupted(
    *,
    writer_sessions: tuple[typing.Any, ...],
    shutdown_request: shutdown.GracefulShutdownRequested,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    writer_finish_thread_count: int,
) -> None:
    """Flush interrupted writer sessions without final Parquet output."""
    writer_finish_start_time = time.perf_counter()
    logger.info("Flushing interrupted output writer(s) after %s.", shutdown_request.signal_name)
    finish_plan = plan_writer_finish_execution(len(writer_sessions), writer_finish_thread_count)
    if not finish_plan.uses_parallel_finish:
        for writer_session in writer_sessions:
            writer_session.finish_interrupted(shutdown_request.signal_name)
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=finish_plan.thread_count,
            thread_name_prefix="g-writer-finish",
        ) as executor:
            futures = tuple(
                executor.submit(writer_session.finish_interrupted, shutdown_request.signal_name)
                for writer_session in writer_sessions
            )
            for future in futures:
                future.result()
    timing.record_stage_duration(stage_timing_recorder, "writer_finish_interrupted", writer_finish_start_time)


def abort_callback(callback: object) -> None:
    """Request callback worker shutdown when supported."""
    abort_callback_method = getattr(callback, "abort", None)
    if callable(abort_callback_method):
        with contextlib.suppress(Exception):
            abort_callback_method()


def abort_writer_session(writer_session: typing.Any) -> None:
    """Abort one writer session."""
    with contextlib.suppress(Exception):
        writer_session.abort()


def abort_writer_sessions(writer_sessions: tuple[typing.Any, ...]) -> None:
    """Abort writer sessions."""
    for writer_session in writer_sessions:
        abort_writer_session(writer_session)
