"""Callback and writer lifecycle helpers for native dispatch."""

from __future__ import annotations

import concurrent.futures
import contextlib
import time
import typing
from pathlib import Path

from g import _core
from g.runner import events, lifecycle, timing

if typing.TYPE_CHECKING:
    from g.engine.native_dispatch import models


def finish_callback_drain(
    *,
    callback: models.BgenDeliveryCallbackProtocol,
    stage_timing_recorder: timing.StageTimingRecorder | None,
) -> None:
    """Wait for queued callback work to drain."""
    callback_finish_start_time = time.perf_counter()
    events.native_dispatch_diagnostic_policy().record_native_dispatch_callback_drain_started_diagnostic_event()
    callback.finish()
    timing.record_stage_duration(stage_timing_recorder, "callback_drain", callback_finish_start_time)


def start_callback(callback: models.BgenDeliveryCallbackProtocol) -> None:
    """Start callback workers before native chunk delivery."""
    callback.start()


def resolve_writer_finish_thread_count(writer_session_count: int, requested_thread_count: int) -> int:
    """Return the bounded number of threads used to finish writer sessions."""
    return int(
        native_schedule_policy().resolve_writer_finish_thread_count(writer_session_count, requested_thread_count)
    )


def plan_writer_finish_execution(
    writer_session_count: int,
    requested_thread_count: int,
) -> _core.NativeWriterFinishExecutionPlan:
    """Return the native writer-finish execution plan."""
    return native_schedule_policy().plan_writer_finish_execution(writer_session_count, requested_thread_count)


def native_schedule_policy() -> _core.NativeSchedulePolicy:
    """Build the native schedule policy handle."""
    return _core.NativeSchedulePolicy()


def native_output_writer_lifecycle_policy() -> _core.NativeOutputWriterLifecyclePolicy:
    """Build the native writer lifecycle policy handle."""
    return _core.NativeOutputWriterLifecyclePolicy()


def all_writer_sessions_native(writer_sessions: tuple[typing.Any, ...]) -> bool:
    """Return whether every writer session is a native writer session."""
    return bool(writer_sessions) and all(
        isinstance(writer_session, _core.OutputWriterSession) for writer_session in writer_sessions
    )


def native_final_parquet_paths(final_parquet_path_values: list[str | None]) -> tuple[Path | None, ...]:
    """Normalize native writer finish paths."""
    return tuple(
        None if final_parquet_path is None else Path(final_parquet_path)
        for final_parquet_path in final_parquet_path_values
    )


def finish_writer_session_to_path(writer_session: typing.Any) -> Path | None:
    """Finish one writer session and normalize its optional final Parquet path."""
    final_parquet_path = typing.cast("str | None", writer_session.finish())
    return None if final_parquet_path is None else Path(final_parquet_path)


def finish_writer_session_interrupted_by_signal(writer_session: typing.Any, signal_name: str) -> None:
    """Flush one interrupted writer session."""
    writer_session.finish_interrupted(signal_name)


def finish_writer_sessions(
    *,
    writer_sessions: tuple[typing.Any, ...],
    stage_timing_recorder: timing.StageTimingRecorder | None,
    writer_finish_thread_count: int,
) -> tuple[Path | None, ...]:
    """Finish writer sessions and optionally finalize Parquet output."""
    writer_finish_start_time = time.perf_counter()
    events.native_dispatch_diagnostic_policy().record_native_dispatch_writer_sessions_finish_started_diagnostic_event(
        requested_thread_count=writer_finish_thread_count,
        writer_session_count=len(writer_sessions),
    )
    finish_plan = plan_writer_finish_execution(len(writer_sessions), writer_finish_thread_count)
    if all_writer_sessions_native(writer_sessions):
        final_parquet_paths = native_final_parquet_paths(
            native_output_writer_lifecycle_policy().finish_writer_sessions(
                list(typing.cast("tuple[_core.OutputWriterSession, ...]", writer_sessions)),
                finish_plan.thread_count,
            )
        )
    elif not finish_plan.uses_parallel_finish:
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


def finish_writer_sessions_interrupted(
    *,
    writer_sessions: tuple[typing.Any, ...],
    shutdown_request: lifecycle.GracefulShutdownRequested,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    writer_finish_thread_count: int,
) -> None:
    """Flush interrupted writer sessions without final Parquet output."""
    writer_finish_start_time = time.perf_counter()
    events.native_dispatch_diagnostic_policy().record_native_dispatch_writer_sessions_interrupted_flush_started_diagnostic_event(
        requested_thread_count=writer_finish_thread_count,
        signal_exit_code=shutdown_request.exit_code,
        signal_name=shutdown_request.signal_name,
        signal_number=shutdown_request.shutdown_signal.number,
        writer_session_count=len(writer_sessions),
    )
    finish_plan = plan_writer_finish_execution(len(writer_sessions), writer_finish_thread_count)
    if all_writer_sessions_native(writer_sessions):
        native_output_writer_lifecycle_policy().finish_writer_sessions_interrupted(
            list(typing.cast("tuple[_core.OutputWriterSession, ...]", writer_sessions)),
            shutdown_request.signal_name,
            finish_plan.thread_count,
        )
    elif not finish_plan.uses_parallel_finish:
        for writer_session in writer_sessions:
            finish_writer_session_interrupted_by_signal(writer_session, shutdown_request.signal_name)
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=finish_plan.thread_count,
            thread_name_prefix="g-writer-finish",
        ) as executor:
            futures = tuple(
                executor.submit(
                    finish_writer_session_interrupted_by_signal, writer_session, shutdown_request.signal_name
                )
                for writer_session in writer_sessions
            )
            for future in futures:
                future.result()
    timing.record_stage_duration(stage_timing_recorder, "writer_finish_interrupted", writer_finish_start_time)


def abort_callback(callback: models.BgenDeliveryCallbackProtocol) -> None:
    """Request callback worker shutdown."""
    with contextlib.suppress(Exception):
        callback.abort()


def abort_writer_session(writer_session: typing.Any) -> None:
    """Abort one writer session."""
    with contextlib.suppress(Exception):
        writer_session.abort()


def abort_writer_sessions(writer_sessions: tuple[typing.Any, ...]) -> None:
    """Abort writer sessions."""
    if all_writer_sessions_native(writer_sessions):
        with contextlib.suppress(Exception):
            native_output_writer_lifecycle_policy().abort_writer_sessions(
                list(typing.cast("tuple[_core.OutputWriterSession, ...]", writer_sessions))
            )
        return
    for writer_session in writer_sessions:
        abort_writer_session(writer_session)
