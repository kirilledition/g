"""Native BGEN chunk delivery helpers."""

from __future__ import annotations

import enum
import time
import typing
from dataclasses import dataclass

from g import _core
from g.engine import shutdown, timing
from g.engine.native_dispatch import models, writers

if typing.TYPE_CHECKING:
    from pathlib import Path


class BgenDeliveryMethod(enum.StrEnum):
    """Native BGEN delivery method selected by engine scheduling policy."""

    DOSAGE_NATIVE_MULTI_ALIGNED_SAMPLES = "dosage_native_multi_aligned_samples"
    DOSAGE_NATIVE_ALIGNED_SAMPLES = "dosage_native_aligned_samples"
    DOSAGE_SAMPLE_INDICES = "dosage_sample_indices"
    PACKED8_NATIVE_MULTI_ALIGNED_SAMPLES = "packed8_native_multi_aligned_samples"
    PACKED8_NATIVE_ALIGNED_SAMPLES = "packed8_native_aligned_samples"
    PACKED8_SAMPLE_INDICES = "packed8_sample_indices"


class BgenDeliveryCleanupOutcome(enum.StrEnum):
    """Native BGEN delivery cleanup outcome selected by engine lifecycle policy."""

    SUCCESS = "success"
    INTERRUPTED = "interrupted"
    FAILURE = "failure"
    INTERRUPTED_CLEANUP_FAILURE = "interrupted_cleanup_failure"


class BgenDeliveryCleanupAction(enum.StrEnum):
    """Native BGEN delivery cleanup action selected by engine lifecycle policy."""

    DRAIN_CALLBACK = "drain_callback"
    FINISH_WRITER_SESSIONS = "finish_writer_sessions"
    FINISH_INTERRUPTED_WRITER_SESSIONS = "finish_interrupted_writer_sessions"
    ABORT_CALLBACK = "abort_callback"
    ABORT_WRITER_SESSIONS = "abort_writer_sessions"
    WRITE_STAGE_TIMING_SNAPSHOT = "write_stage_timing_snapshot"


@dataclass(frozen=True)
class BgenDeliveryCleanupExecution:
    """Result of executing one native BGEN delivery cleanup plan.

    Attributes:
        final_parquet_paths: Final Parquet paths produced by successful writer finalization.
        callback_finished: Whether the callback was drained by this cleanup path.

    """

    final_parquet_paths: tuple[Path | None, ...]
    callback_finished: bool


def plan_bgen_delivery_cleanup(
    *,
    cleanup_outcome: BgenDeliveryCleanupOutcome,
    callback_finished: bool,
) -> _core.NativeBgenDeliveryCleanupPlan:
    """Return the native cleanup plan for one delivery outcome."""
    return _core.plan_bgen_delivery_cleanup(cleanup_outcome.value, callback_finished)


def resolve_native_callback_batch_size(
    callback: object,
    *,
    variant_major_packed8_probability_pairs: bool,
) -> int:
    """Return the validated native callback batch size for one callback object."""
    raw_callback_batch_size = getattr(callback, "native_callback_batch_size", None)
    callback_batch_size = None if raw_callback_batch_size is None else int(raw_callback_batch_size)
    return int(_core.resolve_delivery_callback_batch_size(callback_batch_size, variant_major_packed8_probability_pairs))


def plan_bgen_delivery_invocation(
    callback: object,
    run_input: models.BgenDeliveryRunInputProtocol,
    *,
    variant_major_packed8_probability_pairs: bool,
) -> _core.NativeBgenDeliveryInvocationPlan:
    """Return the native delivery invocation plan for one run input."""
    raw_callback_batch_size = getattr(callback, "native_callback_batch_size", None)
    callback_batch_size = None if raw_callback_batch_size is None else int(raw_callback_batch_size)
    return _core.plan_bgen_delivery_invocation(
        callback_batch_size,
        variant_major_packed8_probability_pairs,
        run_input.native_multi_aligned_sample_data is not None,
        run_input.native_aligned_sample_data is not None,
    )


def execute_bgen_delivery_cleanup_plan(
    *,
    cleanup_plan: _core.NativeBgenDeliveryCleanupPlan,
    callback_finished: bool,
    callback: object,
    writer_sessions: tuple[typing.Any, ...],
    writer_finish_thread_count: int,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    shutdown_request: shutdown.GracefulShutdownRequested | None,
) -> BgenDeliveryCleanupExecution:
    """Execute cleanup side effects in the native lifecycle action order."""
    final_parquet_paths: tuple[Path | None, ...] = ()
    resolved_callback_finished = callback_finished
    for cleanup_action_value in cleanup_plan.cleanup_actions:
        cleanup_action = BgenDeliveryCleanupAction(cleanup_action_value)
        if cleanup_action is BgenDeliveryCleanupAction.DRAIN_CALLBACK:
            writers.finish_callback_drain(callback=callback, stage_timing_recorder=stage_timing_recorder)
            resolved_callback_finished = True
        elif cleanup_action is BgenDeliveryCleanupAction.FINISH_WRITER_SESSIONS:
            final_parquet_paths = writers.finish_writer_sessions(
                writer_sessions=writer_sessions,
                writer_finish_thread_count=writer_finish_thread_count,
                stage_timing_recorder=stage_timing_recorder,
            )
        elif cleanup_action is BgenDeliveryCleanupAction.FINISH_INTERRUPTED_WRITER_SESSIONS:
            if shutdown_request is None:
                message = "Interrupted writer cleanup requires a shutdown request."
                raise RuntimeError(message)
            writers.finish_writer_sessions_interrupted(
                writer_sessions=writer_sessions,
                shutdown_request=shutdown_request,
                writer_finish_thread_count=writer_finish_thread_count,
                stage_timing_recorder=stage_timing_recorder,
            )
        elif cleanup_action is BgenDeliveryCleanupAction.ABORT_CALLBACK:
            writers.abort_callback(callback)
        elif cleanup_action is BgenDeliveryCleanupAction.ABORT_WRITER_SESSIONS:
            writers.abort_writer_sessions(writer_sessions)
        elif cleanup_action is BgenDeliveryCleanupAction.WRITE_STAGE_TIMING_SNAPSHOT:
            # The runner writes final timing outputs once after dispatch.
            continue
    return BgenDeliveryCleanupExecution(
        final_parquet_paths=final_parquet_paths,
        callback_finished=resolved_callback_finished,
    )


def run_variant_major_packed8_delivery(
    *,
    engine: _core.Regenie2RunEngine,
    run_input: models.BgenDeliveryRunInputProtocol,
    callback: object,
    committed_chunk_identifier_list: list[int],
) -> int:
    """Run packed8 delivery using native sample alignment when available."""
    invocation_plan = plan_bgen_delivery_invocation(
        callback,
        run_input,
        variant_major_packed8_probability_pairs=True,
    )
    delivery_method = BgenDeliveryMethod(invocation_plan.delivery_method)
    if delivery_method is BgenDeliveryMethod.PACKED8_NATIVE_MULTI_ALIGNED_SAMPLES:
        native_multi_aligned_sample_data = typing.cast(
            "_core.NativeMultiAlignedSampleData",
            run_input.native_multi_aligned_sample_data,
        )
        return int(
            engine.run_bgen_variant_major_packed8_probability_pair_buffered_chunks_for_native_multi_aligned_samples(
                native_multi_aligned_sample_data,
                callback,
                committed_chunk_identifiers=committed_chunk_identifier_list,
            )
        )
    if delivery_method is BgenDeliveryMethod.PACKED8_NATIVE_ALIGNED_SAMPLES:
        native_aligned_sample_data = typing.cast("_core.NativeAlignedSampleData", run_input.native_aligned_sample_data)
        return int(
            engine.run_bgen_variant_major_packed8_probability_pair_buffered_chunks_for_native_aligned_samples(
                native_aligned_sample_data,
                callback,
                committed_chunk_identifiers=committed_chunk_identifier_list,
            )
        )
    return int(
        engine.run_bgen_variant_major_packed8_probability_pair_buffered_chunks(
            run_input.sample_indices,
            callback,
            committed_chunk_identifiers=committed_chunk_identifier_list,
        )
    )


def run_variant_major_dosage_delivery(
    *,
    engine: _core.Regenie2RunEngine,
    run_input: models.BgenDeliveryRunInputProtocol,
    callback: object,
    committed_chunk_identifier_list: list[int],
) -> int:
    """Run dosage delivery using native sample alignment when available."""
    invocation_plan = plan_bgen_delivery_invocation(
        callback,
        run_input,
        variant_major_packed8_probability_pairs=False,
    )
    native_callback_batch_size = int(invocation_plan.callback_batch_size)
    delivery_method = BgenDeliveryMethod(invocation_plan.delivery_method)
    if delivery_method is BgenDeliveryMethod.DOSAGE_NATIVE_MULTI_ALIGNED_SAMPLES:
        native_multi_aligned_sample_data = typing.cast(
            "_core.NativeMultiAlignedSampleData",
            run_input.native_multi_aligned_sample_data,
        )
        return int(
            engine.run_bgen_variant_major_dosage_buffered_chunks_for_native_multi_aligned_samples(
                native_multi_aligned_sample_data,
                callback,
                committed_chunk_identifiers=committed_chunk_identifier_list,
                callback_batch_size=native_callback_batch_size,
            )
        )
    if delivery_method is BgenDeliveryMethod.DOSAGE_NATIVE_ALIGNED_SAMPLES:
        native_aligned_sample_data = typing.cast("_core.NativeAlignedSampleData", run_input.native_aligned_sample_data)
        return int(
            engine.run_bgen_variant_major_dosage_buffered_chunks_for_native_aligned_samples(
                native_aligned_sample_data,
                callback,
                committed_chunk_identifiers=committed_chunk_identifier_list,
                callback_batch_size=native_callback_batch_size,
            )
        )
    return int(
        engine.run_bgen_variant_major_dosage_buffered_chunks(
            run_input.sample_indices,
            callback,
            committed_chunk_identifiers=committed_chunk_identifier_list,
            callback_batch_size=native_callback_batch_size,
        )
    )


def run_bgen_engine_with_writer_sessions(
    *,
    engine: _core.Regenie2RunEngine,
    run_input: models.BgenDeliveryRunInputProtocol,
    committed_chunk_identifiers: set[int] | None,
    writer_sessions: tuple[typing.Any, ...],
    callback: object,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    writer_finish_thread_count: int,
    variant_major_packed8_probability_pairs: bool,
    pipeline_label: str,
) -> tuple[Path | None, ...]:
    """Run native BGEN chunk delivery and close all output writers."""
    callback_finished = False
    final_parquet_paths: tuple[Path | None, ...] = ()
    try:
        if stage_timing_recorder is not None:
            engine.reset_profile()
        engine_delivery_start_time = time.perf_counter()
        committed_chunk_identifier_list = sorted(committed_chunk_identifiers or set())
        _core.record_native_dispatch_delivery_started_diagnostic_event(
            committed_chunk_count=len(committed_chunk_identifier_list),
            pipeline_label=pipeline_label,
            variant_major_packed8_probability_pairs=variant_major_packed8_probability_pairs,
        )
        writers.start_callback(callback)
        if variant_major_packed8_probability_pairs:
            processed_chunk_count = run_variant_major_packed8_delivery(
                engine=engine,
                run_input=run_input,
                callback=callback,
                committed_chunk_identifier_list=committed_chunk_identifier_list,
            )
        else:
            processed_chunk_count = run_variant_major_dosage_delivery(
                engine=engine,
                run_input=run_input,
                callback=callback,
                committed_chunk_identifier_list=committed_chunk_identifier_list,
            )
        timing.record_stage_duration(stage_timing_recorder, "native_engine_delivery", engine_delivery_start_time)
        _core.record_native_dispatch_delivery_finished_diagnostic_event(
            pipeline_label=pipeline_label,
            processed_chunk_count=processed_chunk_count,
        )
        if stage_timing_recorder is not None:
            stage_timing_recorder.set_native_bgen_profile(engine.profile_snapshot())
        cleanup_plan = plan_bgen_delivery_cleanup(
            cleanup_outcome=BgenDeliveryCleanupOutcome.SUCCESS,
            callback_finished=callback_finished,
        )
        cleanup_execution = execute_bgen_delivery_cleanup_plan(
            cleanup_plan=cleanup_plan,
            callback_finished=callback_finished,
            callback=callback,
            writer_sessions=writer_sessions,
            writer_finish_thread_count=writer_finish_thread_count,
            stage_timing_recorder=stage_timing_recorder,
            shutdown_request=None,
        )
        callback_finished = cleanup_execution.callback_finished
        final_parquet_paths = cleanup_execution.final_parquet_paths
    except shutdown.GracefulShutdownRequested as shutdown_request:
        _core.record_native_dispatch_delivery_interrupted_diagnostic_event(
            pipeline_label=pipeline_label,
            signal_exit_code=shutdown_request.exit_code,
            signal_name=shutdown_request.signal_name,
            signal_number=shutdown_request.shutdown_signal.number,
        )
        cleanup_plan = plan_bgen_delivery_cleanup(
            cleanup_outcome=BgenDeliveryCleanupOutcome.INTERRUPTED,
            callback_finished=callback_finished,
        )
        try:
            cleanup_execution = execute_bgen_delivery_cleanup_plan(
                cleanup_plan=cleanup_plan,
                callback_finished=callback_finished,
                callback=callback,
                writer_sessions=writer_sessions,
                writer_finish_thread_count=writer_finish_thread_count,
                stage_timing_recorder=stage_timing_recorder,
                shutdown_request=shutdown_request,
            )
            callback_finished = cleanup_execution.callback_finished
        except BaseException:
            cleanup_failure_plan = plan_bgen_delivery_cleanup(
                cleanup_outcome=BgenDeliveryCleanupOutcome.INTERRUPTED_CLEANUP_FAILURE,
                callback_finished=callback_finished,
            )
            execute_bgen_delivery_cleanup_plan(
                cleanup_plan=cleanup_failure_plan,
                callback_finished=callback_finished,
                callback=callback,
                writer_sessions=writer_sessions,
                writer_finish_thread_count=writer_finish_thread_count,
                stage_timing_recorder=stage_timing_recorder,
                shutdown_request=shutdown_request,
            )
            raise
        raise
    except BaseException as exception:
        _core.record_native_dispatch_delivery_failed_diagnostic_event(
            exception_message=str(exception),
            exception_type=type(exception).__name__,
            pipeline_label=pipeline_label,
        )
        cleanup_plan = plan_bgen_delivery_cleanup(
            cleanup_outcome=BgenDeliveryCleanupOutcome.FAILURE,
            callback_finished=callback_finished,
        )
        execute_bgen_delivery_cleanup_plan(
            cleanup_plan=cleanup_plan,
            callback_finished=callback_finished,
            callback=callback,
            writer_sessions=writer_sessions,
            writer_finish_thread_count=writer_finish_thread_count,
            stage_timing_recorder=stage_timing_recorder,
            shutdown_request=None,
        )
        raise
    _core.record_native_dispatch_pipeline_finished_diagnostic_event(
        final_parquet_path_count=len(final_parquet_paths),
        pipeline_label=pipeline_label,
    )
    return final_parquet_paths


def run_bgen_engine_with_callback(
    *,
    engine: _core.Regenie2RunEngine,
    run_input: models.NativeBgenRunInput,
    committed_chunk_identifiers: set[int] | None,
    writer_session: typing.Any,
    callback: object,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    variant_major_packed8_probability_pairs: bool,
) -> Path | None:
    """Run native BGEN chunk delivery and close the output writer."""
    final_parquet_paths = run_bgen_engine_with_writer_sessions(
        engine=engine,
        run_input=run_input,
        committed_chunk_identifiers=committed_chunk_identifiers,
        writer_sessions=(writer_session,),
        callback=callback,
        stage_timing_recorder=stage_timing_recorder,
        writer_finish_thread_count=1,
        variant_major_packed8_probability_pairs=variant_major_packed8_probability_pairs,
        pipeline_label="Native BGEN",
    )
    return final_parquet_paths[0]
