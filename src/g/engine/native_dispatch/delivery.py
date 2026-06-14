"""Native BGEN chunk delivery helpers."""

from __future__ import annotations

import logging
import time
import typing

from g.engine import shutdown, timing
from g.engine.native_dispatch import models, writers

if typing.TYPE_CHECKING:
    from pathlib import Path

    from g import _core

logger = logging.getLogger(__name__)


def run_variant_major_packed8_delivery(
    *,
    engine: _core.Regenie2RunEngine,
    run_input: models.BgenDeliveryRunInputProtocol,
    callback: object,
    committed_chunk_identifier_list: list[int],
) -> int:
    """Run packed8 delivery using native sample alignment when available."""
    native_multi_aligned_sample_data = getattr(run_input, "native_multi_aligned_sample_data", None)
    if native_multi_aligned_sample_data is not None:
        return int(
            engine.run_bgen_variant_major_packed8_probability_pair_buffered_chunks_for_native_multi_aligned_samples(
                native_multi_aligned_sample_data,
                callback,
                committed_chunk_identifiers=committed_chunk_identifier_list,
            )
        )
    native_aligned_sample_data = getattr(run_input, "native_aligned_sample_data", None)
    if native_aligned_sample_data is not None:
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
    native_multi_aligned_sample_data = getattr(run_input, "native_multi_aligned_sample_data", None)
    if native_multi_aligned_sample_data is not None:
        return int(
            engine.run_bgen_variant_major_dosage_buffered_chunks_for_native_multi_aligned_samples(
                native_multi_aligned_sample_data,
                callback,
                committed_chunk_identifiers=committed_chunk_identifier_list,
            )
        )
    native_aligned_sample_data = getattr(run_input, "native_aligned_sample_data", None)
    if native_aligned_sample_data is not None:
        return int(
            engine.run_bgen_variant_major_dosage_buffered_chunks_for_native_aligned_samples(
                native_aligned_sample_data,
                callback,
                committed_chunk_identifiers=committed_chunk_identifier_list,
            )
        )
    return int(
        engine.run_bgen_variant_major_dosage_buffered_chunks(
            run_input.sample_indices,
            callback,
            committed_chunk_identifiers=committed_chunk_identifier_list,
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
    stage_timing_snapshot_writer: typing.Callable[[timing.StageTimingRecorder | None, Path | None], None],
) -> tuple[Path | None, ...]:
    """Run native BGEN chunk delivery and close all output writers."""
    callback_finished = False
    try:
        if stage_timing_recorder is not None:
            engine.reset_profile()
        engine_delivery_start_time = time.perf_counter()
        committed_chunk_identifier_list = sorted(committed_chunk_identifiers or set())
        logger.debug(
            "Starting %s delivery: committed_chunk_count=%s variant_major_packed8_probability_pairs=%s.",
            pipeline_label,
            len(committed_chunk_identifier_list),
            variant_major_packed8_probability_pairs,
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
        logger.debug("%s delivery finished: processed_chunk_count=%s.", pipeline_label, processed_chunk_count)
        if stage_timing_recorder is not None:
            stage_timing_recorder.set_native_bgen_profile(engine.profile_snapshot())
        writers.finish_callback_drain(callback=callback, stage_timing_recorder=stage_timing_recorder)
        callback_finished = True
        final_parquet_paths = writers.finish_writer_sessions(
            writer_sessions=writer_sessions,
            writer_finish_thread_count=writer_finish_thread_count,
            stage_timing_recorder=stage_timing_recorder,
        )
    except shutdown.GracefulShutdownRequested as shutdown_request:
        logger.info("%s delivery interrupted by %s.", pipeline_label, shutdown_request.signal_name)
        try:
            if not callback_finished:
                writers.finish_callback_drain(callback=callback, stage_timing_recorder=stage_timing_recorder)
            writers.finish_writer_sessions_interrupted(
                writer_sessions=writer_sessions,
                shutdown_request=shutdown_request,
                writer_finish_thread_count=writer_finish_thread_count,
                stage_timing_recorder=stage_timing_recorder,
            )
        except BaseException:
            writers.abort_callback(callback)
            writers.abort_writer_sessions(writer_sessions)
            stage_timing_snapshot_writer(stage_timing_recorder, None)
            raise
        stage_timing_snapshot_writer(stage_timing_recorder, None)
        raise
    except BaseException:
        logger.exception("%s delivery failed.", pipeline_label)
        writers.abort_callback(callback)
        writers.abort_writer_sessions(writer_sessions)
        stage_timing_snapshot_writer(stage_timing_recorder, None)
        raise
    stage_timing_snapshot_writer(stage_timing_recorder, None)
    logger.info("%s pipeline finished.", pipeline_label)
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
    stage_timing_snapshot_writer: typing.Callable[[timing.StageTimingRecorder | None, Path | None], None],
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
        stage_timing_snapshot_writer=stage_timing_snapshot_writer,
    )
    return final_parquet_paths[0]
