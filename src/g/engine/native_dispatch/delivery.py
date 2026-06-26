"""Native BGEN chunk delivery helpers."""

from __future__ import annotations

import enum
import logging
import time
import typing

from g import _core
from g.engine import shutdown, timing
from g.engine.native_dispatch import models, writers

if typing.TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class BgenDeliveryMethod(enum.StrEnum):
    """Native BGEN delivery method selected by engine scheduling policy."""

    DOSAGE_NATIVE_MULTI_ALIGNED_SAMPLES = "dosage_native_multi_aligned_samples"
    DOSAGE_NATIVE_ALIGNED_SAMPLES = "dosage_native_aligned_samples"
    DOSAGE_SAMPLE_INDICES = "dosage_sample_indices"
    PACKED8_NATIVE_MULTI_ALIGNED_SAMPLES = "packed8_native_multi_aligned_samples"
    PACKED8_NATIVE_ALIGNED_SAMPLES = "packed8_native_aligned_samples"
    PACKED8_SAMPLE_INDICES = "packed8_sample_indices"


def resolve_native_callback_batch_size(
    callback: object,
    *,
    variant_major_packed8_probability_pairs: bool,
) -> int:
    """Return the validated native callback batch size for one callback object."""
    raw_callback_batch_size = getattr(callback, "native_callback_batch_size", None)
    callback_batch_size = None if raw_callback_batch_size is None else int(raw_callback_batch_size)
    return int(_core.resolve_delivery_callback_batch_size(callback_batch_size, variant_major_packed8_probability_pairs))


def resolve_bgen_delivery_method(
    run_input: models.BgenDeliveryRunInputProtocol,
    *,
    variant_major_packed8_probability_pairs: bool,
) -> BgenDeliveryMethod:
    """Resolve the native BGEN delivery method for one run input."""
    native_multi_aligned_sample_data = getattr(run_input, "native_multi_aligned_sample_data", None)
    native_aligned_sample_data = getattr(run_input, "native_aligned_sample_data", None)
    return BgenDeliveryMethod(
        _core.resolve_bgen_delivery_method_value(
            variant_major_packed8_probability_pairs,
            native_multi_aligned_sample_data is not None,
            native_aligned_sample_data is not None,
        )
    )


def run_variant_major_packed8_delivery(
    *,
    engine: _core.Regenie2RunEngine,
    run_input: models.BgenDeliveryRunInputProtocol,
    callback: object,
    committed_chunk_identifier_list: list[int],
) -> int:
    """Run packed8 delivery using native sample alignment when available."""
    resolve_native_callback_batch_size(callback, variant_major_packed8_probability_pairs=True)
    native_multi_aligned_sample_data = getattr(run_input, "native_multi_aligned_sample_data", None)
    native_aligned_sample_data = getattr(run_input, "native_aligned_sample_data", None)
    delivery_method = resolve_bgen_delivery_method(run_input, variant_major_packed8_probability_pairs=True)
    if delivery_method is BgenDeliveryMethod.PACKED8_NATIVE_MULTI_ALIGNED_SAMPLES:
        return int(
            engine.run_bgen_variant_major_packed8_probability_pair_buffered_chunks_for_native_multi_aligned_samples(
                native_multi_aligned_sample_data,
                callback,
                committed_chunk_identifiers=committed_chunk_identifier_list,
            )
        )
    if delivery_method is BgenDeliveryMethod.PACKED8_NATIVE_ALIGNED_SAMPLES:
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
    native_callback_batch_size = resolve_native_callback_batch_size(
        callback, variant_major_packed8_probability_pairs=False
    )
    native_multi_aligned_sample_data = getattr(run_input, "native_multi_aligned_sample_data", None)
    native_aligned_sample_data = getattr(run_input, "native_aligned_sample_data", None)
    delivery_method = resolve_bgen_delivery_method(run_input, variant_major_packed8_probability_pairs=False)
    if delivery_method is BgenDeliveryMethod.DOSAGE_NATIVE_MULTI_ALIGNED_SAMPLES:
        return int(
            engine.run_bgen_variant_major_dosage_buffered_chunks_for_native_multi_aligned_samples(
                native_multi_aligned_sample_data,
                callback,
                committed_chunk_identifiers=committed_chunk_identifier_list,
                callback_batch_size=native_callback_batch_size,
            )
        )
    if delivery_method is BgenDeliveryMethod.DOSAGE_NATIVE_ALIGNED_SAMPLES:
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
