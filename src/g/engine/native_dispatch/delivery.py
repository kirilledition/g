"""Native BGEN chunk delivery helpers."""

from __future__ import annotations

import typing
from pathlib import Path

from g import _core

if typing.TYPE_CHECKING:
    from g.engine import timing as engine_timing
    from g.engine.native_dispatch import models


def run_bgen_engine_with_writer_sessions(
    *,
    engine: _core.Regenie2RunEngine,
    run_input: models.BgenDeliveryRunInputProtocol,
    committed_chunk_identifiers: set[int] | None,
    writer_sessions: tuple[typing.Any, ...],
    callback: models.BgenDeliveryCallbackProtocol,
    stage_timing_recorder: engine_timing.StageTimingRecorder | None,
    writer_finish_thread_count: int,
    variant_major_packed8_probability_pairs: bool,
    pipeline_label: str,
) -> tuple[Path | None, ...]:
    """Run native BGEN chunk delivery and close all output writers."""
    final_parquet_path_values = _core.run_bgen_delivery_with_writer_sessions(
        engine,
        run_input.sample_indices,
        run_input.native_aligned_sample_data,
        run_input.native_multi_aligned_sample_data,
        typing.cast("tuple[_core.OutputWriterSession, ...]", writer_sessions),
        callback,
        None if stage_timing_recorder is None else stage_timing_recorder.native_recorder,
        writer_finish_thread_count,
        sorted(committed_chunk_identifiers or set()),
        variant_major_packed8_probability_pairs,
        pipeline_label,
    )
    return tuple(
        None if final_parquet_path is None else Path(final_parquet_path)
        for final_parquet_path in final_parquet_path_values
    )


def run_bgen_engine_with_callback(
    *,
    engine: _core.Regenie2RunEngine,
    run_input: models.NativeBgenRunInput,
    committed_chunk_identifiers: set[int] | None,
    writer_session: typing.Any,
    callback: models.BgenDeliveryCallbackProtocol,
    stage_timing_recorder: engine_timing.StageTimingRecorder | None,
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
