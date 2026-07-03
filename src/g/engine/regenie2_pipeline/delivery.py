"""Native BGEN delivery helpers for REGENIE step 2 pipelines."""

from __future__ import annotations

import typing

from g.engine.native_dispatch import delivery as native_dispatch_delivery

if typing.TYPE_CHECKING:
    from pathlib import Path

    from g import _core
    from g.engine.regenie2_pipeline import inputs
    from g.runner import timing


def run_bgen_engine_with_callback(
    *,
    engine: _core.Regenie2RunEngine,
    run_input: inputs.NativeBgenRunInput,
    committed_chunk_identifiers: set[int] | None,
    writer_session: _core.OutputWriterSession,
    callback: inputs.BgenDeliveryCallbackProtocol,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    variant_major_packed8_probability_pairs: bool,
) -> Path | None:
    """Run native BGEN chunk delivery and close the output writer."""
    return native_dispatch_delivery.run_bgen_engine_with_callback(
        engine=engine,
        run_input=run_input,
        committed_chunk_identifiers=committed_chunk_identifiers,
        writer_session=writer_session,
        callback=callback,
        stage_timing_recorder=stage_timing_recorder,
        variant_major_packed8_probability_pairs=variant_major_packed8_probability_pairs,
    )


def run_bgen_engine_with_writer_sessions(
    *,
    engine: _core.Regenie2RunEngine,
    run_input: inputs.NativeBgenMultiRunInput | inputs.NativeBgenUnionRunInput,
    committed_chunk_identifiers: set[int] | None,
    writer_sessions: tuple[_core.OutputWriterSession, ...],
    callback: inputs.BgenDeliveryCallbackProtocol,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    writer_finish_thread_count: int,
    variant_major_packed8_probability_pairs: bool,
    pipeline_label: str,
) -> tuple[Path | None, ...]:
    """Run native BGEN chunk delivery and close all output writers."""
    return native_dispatch_delivery.run_bgen_engine_with_writer_sessions(
        engine=engine,
        run_input=run_input,
        committed_chunk_identifiers=committed_chunk_identifiers,
        writer_sessions=writer_sessions,
        callback=callback,
        stage_timing_recorder=stage_timing_recorder,
        writer_finish_thread_count=writer_finish_thread_count,
        variant_major_packed8_probability_pairs=variant_major_packed8_probability_pairs,
        pipeline_label=pipeline_label,
    )
