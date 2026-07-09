"""Prepared multi-phenotype group delivery helpers."""

from __future__ import annotations

import time
import typing

from g import _core
from g.engine import timing as engine_timing
from g.engine.native_dispatch import delivery as native_dispatch_delivery
from g.engine.native_dispatch import models as native_dispatch_models
from g.engine.regenie2_pipeline import callbacks, outputs, preflight
from g.engine.regenie2_pipeline import context as pipeline_context

if typing.TYPE_CHECKING:
    from pathlib import Path

    from g import execution_plan, types

RUN_EVENT_RECORDER: _core.NativeRunEventRecorder = _core.NativeRunEventRecorder()


def prepare_multi_phenotype_output_bundle(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    engine: native_dispatch_models.NativeBgenEngineProtocol,
    run_input: native_dispatch_models.NativeBgenMultiRunInput,
    compute_group: execution_plan.PhenotypeComputeGroup,
    output_sample_mode: types.MultiPhenotypeSampleMode,
) -> _core.NativePreparedOutputBundle:
    """Prepare output runs and writer sessions for a compatible phenotype group."""
    output_group = outputs.build_output_preparation_group(
        phenotype_names=compute_group.phenotype_names,
        covariate_names=tuple(run_input.native_multi_aligned_sample_data.covariate_names),
        sample_count=int(run_input.sample_indices.shape[0]),
        output_sample_mode=output_sample_mode,
        phenotype_compute_group=compute_group,
    )
    return outputs.prepare_output_bundles(context=context, engine=engine, output_groups=(output_group,))[0]


def run_multi_phenotype_group_preflight(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    engine: native_dispatch_models.NativeBgenEngineProtocol,
    run_input: native_dispatch_models.NativeBgenMultiRunInput,
    prediction_source: typing.Any,
) -> None:
    """Run native preflight for one compatible phenotype group."""
    _core.record_prediction_source_loaded_telemetry(
        context.telemetry_session,
        context.association_mode.value,
        None,
        len(run_input.phenotype_names),
    )
    preflight_start_time = time.perf_counter()
    RUN_EVENT_RECORDER.pipeline_multi_group_preflight_started(
        phenotype_count=len(run_input.phenotype_names),
        sample_count=int(run_input.sample_indices.shape[0]),
        trusted_no_missing_diploid=context.effective_trusted_no_missing_diploid,
        variant_limit=context.variant_limit,
    )
    preflight.run_regenie2_multi_preflight(
        run_input=run_input,
        prediction_source=prediction_source,
        engine=engine,
        variant_limit=context.variant_limit,
        is_binary_trait=run_input.is_binary_trait,
        trusted_no_missing_diploid=context.effective_trusted_no_missing_diploid,
    )
    engine_timing.record_stage_duration(context.stage_timing_recorder, "preflight_validation", preflight_start_time)
    RUN_EVENT_RECORDER.pipeline_multi_group_preflight_completed(
        phenotype_count=len(run_input.phenotype_names),
        sample_count=int(run_input.sample_indices.shape[0]),
        trusted_no_missing_diploid=context.effective_trusted_no_missing_diploid,
        variant_limit=context.variant_limit,
    )
    _core.record_multi_phenotype_preflight_completed_telemetry(
        context.telemetry_session,
        context.association_mode.value,
        len(run_input.phenotype_names),
        int(run_input.sample_indices.shape[0]),
    )


def prepare_multi_phenotype_bgen_group_delivery(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    run_input: native_dispatch_models.NativeBgenMultiRunInput,
    prediction_source: typing.Any,
    compute_group: execution_plan.PhenotypeComputeGroup,
    output_bundle: _core.NativePreparedOutputBundle,
    staging_depth: int,
    native_callback_batch_size: int,
    result_in_flight_limit: int | None,
    dosage_buffer_limit: int | None,
    null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy,
) -> pipeline_context.PreparedMultiPhenotypeGroupDelivery:
    """Prepare one compatible phenotype group callback for native BGEN delivery."""
    writer_session_tuple = output_bundle.writer_sessions
    chunk_write_planner = output_bundle.multi_trait_chunk_write_planner()
    callback = callbacks.build_multi_phenotype_group_callback(
        context=context,
        run_input=run_input,
        prediction_source=prediction_source,
        writer_sessions=writer_session_tuple,
        chunk_write_planner=chunk_write_planner,
        staging_depth=staging_depth,
        native_callback_batch_size=native_callback_batch_size,
        result_in_flight_limit=result_in_flight_limit,
        dosage_buffer_limit=dosage_buffer_limit,
        null_logistic_nonconvergence_policy=null_logistic_nonconvergence_policy,
    )
    return pipeline_context.PreparedMultiPhenotypeGroupDelivery(
        compute_group=compute_group,
        phenotype_indices=compute_group.phenotype_indices,
        run_input=run_input,
        callback=callback,
        writer_sessions=writer_session_tuple,
        output_bundle=output_bundle,
    )


def run_prepared_multi_phenotype_bgen_group(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    engine: native_dispatch_models.NativeBgenEngineProtocol,
    run_input: native_dispatch_models.NativeBgenMultiRunInput,
    prediction_source: typing.Any,
    compute_group: execution_plan.PhenotypeComputeGroup,
    output_bundle: _core.NativePreparedOutputBundle,
    staging_depth: int,
    native_callback_batch_size: int,
    result_in_flight_limit: int | None,
    dosage_buffer_limit: int | None,
    null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy,
) -> tuple[Path | None, ...]:
    """Run one prepared compatible phenotype group through one BGEN pass."""
    prepared_delivery = prepare_multi_phenotype_bgen_group_delivery(
        context=context,
        run_input=run_input,
        prediction_source=prediction_source,
        compute_group=compute_group,
        output_bundle=output_bundle,
        staging_depth=staging_depth,
        native_callback_batch_size=native_callback_batch_size,
        result_in_flight_limit=result_in_flight_limit,
        dosage_buffer_limit=dosage_buffer_limit,
        null_logistic_nonconvergence_policy=null_logistic_nonconvergence_policy,
    )
    return native_dispatch_delivery.run_bgen_engine_with_writer_sessions(
        engine=engine,
        run_input=run_input,
        committed_chunk_identifiers={
            int(chunk_identifier)
            for chunk_identifier in prepared_delivery.output_bundle.shared_committed_chunk_identifiers()
        },
        writer_sessions=prepared_delivery.writer_sessions,
        callback=prepared_delivery.callback,
        stage_timing_recorder=context.stage_timing_recorder,
        writer_finish_thread_count=context.writer_settings.writer_thread_count,
        variant_major_packed8_probability_pairs=context.uses_packed8_genotypes,
        pipeline_label="Multi-phenotype native BGEN",
    )
