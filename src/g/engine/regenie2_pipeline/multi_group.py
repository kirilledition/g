"""Prepared multi-phenotype group delivery helpers."""

from __future__ import annotations

import time
import typing

from g import _core
from g.engine import timing as engine_timing
from g.engine.native_dispatch import delivery as native_dispatch_delivery
from g.engine.native_dispatch import models as native_dispatch_models
from g.engine.regenie2_pipeline import (
    callbacks,
    outputs,
    preflight,
)
from g.engine.regenie2_pipeline import context as pipeline_context
from g.runner import events

if typing.TYPE_CHECKING:
    from pathlib import Path

    from g import execution_plan, types


def prepare_multi_phenotype_bgen_group_delivery(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    engine: _core.Regenie2RunEngine,
    run_input: native_dispatch_models.NativeBgenMultiRunInput,
    prediction_source: typing.Any,
    compute_group: execution_plan.PhenotypeComputeGroup,
    prepared_runs_by_phenotype: tuple[_core.NativeRunLifecyclePhenotypeRun, ...],
    staging_depth: int,
    native_callback_batch_size: int,
    result_in_flight_limit: int | None,
    dosage_buffer_limit: int | None,
    null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy,
    output_sample_mode: types.MultiPhenotypeSampleMode,
) -> pipeline_context.PreparedMultiPhenotypeGroupDelivery:
    """Prepare one compatible phenotype group for native BGEN delivery."""
    events.record_prediction_source_loaded_telemetry(
        context.telemetry_session,
        context.association_mode.value,
        None,
        len(run_input.phenotype_names),
    )
    preflight_start_time = time.perf_counter()
    _core.record_pipeline_multi_group_preflight_started_diagnostic_event(
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
    _core.record_pipeline_multi_group_preflight_completed_diagnostic_event(
        phenotype_count=len(run_input.phenotype_names),
        sample_count=int(run_input.sample_indices.shape[0]),
        trusted_no_missing_diploid=context.effective_trusted_no_missing_diploid,
        variant_limit=context.variant_limit,
    )
    events.record_multi_phenotype_preflight_completed_telemetry(
        context.telemetry_session,
        context.association_mode.value,
        len(run_input.phenotype_names),
        int(run_input.sample_indices.shape[0]),
    )
    current_headers = tuple(
        outputs.build_pipeline_manifest_header(
            context=context,
            phenotype_name=phenotype_name,
            covariate_names=tuple(run_input.native_multi_aligned_sample_data.covariate_names),
            sample_count=int(run_input.sample_indices.shape[0]),
            variant_count=int(engine.variant_count),
            multi_phenotype_sample_mode=output_sample_mode,
            phenotype_compute_group=compute_group,
        )
        for phenotype_name in compute_group.phenotype_names
    )
    initialized_outputs = outputs.initialize_pipeline_output_runs(
        context=context,
        phenotype_names=compute_group.phenotype_names,
        current_headers_by_trait=current_headers,
    )
    writer_sessions = outputs.create_pipeline_writer_sessions(
        context=context,
        prepared_runs_by_trait=prepared_runs_by_phenotype,
    )
    writer_session_tuple = writer_sessions
    chunk_write_planner = initialized_outputs.multi_trait_chunk_write_planner(len(writer_session_tuple))
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
        output_initialization=initialized_outputs,
    )


def run_prepared_multi_phenotype_bgen_group(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    engine: _core.Regenie2RunEngine,
    run_input: native_dispatch_models.NativeBgenMultiRunInput,
    prediction_source: typing.Any,
    compute_group: execution_plan.PhenotypeComputeGroup,
    prepared_runs_by_phenotype: tuple[_core.NativeRunLifecyclePhenotypeRun, ...],
    staging_depth: int,
    native_callback_batch_size: int,
    result_in_flight_limit: int | None,
    dosage_buffer_limit: int | None,
    null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy,
    output_sample_mode: types.MultiPhenotypeSampleMode,
) -> tuple[Path | None, ...]:
    """Run one prepared compatible phenotype group through one BGEN pass."""
    prepared_delivery = prepare_multi_phenotype_bgen_group_delivery(
        context=context,
        engine=engine,
        run_input=run_input,
        prediction_source=prediction_source,
        compute_group=compute_group,
        prepared_runs_by_phenotype=prepared_runs_by_phenotype,
        staging_depth=staging_depth,
        native_callback_batch_size=native_callback_batch_size,
        result_in_flight_limit=result_in_flight_limit,
        dosage_buffer_limit=dosage_buffer_limit,
        null_logistic_nonconvergence_policy=null_logistic_nonconvergence_policy,
        output_sample_mode=output_sample_mode,
    )
    return native_dispatch_delivery.run_bgen_engine_with_writer_sessions(
        engine=engine,
        run_input=run_input,
        committed_chunk_identifiers=outputs.shared_committed_chunk_identifiers(prepared_delivery.output_initialization),
        writer_sessions=prepared_delivery.writer_sessions,
        callback=prepared_delivery.callback,
        stage_timing_recorder=context.stage_timing_recorder,
        writer_finish_thread_count=context.writer_settings.writer_thread_count,
        variant_major_packed8_probability_pairs=context.uses_packed8_genotypes,
        pipeline_label="Multi-phenotype native BGEN",
    )
