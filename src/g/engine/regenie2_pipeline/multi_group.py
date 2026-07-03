"""Prepared multi-phenotype group delivery helpers."""

from __future__ import annotations

import time
import typing

import g.engine.callbacks.binary as callback_binary
import g.engine.callbacks.linear as callback_linear
from g import _core, execution_plan, types
from g.engine.native_dispatch import delivery as native_dispatch_delivery
from g.engine.regenie2_pipeline import context as pipeline_context
from g.engine.regenie2_pipeline import inputs, outputs, preflight, telemetry_events, timing

if typing.TYPE_CHECKING:
    from pathlib import Path


def intersect_committed_chunk_identifier_sets(
    committed_chunk_identifier_sets: tuple[set[int], ...],
) -> set[int]:
    """Return chunk identifiers already committed by every output in a delivery."""
    native_committed_chunk_identifier_sets = tuple(
        tuple(committed_chunk_identifier_set) for committed_chunk_identifier_set in committed_chunk_identifier_sets
    )
    return set(
        native_schedule_policy().intersect_committed_chunk_identifier_sets(native_committed_chunk_identifier_sets)
    )


def native_schedule_policy() -> _core.NativeSchedulePolicy:
    """Build the native schedule policy handle."""
    return _core.NativeSchedulePolicy()


def prepare_multi_phenotype_bgen_group_delivery(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    engine: _core.Regenie2RunEngine,
    run_input: inputs.NativeBgenMultiRunInput,
    prediction_source: typing.Any,
    compute_group: execution_plan.PhenotypeComputeGroup,
    output_run_paths_by_phenotype: tuple[outputs.OutputRunPaths, ...],
    staging_depth: int,
    native_callback_batch_size: int,
    result_in_flight_limit: int | None,
    dosage_buffer_limit: int | None,
    existing_manifests: tuple[dict[str, typing.Any] | None, ...],
    resume: bool,
    resume_mode: types.ResumeMode,
    null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy,
    output_sample_mode: outputs.MultiPhenotypeSampleMode,
) -> pipeline_context.PreparedMultiPhenotypeGroupDelivery:
    """Prepare one compatible phenotype group for native BGEN delivery."""
    telemetry_events.log_prediction_source_loaded(
        context=context,
        phenotype_name=None,
        phenotype_count=len(run_input.phenotype_names),
    )
    preflight_start_time = time.perf_counter()
    native_pipeline_diagnostic_policy = telemetry_events.native_pipeline_diagnostic_policy()
    native_pipeline_diagnostic_policy.record_pipeline_multi_group_preflight_started_diagnostic_event(
        phenotype_count=len(run_input.phenotype_names),
        sample_count=int(run_input.sample_indices.shape[0]),
        trusted_no_missing_diploid=context.effective_trusted_no_missing_diploid,
        variant_limit=context.variant_limit,
    )
    run_multi_preflight(
        run_input=run_input,
        prediction_source=prediction_source,
        engine=engine,
        variant_limit=context.variant_limit,
        trusted_no_missing_diploid=context.effective_trusted_no_missing_diploid,
    )
    timing.record_stage_duration(context.stage_timing_recorder, "preflight_validation", preflight_start_time)
    native_pipeline_diagnostic_policy.record_pipeline_multi_group_preflight_completed_diagnostic_event(
        phenotype_count=len(run_input.phenotype_names),
        sample_count=int(run_input.sample_indices.shape[0]),
        trusted_no_missing_diploid=context.effective_trusted_no_missing_diploid,
        variant_limit=context.variant_limit,
    )
    telemetry_events.native_run_event_telemetry_policy().record_multi_phenotype_preflight_completed_telemetry_event(
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
        output_run_paths_by_trait=output_run_paths_by_phenotype,
        existing_manifests_by_trait=existing_manifests,
        current_headers_by_trait=current_headers,
        resume=resume,
        resume_mode=resume_mode,
        runtime_compatibility_token=context.runtime_compatibility_token,
    )
    outputs.notify_output_runs_initialized(context=context, phenotype_names=compute_group.phenotype_names)
    committed_chunk_identifier_sets = initialized_outputs.committed_chunk_identifier_sets
    writer_sessions = outputs.create_pipeline_writer_sessions(
        context=context,
        output_run_paths_by_trait=output_run_paths_by_phenotype,
    )
    writer_session_tuple = writer_sessions
    if context.is_binary_trait:
        binary_kernel_config = pipeline_context.require_binary_kernel_config(context.binary_kernel_config)
        callback = callback_binary.MultiBinaryRegenie2PipelineCallback(
            run_input=run_input,
            prediction_source=prediction_source,
            writer_sessions=writer_session_tuple,
            committed_chunk_identifier_sets=committed_chunk_identifier_sets,
            correction_plan=context.correction_plan,
            kernel_config=binary_kernel_config,
            null_logistic_nonconvergence_policy=null_logistic_nonconvergence_policy,
            staging_depth=staging_depth,
            native_callback_batch_size=native_callback_batch_size,
            result_in_flight_limit=result_in_flight_limit,
            dosage_buffer_limit=dosage_buffer_limit,
            score_dtype=context.score_dtype,
            stage_timing_recorder=context.stage_timing_recorder,
            telemetry_session=context.telemetry_session,
            output_statistic_dtype=context.writer_settings.output_statistic_dtype,
        )
    else:
        callback = callback_linear.MultiLinearRegenie2PipelineCallback(
            run_input=run_input,
            prediction_source=prediction_source,
            writer_sessions=writer_session_tuple,
            committed_chunk_identifier_sets=committed_chunk_identifier_sets,
            staging_depth=staging_depth,
            native_callback_batch_size=native_callback_batch_size,
            result_in_flight_limit=result_in_flight_limit,
            dosage_buffer_limit=dosage_buffer_limit,
            score_dtype=context.score_dtype,
            linear_numerical_config=pipeline_context.require_linear_numerical_config(context.linear_numerical_config),
            stage_timing_recorder=context.stage_timing_recorder,
            telemetry_session=context.telemetry_session,
            output_statistic_dtype=context.writer_settings.output_statistic_dtype,
        )
    return pipeline_context.PreparedMultiPhenotypeGroupDelivery(
        compute_group=compute_group,
        phenotype_indices=compute_group.phenotype_indices,
        run_input=run_input,
        callback=callback,
        writer_sessions=writer_session_tuple,
        committed_chunk_identifier_sets=committed_chunk_identifier_sets,
    )


def run_prepared_multi_phenotype_bgen_group(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    engine: _core.Regenie2RunEngine,
    run_input: inputs.NativeBgenMultiRunInput,
    prediction_source: typing.Any,
    compute_group: execution_plan.PhenotypeComputeGroup,
    output_run_paths_by_phenotype: tuple[outputs.OutputRunPaths, ...],
    staging_depth: int,
    native_callback_batch_size: int,
    result_in_flight_limit: int | None,
    dosage_buffer_limit: int | None,
    existing_manifests: tuple[dict[str, typing.Any] | None, ...],
    resume: bool,
    resume_mode: types.ResumeMode,
    null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy,
    output_sample_mode: outputs.MultiPhenotypeSampleMode,
) -> tuple[Path | None, ...]:
    """Run one prepared compatible phenotype group through one BGEN pass."""
    prepared_delivery = prepare_multi_phenotype_bgen_group_delivery(
        context=context,
        engine=engine,
        run_input=run_input,
        prediction_source=prediction_source,
        compute_group=compute_group,
        output_run_paths_by_phenotype=output_run_paths_by_phenotype,
        staging_depth=staging_depth,
        native_callback_batch_size=native_callback_batch_size,
        result_in_flight_limit=result_in_flight_limit,
        dosage_buffer_limit=dosage_buffer_limit,
        existing_manifests=existing_manifests,
        resume=resume,
        resume_mode=resume_mode,
        null_logistic_nonconvergence_policy=null_logistic_nonconvergence_policy,
        output_sample_mode=output_sample_mode,
    )
    return run_bgen_engine_with_multi_callback(
        engine=engine,
        run_input=run_input,
        committed_chunk_identifiers=intersect_committed_chunk_identifier_sets(
            prepared_delivery.committed_chunk_identifier_sets
        ),
        writer_sessions=prepared_delivery.writer_sessions,
        callback=prepared_delivery.callback,
        stage_timing_recorder=context.stage_timing_recorder,
        writer_finish_thread_count=context.writer_settings.writer_thread_count,
        variant_major_packed8_probability_pairs=context.uses_packed8_genotypes,
    )


def run_multi_preflight(
    *,
    run_input: inputs.NativeBgenMultiRunInput,
    prediction_source: typing.Any,
    engine: _core.Regenie2RunEngine,
    variant_limit: int | None,
    trusted_no_missing_diploid: bool,
) -> None:
    """Run shared batched preflight checks for a multi-trait run."""
    preflight.run_regenie2_multi_preflight(
        run_input=run_input,
        prediction_source=prediction_source,
        engine=engine,
        variant_limit=variant_limit,
        is_binary_trait=run_input.is_binary_trait,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
    )


def run_bgen_engine_with_multi_callback(
    *,
    engine: _core.Regenie2RunEngine,
    run_input: inputs.NativeBgenMultiRunInput,
    committed_chunk_identifiers: set[int] | None,
    writer_sessions: tuple[typing.Any, ...],
    callback: inputs.BgenDeliveryCallbackProtocol,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    writer_finish_thread_count: int,
    variant_major_packed8_probability_pairs: bool,
) -> tuple[Path | None, ...]:
    """Run native BGEN chunk delivery once and close all per-phenotype writers."""
    return native_dispatch_delivery.run_bgen_engine_with_writer_sessions(
        engine=engine,
        run_input=run_input,
        committed_chunk_identifiers=committed_chunk_identifiers,
        writer_sessions=writer_sessions,
        callback=callback,
        stage_timing_recorder=stage_timing_recorder,
        writer_finish_thread_count=writer_finish_thread_count,
        variant_major_packed8_probability_pairs=variant_major_packed8_probability_pairs,
        pipeline_label="Multi-phenotype native BGEN",
    )
