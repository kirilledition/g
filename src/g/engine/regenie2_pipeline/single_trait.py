"""Single-trait REGENIE step 2 pipeline execution."""

from __future__ import annotations

import time
import typing

from g import _core, execution_plan, io, types
from g.engine import timing as engine_timing
from g.engine.native_dispatch import delivery as native_dispatch_delivery
from g.engine.native_dispatch import groups as native_dispatch_groups
from g.engine.native_dispatch import loaders as native_dispatch_loaders
from g.engine.native_dispatch import models as native_dispatch_models
from g.engine.regenie2_pipeline import (
    callbacks,
    compute_config,
    gpu_format,
    outputs,
    preflight,
)
from g.engine.regenie2_pipeline import context as pipeline_context
from g.runner import events

if typing.TYPE_CHECKING:
    from pathlib import Path

    from g.engine import dispatch_requests


def load_single_trait_run_input(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    engine: _core.Regenie2RunEngine,
    phenotype_name: str,
    covariate_names: tuple[str, ...] | None,
    pipeline_label: str,
) -> native_dispatch_models.NativeBgenRunInput:
    """Load one phenotype's aligned native inputs and emit telemetry."""
    alignment_start_time = time.perf_counter()
    _core.record_pipeline_single_trait_input_load_started_diagnostic_event(
        phenotype_name=phenotype_name,
        pipeline_label=pipeline_label,
    )
    run_input = native_dispatch_loaders.load_native_bgen_run_input(
        genotype_source_config=context.genotype_source_config,
        engine=engine,
        phenotype_path=context.phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=context.covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=context.is_binary_trait,
        alignment_config=context.alignment_config,
    )
    engine_timing.record_stage_duration(
        context.stage_timing_recorder, "sample_phenotype_covariate_alignment", alignment_start_time
    )
    sample_count = int(run_input.sample_indices.shape[0])
    covariate_count = len(run_input.native_aligned_sample_data.covariate_names)
    _core.record_pipeline_single_trait_input_aligned_diagnostic_event(
        covariate_count=covariate_count,
        phenotype_name=phenotype_name,
        pipeline_label=pipeline_label,
        sample_count=sample_count,
    )
    events.record_sample_alignment_completed_telemetry(
        context.telemetry_session,
        context.association_mode.value,
        phenotype_name,
        None,
        sample_count,
        covariate_count,
        None,
    )
    return run_input


def build_single_trait_prediction_source(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    run_input: native_dispatch_models.NativeBgenRunInput,
    phenotype_name: str,
    pipeline_label: str,
) -> typing.Any:
    """Load one phenotype's REGENIE prediction source and emit telemetry."""
    prediction_start_time = time.perf_counter()
    _core.record_pipeline_single_trait_prediction_source_load_started_diagnostic_event(
        phenotype_name=phenotype_name,
        pipeline_label=pipeline_label,
    )
    prediction_source = _core.RegeniePredictionSource.from_native_aligned_sample_data(
        str(context.prediction_list_path),
        phenotype_name,
        run_input.native_aligned_sample_data,
        sample_key_mode=native_dispatch_groups.resolve_sample_key_mode(context.alignment_config).value,
    )
    engine_timing.record_stage_duration(context.stage_timing_recorder, "prediction_source_load", prediction_start_time)
    events.record_prediction_source_loaded_telemetry(
        context.telemetry_session,
        context.association_mode.value,
        phenotype_name,
        None,
    )
    return prediction_source


def run_single_trait_preflight(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    run_input: native_dispatch_models.NativeBgenRunInput,
    prediction_source: typing.Any,
    engine: _core.Regenie2RunEngine,
    phenotype_name: str,
    pipeline_label: str,
) -> None:
    """Run preflight validation for one phenotype and emit telemetry."""
    preflight_start_time = time.perf_counter()
    _core.record_pipeline_single_trait_preflight_started_diagnostic_event(
        phenotype_name=phenotype_name,
        pipeline_label=pipeline_label,
        trusted_no_missing_diploid=context.effective_trusted_no_missing_diploid,
        variant_limit=context.variant_limit,
    )
    preflight_report = preflight.run_regenie2_preflight(
        run_input=run_input,
        prediction_source=prediction_source,
        engine=engine,
        variant_limit=context.variant_limit,
        is_binary_trait=context.is_binary_trait,
        trusted_no_missing_diploid=context.effective_trusted_no_missing_diploid,
    )
    engine_timing.record_stage_duration(context.stage_timing_recorder, "preflight_validation", preflight_start_time)
    _core.record_pipeline_single_trait_preflight_completed_diagnostic_event(
        chromosome_count=preflight_report.chromosome_count,
        covariate_count=preflight_report.covariate_count,
        phenotype_name=phenotype_name,
        pipeline_label=pipeline_label,
        sample_count=preflight_report.sample_count,
    )
    events.record_single_trait_preflight_completed_telemetry(
        context.telemetry_session,
        context.association_mode.value,
        phenotype_name,
        preflight_report.sample_count,
        preflight_report.covariate_count,
        preflight_report.chromosome_count,
    )


def run_single_trait_bgen_pipeline(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    phenotype_name: str,
    covariate_names: tuple[str, ...] | None,
    prepared_run: _core.NativeRunLifecyclePhenotypeRun,
    staging_depth: int,
    native_callback_batch_size: int,
    result_in_flight_limit: int | None,
    dosage_buffer_limit: int | None,
    null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy,
    prepared_engine: _core.Regenie2RunEngine | None,
) -> Path | None:
    """Run a single-trait REGENIE step 2 BGEN pipeline lifecycle."""
    pipeline_label = "binary" if context.is_binary_trait else "linear"
    _core.record_pipeline_single_trait_started_diagnostic_event(
        association_mode=context.association_mode.value,
        phenotype_name=phenotype_name,
        pipeline_label=pipeline_label,
    )
    if prepared_engine is None:
        engine = outputs.open_pipeline_bgen_engine(
            context=context,
            pipeline_label=pipeline_label,
            phenotype_name=phenotype_name,
            phenotype_count=None,
        )
    else:
        engine = prepared_engine
        _core.record_pipeline_prevalidated_bgen_engine_used_diagnostic_event(
            phenotype_count=None,
            phenotype_name=phenotype_name,
            pipeline_label=pipeline_label,
        )
        events.record_association_backend_selected_telemetry(
            context.telemetry_session,
            context.association_mode.value,
            context.backend_plan.backend_kind.value,
            context.backend_plan.jax_device.value,
            context.backend_plan.genotype_format.value,
            phenotype_name,
            None,
        )
        _core.record_pipeline_bgen_engine_opened_diagnostic_event(
            phenotype_count=None,
            phenotype_name=phenotype_name,
            pipeline_label=pipeline_label,
            sample_count=int(engine.sample_count),
            variant_count=int(engine.variant_count),
        )
        events.record_bgen_engine_opened_telemetry(
            context.telemetry_session,
            context.association_mode.value,
            context.backend_plan.backend_kind.value,
            int(engine.sample_count),
            int(engine.variant_count),
            phenotype_name,
            None,
        )
    run_input = load_single_trait_run_input(
        context=context,
        engine=engine,
        phenotype_name=phenotype_name,
        covariate_names=covariate_names,
        pipeline_label=pipeline_label,
    )
    prediction_source = build_single_trait_prediction_source(
        context=context,
        run_input=run_input,
        phenotype_name=phenotype_name,
        pipeline_label=pipeline_label,
    )
    resolved_compute_group = native_dispatch_groups.adapt_native_phenotype_compute_group(
        _core.resolve_single_phenotype_compute_group(
            run_input.native_aligned_sample_data,
            phenotype_name,
            str(context.prediction_list_path),
            native_dispatch_groups.resolve_sample_key_mode(context.alignment_config).value,
        )
    )
    run_single_trait_preflight(
        context=context,
        run_input=run_input,
        prediction_source=prediction_source,
        engine=engine,
        phenotype_name=phenotype_name,
        pipeline_label=pipeline_label,
    )
    current_header = outputs.build_pipeline_manifest_header(
        context=context,
        phenotype_name=phenotype_name,
        covariate_names=tuple(run_input.native_aligned_sample_data.covariate_names),
        sample_count=int(run_input.sample_indices.shape[0]),
        variant_count=int(engine.variant_count),
        multi_phenotype_sample_mode=io.MultiPhenotypeSampleMode.SINGLE_PHENOTYPE,
        phenotype_compute_group=resolved_compute_group,
    )
    initialized_outputs = outputs.initialize_pipeline_output_runs(
        context=context,
        phenotype_names=(phenotype_name,),
        current_headers_by_trait=(current_header,),
    )
    writer_sessions = outputs.create_pipeline_writer_sessions(
        context=context,
        prepared_runs_by_trait=(prepared_run,),
    )
    writer_session = writer_sessions[0]
    callback = callbacks.build_single_trait_callback(
        context=context,
        run_input=run_input,
        prediction_source=prediction_source,
        writer_session=writer_session,
        staging_depth=staging_depth,
        native_callback_batch_size=native_callback_batch_size,
        result_in_flight_limit=result_in_flight_limit,
        dosage_buffer_limit=dosage_buffer_limit,
        null_logistic_nonconvergence_policy=null_logistic_nonconvergence_policy,
    )
    return native_dispatch_delivery.run_bgen_engine_with_callback(
        engine=engine,
        run_input=run_input,
        committed_chunk_identifiers=outputs.committed_chunk_identifiers(initialized_outputs, 0),
        writer_session=writer_session,
        callback=callback,
        stage_timing_recorder=context.stage_timing_recorder,
        variant_major_packed8_probability_pairs=context.uses_packed8_genotypes,
    )


def run_regenie2_linear_bgen_pipeline(request: dispatch_requests.SingleTraitLinearPipelineRequest) -> Path | None:
    """Run the native BGEN pipeline for quantitative REGENIE step 2."""
    common_request = request.common
    resolved_gpu_genotype_format = gpu_format.resolve_auto_to_dosage(
        requested_gpu_genotype_format=request.gpu_genotype_format,
        telemetry_session=common_request.telemetry_session,
        resolution_reason="single_trait_linear",
    )
    context = pipeline_context.build_regenie2_pipeline_context(
        association_mode=types.AssociationMode.REGENIE2_LINEAR,
        genotype_source_config=common_request.genotype_source_config,
        phenotype_path=common_request.phenotype_path,
        prediction_list_path=common_request.prediction_list_path,
        covariate_path=common_request.covariate_path,
        chunk_size=common_request.chunk_size,
        variant_limit=common_request.variant_limit,
        trusted_no_missing_diploid=common_request.trusted_no_missing_diploid,
        trusted_bgen_validation_mode=common_request.trusted_bgen_validation_mode,
        bgen_decode_tile_variant_count=common_request.bgen_decode_tile_variant_count,
        jax_device=common_request.jax_device,
        jax_matmul_precision=common_request.jax_matmul_precision,
        score_dtype=common_request.score_dtype,
        firth_dtype=common_request.firth_dtype,
        requested_gpu_genotype_format=request.gpu_genotype_format,
        gpu_genotype_format=resolved_gpu_genotype_format,
        correction_plan=None,
        binary_kernel_config=None,
        linear_numerical_config=request.linear_numerical_config,
        writer_settings=common_request.writer_settings,
        stage_timing_recorder=common_request.stage_timing_recorder,
        telemetry_session=common_request.telemetry_session,
        alignment_config=common_request.alignment_config,
        phenotype_compute_groups=execution_plan.build_phenotype_compute_groups(
            phenotype_names=(request.phenotype_name,),
            multi_phenotype_sample_mode=types.MultiPhenotypeSampleMode.PER_PHENOTYPE,
        ),
        lifecycle_session=common_request.lifecycle_session,
    )
    return run_single_trait_bgen_pipeline(
        context=context,
        phenotype_name=request.phenotype_name,
        covariate_names=common_request.covariate_names,
        prepared_run=request.prepared_run,
        staging_depth=common_request.staging_depth,
        native_callback_batch_size=common_request.native_callback_batch_size,
        result_in_flight_limit=common_request.result_in_flight_limit,
        dosage_buffer_limit=common_request.dosage_buffer_limit,
        null_logistic_nonconvergence_policy=types.NullLogisticNonconvergencePolicy.FAIL,
        prepared_engine=None,
    )


def run_regenie2_binary_bgen_pipeline(request: dispatch_requests.SingleTraitBinaryPipelineRequest) -> Path | None:
    """Run the native BGEN pipeline for binary REGENIE step 2."""
    common_request = request.common
    resolved_kernel_config = compute_config.require_binary_kernel_config(request.binary_kernel_config)
    gpu_genotype_format_resolution = gpu_format.resolve_single_trait_binary_gpu_genotype_format(
        requested_gpu_genotype_format=request.gpu_genotype_format,
        existing_manifest=outputs.existing_manifest_from_prepared_run(request.prepared_run),
        resume=common_request.lifecycle_session.output_resume,
        jax_device=common_request.jax_device,
        genotype_source_config=common_request.genotype_source_config,
        chunk_size=common_request.chunk_size,
        variant_limit=common_request.variant_limit,
        trusted_bgen_validation_mode=common_request.trusted_bgen_validation_mode,
        stage_timing_recorder=common_request.stage_timing_recorder,
        telemetry_session=common_request.telemetry_session,
    )
    context = pipeline_context.build_regenie2_pipeline_context(
        association_mode=types.AssociationMode.REGENIE2_BINARY,
        genotype_source_config=common_request.genotype_source_config,
        phenotype_path=common_request.phenotype_path,
        prediction_list_path=common_request.prediction_list_path,
        covariate_path=common_request.covariate_path,
        chunk_size=common_request.chunk_size,
        variant_limit=common_request.variant_limit,
        trusted_no_missing_diploid=common_request.trusted_no_missing_diploid,
        trusted_bgen_validation_mode=common_request.trusted_bgen_validation_mode,
        bgen_decode_tile_variant_count=common_request.bgen_decode_tile_variant_count,
        jax_device=common_request.jax_device,
        jax_matmul_precision=common_request.jax_matmul_precision,
        score_dtype=common_request.score_dtype,
        firth_dtype=common_request.firth_dtype,
        requested_gpu_genotype_format=gpu_genotype_format_resolution.requested_gpu_genotype_format,
        gpu_genotype_format=gpu_genotype_format_resolution.resolved_gpu_genotype_format,
        correction_plan=request.correction_plan,
        binary_kernel_config=resolved_kernel_config,
        linear_numerical_config=None,
        writer_settings=common_request.writer_settings,
        stage_timing_recorder=common_request.stage_timing_recorder,
        telemetry_session=common_request.telemetry_session,
        alignment_config=common_request.alignment_config,
        phenotype_compute_groups=execution_plan.build_phenotype_compute_groups(
            phenotype_names=(request.phenotype_name,),
            multi_phenotype_sample_mode=types.MultiPhenotypeSampleMode.PER_PHENOTYPE,
        ),
        lifecycle_session=common_request.lifecycle_session,
    )
    return run_single_trait_bgen_pipeline(
        context=context,
        phenotype_name=request.phenotype_name,
        covariate_names=common_request.covariate_names,
        prepared_run=request.prepared_run,
        staging_depth=common_request.staging_depth,
        native_callback_batch_size=common_request.native_callback_batch_size,
        result_in_flight_limit=common_request.result_in_flight_limit,
        dosage_buffer_limit=common_request.dosage_buffer_limit,
        null_logistic_nonconvergence_policy=request.null_logistic_nonconvergence_policy,
        prepared_engine=gpu_genotype_format_resolution.prepared_engine,
    )
