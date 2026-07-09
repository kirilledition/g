"""Multi-phenotype REGENIE step 2 pipeline execution."""

from __future__ import annotations

import time
import typing

from g import _core, types
from g.engine import dispatch_requests
from g.engine import timing as engine_timing
from g.engine.native_dispatch import groups as native_dispatch_groups
from g.engine.native_dispatch import loaders as native_dispatch_loaders
from g.engine.regenie2_pipeline import (
    compute_config,
    gpu_format,
    grouped,
    multi_group,
    outputs,
)
from g.engine.regenie2_pipeline import context as pipeline_context

if typing.TYPE_CHECKING:
    from pathlib import Path

RUN_EVENT_RECORDER: _core.NativeRunEventRecorder = _core.NativeRunEventRecorder()


def run_regenie2_multi_phenotype_bgen_pipeline(
    request: dispatch_requests.MultiTraitLinearPipelineRequest | dispatch_requests.MultiTraitBinaryPipelineRequest,
) -> tuple[Path | None, ...]:
    """Shared implementation for multi-phenotype BGEN pipelines."""
    common_request = request.common
    association_mode = association_mode_from_multi_trait_request(request)
    correction_plan = correction_plan_from_multi_trait_request(request)
    linear_numerical_config = linear_numerical_config_from_multi_trait_request(request)
    null_logistic_nonconvergence_policy = null_logistic_policy_from_multi_trait_request(request)
    resolved_gpu_genotype_format = gpu_format.resolve_auto_to_dosage(
        requested_gpu_genotype_format=request.gpu_genotype_format,
        telemetry_session=common_request.telemetry_session,
        resolution_reason="multi_phenotype",
    )
    resolved_compute_groups = pipeline_context.resolve_multi_phenotype_compute_groups(
        phenotype_names=request.phenotype_names,
        sample_mode=request.sample_mode,
        phenotype_compute_groups=request.phenotype_compute_groups,
    )
    resolved_kernel_config = binary_kernel_config_from_multi_trait_request(request)
    context = pipeline_context.build_regenie2_pipeline_context(
        association_mode=association_mode,
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
        correction_plan=correction_plan,
        binary_kernel_config=resolved_kernel_config,
        linear_numerical_config=linear_numerical_config,
        writer_settings=common_request.writer_settings,
        stage_timing_recorder=common_request.stage_timing_recorder,
        telemetry_session=common_request.telemetry_session,
        alignment_config=common_request.alignment_config,
        phenotype_compute_groups=resolved_compute_groups,
        engine_session=common_request.engine_session,
    )
    if request.sample_mode == types.MultiPhenotypeSampleMode.PER_PHENOTYPE:
        return grouped.run_regenie2_grouped_per_phenotype_bgen_pipeline(
            context=context,
            phenotype_names=request.phenotype_names,
            covariate_names=common_request.covariate_names,
            staging_depth=common_request.staging_depth,
            native_callback_batch_size=common_request.native_callback_batch_size,
            result_in_flight_limit=common_request.result_in_flight_limit,
            dosage_buffer_limit=common_request.dosage_buffer_limit,
            null_logistic_nonconvergence_policy=null_logistic_nonconvergence_policy,
        )
    if request.sample_mode != types.MultiPhenotypeSampleMode.COMPLETE_CASE:
        message = "Multi-phenotype sample mode must be per-phenotype or complete-case."
        raise ValueError(message)
    RUN_EVENT_RECORDER.pipeline_multi_trait_started(
        association_mode=context.association_mode.value,
        phenotype_count=len(request.phenotype_names),
        sample_mode=request.sample_mode.value,
    )
    planned_compute_group = pipeline_context.require_complete_case_compute_group(context.phenotype_compute_groups)
    engine = outputs.open_pipeline_bgen_engine(
        context=context,
        pipeline_label="multi-phenotype",
        phenotype_name=None,
        phenotype_count=len(planned_compute_group.phenotype_names),
    )
    alignment_start_time = time.perf_counter()
    RUN_EVENT_RECORDER.pipeline_multi_trait_input_load_started(
        phenotype_count=len(planned_compute_group.phenotype_names)
    )
    run_input = native_dispatch_loaders.load_native_bgen_multi_run_input(
        genotype_source_config=context.genotype_source_config,
        engine=engine,
        phenotype_path=context.phenotype_path,
        phenotype_names=planned_compute_group.phenotype_names,
        covariate_path=context.covariate_path,
        covariate_names=common_request.covariate_names,
        is_binary_trait=context.is_binary_trait,
        alignment_config=context.alignment_config,
    )
    resolved_compute_group = native_dispatch_groups.adapt_native_phenotype_compute_group(
        _core.resolve_complete_case_compute_group(
            run_input.native_multi_aligned_sample_data,
            list(planned_compute_group.phenotype_indices),
            list(planned_compute_group.phenotype_names),
            str(context.prediction_list_path),
            native_dispatch_groups.resolve_sample_key_mode(context.alignment_config).value,
        )
    )
    engine_timing.record_stage_duration(
        context.stage_timing_recorder, "sample_phenotype_covariate_alignment", alignment_start_time
    )
    sample_count = int(run_input.sample_indices.shape[0])
    phenotype_count = len(run_input.phenotype_names)
    covariate_count = len(run_input.native_multi_aligned_sample_data.covariate_names)
    RUN_EVENT_RECORDER.pipeline_multi_trait_input_aligned(
        covariate_count=covariate_count,
        phenotype_count=phenotype_count,
        sample_count=sample_count,
    )
    _core.record_sample_alignment_completed_telemetry(
        context.telemetry_session,
        context.association_mode.value,
        None,
        phenotype_count,
        sample_count,
        covariate_count,
        None,
    )
    sample_counts = tuple(sample_count for _ in resolved_compute_group.phenotype_names)
    sample_set_fingerprints = tuple(
        resolved_compute_group.sample_set_fingerprint for _ in resolved_compute_group.phenotype_names
    )
    RUN_EVENT_RECORDER.pipeline_multi_phenotype_sample_summary(
        phenotype_count=len(sample_counts),
        phenotype_group_count=1,
        sample_counts_differ=len(set(sample_counts)) > 1,
        sample_mode=types.MultiPhenotypeSampleMode.COMPLETE_CASE.value,
    )
    _core.record_multi_phenotype_sample_summary_telemetry(
        context.telemetry_session,
        context.association_mode.value,
        types.MultiPhenotypeSampleMode.COMPLETE_CASE.value,
        sample_counts,
        sample_set_fingerprints,
        1,
    )
    prediction_start_time = time.perf_counter()
    RUN_EVENT_RECORDER.pipeline_multi_trait_prediction_source_load_started(phenotype_count=phenotype_count)
    prediction_source = _core.MultiRegeniePredictionSource.from_native_multi_aligned_sample_data(
        str(context.prediction_list_path),
        run_input.native_multi_aligned_sample_data,
        sample_key_mode=native_dispatch_groups.resolve_sample_key_mode(context.alignment_config).value,
    )
    engine_timing.record_stage_duration(context.stage_timing_recorder, "prediction_source_load", prediction_start_time)
    multi_group.run_multi_phenotype_group_preflight(
        context=context,
        engine=engine,
        run_input=run_input,
        prediction_source=prediction_source,
    )
    output_bundle = multi_group.prepare_multi_phenotype_output_bundle(
        context=context,
        engine=engine,
        run_input=run_input,
        compute_group=resolved_compute_group,
        output_sample_mode=types.MultiPhenotypeSampleMode.COMPLETE_CASE,
    )
    return multi_group.run_prepared_multi_phenotype_bgen_group(
        context=context,
        engine=engine,
        run_input=run_input,
        prediction_source=prediction_source,
        compute_group=resolved_compute_group,
        output_bundle=output_bundle,
        staging_depth=common_request.staging_depth,
        native_callback_batch_size=common_request.native_callback_batch_size,
        result_in_flight_limit=common_request.result_in_flight_limit,
        dosage_buffer_limit=common_request.dosage_buffer_limit,
        null_logistic_nonconvergence_policy=null_logistic_nonconvergence_policy,
    )


def association_mode_from_multi_trait_request(
    request: dispatch_requests.MultiTraitLinearPipelineRequest | dispatch_requests.MultiTraitBinaryPipelineRequest,
) -> types.AssociationMode:
    """Resolve association mode from the request variant."""
    if isinstance(request, dispatch_requests.MultiTraitBinaryPipelineRequest):
        return types.AssociationMode.REGENIE2_BINARY
    return types.AssociationMode.REGENIE2_LINEAR


def correction_plan_from_multi_trait_request(
    request: dispatch_requests.MultiTraitLinearPipelineRequest | dispatch_requests.MultiTraitBinaryPipelineRequest,
) -> types.BinaryCorrectionPlan | None:
    """Return binary correction settings when the request is binary."""
    if isinstance(request, dispatch_requests.MultiTraitBinaryPipelineRequest):
        return request.correction_plan
    return None


def linear_numerical_config_from_multi_trait_request(
    request: dispatch_requests.MultiTraitLinearPipelineRequest | dispatch_requests.MultiTraitBinaryPipelineRequest,
) -> compute_config.LinearNumericalConfig | None:
    """Return linear numerical settings when the request is linear."""
    if isinstance(request, dispatch_requests.MultiTraitLinearPipelineRequest):
        return request.linear_numerical_config
    return None


def binary_kernel_config_from_multi_trait_request(
    request: dispatch_requests.MultiTraitLinearPipelineRequest | dispatch_requests.MultiTraitBinaryPipelineRequest,
) -> compute_config.BinaryKernelConfig | None:
    """Return binary kernel settings when the request is binary."""
    if isinstance(request, dispatch_requests.MultiTraitBinaryPipelineRequest):
        return compute_config.require_binary_kernel_config(request.binary_kernel_config)
    return None


def null_logistic_policy_from_multi_trait_request(
    request: dispatch_requests.MultiTraitLinearPipelineRequest | dispatch_requests.MultiTraitBinaryPipelineRequest,
) -> types.NullLogisticNonconvergencePolicy:
    """Return null-logistic policy for callback construction."""
    if isinstance(request, dispatch_requests.MultiTraitBinaryPipelineRequest):
        return request.null_logistic_nonconvergence_policy
    return types.NullLogisticNonconvergencePolicy.FAIL
