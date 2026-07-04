"""Multi-phenotype REGENIE step 2 pipeline execution."""

from __future__ import annotations

import time
import typing

from g import types
from g.engine.regenie2_pipeline import (
    compute_config,
    gpu_format,
    grouped,
    inputs,
    multi_group,
    outputs,
    telemetry_events,
    timing,
)
from g.engine.regenie2_pipeline import context as pipeline_context

if typing.TYPE_CHECKING:
    from pathlib import Path

    from g.engine import dispatch_requests


def run_regenie2_multi_phenotype_linear_bgen_pipeline(
    request: dispatch_requests.MultiTraitPipelineRequest,
) -> tuple[Path | None, ...]:
    """Run the complete-case native BGEN pipeline once for multiple quantitative phenotypes."""
    return run_regenie2_multi_phenotype_bgen_pipeline(request)


def run_regenie2_multi_phenotype_binary_bgen_pipeline(
    request: dispatch_requests.MultiTraitPipelineRequest,
) -> tuple[Path | None, ...]:
    """Run the complete-case native BGEN pipeline once for multiple binary phenotypes."""
    return run_regenie2_multi_phenotype_bgen_pipeline(request)


def run_regenie2_multi_phenotype_bgen_pipeline(
    request: dispatch_requests.MultiTraitPipelineRequest,
) -> tuple[Path | None, ...]:
    """Shared implementation for multi-phenotype BGEN pipelines."""
    common_request = request.common
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
    resolved_kernel_config = (
        compute_config.require_binary_kernel_config(request.binary_kernel_config)
        if request.association_mode == types.AssociationMode.REGENIE2_BINARY
        else None
    )
    context = pipeline_context.build_regenie2_pipeline_context(
        association_mode=request.association_mode,
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
        correction_plan=request.correction_plan,
        binary_kernel_config=resolved_kernel_config,
        linear_numerical_config=request.linear_numerical_config,
        writer_settings=common_request.writer_settings,
        stage_timing_recorder=common_request.stage_timing_recorder,
        telemetry_session=common_request.telemetry_session,
        alignment_config=common_request.alignment_config,
        phenotype_compute_groups=resolved_compute_groups,
        runtime_compatibility_token=common_request.runtime_compatibility_token,
        output_initialized_callback=common_request.output_initialized_callback,
    )
    if request.sample_mode == types.MultiPhenotypeSampleMode.PER_PHENOTYPE:
        return grouped.run_regenie2_grouped_per_phenotype_bgen_pipeline(
            context=context,
            phenotype_names=request.phenotype_names,
            covariate_names=common_request.covariate_names,
            output_run_paths_by_phenotype=request.output_run_paths_by_phenotype,
            staging_depth=common_request.staging_depth,
            native_callback_batch_size=common_request.native_callback_batch_size,
            result_in_flight_limit=common_request.result_in_flight_limit,
            dosage_buffer_limit=common_request.dosage_buffer_limit,
            existing_manifests_by_phenotype=request.existing_manifests_by_phenotype,
            resume=common_request.resume,
            resume_mode=common_request.resume_mode,
            null_logistic_nonconvergence_policy=request.null_logistic_nonconvergence_policy,
        )
    if request.sample_mode != types.MultiPhenotypeSampleMode.COMPLETE_CASE:
        message = "Multi-phenotype sample mode must be per-phenotype or complete-case."
        raise ValueError(message)
    telemetry_events.record_pipeline_multi_trait_started(
        association_mode=context.association_mode,
        phenotype_count=len(request.phenotype_names),
        sample_mode=request.sample_mode,
    )
    existing_manifests = request.existing_manifests_by_phenotype or tuple(None for _ in request.phenotype_names)
    planned_compute_group = pipeline_context.require_complete_case_compute_group(context.phenotype_compute_groups)
    engine = outputs.open_pipeline_bgen_engine(
        context=context,
        pipeline_label="multi-phenotype",
        phenotype_name=None,
        phenotype_count=len(planned_compute_group.phenotype_names),
    )
    alignment_start_time = time.perf_counter()
    telemetry_events.record_pipeline_multi_trait_input_load_started(len(planned_compute_group.phenotype_names))
    run_input = inputs.load_native_bgen_multi_run_input(
        genotype_source_config=context.genotype_source_config,
        engine=engine,
        phenotype_path=context.phenotype_path,
        phenotype_names=planned_compute_group.phenotype_names,
        covariate_path=context.covariate_path,
        covariate_names=common_request.covariate_names,
        is_binary_trait=context.is_binary_trait,
        alignment_config=context.alignment_config,
    )
    resolved_compute_group = inputs.build_resolved_complete_case_phenotype_compute_group(
        run_input=run_input,
        prediction_list_path=context.prediction_list_path,
        planned_compute_groups=context.phenotype_compute_groups,
        alignment_config=context.alignment_config,
    )
    timing.record_stage_duration(
        context.stage_timing_recorder, "sample_phenotype_covariate_alignment", alignment_start_time
    )
    sample_count = int(run_input.sample_indices.shape[0])
    phenotype_count = len(run_input.phenotype_names)
    covariate_count = len(run_input.native_multi_aligned_sample_data.covariate_names)
    telemetry_events.record_pipeline_multi_trait_input_aligned(
        covariate_count=covariate_count,
        phenotype_count=phenotype_count,
        sample_count=sample_count,
    )
    telemetry_events.log_sample_alignment_completed(
        context=context,
        sample_count=sample_count,
        covariate_count=covariate_count,
        phenotype_name=None,
        phenotype_count=phenotype_count,
        phenotype_group_count=None,
    )
    telemetry_events.log_multi_phenotype_sample_summary(
        context=context,
        sample_mode=types.MultiPhenotypeSampleMode.COMPLETE_CASE,
        sample_counts=tuple(sample_count for _ in resolved_compute_group.phenotype_names),
        sample_set_fingerprints=tuple(
            resolved_compute_group.sample_set_fingerprint for _ in resolved_compute_group.phenotype_names
        ),
        phenotype_group_count=1,
    )
    prediction_start_time = time.perf_counter()
    telemetry_events.record_pipeline_multi_trait_prediction_source_load_started(phenotype_count)
    prediction_source = inputs.build_multi_regenie_prediction_source(
        prediction_list_path=context.prediction_list_path,
        run_input=run_input,
        alignment_config=context.alignment_config,
    )
    timing.record_stage_duration(context.stage_timing_recorder, "prediction_source_load", prediction_start_time)
    return multi_group.run_prepared_multi_phenotype_bgen_group(
        context=context,
        engine=engine,
        run_input=run_input,
        prediction_source=prediction_source,
        compute_group=resolved_compute_group,
        output_run_paths_by_phenotype=typing.cast(
            "tuple[outputs.OutputRunPaths, ...]",
            pipeline_context.select_by_phenotype_indices(
                request.output_run_paths_by_phenotype,
                resolved_compute_group.phenotype_indices,
            ),
        ),
        staging_depth=common_request.staging_depth,
        native_callback_batch_size=common_request.native_callback_batch_size,
        result_in_flight_limit=common_request.result_in_flight_limit,
        dosage_buffer_limit=common_request.dosage_buffer_limit,
        existing_manifests=typing.cast(
            "tuple[dict[str, typing.Any] | None, ...]",
            pipeline_context.select_by_phenotype_indices(existing_manifests, resolved_compute_group.phenotype_indices),
        ),
        resume=common_request.resume,
        resume_mode=common_request.resume_mode,
        null_logistic_nonconvergence_policy=request.null_logistic_nonconvergence_policy,
        output_sample_mode=outputs.COMPLETE_CASE_SAMPLE_MODE,
    )
