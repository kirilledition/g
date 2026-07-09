"""Single-trait REGENIE step 2 pipeline execution."""

from __future__ import annotations

import typing
from pathlib import Path

from g import _core, execution_plan, types
from g.engine.regenie2_pipeline import (
    callbacks,
    compute_config,
    gpu_format,
    outputs,
)
from g.engine.regenie2_pipeline import context as pipeline_context

if typing.TYPE_CHECKING:
    from g.engine import dispatch_requests


def run_single_trait_bgen_pipeline(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    phenotype_name: str,
    covariate_names: tuple[str, ...] | None,
    staging_depth: int,
    native_callback_batch_size: int,
    result_in_flight_limit: int | None,
    dosage_buffer_limit: int | None,
    null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy,
    prepared_engine: _core.NativeRunEngineSession | None,
) -> Path | None:
    """Run a single-trait REGENIE step 2 BGEN pipeline lifecycle."""
    pipeline_label = "binary" if context.is_binary_trait else "linear"
    if prepared_engine is not None and prepared_engine is not context.engine_session:
        message = "Prepared BGEN engine must belong to the current native run engine session."
        raise RuntimeError(message)
    sample_key_mode = (
        types.SampleKeyMode.IID.value
        if context.alignment_config is None
        else context.alignment_config.sample_key_mode.value
    )
    prepared_bundle = context.engine_session.prepare_single_trait_pipeline_bundle(
        phenotype_name,
        None if covariate_names is None else list(covariate_names),
        association_mode=context.association_mode.value,
        association_backend_kind=context.backend_plan.backend_kind.value,
        jax_device=context.backend_plan.jax_device.value,
        genotype_format=context.backend_plan.genotype_format.value,
        requested_gpu_genotype_format=context.requested_gpu_genotype_format.value,
        score_dtype=context.score_dtype.value,
        firth_dtype=context.firth_dtype.value,
        binary_kernel_config_json=outputs.build_binary_kernel_config_json(context=context),
        sample_key_mode=sample_key_mode,
        is_binary_trait=context.is_binary_trait,
        pipeline_label=pipeline_label,
        bgen_path=str(context.genotype_source_config.source_path),
        sample_path=(
            None
            if context.genotype_source_config.sample_path is None
            else str(context.genotype_source_config.sample_path)
        ),
        phenotype_path=str(context.phenotype_path),
        covariate_path=None if context.covariate_path is None else str(context.covariate_path),
        prediction_list_path=str(context.prediction_list_path),
        chunk_size=context.chunk_size,
        variant_limit=context.variant_limit,
        effective_trusted_no_missing_diploid=context.effective_trusted_no_missing_diploid,
        trusted_bgen_validation_mode=context.trusted_bgen_validation_mode.value,
        telemetry_session=context.telemetry_session,
        stage_timing_recorder=None
        if context.stage_timing_recorder is None
        else context.stage_timing_recorder.native_recorder,
    )
    callback = callbacks.build_single_trait_callback(
        context=context,
        run_input=prepared_bundle.run_input,
        prediction_source=prepared_bundle.prediction_source,
        writer_session=prepared_bundle.writer_session,
        staging_depth=staging_depth,
        native_callback_batch_size=native_callback_batch_size,
        result_in_flight_limit=result_in_flight_limit,
        dosage_buffer_limit=dosage_buffer_limit,
        null_logistic_nonconvergence_policy=null_logistic_nonconvergence_policy,
    )
    final_output_path_value = context.engine_session.run_single_trait_pipeline_bundle(
        prepared_bundle,
        callback=callback,
        stage_timing_recorder=None
        if context.stage_timing_recorder is None
        else context.stage_timing_recorder.native_recorder,
        variant_major_packed8_probability_pairs=context.uses_packed8_genotypes,
        pipeline_label="Native BGEN",
    )
    return None if final_output_path_value is None else Path(final_output_path_value)


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
        engine_session=common_request.engine_session,
    )
    return run_single_trait_bgen_pipeline(
        context=context,
        phenotype_name=request.phenotype_name,
        covariate_names=common_request.covariate_names,
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
        resume=common_request.engine_session.output_resume,
        jax_device=common_request.jax_device,
        engine_session=common_request.engine_session,
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
        engine_session=common_request.engine_session,
    )
    return run_single_trait_bgen_pipeline(
        context=context,
        phenotype_name=request.phenotype_name,
        covariate_names=common_request.covariate_names,
        staging_depth=common_request.staging_depth,
        native_callback_batch_size=common_request.native_callback_batch_size,
        result_in_flight_limit=common_request.result_in_flight_limit,
        dosage_buffer_limit=common_request.dosage_buffer_limit,
        null_logistic_nonconvergence_policy=request.null_logistic_nonconvergence_policy,
        prepared_engine=gpu_genotype_format_resolution.prepared_engine,
    )
