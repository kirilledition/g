"""Multi-phenotype REGENIE step 2 pipeline execution."""

from __future__ import annotations

import logging
import time
import typing

from g import execution_plan, types
from g.engine import telemetry, timing
from g.engine.native_dispatch import groups as native_dispatch_groups
from g.engine.native_dispatch import loaders as native_dispatch_loaders
from g.engine.native_dispatch import models as native_dispatch_models
from g.engine.regenie2_pipeline import context as pipeline_context
from g.engine.regenie2_pipeline import gpu_format, grouped, multi_group, outputs, telemetry_events
from g.io import output

if typing.TYPE_CHECKING:
    from pathlib import Path

    from g.compute.regenie2_binary import config as regenie2_binary_config
    from g.compute.regenie2_linear import config as regenie2_linear_config
    from g.io import source

logger = logging.getLogger(__name__)


def run_regenie2_multi_phenotype_linear_bgen_pipeline(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    phenotype_path: Path,
    phenotype_names: tuple[str, ...],
    prediction_list_path: Path,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    chunk_size: int,
    variant_limit: int | None,
    output_run_paths_by_phenotype: tuple[output.OutputRunPaths, ...],
    staging_depth: int,
    native_callback_batch_size: int,
    result_in_flight_limit: int | None,
    dosage_buffer_limit: int | None,
    existing_manifests_by_phenotype: tuple[dict[str, typing.Any] | None, ...] | None,
    resume: bool,
    resume_mode: types.ResumeMode,
    writer_settings: output.OutputWriterSettings,
    trusted_no_missing_diploid: bool,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode,
    bgen_decode_tile_variant_count: int,
    jax_device: types.Device,
    jax_matmul_precision: types.JaxMatmulPrecision | None,
    score_dtype: types.FloatingPointDtype,
    firth_dtype: types.FloatingPointDtype,
    linear_numerical_config: regenie2_linear_config.LinearNumericalConfig | None,
    gpu_genotype_format: types.GpuGenotypeFormat,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    telemetry_session: telemetry.TelemetrySession | None,
    alignment_config: native_dispatch_models.SampleAlignmentConfigProtocol | None,
    sample_mode: types.MultiPhenotypeSampleMode | None,
    phenotype_compute_groups: tuple[execution_plan.PhenotypeComputeGroup, ...] | None,
    output_initialized_callback: typing.Callable[[tuple[str, ...]], None] | None,
) -> tuple[Path | None, ...]:
    """Run the complete-case native BGEN pipeline once for multiple quantitative phenotypes."""
    return run_regenie2_multi_phenotype_bgen_pipeline(
        genotype_source_config=genotype_source_config,
        phenotype_path=phenotype_path,
        phenotype_names=phenotype_names,
        prediction_list_path=prediction_list_path,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        output_run_paths_by_phenotype=output_run_paths_by_phenotype,
        staging_depth=staging_depth,
        native_callback_batch_size=native_callback_batch_size,
        result_in_flight_limit=result_in_flight_limit,
        dosage_buffer_limit=dosage_buffer_limit,
        existing_manifests_by_phenotype=existing_manifests_by_phenotype,
        resume=resume,
        resume_mode=resume_mode,
        writer_settings=writer_settings,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
        trusted_bgen_validation_mode=trusted_bgen_validation_mode,
        bgen_decode_tile_variant_count=bgen_decode_tile_variant_count,
        jax_device=jax_device,
        jax_matmul_precision=jax_matmul_precision,
        score_dtype=score_dtype,
        firth_dtype=firth_dtype,
        gpu_genotype_format=gpu_genotype_format,
        correction_plan=types.BinaryCorrectionPlan(
            method=types.BinaryFallbackMethod.SCORE_ONLY,
            p_threshold=0.05,
            firth_se=False,
        ),
        kernel_config=None,
        linear_numerical_config=linear_numerical_config,
        null_logistic_nonconvergence_policy=types.NullLogisticNonconvergencePolicy.FAIL,
        stage_timing_recorder=stage_timing_recorder,
        telemetry_session=telemetry_session,
        alignment_config=alignment_config,
        sample_mode=sample_mode,
        phenotype_compute_groups=phenotype_compute_groups,
        output_initialized_callback=output_initialized_callback,
        association_mode=types.AssociationMode.REGENIE2_LINEAR,
    )


def run_regenie2_multi_phenotype_binary_bgen_pipeline(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    phenotype_path: Path,
    phenotype_names: tuple[str, ...],
    prediction_list_path: Path,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    chunk_size: int,
    variant_limit: int | None,
    output_run_paths_by_phenotype: tuple[output.OutputRunPaths, ...],
    staging_depth: int,
    native_callback_batch_size: int,
    result_in_flight_limit: int | None,
    dosage_buffer_limit: int | None,
    existing_manifests_by_phenotype: tuple[dict[str, typing.Any] | None, ...] | None,
    resume: bool,
    resume_mode: types.ResumeMode,
    writer_settings: output.OutputWriterSettings,
    trusted_no_missing_diploid: bool,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode,
    bgen_decode_tile_variant_count: int,
    jax_device: types.Device,
    jax_matmul_precision: types.JaxMatmulPrecision | None,
    score_dtype: types.FloatingPointDtype,
    firth_dtype: types.FloatingPointDtype,
    correction_plan: types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig,
    gpu_genotype_format: types.GpuGenotypeFormat,
    null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    telemetry_session: telemetry.TelemetrySession | None,
    alignment_config: native_dispatch_models.SampleAlignmentConfigProtocol | None,
    sample_mode: types.MultiPhenotypeSampleMode | None,
    phenotype_compute_groups: tuple[execution_plan.PhenotypeComputeGroup, ...] | None,
    output_initialized_callback: typing.Callable[[tuple[str, ...]], None] | None,
) -> tuple[Path | None, ...]:
    """Run the complete-case native BGEN pipeline once for multiple binary phenotypes."""
    resolved_kernel_config = pipeline_context.require_binary_kernel_config(kernel_config)
    return run_regenie2_multi_phenotype_bgen_pipeline(
        genotype_source_config=genotype_source_config,
        phenotype_path=phenotype_path,
        phenotype_names=phenotype_names,
        prediction_list_path=prediction_list_path,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        output_run_paths_by_phenotype=output_run_paths_by_phenotype,
        staging_depth=staging_depth,
        native_callback_batch_size=native_callback_batch_size,
        result_in_flight_limit=result_in_flight_limit,
        dosage_buffer_limit=dosage_buffer_limit,
        existing_manifests_by_phenotype=existing_manifests_by_phenotype,
        resume=resume,
        resume_mode=resume_mode,
        writer_settings=writer_settings,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
        trusted_bgen_validation_mode=trusted_bgen_validation_mode,
        bgen_decode_tile_variant_count=bgen_decode_tile_variant_count,
        jax_device=jax_device,
        jax_matmul_precision=jax_matmul_precision,
        score_dtype=score_dtype,
        firth_dtype=firth_dtype,
        gpu_genotype_format=gpu_genotype_format,
        correction_plan=correction_plan,
        kernel_config=resolved_kernel_config,
        linear_numerical_config=None,
        null_logistic_nonconvergence_policy=null_logistic_nonconvergence_policy,
        stage_timing_recorder=stage_timing_recorder,
        telemetry_session=telemetry_session,
        alignment_config=alignment_config,
        sample_mode=sample_mode,
        phenotype_compute_groups=phenotype_compute_groups,
        output_initialized_callback=output_initialized_callback,
        association_mode=types.AssociationMode.REGENIE2_BINARY,
    )


def run_regenie2_multi_phenotype_bgen_pipeline(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    phenotype_path: Path,
    phenotype_names: tuple[str, ...],
    prediction_list_path: Path,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    chunk_size: int,
    variant_limit: int | None,
    output_run_paths_by_phenotype: tuple[output.OutputRunPaths, ...],
    staging_depth: int,
    native_callback_batch_size: int,
    result_in_flight_limit: int | None,
    dosage_buffer_limit: int | None,
    existing_manifests_by_phenotype: tuple[dict[str, typing.Any] | None, ...] | None,
    resume: bool,
    resume_mode: types.ResumeMode,
    writer_settings: output.OutputWriterSettings,
    trusted_no_missing_diploid: bool,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode,
    bgen_decode_tile_variant_count: int,
    jax_device: types.Device,
    jax_matmul_precision: types.JaxMatmulPrecision | None,
    score_dtype: types.FloatingPointDtype,
    firth_dtype: types.FloatingPointDtype,
    gpu_genotype_format: types.GpuGenotypeFormat,
    correction_plan: types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig | None,
    null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    telemetry_session: telemetry.TelemetrySession | None,
    alignment_config: native_dispatch_models.SampleAlignmentConfigProtocol | None,
    sample_mode: types.MultiPhenotypeSampleMode | None,
    phenotype_compute_groups: tuple[execution_plan.PhenotypeComputeGroup, ...] | None,
    association_mode: types.AssociationMode,
    linear_numerical_config: regenie2_linear_config.LinearNumericalConfig | None,
    output_initialized_callback: typing.Callable[[tuple[str, ...]], None] | None,
) -> tuple[Path | None, ...]:
    """Shared implementation for multi-phenotype BGEN pipelines."""
    resolved_gpu_genotype_format = gpu_format.resolve_auto_to_dosage(
        requested_gpu_genotype_format=gpu_genotype_format,
        telemetry_session=telemetry_session,
        resolution_reason="multi_phenotype",
    )
    resolved_compute_groups = pipeline_context.resolve_multi_phenotype_compute_groups(
        phenotype_names=phenotype_names,
        sample_mode=sample_mode,
        phenotype_compute_groups=phenotype_compute_groups,
    )
    resolved_kernel_config = (
        pipeline_context.require_binary_kernel_config(kernel_config)
        if association_mode == types.AssociationMode.REGENIE2_BINARY
        else None
    )
    context = pipeline_context.build_regenie2_pipeline_context(
        association_mode=association_mode,
        genotype_source_config=genotype_source_config,
        phenotype_path=phenotype_path,
        prediction_list_path=prediction_list_path,
        covariate_path=covariate_path,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
        trusted_bgen_validation_mode=trusted_bgen_validation_mode,
        bgen_decode_tile_variant_count=bgen_decode_tile_variant_count,
        jax_device=jax_device,
        jax_matmul_precision=jax_matmul_precision,
        score_dtype=score_dtype,
        firth_dtype=firth_dtype,
        requested_gpu_genotype_format=gpu_genotype_format,
        gpu_genotype_format=resolved_gpu_genotype_format,
        correction_plan=correction_plan,
        binary_kernel_config=resolved_kernel_config,
        linear_numerical_config=linear_numerical_config,
        writer_settings=writer_settings,
        stage_timing_recorder=stage_timing_recorder,
        telemetry_session=telemetry_session,
        alignment_config=alignment_config,
        phenotype_compute_groups=resolved_compute_groups,
        output_initialized_callback=output_initialized_callback,
    )
    if sample_mode == types.MultiPhenotypeSampleMode.PER_PHENOTYPE:
        return grouped.run_regenie2_grouped_per_phenotype_bgen_pipeline(
            context=context,
            phenotype_names=phenotype_names,
            covariate_names=covariate_names,
            output_run_paths_by_phenotype=output_run_paths_by_phenotype,
            staging_depth=staging_depth,
            native_callback_batch_size=native_callback_batch_size,
            result_in_flight_limit=result_in_flight_limit,
            dosage_buffer_limit=dosage_buffer_limit,
            existing_manifests_by_phenotype=existing_manifests_by_phenotype,
            resume=resume,
            resume_mode=resume_mode,
            null_logistic_nonconvergence_policy=null_logistic_nonconvergence_policy,
        )
    if sample_mode != types.MultiPhenotypeSampleMode.COMPLETE_CASE:
        message = "Multi-phenotype sample mode must be per-phenotype or complete-case."
        raise ValueError(message)
    logger.info("Starting multi-phenotype REGENIE step 2 BGEN pipeline.")
    existing_manifests = existing_manifests_by_phenotype or tuple(None for _ in phenotype_names)
    planned_compute_group = pipeline_context.require_complete_case_compute_group(context.phenotype_compute_groups)
    engine = outputs.open_pipeline_bgen_engine(
        context=context,
        pipeline_label="multi-phenotype",
        phenotype_name=None,
        phenotype_count=len(planned_compute_group.phenotype_names),
    )
    alignment_start_time = time.perf_counter()
    logger.debug("Loading aligned native sample, phenotype, and covariate inputs for multi-phenotype pipeline.")
    run_input = native_dispatch_loaders.load_native_bgen_multi_run_input(
        genotype_source_config=context.genotype_source_config,
        engine=engine,
        phenotype_path=context.phenotype_path,
        phenotype_names=planned_compute_group.phenotype_names,
        covariate_path=context.covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=context.is_binary_trait,
        alignment_config=context.alignment_config,
    )
    resolved_compute_group = native_dispatch_groups.build_resolved_complete_case_phenotype_compute_group(
        run_input=run_input,
        prediction_list_path=context.prediction_list_path,
        planned_compute_groups=context.phenotype_compute_groups,
        alignment_config=context.alignment_config,
    )
    timing.record_stage_duration(
        context.stage_timing_recorder, "sample_phenotype_covariate_alignment", alignment_start_time
    )
    logger.debug(
        "Aligned multi-phenotype pipeline inputs: sample_count=%s phenotype_count=%s covariate_count=%s.",
        int(run_input.sample_indices.shape[0]),
        len(run_input.phenotype_names),
        len(run_input.native_multi_aligned_sample_data.covariate_names),
    )
    telemetry_events.log_sample_alignment_completed(
        context=context,
        sample_count=int(run_input.sample_indices.shape[0]),
        covariate_count=len(run_input.native_multi_aligned_sample_data.covariate_names),
        phenotype_name=None,
        phenotype_count=len(run_input.phenotype_names),
        phenotype_group_count=None,
    )
    telemetry_events.log_multi_phenotype_sample_summary(
        context=context,
        sample_mode=types.MultiPhenotypeSampleMode.COMPLETE_CASE,
        sample_counts=tuple(int(run_input.sample_indices.shape[0]) for _ in resolved_compute_group.phenotype_names),
        sample_set_fingerprints=tuple(
            resolved_compute_group.sample_set_fingerprint for _ in resolved_compute_group.phenotype_names
        ),
        phenotype_group_count=1,
    )
    prediction_start_time = time.perf_counter()
    logger.debug("Loading REGENIE prediction source for multi-phenotype pipeline.")
    prediction_source = native_dispatch_loaders.build_multi_regenie_prediction_source(
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
            "tuple[output.OutputRunPaths, ...]",
            pipeline_context.select_by_phenotype_indices(
                output_run_paths_by_phenotype,
                resolved_compute_group.phenotype_indices,
            ),
        ),
        staging_depth=staging_depth,
        native_callback_batch_size=native_callback_batch_size,
        result_in_flight_limit=result_in_flight_limit,
        dosage_buffer_limit=dosage_buffer_limit,
        existing_manifests=typing.cast(
            "tuple[dict[str, typing.Any] | None, ...]",
            pipeline_context.select_by_phenotype_indices(existing_manifests, resolved_compute_group.phenotype_indices),
        ),
        resume=resume,
        resume_mode=resume_mode,
        null_logistic_nonconvergence_policy=null_logistic_nonconvergence_policy,
        output_sample_mode=output.MultiPhenotypeSampleMode.COMPLETE_CASE,
    )
