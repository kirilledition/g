"""Single-trait REGENIE step 2 pipeline execution."""

from __future__ import annotations

import time
import typing

import g.engine.callbacks.binary as callback_binary
import g.engine.callbacks.linear as callback_linear
from g import _core, execution_plan, types
from g.engine import preflight, telemetry, timing
from g.engine.native_dispatch import delivery as native_dispatch_delivery
from g.engine.native_dispatch import groups as native_dispatch_groups
from g.engine.native_dispatch import loaders as native_dispatch_loaders
from g.engine.native_dispatch import models as native_dispatch_models
from g.engine.regenie2_pipeline import context as pipeline_context
from g.engine.regenie2_pipeline import gpu_format, outputs, telemetry_events
from g.io import output

if typing.TYPE_CHECKING:
    from pathlib import Path

    from g.compute.regenie2_binary import config as regenie2_binary_config
    from g.compute.regenie2_linear import config as regenie2_linear_config
    from g.io import source


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
        build_native_bgen_run_input_callable=None,
        load_aligned_sample_data_callable=None,
    )
    timing.record_stage_duration(
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
    telemetry_events.log_sample_alignment_completed(
        context=context,
        sample_count=sample_count,
        covariate_count=covariate_count,
        phenotype_name=phenotype_name,
        phenotype_count=None,
        phenotype_group_count=None,
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
    prediction_source = native_dispatch_loaders.build_regenie_prediction_source(
        prediction_list_path=context.prediction_list_path,
        phenotype_name=phenotype_name,
        run_input=run_input,
        alignment_config=context.alignment_config,
    )
    timing.record_stage_duration(context.stage_timing_recorder, "prediction_source_load", prediction_start_time)
    telemetry_events.log_prediction_source_loaded(
        context=context,
        phenotype_name=phenotype_name,
        phenotype_count=None,
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
    timing.record_stage_duration(context.stage_timing_recorder, "preflight_validation", preflight_start_time)
    _core.record_pipeline_single_trait_preflight_completed_diagnostic_event(
        chromosome_count=preflight_report.chromosome_count,
        covariate_count=preflight_report.covariate_count,
        phenotype_name=phenotype_name,
        pipeline_label=pipeline_label,
        sample_count=preflight_report.sample_count,
    )
    _core.record_single_trait_preflight_completed_telemetry_event(
        context.telemetry_session,
        context.association_mode.value,
        phenotype_name,
        preflight_report.sample_count,
        preflight_report.covariate_count,
        preflight_report.chromosome_count,
    )


def build_single_trait_callback(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    run_input: native_dispatch_models.NativeBgenRunInput,
    prediction_source: typing.Any,
    writer_session: typing.Any,
    staging_depth: int,
    native_callback_batch_size: int,
    result_in_flight_limit: int | None,
    dosage_buffer_limit: int | None,
    null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy,
) -> object:
    """Build the association-specific single-trait callback."""
    if context.is_binary_trait:
        return callback_binary.BinaryRegenie2PipelineCallback(
            run_input=run_input,
            prediction_source=prediction_source,
            writer_session=writer_session,
            correction_plan=context.correction_plan,
            kernel_config=pipeline_context.require_binary_kernel_config(context.binary_kernel_config),
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
    return callback_linear.LinearRegenie2PipelineCallback(
        run_input=run_input,
        prediction_source=prediction_source,
        writer_session=writer_session,
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


def run_single_trait_bgen_pipeline(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    phenotype_name: str,
    covariate_names: tuple[str, ...] | None,
    output_run_paths: output.OutputRunPaths,
    existing_manifest: dict[str, typing.Any] | None,
    resume: bool,
    resume_mode: types.ResumeMode,
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
        engine = outputs.use_prepared_pipeline_bgen_engine(
            context=context,
            engine=prepared_engine,
            pipeline_label=pipeline_label,
            phenotype_name=phenotype_name,
            phenotype_count=None,
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
    resolved_compute_group = native_dispatch_groups.build_resolved_single_phenotype_compute_group(
        phenotype_name=phenotype_name,
        run_input=run_input,
        prediction_list_path=context.prediction_list_path,
        alignment_config=context.alignment_config,
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
        multi_phenotype_sample_mode=output.MultiPhenotypeSampleMode.SINGLE_PHENOTYPE,
        phenotype_compute_group=resolved_compute_group,
    )
    initialized_outputs = outputs.initialize_pipeline_output_runs(
        output_run_paths_by_trait=(output_run_paths,),
        existing_manifests_by_trait=(existing_manifest,),
        current_headers_by_trait=(current_header,),
        resume=resume,
        resume_mode=resume_mode,
        runtime_compatibility_token=context.runtime_compatibility_token,
    )
    outputs.notify_output_runs_initialized(context=context, phenotype_names=(phenotype_name,))
    writer_sessions = outputs.create_pipeline_writer_sessions(
        context=context,
        output_run_paths_by_trait=(output_run_paths,),
    )
    writer_session = writer_sessions[0]
    callback = build_single_trait_callback(
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
        committed_chunk_identifiers=initialized_outputs.committed_chunk_identifiers(0),
        writer_session=writer_session,
        callback=callback,
        stage_timing_recorder=context.stage_timing_recorder,
        variant_major_packed8_probability_pairs=context.uses_packed8_genotypes,
    )


def run_regenie2_linear_bgen_pipeline(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    phenotype_path: Path,
    phenotype_name: str,
    prediction_list_path: Path,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    chunk_size: int,
    variant_limit: int | None,
    output_run_paths: output.OutputRunPaths,
    staging_depth: int,
    native_callback_batch_size: int,
    result_in_flight_limit: int | None,
    dosage_buffer_limit: int | None,
    existing_manifest: dict[str, typing.Any] | None,
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
    runtime_compatibility_token: _core.NativeRuntimeCompatibilityToken,
    output_initialized_callback: typing.Callable[[tuple[str, ...]], None] | None,
) -> Path | None:
    """Run the native BGEN pipeline for quantitative REGENIE step 2."""
    resolved_gpu_genotype_format = gpu_format.resolve_auto_to_dosage(
        requested_gpu_genotype_format=gpu_genotype_format,
        telemetry_session=telemetry_session,
        resolution_reason="single_trait_linear",
    )
    context = pipeline_context.build_regenie2_pipeline_context(
        association_mode=types.AssociationMode.REGENIE2_LINEAR,
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
        correction_plan=types.BinaryCorrectionPlan(
            method=types.BinaryFallbackMethod.SCORE_ONLY,
            p_threshold=0.05,
            firth_se=False,
        ),
        binary_kernel_config=None,
        linear_numerical_config=linear_numerical_config,
        writer_settings=writer_settings,
        stage_timing_recorder=stage_timing_recorder,
        telemetry_session=telemetry_session,
        alignment_config=alignment_config,
        phenotype_compute_groups=execution_plan.build_phenotype_compute_groups(
            phenotype_names=(phenotype_name,),
            multi_phenotype_sample_mode=types.MultiPhenotypeSampleMode.PER_PHENOTYPE,
        ),
        runtime_compatibility_token=runtime_compatibility_token,
        output_initialized_callback=output_initialized_callback,
    )
    return run_single_trait_bgen_pipeline(
        context=context,
        phenotype_name=phenotype_name,
        covariate_names=covariate_names,
        output_run_paths=output_run_paths,
        existing_manifest=existing_manifest,
        resume=resume,
        resume_mode=resume_mode,
        staging_depth=staging_depth,
        native_callback_batch_size=native_callback_batch_size,
        result_in_flight_limit=result_in_flight_limit,
        dosage_buffer_limit=dosage_buffer_limit,
        null_logistic_nonconvergence_policy=types.NullLogisticNonconvergencePolicy.FAIL,
        prepared_engine=None,
    )


def run_regenie2_binary_bgen_pipeline(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    phenotype_path: Path,
    phenotype_name: str,
    prediction_list_path: Path,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    chunk_size: int,
    variant_limit: int | None,
    output_run_paths: output.OutputRunPaths,
    staging_depth: int,
    native_callback_batch_size: int,
    result_in_flight_limit: int | None,
    dosage_buffer_limit: int | None,
    existing_manifest: dict[str, typing.Any] | None,
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
    runtime_compatibility_token: _core.NativeRuntimeCompatibilityToken,
    output_initialized_callback: typing.Callable[[tuple[str, ...]], None] | None,
) -> Path | None:
    """Run the native BGEN pipeline for binary REGENIE step 2."""
    resolved_kernel_config = pipeline_context.require_binary_kernel_config(kernel_config)
    gpu_genotype_format_resolution = gpu_format.resolve_single_trait_binary_gpu_genotype_format(
        requested_gpu_genotype_format=gpu_genotype_format,
        existing_manifest=existing_manifest,
        resume=resume,
        jax_device=jax_device,
        genotype_source_config=genotype_source_config,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        trusted_bgen_validation_mode=trusted_bgen_validation_mode,
        stage_timing_recorder=stage_timing_recorder,
        telemetry_session=telemetry_session,
    )
    context = pipeline_context.build_regenie2_pipeline_context(
        association_mode=types.AssociationMode.REGENIE2_BINARY,
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
        requested_gpu_genotype_format=gpu_genotype_format_resolution.requested_gpu_genotype_format,
        gpu_genotype_format=gpu_genotype_format_resolution.resolved_gpu_genotype_format,
        correction_plan=correction_plan,
        binary_kernel_config=resolved_kernel_config,
        linear_numerical_config=None,
        writer_settings=writer_settings,
        stage_timing_recorder=stage_timing_recorder,
        telemetry_session=telemetry_session,
        alignment_config=alignment_config,
        phenotype_compute_groups=execution_plan.build_phenotype_compute_groups(
            phenotype_names=(phenotype_name,),
            multi_phenotype_sample_mode=types.MultiPhenotypeSampleMode.PER_PHENOTYPE,
        ),
        runtime_compatibility_token=runtime_compatibility_token,
        output_initialized_callback=output_initialized_callback,
    )
    return run_single_trait_bgen_pipeline(
        context=context,
        phenotype_name=phenotype_name,
        covariate_names=covariate_names,
        output_run_paths=output_run_paths,
        existing_manifest=existing_manifest,
        resume=resume,
        resume_mode=resume_mode,
        staging_depth=staging_depth,
        native_callback_batch_size=native_callback_batch_size,
        result_in_flight_limit=result_in_flight_limit,
        dosage_buffer_limit=dosage_buffer_limit,
        null_logistic_nonconvergence_policy=null_logistic_nonconvergence_policy,
        prepared_engine=gpu_genotype_format_resolution.prepared_engine,
    )
