"""Execution layer for REGENIE-compatible runs."""

from __future__ import annotations

import logging
import time
import typing
from dataclasses import dataclass

from g import _core, execution_plan, types
from g.engine import run_events, shutdown, telemetry, timing
from g.interface import config
from g.runner import metadata, runtime

if typing.TYPE_CHECKING:
    from pathlib import Path

    from g.io import output, source

logger = logging.getLogger(__name__)
RunArtifacts = run_events.RunArtifacts


@dataclass(frozen=True)
class CommonEngineDispatchRequest:
    """Arguments shared by single- and multi-phenotype engine dispatch.

    Attributes:
        genotype_source_config: BGEN source configuration.
        phenotype_path: Phenotype file path.
        prediction_list_path: REGENIE step 1 prediction list.
        covariate_path: Optional covariate file path.
        covariate_names: Optional covariate column names.
        chunk_size: Native variant chunk size.
        variant_limit: Optional variant cap.
        staging_depth: Native callback staging depth.
        native_callback_batch_size: Native-to-Python callback chunk batch size.
        result_in_flight_limit: Optional cap for materialization backlog.
        dosage_buffer_limit: Optional cap for native dosage decode buffers.
        resume: Whether output resume is enabled.
        resume_mode: Resume validation policy.
        writer_settings: Output writer settings.
        trusted_no_missing_diploid: Trusted BGEN fast-path policy.
        trusted_bgen_validation_mode: Trusted BGEN validation policy.
        bgen_decode_tile_variant_count: Native BGEN decode tile size.
        jax_device: Requested JAX device.
        jax_matmul_precision: Optional JAX matmul precision.
        score_dtype: Score-test compute dtype.
        firth_dtype: Firth compute dtype.
        stage_timing_recorder: Optional stage timing recorder.
        telemetry_session: Optional telemetry session.
        alignment_config: Sample alignment settings.
        runtime_compatibility_token: Native token proving process-global runtime checks passed.
        output_initialized_callback: Callback after manifest initialization.

    """

    genotype_source_config: source.GenotypeSourceConfig
    phenotype_path: Path
    prediction_list_path: Path
    covariate_path: Path | None
    covariate_names: tuple[str, ...] | None
    chunk_size: int
    variant_limit: int | None
    staging_depth: int
    native_callback_batch_size: int
    result_in_flight_limit: int | None
    dosage_buffer_limit: int | None
    resume: bool
    resume_mode: types.ResumeMode
    writer_settings: output.OutputWriterSettings
    trusted_no_missing_diploid: bool
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode
    bgen_decode_tile_variant_count: int
    jax_device: types.Device
    jax_matmul_precision: types.JaxMatmulPrecision | None
    score_dtype: types.FloatingPointDtype
    firth_dtype: types.FloatingPointDtype
    stage_timing_recorder: timing.StageTimingRecorder | None
    telemetry_session: telemetry.TelemetrySession | None
    alignment_config: typing.Any
    runtime_compatibility_token: _core.NativeRuntimeCompatibilityToken
    output_initialized_callback: typing.Callable[[tuple[str, ...]], None] | None


def regenie(
    regenie_config: config.RegenieConfig,
    *,
    run_telemetry_session: telemetry.TelemetrySession | None,
    close_telemetry_session_on_exit: bool,
    initialize_logging_on_entry: bool,
) -> RunArtifacts:
    """Run the shared REGENIE-compatible config path."""
    active_telemetry_session = run_telemetry_session or telemetry.build_telemetry_session(regenie_config)
    try:
        config.validate_config_for_run(regenie_config)
        runtime_policy = runtime.build_runtime_policy(regenie_config, active_telemetry_session.paths)
        run_runtime = runtime.build_run_runtime(runtime_policy)
        if initialize_logging_on_entry:
            runtime.initialize_logging(regenie_config.g_diagnostics, active_telemetry_session.paths)
        association_mode = execution_plan.resolve_association_mode(regenie_config.trait.trait_type)
        phenotype_count = len(regenie_config.input.pheno_columns)
        active_telemetry_session.log_event(
            "run_started",
            level="info",
            association_mode=association_mode.value,
            trait_type=regenie_config.trait.trait_type.value,
            phenotype_count=phenotype_count,
            output_run_root=str(telemetry.resolve_output_run_root(regenie_config)),
        )
        logger.info("Starting REGENIE run.")
        runtime.configure_runtime(regenie_config.g_compute, regenie_config.trait)
        artifacts = run_validated_regenie_config(
            regenie_config,
            telemetry_session=active_telemetry_session,
            runtime_compatibility_token=run_runtime.runtime_compatibility_token,
        )
    except shutdown.GracefulShutdownRequested as shutdown_request:
        interrupted_event = run_events.build_run_interrupted_event(shutdown_request)
        active_telemetry_session.log_run_interrupted(interrupted_event)
        logger.warning("REGENIE run interrupted by %s.", interrupted_event.signal_name)
        raise
    except Exception as error:
        failed_event = run_events.build_run_failed_event(error)
        active_telemetry_session.log_run_failed(failed_event)
        logger.exception("REGENIE run failed.")
        raise
    else:
        artifacts = run_events.attach_run_metadata(
            artifacts,
            run_id=active_telemetry_session.run_id,
            association_mode=association_mode,
            phenotype_count=phenotype_count,
        )
        completed_event = run_events.build_run_completed_event(artifacts)
        active_telemetry_session.log_run_completed(completed_event)
        logger.info("Finished REGENIE run.")
        return artifacts
    finally:
        if close_telemetry_session_on_exit:
            telemetry.close_telemetry_session(active_telemetry_session)


def run_validated_regenie_config(
    regenie_config: config.RegenieConfig,
    telemetry_session: telemetry.TelemetrySession | None,
    runtime_compatibility_token: _core.NativeRuntimeCompatibilityToken,
) -> RunArtifacts:
    """Plan, execute, and finalize a validated REGENIE-compatible config."""
    api_entry_start_time = time.perf_counter()
    stage_timing_recorder = None
    stage_timing_path = (
        regenie_config.g_diagnostics.stage_timings_json
        if telemetry_session is None
        else telemetry_session.paths.stage_timings_json
    )
    profile_summary_path = None if telemetry_session is None else telemetry_session.paths.profile_summary_json
    try:
        device_start_time = time.perf_counter()
        logger.debug("Configuring JAX runtime before backend initialization.")
        runtime.configure_runtime_before_jax_import(regenie_config.g_compute, telemetry_session=telemetry_session)
        stage_timing_recorder = timing.build_stage_timing_recorder(
            stage_timing_path,
            force=telemetry_session is not None and telemetry_session.profile_enabled,
        )
        timing.record_stage_duration(stage_timing_recorder, "jax_device_configuration_backend_init", device_start_time)
        output_start_time = time.perf_counter()
        logger.debug("Building REGENIE execution plan.")
        plan = execution_plan.build_regenie_execution_plan(
            regenie_config,
            runtime_compatibility_token=runtime_compatibility_token,
        )
        if telemetry_session is not None:
            telemetry_session.log_event(
                "execution_plan_prepared",
                level="info",
                association_mode=plan.association_mode.value,
                trait_type=regenie_config.trait.trait_type.value,
                phenotype_count=len(plan.phenotype_run_plans),
                chunk_size=plan.kernel_config.chunk_size,
                variant_limit=plan.kernel_config.variant_limit,
                device=plan.kernel_config.device.value,
            )
        logger.info("Prepared REGENIE execution plan for %s phenotype(s).", len(plan.phenotype_run_plans))
        timing.record_stage_duration(stage_timing_recorder, "output_run_preparation", output_start_time)
        logger.debug("Dispatching REGENIE execution plan.")
        final_output_paths = dispatch_execution_plan(
            regenie_config=regenie_config,
            plan=plan,
            stage_timing_recorder=stage_timing_recorder,
            telemetry_session=telemetry_session,
            runtime_compatibility_token=runtime_compatibility_token,
        )
        logger.debug("Finalizing REGENIE execution plan.")
        return metadata.finalize_execution_plan(
            regenie_config=regenie_config,
            plan=plan,
            final_output_paths=final_output_paths,
        )
    finally:
        if stage_timing_recorder is not None:
            timing.record_stage_duration(stage_timing_recorder, "python_api_entry", api_entry_start_time)
            logger.debug("Writing final stage timing snapshot.")
            timing.write_stage_timing_snapshot(stage_timing_recorder, stage_timing_path)
            timing.write_profile_summary(
                stage_timing_recorder,
                profile_summary_path,
                run_id=None if telemetry_session is None else telemetry_session.run_id,
            )


def dispatch_execution_plan(
    *,
    regenie_config: config.RegenieConfig,
    plan: execution_plan.RegenieExecutionPlan,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    telemetry_session: telemetry.TelemetrySession | None,
    runtime_compatibility_token: _core.NativeRuntimeCompatibilityToken,
) -> tuple[Path | None, ...]:
    """Dispatch an execution plan to the native engine layer."""
    output_initialized_callback = metadata.build_output_initialized_metadata_callback(
        regenie_config=regenie_config,
        plan=plan,
        telemetry_session=telemetry_session,
    )
    if len(plan.phenotype_run_plans) > 1:
        logger.debug("Dispatching multi-phenotype native engine pipeline.")
        return dispatch_multi_phenotype_engine_pipeline(
            plan=plan,
            stage_timing_recorder=stage_timing_recorder,
            telemetry_session=telemetry_session,
            runtime_compatibility_token=runtime_compatibility_token,
            output_initialized_callback=output_initialized_callback,
        )
    logger.debug("Dispatching single-phenotype native engine pipeline.")
    return (
        dispatch_one_phenotype_engine_pipeline(
            plan=plan,
            phenotype_run_plan=plan.phenotype_run_plans[0],
            stage_timing_recorder=stage_timing_recorder,
            telemetry_session=telemetry_session,
            runtime_compatibility_token=runtime_compatibility_token,
            output_initialized_callback=output_initialized_callback,
        ),
    )


def build_common_engine_dispatch_request(
    *,
    plan: execution_plan.RegenieExecutionPlan,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    telemetry_session: telemetry.TelemetrySession | None,
    runtime_compatibility_token: _core.NativeRuntimeCompatibilityToken,
    output_initialized_callback: typing.Callable[[tuple[str, ...]], None] | None,
) -> CommonEngineDispatchRequest:
    """Build shared engine dispatch arguments."""
    return CommonEngineDispatchRequest(
        genotype_source_config=plan.genotype_source_config,
        phenotype_path=plan.phenotype_path,
        prediction_list_path=plan.prediction_list_path,
        covariate_path=plan.covariate_path,
        covariate_names=plan.covariate_names,
        chunk_size=plan.kernel_config.chunk_size,
        variant_limit=plan.kernel_config.variant_limit,
        staging_depth=plan.kernel_config.staging_depth,
        native_callback_batch_size=plan.kernel_config.native_callback_batch_size,
        result_in_flight_limit=plan.kernel_config.result_in_flight_limit,
        dosage_buffer_limit=plan.kernel_config.dosage_buffer_limit,
        resume=plan.output_plan.resume,
        resume_mode=plan.output_plan.resume_mode,
        writer_settings=plan.output_plan.writer_settings,
        trusted_no_missing_diploid=plan.kernel_config.trusted_no_missing_diploid,
        trusted_bgen_validation_mode=plan.kernel_config.trusted_bgen_validation_mode,
        bgen_decode_tile_variant_count=plan.kernel_config.bgen_decode_tile_variant_count,
        jax_device=plan.kernel_config.device,
        jax_matmul_precision=plan.kernel_config.alignment_config.jax_matmul_precision,
        score_dtype=plan.kernel_config.alignment_config.score_dtype,
        firth_dtype=plan.kernel_config.alignment_config.firth_dtype,
        stage_timing_recorder=stage_timing_recorder,
        telemetry_session=telemetry_session,
        alignment_config=plan.kernel_config.alignment_config,
        runtime_compatibility_token=runtime_compatibility_token,
        output_initialized_callback=output_initialized_callback,
    )


def dispatch_one_phenotype_engine_pipeline(
    *,
    plan: execution_plan.RegenieExecutionPlan,
    phenotype_run_plan: execution_plan.PhenotypeRunPlan,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    telemetry_session: telemetry.TelemetrySession | None,
    runtime_compatibility_token: _core.NativeRuntimeCompatibilityToken,
    output_initialized_callback: typing.Callable[[tuple[str, ...]], None] | None,
) -> Path | None:
    """Dispatch one phenotype to the native linear or binary pipeline."""
    common_request = build_common_engine_dispatch_request(
        plan=plan,
        stage_timing_recorder=stage_timing_recorder,
        telemetry_session=telemetry_session,
        runtime_compatibility_token=runtime_compatibility_token,
        output_initialized_callback=output_initialized_callback,
    )
    if plan.association_mode == types.AssociationMode.REGENIE2_BINARY:
        logger.debug("Dispatching binary native engine pipeline.")
        final_output_path = runtime.run_regenie2_binary_bgen_pipeline(
            genotype_source_config=common_request.genotype_source_config,
            phenotype_path=common_request.phenotype_path,
            phenotype_name=phenotype_run_plan.phenotype_name,
            prediction_list_path=common_request.prediction_list_path,
            covariate_path=common_request.covariate_path,
            covariate_names=common_request.covariate_names,
            chunk_size=common_request.chunk_size,
            variant_limit=common_request.variant_limit,
            output_run_paths=phenotype_run_plan.output_run_paths,
            staging_depth=common_request.staging_depth,
            native_callback_batch_size=common_request.native_callback_batch_size,
            result_in_flight_limit=common_request.result_in_flight_limit,
            dosage_buffer_limit=common_request.dosage_buffer_limit,
            existing_manifest=phenotype_run_plan.existing_manifest,
            resume=common_request.resume,
            resume_mode=common_request.resume_mode,
            writer_settings=common_request.writer_settings,
            trusted_no_missing_diploid=common_request.trusted_no_missing_diploid,
            trusted_bgen_validation_mode=common_request.trusted_bgen_validation_mode,
            bgen_decode_tile_variant_count=common_request.bgen_decode_tile_variant_count,
            jax_device=common_request.jax_device,
            jax_matmul_precision=common_request.jax_matmul_precision,
            score_dtype=common_request.score_dtype,
            firth_dtype=common_request.firth_dtype,
            correction_plan=plan.binary_correction_plan,
            kernel_config=plan.kernel_config.binary_kernel_config,
            null_logistic_nonconvergence_policy=(
                plan.kernel_config.alignment_config.null_logistic_nonconvergence_policy
            ),
            gpu_genotype_format=plan.kernel_config.gpu_genotype_format,
            stage_timing_recorder=common_request.stage_timing_recorder,
            telemetry_session=common_request.telemetry_session,
            alignment_config=common_request.alignment_config,
            runtime_compatibility_token=common_request.runtime_compatibility_token,
            output_initialized_callback=common_request.output_initialized_callback,
        )
        metadata.log_writer_finished(
            telemetry_session=telemetry_session,
            association_mode=plan.association_mode,
            phenotype=phenotype_run_plan.phenotype_name,
            final_output_path=final_output_path,
        )
        return final_output_path
    logger.debug("Dispatching linear native engine pipeline.")
    final_output_path = runtime.run_regenie2_linear_bgen_pipeline(
        genotype_source_config=common_request.genotype_source_config,
        phenotype_path=common_request.phenotype_path,
        phenotype_name=phenotype_run_plan.phenotype_name,
        prediction_list_path=common_request.prediction_list_path,
        covariate_path=common_request.covariate_path,
        covariate_names=common_request.covariate_names,
        chunk_size=common_request.chunk_size,
        variant_limit=common_request.variant_limit,
        output_run_paths=phenotype_run_plan.output_run_paths,
        staging_depth=common_request.staging_depth,
        native_callback_batch_size=common_request.native_callback_batch_size,
        result_in_flight_limit=common_request.result_in_flight_limit,
        dosage_buffer_limit=common_request.dosage_buffer_limit,
        existing_manifest=phenotype_run_plan.existing_manifest,
        resume=common_request.resume,
        resume_mode=common_request.resume_mode,
        writer_settings=common_request.writer_settings,
        trusted_no_missing_diploid=common_request.trusted_no_missing_diploid,
        trusted_bgen_validation_mode=common_request.trusted_bgen_validation_mode,
        bgen_decode_tile_variant_count=common_request.bgen_decode_tile_variant_count,
        jax_device=common_request.jax_device,
        jax_matmul_precision=common_request.jax_matmul_precision,
        score_dtype=common_request.score_dtype,
        firth_dtype=common_request.firth_dtype,
        gpu_genotype_format=plan.kernel_config.gpu_genotype_format,
        linear_numerical_config=plan.kernel_config.linear_numerical_config,
        stage_timing_recorder=common_request.stage_timing_recorder,
        telemetry_session=common_request.telemetry_session,
        alignment_config=common_request.alignment_config,
        runtime_compatibility_token=common_request.runtime_compatibility_token,
        output_initialized_callback=common_request.output_initialized_callback,
    )
    metadata.log_writer_finished(
        telemetry_session=telemetry_session,
        association_mode=plan.association_mode,
        phenotype=phenotype_run_plan.phenotype_name,
        final_output_path=final_output_path,
    )
    return final_output_path


def dispatch_multi_phenotype_engine_pipeline(
    *,
    plan: execution_plan.RegenieExecutionPlan,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    telemetry_session: telemetry.TelemetrySession | None,
    runtime_compatibility_token: _core.NativeRuntimeCompatibilityToken,
    output_initialized_callback: typing.Callable[[tuple[str, ...]], None] | None,
) -> tuple[Path | None, ...]:
    """Dispatch multiple phenotypes to the shared native pipeline."""
    common_request = build_common_engine_dispatch_request(
        plan=plan,
        stage_timing_recorder=stage_timing_recorder,
        telemetry_session=telemetry_session,
        runtime_compatibility_token=runtime_compatibility_token,
        output_initialized_callback=output_initialized_callback,
    )
    phenotype_names = tuple(phenotype_run_plan.phenotype_name for phenotype_run_plan in plan.phenotype_run_plans)
    output_run_paths_by_phenotype = tuple(
        phenotype_run_plan.output_run_paths for phenotype_run_plan in plan.phenotype_run_plans
    )
    existing_manifests_by_phenotype = tuple(
        phenotype_run_plan.existing_manifest for phenotype_run_plan in plan.phenotype_run_plans
    )
    if plan.association_mode == types.AssociationMode.REGENIE2_BINARY:
        logger.debug("Dispatching multi-phenotype binary native engine pipeline.")
        final_output_paths = runtime.run_regenie2_multi_phenotype_binary_bgen_pipeline(
            genotype_source_config=common_request.genotype_source_config,
            phenotype_path=common_request.phenotype_path,
            phenotype_names=phenotype_names,
            prediction_list_path=common_request.prediction_list_path,
            covariate_path=common_request.covariate_path,
            covariate_names=common_request.covariate_names,
            chunk_size=common_request.chunk_size,
            variant_limit=common_request.variant_limit,
            output_run_paths_by_phenotype=output_run_paths_by_phenotype,
            staging_depth=common_request.staging_depth,
            native_callback_batch_size=common_request.native_callback_batch_size,
            result_in_flight_limit=common_request.result_in_flight_limit,
            dosage_buffer_limit=common_request.dosage_buffer_limit,
            existing_manifests_by_phenotype=existing_manifests_by_phenotype,
            resume=common_request.resume,
            resume_mode=common_request.resume_mode,
            writer_settings=common_request.writer_settings,
            trusted_no_missing_diploid=common_request.trusted_no_missing_diploid,
            trusted_bgen_validation_mode=common_request.trusted_bgen_validation_mode,
            bgen_decode_tile_variant_count=common_request.bgen_decode_tile_variant_count,
            jax_device=common_request.jax_device,
            jax_matmul_precision=common_request.jax_matmul_precision,
            score_dtype=common_request.score_dtype,
            firth_dtype=common_request.firth_dtype,
            correction_plan=plan.binary_correction_plan,
            kernel_config=plan.kernel_config.binary_kernel_config,
            null_logistic_nonconvergence_policy=(
                plan.kernel_config.alignment_config.null_logistic_nonconvergence_policy
            ),
            gpu_genotype_format=plan.kernel_config.gpu_genotype_format,
            stage_timing_recorder=common_request.stage_timing_recorder,
            telemetry_session=common_request.telemetry_session,
            alignment_config=common_request.alignment_config,
            runtime_compatibility_token=common_request.runtime_compatibility_token,
            sample_mode=plan.kernel_config.multi_phenotype_sample_mode,
            phenotype_compute_groups=plan.phenotype_compute_groups,
            output_initialized_callback=common_request.output_initialized_callback,
        )
    else:
        logger.debug("Dispatching multi-phenotype linear native engine pipeline.")
        final_output_paths = runtime.run_regenie2_multi_phenotype_linear_bgen_pipeline(
            genotype_source_config=common_request.genotype_source_config,
            phenotype_path=common_request.phenotype_path,
            phenotype_names=phenotype_names,
            prediction_list_path=common_request.prediction_list_path,
            covariate_path=common_request.covariate_path,
            covariate_names=common_request.covariate_names,
            chunk_size=common_request.chunk_size,
            variant_limit=common_request.variant_limit,
            output_run_paths_by_phenotype=output_run_paths_by_phenotype,
            staging_depth=common_request.staging_depth,
            native_callback_batch_size=common_request.native_callback_batch_size,
            result_in_flight_limit=common_request.result_in_flight_limit,
            dosage_buffer_limit=common_request.dosage_buffer_limit,
            existing_manifests_by_phenotype=existing_manifests_by_phenotype,
            resume=common_request.resume,
            resume_mode=common_request.resume_mode,
            writer_settings=common_request.writer_settings,
            trusted_no_missing_diploid=common_request.trusted_no_missing_diploid,
            trusted_bgen_validation_mode=common_request.trusted_bgen_validation_mode,
            bgen_decode_tile_variant_count=common_request.bgen_decode_tile_variant_count,
            jax_device=common_request.jax_device,
            jax_matmul_precision=common_request.jax_matmul_precision,
            score_dtype=common_request.score_dtype,
            firth_dtype=common_request.firth_dtype,
            linear_numerical_config=plan.kernel_config.linear_numerical_config,
            gpu_genotype_format=plan.kernel_config.gpu_genotype_format,
            stage_timing_recorder=common_request.stage_timing_recorder,
            telemetry_session=common_request.telemetry_session,
            alignment_config=common_request.alignment_config,
            runtime_compatibility_token=common_request.runtime_compatibility_token,
            sample_mode=plan.kernel_config.multi_phenotype_sample_mode,
            phenotype_compute_groups=plan.phenotype_compute_groups,
            output_initialized_callback=common_request.output_initialized_callback,
        )
    if telemetry_session is not None:
        telemetry_session.log_event(
            "writer_finished",
            level="info",
            association_mode=plan.association_mode.value,
            phenotype_count=len(plan.phenotype_run_plans),
            final_output_paths=tuple(None if path is None else str(path) for path in final_output_paths),
        )
    return final_output_paths
