"""Execution layer for REGENIE-compatible runs."""

from __future__ import annotations

import time
import typing

from g import _core, execution_plan, types
from g.engine import dispatch_requests
from g.interface import config
from g.runner import events, lifecycle, metadata, outputs, runtime, timing

if typing.TYPE_CHECKING:
    from pathlib import Path


def regenie(
    regenie_config: config.RegenieConfig,
    *,
    run_telemetry_session: events.TelemetrySession | None,
    close_telemetry_session_on_exit: bool,
    initialize_logging_on_entry: bool,
) -> events.RunArtifacts:
    """Run the shared REGENIE-compatible config path."""
    active_telemetry_session = run_telemetry_session or events.build_telemetry_session(regenie_config)
    try:
        config.validate_config_for_run(regenie_config)
        runtime_policy = runtime.build_runtime_policy(regenie_config, active_telemetry_session.paths)
        run_runtime = runtime.build_run_runtime(runtime_policy)
        if initialize_logging_on_entry:
            runtime.initialize_logging(regenie_config.g_diagnostics, active_telemetry_session.paths)
        association_mode = execution_plan.resolve_association_mode(regenie_config.trait.trait_type)
        phenotype_count = len(regenie_config.input.pheno_columns)
        events.record_runner_run_started(
            active_telemetry_session,
            association_mode=association_mode,
            trait_type=regenie_config.trait.trait_type,
            phenotype_count=phenotype_count,
            output_run_root=events.resolve_output_run_root(regenie_config),
        )
        runtime.configure_runtime(regenie_config.g_compute, regenie_config.trait)
        artifacts = run_validated_regenie_config(
            regenie_config,
            telemetry_session=active_telemetry_session,
            runtime_compatibility_token=run_runtime.runtime_compatibility_token,
        )
    except lifecycle.GracefulShutdownRequested as shutdown_request:
        interrupted_event = events.build_run_interrupted_event(shutdown_request)
        events.record_runner_run_interrupted(active_telemetry_session, interrupted_event)
        raise
    except Exception as error:
        failed_event = events.build_run_failed_event(error)
        events.record_runner_run_failed(active_telemetry_session, failed_event)
        raise
    else:
        artifacts = events.attach_run_metadata(
            artifacts,
            run_id=active_telemetry_session.run_id,
            association_mode=association_mode,
            phenotype_count=phenotype_count,
        )
        completed_event = events.build_run_completed_event(artifacts)
        events.record_runner_run_completed(active_telemetry_session, completed_event)
        return artifacts
    finally:
        if close_telemetry_session_on_exit:
            events.close_telemetry_session(active_telemetry_session)


def run_validated_regenie_config(
    regenie_config: config.RegenieConfig,
    telemetry_session: events.TelemetrySession | None,
    runtime_compatibility_token: _core.NativeRuntimeCompatibilityToken,
) -> events.RunArtifacts:
    """Plan, execute, and finalize a validated REGENIE-compatible config."""
    api_entry_start_time = time.perf_counter()
    stage_timing_recorder = None
    final_timing_context = timing.resolve_final_timing_output_context(
        regenie_config.g_diagnostics.stage_timings_json,
        telemetry_session,
    )
    try:
        device_start_time = time.perf_counter()
        events.record_runner_jax_runtime_configuration_started()
        runtime.configure_runtime_before_jax_import(regenie_config.g_compute, telemetry_session=telemetry_session)
        stage_timing_recorder = timing.build_stage_timing_recorder(
            final_timing_context.stage_timing_path,
            force=final_timing_context.force_stage_timing_recorder,
        )
        timing.record_stage_duration(stage_timing_recorder, "jax_device_configuration_backend_init", device_start_time)
        output_start_time = time.perf_counter()
        events.record_runner_execution_plan_build_started()
        plan = execution_plan.build_regenie_execution_plan(regenie_config)
        phenotype_run_plans = outputs.prepare_execution_plan_outputs(
            plan=plan,
            runtime_compatibility_token=runtime_compatibility_token,
        )
        events.record_execution_plan_prepared(
            telemetry_session,
            association_mode=plan.association_mode,
            trait_type=regenie_config.trait.trait_type,
            phenotype_count=len(phenotype_run_plans),
            chunk_size=plan.kernel_config.chunk_size,
            variant_limit=plan.kernel_config.variant_limit,
            device=plan.kernel_config.device,
        )
        timing.record_stage_duration(stage_timing_recorder, "output_run_preparation", output_start_time)
        events.record_runner_execution_plan_dispatch_started(
            phenotype_count=len(phenotype_run_plans),
            association_mode=plan.association_mode,
        )
        final_output_paths = dispatch_execution_plan(
            regenie_config=regenie_config,
            plan=plan,
            phenotype_run_plans=phenotype_run_plans,
            stage_timing_recorder=stage_timing_recorder,
            telemetry_session=telemetry_session,
            runtime_compatibility_token=runtime_compatibility_token,
        )
        events.record_runner_execution_plan_finalization_started(
            phenotype_count=len(phenotype_run_plans),
            association_mode=plan.association_mode,
        )
        return metadata.finalize_execution_plan(
            regenie_config=regenie_config,
            plan=plan,
            phenotype_run_plans=phenotype_run_plans,
            final_output_paths=final_output_paths,
        )
    finally:
        if stage_timing_recorder is not None:
            timing.record_stage_duration(stage_timing_recorder, "python_api_entry", api_entry_start_time)
            timing.record_final_timing_outputs_write_started_diagnostic_event(
                final_timing_context.stage_timing_path,
                final_timing_context.profile_summary_path,
                final_timing_context.run_id,
            )
            timing.write_final_timing_outputs(
                stage_timing_recorder,
                stage_timing_path=final_timing_context.stage_timing_path,
                profile_summary_path=final_timing_context.profile_summary_path,
                run_id=final_timing_context.run_id,
            )


def dispatch_execution_plan(
    *,
    regenie_config: config.RegenieConfig,
    plan: execution_plan.RegenieExecutionPlan,
    phenotype_run_plans: tuple[outputs.PreparedPhenotypeRunPlan, ...],
    stage_timing_recorder: timing.StageTimingRecorder | None,
    telemetry_session: events.TelemetrySession | None,
    runtime_compatibility_token: _core.NativeRuntimeCompatibilityToken,
) -> tuple[Path | None, ...]:
    """Dispatch an execution plan to the native engine layer."""
    output_initialized_callback = metadata.build_output_initialized_metadata_callback(
        regenie_config=regenie_config,
        plan=plan,
        phenotype_run_plans=phenotype_run_plans,
        telemetry_session=telemetry_session,
    )
    if len(phenotype_run_plans) > 1:
        events.record_runner_multi_phenotype_dispatch_started(
            phenotype_count=len(phenotype_run_plans),
            association_mode=plan.association_mode,
        )
        return dispatch_multi_phenotype_engine_pipeline(
            plan=plan,
            phenotype_run_plans=phenotype_run_plans,
            stage_timing_recorder=stage_timing_recorder,
            telemetry_session=telemetry_session,
            runtime_compatibility_token=runtime_compatibility_token,
            output_initialized_callback=output_initialized_callback,
        )
    events.record_runner_single_phenotype_dispatch_started(
        association_mode=plan.association_mode,
        phenotype=phenotype_run_plans[0].phenotype_name,
    )
    return (
        dispatch_one_phenotype_engine_pipeline(
            plan=plan,
            phenotype_run_plan=phenotype_run_plans[0],
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
    telemetry_session: events.TelemetrySession | None,
    runtime_compatibility_token: _core.NativeRuntimeCompatibilityToken,
    output_initialized_callback: typing.Callable[[tuple[str, ...]], None] | None,
) -> dispatch_requests.PipelineCommonRequest:
    """Build shared engine dispatch arguments."""
    return dispatch_requests.PipelineCommonRequest(
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
        writer_settings=outputs.output_writer_settings_from_plan(plan.output_plan.writer_settings),
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
    phenotype_run_plan: outputs.PreparedPhenotypeRunPlan,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    telemetry_session: events.TelemetrySession | None,
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
        events.record_runner_binary_engine_dispatch_started(phenotype_run_plan.phenotype_name)
        final_output_path = runtime.run_regenie2_binary_bgen_pipeline(
            dispatch_requests.SingleTraitPipelineRequest(
                common=common_request,
                phenotype_name=phenotype_run_plan.phenotype_name,
                output_run_paths=phenotype_run_plan.output_run_paths,
                existing_manifest=phenotype_run_plan.existing_manifest,
                association_mode=plan.association_mode,
                correction_plan=plan.binary_correction_plan,
                binary_kernel_config=plan.kernel_config.binary_kernel_config,
                linear_numerical_config=None,
                gpu_genotype_format=plan.kernel_config.gpu_genotype_format,
                null_logistic_nonconvergence_policy=(
                    plan.kernel_config.alignment_config.null_logistic_nonconvergence_policy
                ),
            ),
        )
        metadata.log_writer_finished(
            telemetry_session=telemetry_session,
            association_mode=plan.association_mode,
            phenotype=phenotype_run_plan.phenotype_name,
            final_output_path=final_output_path,
        )
        return final_output_path
    events.record_runner_linear_engine_dispatch_started(phenotype_run_plan.phenotype_name)
    final_output_path = runtime.run_regenie2_linear_bgen_pipeline(
        dispatch_requests.SingleTraitPipelineRequest(
            common=common_request,
            phenotype_name=phenotype_run_plan.phenotype_name,
            output_run_paths=phenotype_run_plan.output_run_paths,
            existing_manifest=phenotype_run_plan.existing_manifest,
            association_mode=plan.association_mode,
            correction_plan=types.BinaryCorrectionPlan(
                method=types.BinaryFallbackMethod.SCORE_ONLY,
                p_threshold=0.05,
                firth_se=False,
            ),
            binary_kernel_config=None,
            linear_numerical_config=plan.kernel_config.linear_numerical_config,
            gpu_genotype_format=plan.kernel_config.gpu_genotype_format,
            null_logistic_nonconvergence_policy=types.NullLogisticNonconvergencePolicy.FAIL,
        )
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
    phenotype_run_plans: tuple[outputs.PreparedPhenotypeRunPlan, ...],
    stage_timing_recorder: timing.StageTimingRecorder | None,
    telemetry_session: events.TelemetrySession | None,
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
    phenotype_names = tuple(phenotype_run_plan.phenotype_name for phenotype_run_plan in phenotype_run_plans)
    output_run_paths_by_phenotype = tuple(
        phenotype_run_plan.output_run_paths for phenotype_run_plan in phenotype_run_plans
    )
    existing_manifests_by_phenotype = tuple(
        phenotype_run_plan.existing_manifest for phenotype_run_plan in phenotype_run_plans
    )
    if plan.association_mode == types.AssociationMode.REGENIE2_BINARY:
        events.record_runner_multi_phenotype_binary_engine_dispatch_started(len(phenotype_names))
        final_output_paths = runtime.run_regenie2_multi_phenotype_binary_bgen_pipeline(
            dispatch_requests.MultiTraitPipelineRequest(
                common=common_request,
                phenotype_names=phenotype_names,
                output_run_paths_by_phenotype=output_run_paths_by_phenotype,
                existing_manifests_by_phenotype=existing_manifests_by_phenotype,
                association_mode=plan.association_mode,
                correction_plan=plan.binary_correction_plan,
                binary_kernel_config=plan.kernel_config.binary_kernel_config,
                linear_numerical_config=None,
                gpu_genotype_format=plan.kernel_config.gpu_genotype_format,
                null_logistic_nonconvergence_policy=(
                    plan.kernel_config.alignment_config.null_logistic_nonconvergence_policy
                ),
                sample_mode=plan.kernel_config.multi_phenotype_sample_mode,
                phenotype_compute_groups=plan.phenotype_compute_groups,
            ),
        )
    else:
        events.record_runner_multi_phenotype_linear_engine_dispatch_started(len(phenotype_names))
        final_output_paths = runtime.run_regenie2_multi_phenotype_linear_bgen_pipeline(
            dispatch_requests.MultiTraitPipelineRequest(
                common=common_request,
                phenotype_names=phenotype_names,
                output_run_paths_by_phenotype=output_run_paths_by_phenotype,
                existing_manifests_by_phenotype=existing_manifests_by_phenotype,
                association_mode=plan.association_mode,
                correction_plan=types.BinaryCorrectionPlan(
                    method=types.BinaryFallbackMethod.SCORE_ONLY,
                    p_threshold=0.05,
                    firth_se=False,
                ),
                binary_kernel_config=None,
                linear_numerical_config=plan.kernel_config.linear_numerical_config,
                gpu_genotype_format=plan.kernel_config.gpu_genotype_format,
                null_logistic_nonconvergence_policy=types.NullLogisticNonconvergencePolicy.FAIL,
                sample_mode=plan.kernel_config.multi_phenotype_sample_mode,
                phenotype_compute_groups=plan.phenotype_compute_groups,
            )
        )
    events.record_multi_writer_finished(
        telemetry_session,
        association_mode=plan.association_mode,
        phenotype_count=len(phenotype_run_plans),
        final_output_paths=final_output_paths,
    )
    return final_output_paths
