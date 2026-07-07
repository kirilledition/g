"""Execution layer for REGENIE-compatible runs."""

from __future__ import annotations

import time
import typing
from pathlib import Path

from g import _core, execution_plan, types
from g.engine import dispatch_requests
from g.engine import timing as engine_timing
from g.runner import events, lifecycle, runtime


def regenie(
    regenie_config: _core.RegenieConfig,
    *,
    run_telemetry_session: events.TelemetrySession | None,
    close_telemetry_session_on_exit: bool,
    initialize_logging_on_entry: bool,
) -> events.RunArtifacts:
    """Run the shared REGENIE-compatible config path."""
    active_telemetry_session = run_telemetry_session or events.build_telemetry_session(regenie_config)
    try:
        _core.validate_regenie_config_for_run(regenie_config)
        runtime_policy = runtime.build_runtime_policy(
            runtime.RuntimePolicyRequest(
                diagnostics_config=regenie_config.g_diagnostics,
                compute_config=regenie_config.g_compute,
                rayon_thread_count=regenie_config.trait.threads,
                telemetry_paths=active_telemetry_session.paths,
            )
        )
        run_runtime = runtime.build_run_runtime(runtime_policy)
        if initialize_logging_on_entry:
            runtime.initialize_logging(regenie_config.g_diagnostics, active_telemetry_session.paths)
        association_mode = association_mode_from_trait_type(regenie_config.trait.trait_type)
        phenotype_count = len(regenie_config.input.pheno_columns)
        output_run_directory = regenie_config.g_output.output_run_directory
        _core.record_runner_run_started_events(
            active_telemetry_session,
            association_mode.value,
            regenie_config.trait.trait_type.value,
            phenotype_count,
            _core.resolve_output_run_root_value(
                str(typing.cast("Path", regenie_config.g_output.out)),
                None if output_run_directory is None else str(output_run_directory),
            ),
        )
        runtime.PROCESS_RUNTIME_STATE.configure_runtime_knobs(
            regenie_config.g_compute.bgen_decode_tile_variant_count,
            regenie_config.trait.threads,
        )
        artifacts = run_validated_regenie_config(
            regenie_config,
            telemetry_session=active_telemetry_session,
            runtime_compatibility_token=run_runtime.runtime_compatibility_token,
        )
    except lifecycle.GracefulShutdownRequested as shutdown_request:
        _core.record_runner_run_interrupted_events(active_telemetry_session, shutdown_request)
        raise
    except Exception as error:
        _core.record_runner_run_failed_events(active_telemetry_session, error)
        raise
    else:
        native_artifacts = _core.attach_run_metadata_and_record_completed_events(
            active_telemetry_session,
            artifacts,
            active_telemetry_session.run_id,
            association_mode.value,
            phenotype_count,
        )
        return events.run_artifacts_from_native_artifacts(native_artifacts)
    finally:
        if close_telemetry_session_on_exit:
            native_telemetry_session = active_telemetry_session.native_telemetry_session
            if native_telemetry_session is not None:
                native_telemetry_session.finish_with_current_close_event_metadata()


def association_mode_from_trait_type(trait_type: types.RegenieTraitType) -> types.AssociationMode:
    """Resolve the association mode implied by the configured trait type."""
    if trait_type == types.RegenieTraitType.BINARY:
        return types.AssociationMode.REGENIE2_BINARY
    return types.AssociationMode.REGENIE2_LINEAR


def run_validated_regenie_config(
    regenie_config: _core.RegenieConfig,
    telemetry_session: events.TelemetrySession | None,
    runtime_compatibility_token: _core.NativeRuntimeCompatibilityToken,
) -> events.RunArtifacts:
    """Plan, execute, and finalize a validated REGENIE-compatible config."""
    api_entry_start_time = time.perf_counter()
    stage_timing_recorder = None
    native_final_timing_context = _core.resolve_final_timing_output_context(
        None
        if regenie_config.g_diagnostics.stage_timings_json is None
        else str(regenie_config.g_diagnostics.stage_timings_json),
        telemetry_session,
    )
    final_stage_timing_path = (
        None
        if native_final_timing_context.stage_timing_path is None
        else Path(native_final_timing_context.stage_timing_path)
    )
    final_profile_summary_path = (
        None
        if native_final_timing_context.profile_summary_path is None
        else Path(native_final_timing_context.profile_summary_path)
    )
    try:
        device_start_time = time.perf_counter()
        _core.record_runner_jax_runtime_configuration_started_diagnostic_event()
        runtime.configure_runtime_before_jax_import(regenie_config.g_compute, telemetry_session=telemetry_session)
        stage_timing_recorder = engine_timing.build_stage_timing_recorder(
            final_stage_timing_path,
            force=native_final_timing_context.force_stage_timing_recorder,
        )
        engine_timing.record_stage_duration(
            stage_timing_recorder, "jax_device_configuration_backend_init", device_start_time
        )
        output_start_time = time.perf_counter()
        _core.record_runner_execution_plan_build_started_diagnostic_event()
        lifecycle_session = _core.NativeRunLifecycleSession(regenie_config, runtime_compatibility_token)
        plan = execution_plan.build_regenie_execution_plan_from_run_request(
            regenie_config,
            lifecycle_session.run_request,
        )
        phenotype_run_plans = lifecycle_session.prepared_phenotype_runs()
        _core.record_execution_plan_prepared_events(
            telemetry_session,
            plan.association_mode.value,
            regenie_config.trait.trait_type.value,
            len(phenotype_run_plans),
            plan.kernel_config.chunk_size,
            plan.kernel_config.variant_limit,
            plan.kernel_config.device.value,
        )
        engine_timing.record_stage_duration(stage_timing_recorder, "output_run_preparation", output_start_time)
        _core.record_runner_execution_plan_dispatch_started_diagnostic_event(
            phenotype_count=len(phenotype_run_plans),
            association_mode=plan.association_mode.value,
        )
        final_output_paths = dispatch_execution_plan(
            plan=plan,
            phenotype_run_plans=phenotype_run_plans,
            lifecycle_session=lifecycle_session,
            stage_timing_recorder=stage_timing_recorder,
            telemetry_session=telemetry_session,
        )
        _core.record_runner_execution_plan_finalization_started_diagnostic_event(
            phenotype_count=len(phenotype_run_plans),
            association_mode=plan.association_mode.value,
        )
        native_artifacts = lifecycle_session.finalize_success(
            tuple(
                None if final_output_path is None else str(final_output_path)
                for final_output_path in final_output_paths
            )
        )
        artifacts = events.run_artifacts_from_native_artifacts(native_artifacts)
        _core.record_runner_metadata_artifacts_finalized_diagnostic_event(
            association_mode=plan.association_mode.value,
            phenotype_count=len(phenotype_run_plans),
        )
        return artifacts
    finally:
        if stage_timing_recorder is not None:
            engine_timing.record_stage_duration(stage_timing_recorder, "python_api_entry", api_entry_start_time)
            _core.record_final_timing_outputs_write_started_diagnostic_event(
                None if final_stage_timing_path is None else str(final_stage_timing_path),
                None if final_profile_summary_path is None else str(final_profile_summary_path),
                native_final_timing_context.run_id,
            )
            engine_timing.write_final_timing_outputs(
                stage_timing_recorder,
                stage_timing_path=final_stage_timing_path,
                profile_summary_path=final_profile_summary_path,
                run_id=native_final_timing_context.run_id,
            )


def dispatch_execution_plan(
    *,
    plan: execution_plan.RegenieExecutionPlan,
    phenotype_run_plans: tuple[_core.NativeRunLifecyclePhenotypeRun, ...],
    lifecycle_session: _core.NativeRunLifecycleSession,
    stage_timing_recorder: engine_timing.StageTimingRecorder | None,
    telemetry_session: events.TelemetrySession | None,
) -> tuple[Path | None, ...]:
    """Dispatch an execution plan to the native engine layer."""
    lifecycle_session.mark_dispatch_started()
    if len(phenotype_run_plans) > 1:
        _core.record_runner_multi_phenotype_dispatch_started_diagnostic_event(
            phenotype_count=len(phenotype_run_plans),
            association_mode=plan.association_mode.value,
        )
        return dispatch_multi_phenotype_engine_pipeline(
            plan=plan,
            phenotype_run_plans=phenotype_run_plans,
            lifecycle_session=lifecycle_session,
            stage_timing_recorder=stage_timing_recorder,
            telemetry_session=telemetry_session,
        )
    _core.record_runner_single_phenotype_dispatch_started_diagnostic_event(
        association_mode=plan.association_mode.value,
        phenotype=phenotype_run_plans[0].phenotype_name,
    )
    return (
        dispatch_one_phenotype_engine_pipeline(
            plan=plan,
            phenotype_run_plan=phenotype_run_plans[0],
            lifecycle_session=lifecycle_session,
            stage_timing_recorder=stage_timing_recorder,
            telemetry_session=telemetry_session,
        ),
    )


def build_common_engine_dispatch_request(
    *,
    plan: execution_plan.RegenieExecutionPlan,
    lifecycle_session: _core.NativeRunLifecycleSession,
    stage_timing_recorder: engine_timing.StageTimingRecorder | None,
    telemetry_session: events.TelemetrySession | None,
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
        lifecycle_session=lifecycle_session,
    )


def dispatch_one_phenotype_engine_pipeline(
    *,
    plan: execution_plan.RegenieExecutionPlan,
    phenotype_run_plan: _core.NativeRunLifecyclePhenotypeRun,
    lifecycle_session: _core.NativeRunLifecycleSession,
    stage_timing_recorder: engine_timing.StageTimingRecorder | None,
    telemetry_session: events.TelemetrySession | None,
) -> Path | None:
    """Dispatch one phenotype to the native linear or binary pipeline."""
    common_request = build_common_engine_dispatch_request(
        plan=plan,
        lifecycle_session=lifecycle_session,
        stage_timing_recorder=stage_timing_recorder,
        telemetry_session=telemetry_session,
    )
    if plan.association_mode == types.AssociationMode.REGENIE2_BINARY:
        _core.record_runner_binary_engine_dispatch_started_diagnostic_event(phenotype=phenotype_run_plan.phenotype_name)
        final_output_path = runtime.run_regenie2_binary_bgen_pipeline(
            dispatch_requests.SingleTraitBinaryPipelineRequest(
                common=common_request,
                phenotype_name=phenotype_run_plan.phenotype_name,
                prepared_run=phenotype_run_plan,
                correction_plan=plan.binary_correction_plan,
                binary_kernel_config=plan.kernel_config.binary_kernel_config,
                gpu_genotype_format=plan.kernel_config.gpu_genotype_format,
                null_logistic_nonconvergence_policy=(
                    plan.kernel_config.alignment_config.null_logistic_nonconvergence_policy
                ),
            ),
        )
        _core.record_phenotype_writer_finished_telemetry(
            telemetry_session,
            plan.association_mode.value,
            phenotype_run_plan.phenotype_name,
            None if final_output_path is None else str(final_output_path),
        )
        return final_output_path
    _core.record_runner_linear_engine_dispatch_started_diagnostic_event(phenotype=phenotype_run_plan.phenotype_name)
    final_output_path = runtime.run_regenie2_linear_bgen_pipeline(
        dispatch_requests.SingleTraitLinearPipelineRequest(
            common=common_request,
            phenotype_name=phenotype_run_plan.phenotype_name,
            prepared_run=phenotype_run_plan,
            linear_numerical_config=plan.kernel_config.linear_numerical_config,
            gpu_genotype_format=plan.kernel_config.gpu_genotype_format,
        )
    )
    _core.record_phenotype_writer_finished_telemetry(
        telemetry_session,
        plan.association_mode.value,
        phenotype_run_plan.phenotype_name,
        None if final_output_path is None else str(final_output_path),
    )
    return final_output_path


def dispatch_multi_phenotype_engine_pipeline(
    *,
    plan: execution_plan.RegenieExecutionPlan,
    phenotype_run_plans: tuple[_core.NativeRunLifecyclePhenotypeRun, ...],
    lifecycle_session: _core.NativeRunLifecycleSession,
    stage_timing_recorder: engine_timing.StageTimingRecorder | None,
    telemetry_session: events.TelemetrySession | None,
) -> tuple[Path | None, ...]:
    """Dispatch multiple phenotypes to the shared native pipeline."""
    common_request = build_common_engine_dispatch_request(
        plan=plan,
        lifecycle_session=lifecycle_session,
        stage_timing_recorder=stage_timing_recorder,
        telemetry_session=telemetry_session,
    )
    phenotype_names = tuple(phenotype_run_plan.phenotype_name for phenotype_run_plan in phenotype_run_plans)
    if plan.association_mode == types.AssociationMode.REGENIE2_BINARY:
        _core.record_runner_multi_phenotype_binary_engine_dispatch_started_diagnostic_event(
            phenotype_count=len(phenotype_names)
        )
        final_output_paths = runtime.run_regenie2_multi_phenotype_binary_bgen_pipeline(
            dispatch_requests.MultiTraitBinaryPipelineRequest(
                common=common_request,
                phenotype_names=phenotype_names,
                prepared_runs=phenotype_run_plans,
                correction_plan=plan.binary_correction_plan,
                binary_kernel_config=plan.kernel_config.binary_kernel_config,
                gpu_genotype_format=plan.kernel_config.gpu_genotype_format,
                null_logistic_nonconvergence_policy=(
                    plan.kernel_config.alignment_config.null_logistic_nonconvergence_policy
                ),
                sample_mode=plan.kernel_config.multi_phenotype_sample_mode,
                phenotype_compute_groups=plan.phenotype_compute_groups,
            ),
        )
    else:
        _core.record_runner_multi_phenotype_linear_engine_dispatch_started_diagnostic_event(
            phenotype_count=len(phenotype_names)
        )
        final_output_paths = runtime.run_regenie2_multi_phenotype_linear_bgen_pipeline(
            dispatch_requests.MultiTraitLinearPipelineRequest(
                common=common_request,
                phenotype_names=phenotype_names,
                prepared_runs=phenotype_run_plans,
                linear_numerical_config=plan.kernel_config.linear_numerical_config,
                gpu_genotype_format=plan.kernel_config.gpu_genotype_format,
                sample_mode=plan.kernel_config.multi_phenotype_sample_mode,
                phenotype_compute_groups=plan.phenotype_compute_groups,
            )
        )
    _core.record_multi_phenotype_writer_finished_telemetry(
        telemetry_session,
        plan.association_mode.value,
        len(phenotype_run_plans),
        tuple(
            None if final_output_path is None else str(final_output_path) for final_output_path in final_output_paths
        ),
    )
    return final_output_paths
