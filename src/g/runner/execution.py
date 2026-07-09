"""Execution layer for REGENIE-compatible runs."""

from __future__ import annotations

import time
import typing
from pathlib import Path

import g.engine.callbacks.factory as callback_factory
from g import _core, types
from g.engine import timing as engine_timing
from g.runner import events, lifecycle, runtime

RUN_EVENT_RECORDER: _core.NativeRunEventRecorder = _core.NativeRunEventRecorder()


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
    native_final_timing_context = _core.NativeFinalTimingOutputContext.from_sources(
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
        RUN_EVENT_RECORDER.runner_jax_runtime_configuration_started()
        runtime.configure_runtime_before_jax_import(regenie_config.g_compute, telemetry_session=telemetry_session)
        stage_timing_recorder = engine_timing.build_stage_timing_recorder(
            final_stage_timing_path,
            force=native_final_timing_context.force_stage_timing_recorder,
        )
        engine_timing.record_stage_duration(
            stage_timing_recorder, "jax_device_configuration_backend_init", device_start_time
        )
        engine_session = _core.NativeRunEngineSession(regenie_config, runtime_compatibility_token)
        native_artifacts = engine_session.run_to_completion(
            callback_factory.NativeRunCallbackFactory(regenie_config, telemetry_session, stage_timing_recorder),
            telemetry_session,
            stage_timing_recorder.native_recorder,
        )
        return events.run_artifacts_from_native_artifacts(native_artifacts)
    finally:
        if stage_timing_recorder is not None:
            engine_timing.record_stage_duration(stage_timing_recorder, "python_api_entry", api_entry_start_time)
            native_final_timing_context.record_outputs_write_started_diagnostic()
            engine_timing.write_final_timing_outputs(
                stage_timing_recorder,
                stage_timing_path=final_stage_timing_path,
                profile_summary_path=final_profile_summary_path,
                run_id=native_final_timing_context.run_id,
            )
