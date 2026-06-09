"""Execution layer for REGENIE-compatible runs."""

from __future__ import annotations

import importlib
import logging
import time
import typing
from dataclasses import dataclass

from g import _core, execution_plan, jax_runtime, types
from g.engine import run_events, shutdown, telemetry, timing
from g.interface import config
from g.io import output

if typing.TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


RunArtifacts = run_events.RunArtifacts


@dataclass(frozen=True)
class LoggingRuntimePolicy:
    """Process-global logging sink configuration.

    Attributes:
        log_filter: Rust/Python tracing filter.
        log_file: Optional JSON tracing file.
        log_stderr: Whether compact stderr logs are enabled.
        log_queue_size: Non-blocking writer queue size.
        log_lossy: Whether full queues drop log lines instead of blocking.
        include_source_location: Whether JSON tracing events include source locations.
        include_span_events: Whether tracing span lifecycle events are emitted.
        trace_file: Optional high-volume trace file.
        trace_filter: Trace file filter.
        trace_event_cap: Optional trace-mode event cap. None disables native cap enforcement.

    """

    log_filter: str
    log_file: Path | None
    log_stderr: bool
    log_queue_size: int
    log_lossy: bool
    include_source_location: bool
    include_span_events: bool
    trace_file: Path | None
    trace_filter: str
    trace_event_cap: int | None


CONFIGURED_LOGGING_RUNTIME_POLICY: LoggingRuntimePolicy | None = None
CONFIGURED_RAYON_THREAD_COUNT: int | None = None


# These lazy boundaries intentionally keep JAX and JAX-decorated pipeline modules out
# of API/CLI startup until the run's process-global runtime policy is applied.
def load_regenie2_pipeline_module_after_jax_runtime_setup() -> typing.Any:
    """Load the JAX-heavy REGENIE pipeline after runtime policy is configured."""
    return importlib.import_module("g.engine.regenie2_pipeline")


def load_jax_setup_module_at_runtime_boundary() -> typing.Any:
    """Load JAX setup only from explicit JAX runtime configuration paths."""
    return importlib.import_module("g.jax_setup")


def record_jax_runtime_diagnostic_event(
    diagnostic_event: jax_runtime.JaxRuntimeDiagnosticEvent,
    *,
    telemetry_session: telemetry.TelemetrySession | None = None,
) -> None:
    """Record one structured JAX runtime diagnostic event.

    Args:
        diagnostic_event: Runtime diagnostic event to record.
        telemetry_session: Optional run telemetry session.

    """
    event_fields = jax_runtime.diagnostic_event_fields(diagnostic_event)
    logger.log(
        jax_runtime.diagnostic_logging_level(diagnostic_event.level),
        "%s",
        diagnostic_event.message,
        extra={
            "g_event": diagnostic_event.event_name,
            "g_fields": event_fields,
        },
    )
    if telemetry_session is None:
        return
    telemetry_session.log_event(diagnostic_event.event_name, level=diagnostic_event.level.value, **event_fields)


def configure_runtime_before_jax_import(
    compute_config: config.GComputeConfig,
    telemetry_session: telemetry.TelemetrySession | None = None,
) -> jax_runtime.JaxRuntimeSetupReport | None:
    """Configure JAX platform and runtime before compute modules are imported."""
    requested_policy = jax_runtime.require_compatible_jax_runtime_policy(compute_config)
    if jax_runtime.jax_runtime_policy_is_configured(requested_policy):
        return None

    def record_diagnostic_event(diagnostic_event: jax_runtime.JaxRuntimeDiagnosticEvent) -> None:
        record_jax_runtime_diagnostic_event(diagnostic_event, telemetry_session=telemetry_session)

    setup_report = load_jax_setup_module_at_runtime_boundary().configure_jax_runtime_before_backend_init(
        requested_policy,
        diagnostic_sink=record_diagnostic_event,
    )
    jax_runtime.mark_jax_runtime_policy_configured(requested_policy)
    return typing.cast("jax_runtime.JaxRuntimeSetupReport", setup_report)


def configure_jax_runtime(compute_config: config.GComputeConfig) -> jax_runtime.JaxRuntimeSetupReport | None:
    """Configure JAX lazily."""
    return configure_runtime_before_jax_import(compute_config)


def build_stage_timing_recorder(stage_timing_path: Path | None, *, force: bool = False) -> typing.Any:
    """Build a stage timing recorder."""
    if force:
        return timing.build_stage_timing_recorder(stage_timing_path, force=True)
    return timing.build_stage_timing_recorder(stage_timing_path)


def record_stage_duration(stage_timing_recorder: typing.Any, stage_name: str, start_time: float) -> None:
    """Record one stage duration."""
    timing.record_stage_duration(stage_timing_recorder, stage_name, start_time)


def write_stage_timing_snapshot(stage_timing_recorder: typing.Any, stage_timing_path: Path | None) -> None:
    """Write a stage timing snapshot."""
    timing.write_stage_timing_snapshot(stage_timing_recorder, stage_timing_path)


def write_profile_summary(
    stage_timing_recorder: typing.Any,
    profile_summary_path: Path | None,
    run_id: str | None,
) -> None:
    """Write an aggregate profile summary."""
    timing.write_profile_summary(stage_timing_recorder, profile_summary_path, run_id=run_id)


def run_regenie2_linear_bgen_pipeline(**kwargs: typing.Any) -> Path | None:
    """Run the linear native pipeline after JAX runtime setup."""
    return typing.cast(
        "Path | None",
        load_regenie2_pipeline_module_after_jax_runtime_setup().run_regenie2_linear_bgen_pipeline(**kwargs),
    )


def run_regenie2_binary_bgen_pipeline(**kwargs: typing.Any) -> Path | None:
    """Run the binary native pipeline after JAX runtime setup."""
    return typing.cast(
        "Path | None",
        load_regenie2_pipeline_module_after_jax_runtime_setup().run_regenie2_binary_bgen_pipeline(**kwargs),
    )


def run_regenie2_multi_phenotype_linear_bgen_pipeline(**kwargs: typing.Any) -> tuple[Path | None, ...]:
    """Run the multi-phenotype linear native pipeline after JAX runtime setup."""
    return typing.cast(
        "tuple[Path | None, ...]",
        load_regenie2_pipeline_module_after_jax_runtime_setup().run_regenie2_multi_phenotype_linear_bgen_pipeline(
            **kwargs
        ),
    )


def run_regenie2_multi_phenotype_binary_bgen_pipeline(**kwargs: typing.Any) -> tuple[Path | None, ...]:
    """Run the multi-phenotype binary native pipeline after JAX runtime setup."""
    return typing.cast(
        "tuple[Path | None, ...]",
        load_regenie2_pipeline_module_after_jax_runtime_setup().run_regenie2_multi_phenotype_binary_bgen_pipeline(
            **kwargs
        ),
    )


def configure_rayon_thread_pool(core_module: typing.Any, thread_count: int) -> None:
    """Configure Rayon global thread count once and reject incompatible repeats."""
    global CONFIGURED_RAYON_THREAD_COUNT
    if thread_count == CONFIGURED_RAYON_THREAD_COUNT:
        return
    if CONFIGURED_RAYON_THREAD_COUNT is not None:
        message = (
            "Rayon global thread pool is already configured with "
            f"{CONFIGURED_RAYON_THREAD_COUNT} thread(s); cannot apply requested --threads={thread_count}."
        )
        raise RuntimeError(message)
    try:
        core_module.configure_rayon_global_thread_pool(thread_count)
    except RuntimeError as error:
        message = (
            f"Unable to configure Rayon global thread pool for --threads={thread_count}; "
            f"existing Rayon settings are unknown: {error}"
        )
        raise RuntimeError(message) from error
    CONFIGURED_RAYON_THREAD_COUNT = thread_count


def effective_rayon_thread_count(requested_thread_count: int | None) -> int | None:
    """Return the Rayon thread count known to be effective in this process."""
    if CONFIGURED_RAYON_THREAD_COUNT is not None:
        return CONFIGURED_RAYON_THREAD_COUNT
    return requested_thread_count


def configure_runtime(compute_config: config.GComputeConfig, trait_config: config.TraitConfig) -> None:
    """Apply native runtime knobs before engine execution."""
    logger.debug("Configuring native runtime knobs.")
    _core.configure_bgen_decode_tile_variant_count(compute_config.bgen_decode_tile_variant_count)
    if trait_config.threads is not None:
        configure_rayon_thread_pool(_core, trait_config.threads)


def initialize_logging(
    diagnostics_config: config.GDiagnosticsConfig,
    telemetry_paths: telemetry.TelemetryPaths | None = None,
) -> None:
    """Initialize unified Rust/Python logging before runtime setup."""
    global CONFIGURED_LOGGING_RUNTIME_POLICY
    telemetry_stream_file = None if telemetry_paths is None else telemetry_paths.stream_file
    log_file = diagnostics_config.log_file if telemetry_stream_file is None else None
    trace_file = diagnostics_config.trace_file if telemetry_stream_file is None else telemetry_stream_file
    trace_event_cap = (
        diagnostics_config.trace_event_cap if diagnostics_config.telemetry == types.TelemetryMode.TRACE else None
    )
    runtime_policy = LoggingRuntimePolicy(
        log_filter=diagnostics_config.log_filter,
        log_file=log_file,
        log_stderr=diagnostics_config.log_stderr,
        log_queue_size=diagnostics_config.log_queue_size,
        log_lossy=diagnostics_config.log_lossy,
        include_source_location=diagnostics_config.include_source_location,
        include_span_events=diagnostics_config.include_span_events,
        trace_file=trace_file,
        trace_filter=diagnostics_config.trace_filter,
        trace_event_cap=trace_event_cap,
    )
    initialized_logging = _core.initialize_logging(
        log_filter=diagnostics_config.log_filter,
        log_file=None if log_file is None else str(log_file),
        log_stderr=diagnostics_config.log_stderr,
        log_queue_size=diagnostics_config.log_queue_size,
        log_lossy=diagnostics_config.log_lossy,
        include_source_location=diagnostics_config.include_source_location,
        include_span_events=diagnostics_config.include_span_events,
        trace_file=None if trace_file is None else str(trace_file),
        trace_filter=diagnostics_config.trace_filter,
        trace_event_cap=trace_event_cap,
    )
    if initialized_logging is False:
        if CONFIGURED_LOGGING_RUNTIME_POLICY is not None and runtime_policy != CONFIGURED_LOGGING_RUNTIME_POLICY:
            message = "Logging is process-global; start a new process or reuse the first run's logging configuration."
            raise RuntimeError(message)
        CONFIGURED_LOGGING_RUNTIME_POLICY = runtime_policy
        return
    if initialized_logging is True:
        CONFIGURED_LOGGING_RUNTIME_POLICY = runtime_policy


def regenie(regenie_config: config.RegenieConfig) -> RunArtifacts:
    """Run the shared REGENIE-compatible config path."""
    config.validate_config_for_run(regenie_config)
    telemetry_session = telemetry.build_telemetry_session(regenie_config)
    initialize_logging(regenie_config.g_diagnostics, telemetry_session.paths)
    association_mode = execution_plan.resolve_association_mode(regenie_config.trait.trait_type)
    phenotype_count = len(regenie_config.input.pheno_columns)
    telemetry_session.log_event(
        "run_started",
        association_mode=association_mode.value,
        trait_type=regenie_config.trait.trait_type.value,
        phenotype_count=phenotype_count,
        output_run_root=str(telemetry.resolve_output_run_root(regenie_config)),
    )
    logger.info("Starting REGENIE run.")
    configure_runtime(regenie_config.g_compute, regenie_config.trait)
    try:
        artifacts = run_validated_regenie_config(regenie_config, telemetry_session=telemetry_session)
    except shutdown.GracefulShutdownRequested as shutdown_request:
        interrupted_event = run_events.build_run_interrupted_event(shutdown_request)
        telemetry_session.log_event(
            "run_failed",
            level="warn",
            **run_events.run_interrupted_telemetry_fields(interrupted_event),
        )
        logger.warning("REGENIE run interrupted by %s.", interrupted_event.signal_name)
        raise
    except Exception as error:
        failed_event = run_events.build_run_failed_event(error)
        telemetry_session.log_event("run_failed", level="error", **run_events.run_failed_telemetry_fields(failed_event))
        logger.exception("REGENIE run failed.")
        raise
    else:
        artifacts = run_events.attach_run_metadata(
            artifacts,
            run_id=telemetry_session.run_id,
            association_mode=association_mode,
            phenotype_count=phenotype_count,
        )
        completed_event = run_events.build_run_completed_event(artifacts)
        telemetry_session.log_event(
            "run_completed",
            level="info",
            **run_events.run_completed_telemetry_fields(completed_event),
        )
        logger.info("Finished REGENIE run.")
        return artifacts
    finally:
        telemetry.close_telemetry_session(telemetry_session)


def run_validated_regenie_config(
    regenie_config: config.RegenieConfig,
    telemetry_session: telemetry.TelemetrySession | None = None,
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
        configure_runtime_before_jax_import(regenie_config.g_compute, telemetry_session=telemetry_session)
        stage_timing_recorder = build_stage_timing_recorder(
            stage_timing_path,
            force=telemetry_session is not None and telemetry_session.profile_enabled,
        )
        record_stage_duration(stage_timing_recorder, "jax_device_configuration_backend_init", device_start_time)
        output_start_time = time.perf_counter()
        logger.debug("Building REGENIE execution plan.")
        plan = execution_plan.build_regenie_execution_plan(regenie_config)
        if telemetry_session is not None:
            telemetry_session.log_event(
                "execution_plan_prepared",
                association_mode=plan.association_mode.value,
                trait_type=regenie_config.trait.trait_type.value,
                phenotype_count=len(plan.phenotype_run_plans),
                chunk_size=plan.kernel_config.chunk_size,
                variant_limit=plan.kernel_config.variant_limit,
                device=plan.kernel_config.device.value,
            )
        logger.info("Prepared REGENIE execution plan for %s phenotype(s).", len(plan.phenotype_run_plans))
        logger.debug("Writing execution plan start metadata.")
        write_execution_plan_start_metadata(
            regenie_config=regenie_config,
            plan=plan,
            telemetry_session=telemetry_session,
        )
        record_stage_duration(stage_timing_recorder, "output_run_preparation", output_start_time)
        logger.debug("Dispatching REGENIE execution plan.")
        final_output_paths = dispatch_execution_plan(
            plan=plan,
            stage_timing_recorder=stage_timing_recorder,
            telemetry_session=telemetry_session,
        )
        logger.debug("Finalizing REGENIE execution plan.")
        return finalize_execution_plan(
            regenie_config=regenie_config,
            plan=plan,
            final_output_paths=final_output_paths,
        )
    finally:
        if stage_timing_recorder is not None:
            record_stage_duration(stage_timing_recorder, "python_api_entry", api_entry_start_time)
            logger.debug("Writing final stage timing snapshot.")
            write_stage_timing_snapshot(stage_timing_recorder, stage_timing_path)
            write_profile_summary(
                stage_timing_recorder,
                profile_summary_path,
                run_id=None if telemetry_session is None else telemetry_session.run_id,
            )


def dispatch_execution_plan(
    *,
    plan: execution_plan.RegenieExecutionPlan,
    stage_timing_recorder: typing.Any,
    telemetry_session: telemetry.TelemetrySession | None = None,
) -> tuple[Path | None, ...]:
    """Dispatch an execution plan to the native engine layer."""
    if len(plan.phenotype_run_plans) > 1:
        logger.debug("Dispatching multi-phenotype native engine pipeline.")
        return dispatch_multi_phenotype_engine_pipeline(
            plan=plan,
            stage_timing_recorder=stage_timing_recorder,
            telemetry_session=telemetry_session,
        )
    logger.debug("Dispatching single-phenotype native engine pipeline.")
    return (
        dispatch_one_phenotype_engine_pipeline(
            plan=plan,
            phenotype_run_plan=plan.phenotype_run_plans[0],
            stage_timing_recorder=stage_timing_recorder,
            telemetry_session=telemetry_session,
        ),
    )


def build_common_engine_arguments(
    *,
    plan: execution_plan.RegenieExecutionPlan,
    stage_timing_recorder: typing.Any,
    telemetry_session: telemetry.TelemetrySession | None,
) -> dict[str, typing.Any]:
    """Build arguments shared by single- and multi-phenotype native wrappers."""
    return {
        "genotype_source_config": plan.genotype_source_config,
        "phenotype_path": plan.phenotype_path,
        "prediction_list_path": plan.prediction_list_path,
        "covariate_path": plan.covariate_path,
        "covariate_names": plan.covariate_names,
        "chunk_size": plan.kernel_config.chunk_size,
        "variant_limit": plan.kernel_config.variant_limit,
        "staging_depth": plan.kernel_config.staging_depth,
        "result_in_flight_limit": plan.kernel_config.result_in_flight_limit,
        "dosage_buffer_limit": plan.kernel_config.dosage_buffer_limit,
        "resume": plan.output_plan.resume,
        "resume_mode": plan.output_plan.resume_mode,
        "finalize_parquet": plan.output_plan.finalize_parquet,
        "writer_thread_count": plan.output_plan.writer_threads,
        "writer_queue_depth": plan.output_plan.writer_queue_depth,
        "chunks_per_arrow_file": plan.output_plan.chunks_per_arrow_file,
        "arrow_compression": plan.output_plan.arrow_compression,
        "parquet_compression": plan.output_plan.parquet_compression,
        "trusted_no_missing_diploid": plan.kernel_config.trusted_no_missing_diploid,
        "trusted_bgen_validation_mode": plan.kernel_config.trusted_bgen_validation_mode,
        "bgen_decode_tile_variant_count": plan.kernel_config.bgen_decode_tile_variant_count,
        "jax_device": plan.kernel_config.device,
        "jax_matmul_precision": plan.kernel_config.alignment_config.jax_matmul_precision,
        "score_dtype": plan.kernel_config.alignment_config.score_dtype,
        "firth_dtype": plan.kernel_config.alignment_config.firth_dtype,
        "output_format": plan.output_plan.output_format,
        "stage_timing_recorder": stage_timing_recorder,
        "telemetry_session": telemetry_session,
        "alignment_config": plan.kernel_config.alignment_config,
    }


def dispatch_one_phenotype_engine_pipeline(
    *,
    plan: execution_plan.RegenieExecutionPlan,
    phenotype_run_plan: execution_plan.PhenotypeRunPlan,
    stage_timing_recorder: typing.Any,
    telemetry_session: telemetry.TelemetrySession | None = None,
) -> Path | None:
    """Dispatch one phenotype to the native linear or binary pipeline."""
    common_arguments = build_common_engine_arguments(
        plan=plan,
        stage_timing_recorder=stage_timing_recorder,
        telemetry_session=telemetry_session,
    )
    common_arguments.update(
        {
            "phenotype_name": phenotype_run_plan.phenotype_name,
            "output_run_paths": phenotype_run_plan.output_run_paths,
            "existing_manifest": phenotype_run_plan.existing_manifest,
        }
    )
    if plan.association_mode == types.AssociationMode.REGENIE2_BINARY:
        logger.debug("Dispatching binary native engine pipeline.")
        final_output_path = run_regenie2_binary_bgen_pipeline(
            **common_arguments,
            correction_plan=plan.binary_correction_plan,
            kernel_config=plan.kernel_config.binary_kernel_config,
            null_logistic_nonconvergence_policy=(
                plan.kernel_config.alignment_config.null_logistic_nonconvergence_policy
            ),
            gpu_genotype_format=plan.kernel_config.gpu_genotype_format,
        )
        log_writer_finished(
            telemetry_session=telemetry_session,
            association_mode=plan.association_mode,
            phenotype=phenotype_run_plan.phenotype_name,
            final_output_path=final_output_path,
        )
        return final_output_path
    logger.debug("Dispatching linear native engine pipeline.")
    final_output_path = run_regenie2_linear_bgen_pipeline(
        **common_arguments,
        gpu_genotype_format=plan.kernel_config.gpu_genotype_format,
        linear_numerical_config=plan.kernel_config.linear_numerical_config,
    )
    log_writer_finished(
        telemetry_session=telemetry_session,
        association_mode=plan.association_mode,
        phenotype=phenotype_run_plan.phenotype_name,
        final_output_path=final_output_path,
    )
    return final_output_path


def dispatch_multi_phenotype_engine_pipeline(
    *,
    plan: execution_plan.RegenieExecutionPlan,
    stage_timing_recorder: typing.Any,
    telemetry_session: telemetry.TelemetrySession | None = None,
) -> tuple[Path | None, ...]:
    """Dispatch multiple phenotypes to the shared native pipeline."""
    common_arguments = build_common_engine_arguments(
        plan=plan,
        stage_timing_recorder=stage_timing_recorder,
        telemetry_session=telemetry_session,
    )
    common_arguments.update(
        {
            "sample_mode": plan.kernel_config.multi_phenotype_sample_mode,
            "phenotype_names": tuple(
                phenotype_run_plan.phenotype_name for phenotype_run_plan in plan.phenotype_run_plans
            ),
            "output_run_paths_by_phenotype": tuple(
                phenotype_run_plan.output_run_paths for phenotype_run_plan in plan.phenotype_run_plans
            ),
            "existing_manifests_by_phenotype": tuple(
                phenotype_run_plan.existing_manifest for phenotype_run_plan in plan.phenotype_run_plans
            ),
            "phenotype_compute_groups": plan.phenotype_compute_groups,
            "gpu_genotype_format": plan.kernel_config.gpu_genotype_format,
        }
    )
    if plan.association_mode == types.AssociationMode.REGENIE2_BINARY:
        logger.debug("Dispatching multi-phenotype binary native engine pipeline.")
        final_output_paths = run_regenie2_multi_phenotype_binary_bgen_pipeline(
            **common_arguments,
            correction_plan=plan.binary_correction_plan,
            kernel_config=plan.kernel_config.binary_kernel_config,
            null_logistic_nonconvergence_policy=(
                plan.kernel_config.alignment_config.null_logistic_nonconvergence_policy
            ),
        )
    else:
        logger.debug("Dispatching multi-phenotype linear native engine pipeline.")
        final_output_paths = run_regenie2_multi_phenotype_linear_bgen_pipeline(
            **common_arguments,
            linear_numerical_config=plan.kernel_config.linear_numerical_config,
        )
    if telemetry_session is not None:
        telemetry_session.log_event(
            "writer_finished",
            association_mode=plan.association_mode.value,
            phenotype_count=len(plan.phenotype_run_plans),
            final_output_paths=tuple(None if path is None else str(path) for path in final_output_paths),
        )
    return final_output_paths


def log_writer_finished(
    *,
    telemetry_session: telemetry.TelemetrySession | None,
    association_mode: types.AssociationMode,
    phenotype: str,
    final_output_path: Path | None,
) -> None:
    """Record output writer completion."""
    if telemetry_session is None:
        return
    telemetry_session.log_event(
        "writer_finished",
        association_mode=association_mode.value,
        phenotype=phenotype,
        final_output_path=None if final_output_path is None else str(final_output_path),
    )


def write_execution_plan_start_metadata(
    *,
    regenie_config: config.RegenieConfig,
    plan: execution_plan.RegenieExecutionPlan,
    telemetry_session: telemetry.TelemetrySession | None = None,
) -> None:
    """Write per-phenotype metadata before native engine execution starts."""
    for phenotype_run_plan in plan.phenotype_run_plans:
        write_run_start_metadata(
            regenie_config=regenie_config,
            plan=plan,
            phenotype_run_plan=phenotype_run_plan,
            telemetry_session=telemetry_session,
        )


def write_run_start_metadata(
    *,
    regenie_config: config.RegenieConfig,
    plan: execution_plan.RegenieExecutionPlan,
    phenotype_run_plan: execution_plan.PhenotypeRunPlan,
    telemetry_session: telemetry.TelemetrySession | None = None,
) -> None:
    """Write run metadata before native engine execution starts."""
    config.write_toml(regenie_config, phenotype_run_plan.effective_config_path)
    extend_run_manifest(
        plan=plan,
        phenotype_run_plan=phenotype_run_plan,
    )
    if telemetry_session is not None:
        telemetry_session.log_event(
            "effective_config_written",
            association_mode=plan.association_mode.value,
            phenotype=phenotype_run_plan.phenotype_name,
            effective_config=str(phenotype_run_plan.effective_config_path),
            output_run_directory=str(phenotype_run_plan.output_run_paths.run_directory),
        )


def finalize_execution_plan(
    *,
    regenie_config: config.RegenieConfig,
    plan: execution_plan.RegenieExecutionPlan,
    final_output_paths: tuple[Path | None, ...],
) -> RunArtifacts:
    """Build user-facing artifacts after native execution."""
    phenotype_artifacts = tuple(
        finalize_phenotype_run(
            regenie_config=regenie_config,
            plan=plan,
            phenotype_run_plan=phenotype_run_plan,
            final_output_path=final_output_path,
        )
        for phenotype_run_plan, final_output_path in zip(
            plan.phenotype_run_plans,
            final_output_paths,
            strict=True,
        )
    )
    logger.info("Finalized REGENIE run artifacts for %s phenotype(s).", len(phenotype_artifacts))
    if len(phenotype_artifacts) == 1:
        return phenotype_artifacts[0]
    return RunArtifacts(
        phenotype_artifacts=phenotype_artifacts,
        association_mode=plan.association_mode,
        phenotype_count=len(phenotype_artifacts),
    )


def finalize_phenotype_run(
    *,
    regenie_config: config.RegenieConfig,
    plan: execution_plan.RegenieExecutionPlan,
    phenotype_run_plan: execution_plan.PhenotypeRunPlan,
    final_output_path: Path | None,
) -> RunArtifacts:
    """Build artifacts for one phenotype."""
    del regenie_config
    final_dataset = (
        phenotype_run_plan.output_run_paths.chunks_directory
        if plan.output_plan.output_format == types.OutputFormat.PARQUET
        else None
    )
    final_parquet_path = None
    final_regenie_path = None
    if plan.output_plan.output_format == types.OutputFormat.REGENIE:
        final_regenie_path = final_output_path
    else:
        final_parquet_path = final_output_path
    return RunArtifacts(
        output_run_directory=phenotype_run_plan.output_run_paths.run_directory,
        final_dataset=final_dataset,
        final_parquet=final_parquet_path,
        final_regenie=final_regenie_path,
        effective_config=phenotype_run_plan.effective_config_path,
        phenotype_name=phenotype_run_plan.phenotype_name,
        association_mode=plan.association_mode,
        phenotype_count=len(plan.phenotype_run_plans),
    )


def extend_run_manifest(
    *,
    plan: execution_plan.RegenieExecutionPlan,
    phenotype_run_plan: execution_plan.PhenotypeRunPlan,
) -> None:
    """Add command and runtime metadata to a run manifest."""
    manifest = output.load_run_manifest(phenotype_run_plan.output_run_paths) or {}
    manifest["command"] = {
        "interface": "g regenie",
        "phenotype": phenotype_run_plan.phenotype_name,
        "effective_config": str(phenotype_run_plan.effective_config_path),
        "output_format": plan.output_plan.output_format.value,
    }
    manifest["runtime"] = {
        "device": plan.kernel_config.device.value,
        "staging_depth": plan.kernel_config.staging_depth,
        "threads": plan.kernel_config.thread_count,
        "writer_threads": plan.output_plan.writer_threads,
        "writer_queue_depth": plan.output_plan.writer_queue_depth,
        "chunks_per_arrow_file": plan.output_plan.chunks_per_arrow_file,
        "arrow_compression": plan.output_plan.arrow_compression.value,
        "parquet_compression": plan.output_plan.parquet_compression.value,
        "bgen_decode_tile_variant_count": plan.kernel_config.bgen_decode_tile_variant_count,
        "trusted_no_missing_diploid": plan.kernel_config.trusted_no_missing_diploid,
        "trusted_bgen_validation_mode": plan.kernel_config.trusted_bgen_validation_mode.value,
    }
    output.write_run_manifest(phenotype_run_plan.output_run_paths, manifest)
