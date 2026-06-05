"""Execution layer for REGENIE-compatible runs."""

from __future__ import annotations

import contextlib
import importlib
import logging
import time
import typing
from dataclasses import dataclass

from g import execution_plan, types
from g.interface import config
from g.io import output

if typing.TYPE_CHECKING:
    from pathlib import Path

    from g.engine import telemetry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunArtifacts:
    """Immutable pointers to generated output files.

    Attributes:
        output_run_directory: Chunked output run directory.
        final_parquet: Finalized Parquet output path.
        effective_config: Written effective TOML config path.
        phenotype_artifacts: Per-phenotype artifacts for multi-phenotype runs.

    """

    output_run_directory: Path | None = None
    final_parquet: Path | None = None
    effective_config: Path | None = None
    phenotype_artifacts: tuple[RunArtifacts, ...] = ()


@dataclass(frozen=True)
class JaxRuntimePolicy:
    """Process-global JAX runtime settings selected by the first run.

    Attributes:
        device: Requested JAX platform.
        cache_directory: Persistent compilation cache directory.
        matmul_precision: Requested matmul precision.
        enable_x64: Whether JAX 64-bit arrays are enabled.
        persistent_cache: Whether persistent compilation caching is enabled.
        persistent_cache_min_entry_size_bytes: Minimum cache entry size.
        persistent_cache_min_compile_time_seconds: Minimum compile time for cache entries.
        xla_autotune_cache: Whether XLA autotune caches are enabled.
        transfer_guard: Whether transfer guard diagnostics are enabled.

    """

    device: types.Device
    cache_directory: Path | None
    matmul_precision: types.JaxMatmulPrecision | None
    enable_x64: bool
    persistent_cache: bool
    persistent_cache_min_entry_size_bytes: int
    persistent_cache_min_compile_time_seconds: int
    xla_autotune_cache: bool
    transfer_guard: bool


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


CONFIGURED_JAX_RUNTIME_POLICY: JaxRuntimePolicy | None = None
CONFIGURED_LOGGING_RUNTIME_POLICY: LoggingRuntimePolicy | None = None


def load_regenie2_pipeline_module() -> typing.Any:
    """Load the JAX-heavy REGENIE pipeline module lazily."""
    return importlib.import_module("g.engine.regenie2_pipeline")


def load_timing_module() -> typing.Any:
    """Load stage-timing helpers lazily."""
    return importlib.import_module("g.engine.timing")


def load_telemetry_module() -> typing.Any:
    """Load telemetry helpers lazily."""
    return importlib.import_module("g.engine.telemetry")


def load_jax_setup_module() -> typing.Any:
    """Load JAX setup lazily after runtime environment is configured."""
    return importlib.import_module("g.jax_setup")


def configure_jax_platform_before_setup_import(device: types.Device) -> None:
    """Configure JAX platform selection before importing setup helpers."""
    jax_module = importlib.import_module("jax")
    platform_name = "cuda" if device == types.Device.GPU else "cpu"
    jax_module.config.update("jax_platforms", platform_name)


def build_jax_runtime_policy(compute_config: config.GComputeConfig) -> JaxRuntimePolicy:
    """Build the process-global JAX runtime policy requested by a run."""
    cache_directory = None
    if compute_config.jax_cache_dir is not None:
        cache_directory = compute_config.jax_cache_dir.expanduser()
    return JaxRuntimePolicy(
        device=compute_config.device,
        cache_directory=cache_directory,
        matmul_precision=compute_config.jax_matmul_precision,
        enable_x64=compute_config.jax_enable_x64,
        persistent_cache=compute_config.jax_persistent_cache,
        persistent_cache_min_entry_size_bytes=compute_config.jax_persistent_cache_min_entry_size_bytes,
        persistent_cache_min_compile_time_seconds=compute_config.jax_persistent_cache_min_compile_time_seconds,
        xla_autotune_cache=compute_config.jax_xla_autotune_cache,
        transfer_guard=compute_config.jax_transfer_guard,
    )


def describe_jax_runtime_policy(policy: JaxRuntimePolicy) -> str:
    """Format a JAX runtime policy for diagnostics."""
    cache_directory = "<default>" if policy.cache_directory is None else str(policy.cache_directory)
    matmul_precision = "<default>" if policy.matmul_precision is None else policy.matmul_precision.value
    return (
        f"device={policy.device.value}, "
        f"jax-cache-dir={cache_directory}, "
        f"jax-matmul-precision={matmul_precision}, "
        f"jax-enable-x64={policy.enable_x64}, "
        f"jax-persistent-cache={policy.persistent_cache}, "
        f"jax-persistent-cache-min-entry-size-bytes={policy.persistent_cache_min_entry_size_bytes}, "
        f"jax-persistent-cache-min-compile-time-seconds={policy.persistent_cache_min_compile_time_seconds}, "
        f"jax-xla-autotune-cache={policy.xla_autotune_cache}, "
        f"jax-transfer-guard={policy.transfer_guard}"
    )


def require_compatible_jax_runtime_policy(compute_config: config.GComputeConfig) -> JaxRuntimePolicy:
    """Return the requested policy or raise when it conflicts with the first run."""
    requested_policy = build_jax_runtime_policy(compute_config)
    if CONFIGURED_JAX_RUNTIME_POLICY is None or requested_policy == CONFIGURED_JAX_RUNTIME_POLICY:
        return requested_policy
    message = (
        "JAX runtime is already configured for this Python process with "
        f"{describe_jax_runtime_policy(CONFIGURED_JAX_RUNTIME_POLICY)}. "
        "A later run requested incompatible settings: "
        f"{describe_jax_runtime_policy(requested_policy)}. "
        "JAX backend, platform, and compilation cache settings are process-global; start a fresh Python process "
        "for incompatible runtime settings."
    )
    raise RuntimeError(message)


def mark_jax_runtime_policy_configured(policy: JaxRuntimePolicy) -> None:
    """Record that JAX has been configured for this process."""
    global CONFIGURED_JAX_RUNTIME_POLICY
    CONFIGURED_JAX_RUNTIME_POLICY = policy


def configure_runtime_before_jax_import(compute_config: config.GComputeConfig) -> None:
    """Configure JAX platform and runtime before compute modules are imported."""
    requested_policy = require_compatible_jax_runtime_policy(compute_config)
    if requested_policy == CONFIGURED_JAX_RUNTIME_POLICY:
        return
    configure_jax_platform_before_setup_import(compute_config.device)
    load_jax_setup_module().configure_jax_runtime_before_backend_init(
        device=compute_config.device,
        cache_directory=compute_config.jax_cache_dir,
        matmul_precision=compute_config.jax_matmul_precision,
        enable_x64=compute_config.jax_enable_x64,
        persistent_cache=compute_config.jax_persistent_cache,
        persistent_cache_min_entry_size_bytes=compute_config.jax_persistent_cache_min_entry_size_bytes,
        persistent_cache_min_compile_time_seconds=compute_config.jax_persistent_cache_min_compile_time_seconds,
        xla_autotune_cache=compute_config.jax_xla_autotune_cache,
        transfer_guard=compute_config.jax_transfer_guard,
    )
    mark_jax_runtime_policy_configured(requested_policy)


def configure_jax_runtime(compute_config: config.GComputeConfig) -> None:
    """Configure JAX lazily."""
    configure_runtime_before_jax_import(compute_config)


def configure_jax_device(device: types.Device) -> None:
    """Configure JAX lazily."""
    load_jax_setup_module().configure_jax_device(device)


def build_stage_timing_recorder(stage_timing_path: Path | None, *, force: bool = False) -> typing.Any:
    """Build a stage timing recorder lazily."""
    timing_module = load_timing_module()
    if force:
        return timing_module.build_stage_timing_recorder(stage_timing_path, force=True)
    return timing_module.build_stage_timing_recorder(stage_timing_path)


def record_stage_duration(stage_timing_recorder: typing.Any, stage_name: str, start_time: float) -> None:
    """Record one stage duration lazily."""
    load_timing_module().record_stage_duration(stage_timing_recorder, stage_name, start_time)


def write_stage_timing_snapshot(stage_timing_recorder: typing.Any, stage_timing_path: Path | None) -> None:
    """Write a stage timing snapshot lazily."""
    load_timing_module().write_stage_timing_snapshot(stage_timing_recorder, stage_timing_path)


def write_profile_summary(
    stage_timing_recorder: typing.Any,
    profile_summary_path: Path | None,
    run_id: str | None,
) -> None:
    """Write an aggregate profile summary lazily."""
    load_timing_module().write_profile_summary(stage_timing_recorder, profile_summary_path, run_id=run_id)


def run_regenie2_linear_bgen_pipeline(**kwargs: typing.Any) -> Path | None:
    """Run the linear native pipeline lazily."""
    return typing.cast("Path | None", load_regenie2_pipeline_module().run_regenie2_linear_bgen_pipeline(**kwargs))


def run_regenie2_binary_bgen_pipeline(**kwargs: typing.Any) -> Path | None:
    """Run the binary native pipeline lazily."""
    return typing.cast("Path | None", load_regenie2_pipeline_module().run_regenie2_binary_bgen_pipeline(**kwargs))


def run_regenie2_multi_phenotype_linear_bgen_pipeline(**kwargs: typing.Any) -> tuple[Path | None, ...]:
    """Run the multi-phenotype linear native pipeline lazily."""
    return typing.cast(
        "tuple[Path | None, ...]",
        load_regenie2_pipeline_module().run_regenie2_multi_phenotype_linear_bgen_pipeline(**kwargs),
    )


def run_regenie2_multi_phenotype_binary_bgen_pipeline(**kwargs: typing.Any) -> tuple[Path | None, ...]:
    """Run the multi-phenotype binary native pipeline lazily."""
    return typing.cast(
        "tuple[Path | None, ...]",
        load_regenie2_pipeline_module().run_regenie2_multi_phenotype_binary_bgen_pipeline(**kwargs),
    )


def configure_runtime(compute_config: config.GComputeConfig, trait_config: config.TraitConfig) -> None:
    """Apply native runtime knobs before engine execution."""
    core_module = importlib.import_module("g._core")
    logger.debug("Configuring native runtime knobs.")
    core_module.configure_bgen_decode_tile_variant_count(compute_config.bgen_decode_tile_variant_count)
    core_module.configure_bgen_simd_mode(compute_config.bgen_simd.value)
    if trait_config.threads is not None:
        with contextlib.suppress(RuntimeError):
            core_module.configure_rayon_global_thread_pool(trait_config.threads)


def initialize_logging(
    diagnostics_config: config.GDiagnosticsConfig,
    telemetry_paths: telemetry.TelemetryPaths | None = None,
) -> None:
    """Initialize unified Rust/Python logging before runtime setup."""
    global CONFIGURED_LOGGING_RUNTIME_POLICY
    log_file = diagnostics_config.log_file
    trace_file = diagnostics_config.trace_file if telemetry_paths is None else telemetry_paths.trace_file
    validate_logging_stream_ownership(log_file=log_file, trace_file=trace_file, telemetry_paths=telemetry_paths)
    core_module = importlib.import_module("g._core")
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
    )
    initialized_logging = core_module.initialize_logging(
        log_filter=diagnostics_config.log_filter,
        log_file=None if log_file is None else str(log_file),
        log_stderr=diagnostics_config.log_stderr,
        log_queue_size=diagnostics_config.log_queue_size,
        log_lossy=diagnostics_config.log_lossy,
        include_source_location=diagnostics_config.include_source_location,
        include_span_events=diagnostics_config.include_span_events,
        trace_file=None if trace_file is None else str(trace_file),
        trace_filter=diagnostics_config.trace_filter,
    )
    if initialized_logging is False:
        if CONFIGURED_LOGGING_RUNTIME_POLICY is not None and runtime_policy != CONFIGURED_LOGGING_RUNTIME_POLICY:
            message = "Logging is process-global; start a new process or reuse the first run's logging configuration."
            raise RuntimeError(message)
        CONFIGURED_LOGGING_RUNTIME_POLICY = runtime_policy
        return
    if initialized_logging is True:
        CONFIGURED_LOGGING_RUNTIME_POLICY = runtime_policy


def validate_logging_stream_ownership(
    *,
    log_file: Path | None,
    trace_file: Path | None,
    telemetry_paths: telemetry.TelemetryPaths | None,
) -> None:
    """Reject configurations that make Rust and Python append to one file."""
    if telemetry_paths is None or telemetry_paths.event_file is None:
        return
    python_event_file = telemetry_paths.event_file
    if paths_refer_to_same_file(log_file, python_event_file):
        message = (
            "g-log-file points at the Python telemetry event stream. "
            "Use a separate Rust tracing path such as rust.events.jsonl."
        )
        raise ValueError(message)
    if paths_refer_to_same_file(trace_file, python_event_file):
        message = (
            "g-trace-file points at the Python telemetry event stream. "
            "Use a separate Rust tracing path such as rust.events.jsonl."
        )
        raise ValueError(message)


def paths_refer_to_same_file(first_path: Path | None, second_path: Path | None) -> bool:
    """Return whether two configured paths resolve to the same file."""
    if first_path is None or second_path is None:
        return False
    return first_path.expanduser().resolve() == second_path.expanduser().resolve()


def regenie(regenie_config: config.RegenieConfig) -> RunArtifacts:
    """Run the shared REGENIE-compatible config path."""
    config.validate_config(regenie_config)
    telemetry_module = load_telemetry_module()
    telemetry_session = telemetry_module.build_telemetry_session(regenie_config)
    initialize_logging(regenie_config.g_diagnostics, telemetry_session.paths)
    telemetry_session.log_event(
        "run_started",
        association_mode=execution_plan.resolve_association_mode(regenie_config.trait.trait_type).value,
        trait_type=regenie_config.trait.trait_type.value,
        phenotype_count=len(regenie_config.input.pheno_columns),
        output_run_root=str(telemetry_module.resolve_output_run_root(regenie_config)),
    )
    logger.info("Starting REGENIE run.")
    configure_runtime(regenie_config.g_compute, regenie_config.trait)
    try:
        artifacts = run_validated_regenie_config(regenie_config, telemetry_session=telemetry_session)
    except Exception:
        telemetry_session.log_event("run_failed", level="error")
        logger.exception("REGENIE run failed.")
        raise
    else:
        telemetry_session.log_event("run_completed")
        logger.info("Finished REGENIE run.")
        return artifacts
    finally:
        telemetry_module.close_telemetry_session(telemetry_session)


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
        configure_runtime_before_jax_import(regenie_config.g_compute)
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
        final_parquet_paths = dispatch_execution_plan(
            plan=plan,
            stage_timing_recorder=stage_timing_recorder,
            telemetry_session=telemetry_session,
        )
        logger.debug("Finalizing REGENIE execution plan.")
        return finalize_execution_plan(
            regenie_config=regenie_config,
            plan=plan,
            final_parquet_paths=final_parquet_paths,
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
        "resume": plan.output_plan.resume,
        "resume_mode": plan.output_plan.resume_mode,
        "finalize_parquet": plan.output_plan.finalize_parquet,
        "writer_thread_count": plan.output_plan.writer_threads,
        "writer_queue_depth": plan.output_plan.writer_queue_depth,
        "chunks_per_arrow_file": plan.output_plan.chunks_per_arrow_file,
        "arrow_compression": plan.output_plan.arrow_compression,
        "trusted_no_missing_diploid": plan.kernel_config.trusted_no_missing_diploid,
        "trusted_bgen_validation_mode": plan.kernel_config.trusted_bgen_validation_mode,
        "bgen_decode_tile_variant_count": plan.kernel_config.bgen_decode_tile_variant_count,
        "bgen_simd": plan.kernel_config.bgen_simd,
        "jax_device": plan.kernel_config.device,
        "jax_matmul_precision": plan.kernel_config.alignment_config.jax_matmul_precision,
        "jax_enable_x64": plan.kernel_config.alignment_config.jax_enable_x64,
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
        final_parquet_path = run_regenie2_binary_bgen_pipeline(
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
            final_parquet_path=final_parquet_path,
        )
        return final_parquet_path
    logger.debug("Dispatching linear native engine pipeline.")
    final_parquet_path = run_regenie2_linear_bgen_pipeline(**common_arguments)
    log_writer_finished(
        telemetry_session=telemetry_session,
        association_mode=plan.association_mode,
        phenotype=phenotype_run_plan.phenotype_name,
        final_parquet_path=final_parquet_path,
    )
    return final_parquet_path


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
        }
    )
    if plan.association_mode == types.AssociationMode.REGENIE2_BINARY:
        logger.debug("Dispatching multi-phenotype binary native engine pipeline.")
        final_parquet_paths = run_regenie2_multi_phenotype_binary_bgen_pipeline(
            **common_arguments,
            correction_plan=plan.binary_correction_plan,
            kernel_config=plan.kernel_config.binary_kernel_config,
            null_logistic_nonconvergence_policy=(
                plan.kernel_config.alignment_config.null_logistic_nonconvergence_policy
            ),
        )
    else:
        logger.debug("Dispatching multi-phenotype linear native engine pipeline.")
        final_parquet_paths = run_regenie2_multi_phenotype_linear_bgen_pipeline(**common_arguments)
    if telemetry_session is not None:
        telemetry_session.log_event(
            "writer_finished",
            association_mode=plan.association_mode.value,
            phenotype_count=len(plan.phenotype_run_plans),
            final_parquet_paths=tuple(None if path is None else str(path) for path in final_parquet_paths),
        )
    return final_parquet_paths


def log_writer_finished(
    *,
    telemetry_session: telemetry.TelemetrySession | None,
    association_mode: types.AssociationMode,
    phenotype: str,
    final_parquet_path: Path | None,
) -> None:
    """Record output writer completion."""
    if telemetry_session is None:
        return
    telemetry_session.log_event(
        "writer_finished",
        association_mode=association_mode.value,
        phenotype=phenotype,
        final_parquet_path=None if final_parquet_path is None else str(final_parquet_path),
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
    final_parquet_paths: tuple[Path | None, ...],
) -> RunArtifacts:
    """Build user-facing artifacts after native execution."""
    phenotype_artifacts = tuple(
        finalize_phenotype_run(
            regenie_config=regenie_config,
            plan=plan,
            phenotype_run_plan=phenotype_run_plan,
            final_parquet_path=final_parquet_path,
        )
        for phenotype_run_plan, final_parquet_path in zip(
            plan.phenotype_run_plans,
            final_parquet_paths,
            strict=True,
        )
    )
    logger.info("Finalized REGENIE run artifacts for %s phenotype(s).", len(phenotype_artifacts))
    if len(phenotype_artifacts) == 1:
        return phenotype_artifacts[0]
    return RunArtifacts(phenotype_artifacts=phenotype_artifacts)


def finalize_phenotype_run(
    *,
    regenie_config: config.RegenieConfig,
    plan: execution_plan.RegenieExecutionPlan,
    phenotype_run_plan: execution_plan.PhenotypeRunPlan,
    final_parquet_path: Path | None,
) -> RunArtifacts:
    """Build artifacts for one phenotype."""
    del regenie_config, plan
    return RunArtifacts(
        output_run_directory=phenotype_run_plan.output_run_paths.run_directory,
        final_parquet=final_parquet_path,
        effective_config=phenotype_run_plan.effective_config_path,
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
        "bgen_decode_tile_variant_count": plan.kernel_config.bgen_decode_tile_variant_count,
        "bgen_simd": plan.kernel_config.bgen_simd.value,
        "trusted_no_missing_diploid": plan.kernel_config.trusted_no_missing_diploid,
        "trusted_bgen_validation_mode": plan.kernel_config.trusted_bgen_validation_mode.value,
    }
    output.write_run_manifest(phenotype_run_plan.output_run_paths, manifest)
