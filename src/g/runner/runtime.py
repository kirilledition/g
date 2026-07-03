"""Runtime setup and process-global policy for REGENIE-compatible runs."""

from __future__ import annotations

import importlib
import typing
from dataclasses import dataclass
from pathlib import Path

from g import _core, types
from g.jax_runtime import models as jax_runtime_models
from g.jax_runtime import resolution as jax_runtime_resolution

if typing.TYPE_CHECKING:
    from g import execution_plan
    from g.interface import config
    from g.runner import events, outputs, timing


class SingleTraitPipelineKwargs(typing.TypedDict):
    """Shared late-import kwargs for one single-trait pipeline dispatch."""

    genotype_source_config: execution_plan.GenotypeSourceConfig
    phenotype_path: Path
    phenotype_name: str
    prediction_list_path: Path
    covariate_path: Path | None
    covariate_names: tuple[str, ...] | None
    chunk_size: int
    variant_limit: int | None
    output_run_paths: outputs.OutputRunPaths
    staging_depth: int
    native_callback_batch_size: int
    result_in_flight_limit: int | None
    dosage_buffer_limit: int | None
    existing_manifest: dict[str, object] | None
    resume: bool
    resume_mode: types.ResumeMode
    writer_settings: outputs.OutputWriterSettings
    trusted_no_missing_diploid: bool
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode
    bgen_decode_tile_variant_count: int
    jax_device: types.Device
    jax_matmul_precision: types.JaxMatmulPrecision | None
    score_dtype: types.FloatingPointDtype
    firth_dtype: types.FloatingPointDtype
    gpu_genotype_format: types.GpuGenotypeFormat
    stage_timing_recorder: timing.StageTimingRecorder | None
    telemetry_session: events.TelemetrySession | None
    alignment_config: config.GComputeConfig | None
    runtime_compatibility_token: _core.NativeRuntimeCompatibilityToken
    output_initialized_callback: typing.Callable[[tuple[str, ...]], None] | None


class LinearSingleTraitPipelineKwargs(SingleTraitPipelineKwargs):
    """Late-import kwargs for the linear single-trait pipeline."""

    linear_numerical_config: execution_plan.LinearNumericalConfig | None


class BinarySingleTraitPipelineKwargs(SingleTraitPipelineKwargs):
    """Late-import kwargs for the binary single-trait pipeline."""

    correction_plan: types.BinaryCorrectionPlan
    kernel_config: execution_plan.BinaryKernelConfig
    null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy


class MultiPhenotypePipelineKwargs(typing.TypedDict):
    """Shared late-import kwargs for multi-phenotype pipeline dispatch."""

    genotype_source_config: execution_plan.GenotypeSourceConfig
    phenotype_path: Path
    phenotype_names: tuple[str, ...]
    prediction_list_path: Path
    covariate_path: Path | None
    covariate_names: tuple[str, ...] | None
    chunk_size: int
    variant_limit: int | None
    output_run_paths_by_phenotype: tuple[outputs.OutputRunPaths, ...]
    staging_depth: int
    native_callback_batch_size: int
    result_in_flight_limit: int | None
    dosage_buffer_limit: int | None
    existing_manifests_by_phenotype: tuple[dict[str, object] | None, ...] | None
    resume: bool
    resume_mode: types.ResumeMode
    writer_settings: outputs.OutputWriterSettings
    trusted_no_missing_diploid: bool
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode
    bgen_decode_tile_variant_count: int
    jax_device: types.Device
    jax_matmul_precision: types.JaxMatmulPrecision | None
    score_dtype: types.FloatingPointDtype
    firth_dtype: types.FloatingPointDtype
    gpu_genotype_format: types.GpuGenotypeFormat
    stage_timing_recorder: timing.StageTimingRecorder | None
    telemetry_session: events.TelemetrySession | None
    alignment_config: config.GComputeConfig | None
    runtime_compatibility_token: _core.NativeRuntimeCompatibilityToken
    sample_mode: types.MultiPhenotypeSampleMode | None
    phenotype_compute_groups: tuple[execution_plan.PhenotypeComputeGroup, ...] | None
    output_initialized_callback: typing.Callable[[tuple[str, ...]], None] | None


class LinearMultiPhenotypePipelineKwargs(MultiPhenotypePipelineKwargs):
    """Late-import kwargs for the linear multi-phenotype pipeline."""

    linear_numerical_config: execution_plan.LinearNumericalConfig | None


class BinaryMultiPhenotypePipelineKwargs(MultiPhenotypePipelineKwargs):
    """Late-import kwargs for the binary multi-phenotype pipeline."""

    correction_plan: types.BinaryCorrectionPlan
    kernel_config: execution_plan.BinaryKernelConfig
    null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy


class SingleTraitPipelineModule(typing.Protocol):
    """Single-trait pipeline entrypoints loaded after runtime setup."""

    def run_regenie2_linear_bgen_pipeline(
        self,
        **kwargs: typing.Unpack[LinearSingleTraitPipelineKwargs],
    ) -> Path | None:
        """Run the linear single-trait pipeline."""
        ...

    def run_regenie2_binary_bgen_pipeline(
        self,
        **kwargs: typing.Unpack[BinarySingleTraitPipelineKwargs],
    ) -> Path | None:
        """Run the binary single-trait pipeline."""
        ...


class MultiPhenotypePipelineModule(typing.Protocol):
    """Multi-phenotype pipeline entrypoints loaded after runtime setup."""

    def run_regenie2_multi_phenotype_linear_bgen_pipeline(
        self,
        **kwargs: typing.Unpack[LinearMultiPhenotypePipelineKwargs],
    ) -> tuple[Path | None, ...]:
        """Run the linear multi-phenotype pipeline."""
        ...

    def run_regenie2_multi_phenotype_binary_bgen_pipeline(
        self,
        **kwargs: typing.Unpack[BinaryMultiPhenotypePipelineKwargs],
    ) -> tuple[Path | None, ...]:
        """Run the binary multi-phenotype pipeline."""
        ...


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


@dataclass(frozen=True)
class RuntimePolicy:
    """Process-global runtime choices requested by one run.

    Attributes:
        native_policy: Native aggregate process-runtime policy handle.

    """

    native_policy: _core.NativeRuntimePolicy

    @property
    def logging_policy(self) -> LoggingRuntimePolicy:
        """Return the requested logging and tracing sink policy view."""
        return logging_runtime_policy_from_native_policy(self.native_policy.logging_runtime_policy())

    @property
    def rayon_thread_count(self) -> int | None:
        """Return the requested Rayon thread count, or None when unset."""
        return self.native_policy.rayon_thread_count

    @property
    def jax_policy(self) -> jax_runtime_models.JaxRuntimePolicy:
        """Return the requested JAX runtime policy view."""
        return jax_runtime_resolution.jax_runtime_policy_from_native_policy(self.native_policy.jax_runtime_policy())


@dataclass(frozen=True)
class RunRuntime:
    """Run-scoped native runtime handle after compatibility checks pass.

    Attributes:
        native_runtime: Native run runtime handle.

    """

    native_runtime: _core.NativeRunRuntime

    @property
    def runtime_compatibility_token(self) -> _core.NativeRuntimeCompatibilityToken:
        """Return the native token proving runtime compatibility checks passed."""
        return self.native_runtime.runtime_compatibility_token()

    @property
    def logging_policy(self) -> LoggingRuntimePolicy:
        """Return the checked logging and tracing sink policy view."""
        return logging_runtime_policy_from_native_policy(self.native_runtime.logging_runtime_policy())

    @property
    def rayon_thread_count(self) -> int | None:
        """Return the checked Rayon thread count, or None when unset."""
        return self.native_runtime.rayon_thread_count

    @property
    def jax_policy(self) -> jax_runtime_models.JaxRuntimePolicy:
        """Return the checked JAX runtime policy view."""
        return jax_runtime_resolution.jax_runtime_policy_from_native_policy(self.native_runtime.jax_runtime_policy())


@dataclass(frozen=True)
class RuntimeState:
    """Process-global runtime choices already configured in this Python process.

    Attributes:
        logging_policy: Configured logging and tracing sink policy.
        rayon_thread_count: Configured Rayon thread count.
        jax_policy: Configured JAX runtime policy.

    """

    logging_policy: LoggingRuntimePolicy | None
    rayon_thread_count: int | None
    jax_policy: jax_runtime_models.JaxRuntimePolicy | None


PROCESS_RUNTIME_STATE: _core.NativeRuntimeState = _core.NativeRuntimeState.global_process_runtime_state()


def native_jax_runtime_diagnostic_policy() -> _core.NativeJaxRuntimeDiagnosticPolicy:
    """Build the native JAX runtime diagnostic policy handle."""
    return _core.NativeJaxRuntimeDiagnosticPolicy()


def record_jax_runtime_diagnostic_event(
    diagnostic_event: jax_runtime_models.JaxRuntimeDiagnosticEvent,
    *,
    telemetry_session: events.TelemetrySession | None,
) -> None:
    """Record one structured JAX runtime diagnostic event.

    Args:
        diagnostic_event: Runtime diagnostic event to record.
        telemetry_session: Optional run telemetry session.

    """
    native_jax_runtime_diagnostic_policy().record_jax_runtime_diagnostic_event(
        diagnostic_event,
        telemetry_session,
    )


def configure_runtime_before_jax_import(
    compute_config: config.GComputeConfig,
    telemetry_session: events.TelemetrySession | None,
) -> jax_runtime_models.JaxRuntimeSetupReport | None:
    """Configure JAX platform and runtime before compute modules are imported."""
    requested_policy = jax_runtime_resolution.resolve_jax_runtime_policy(compute_config)
    native_requested_policy = jax_runtime_resolution.jax_runtime_policy_to_native_policy(requested_policy)
    native_setup_session = PROCESS_RUNTIME_STATE.build_jax_runtime_setup_session_resolving_cache_directory(
        native_requested_policy,
    )
    if not native_setup_session.should_configure:
        return None

    def record_diagnostic_event(diagnostic_event: jax_runtime_models.JaxRuntimeDiagnosticEvent) -> None:
        record_jax_runtime_diagnostic_event(diagnostic_event, telemetry_session=telemetry_session)

    setup_module = importlib.import_module("g.jax_runtime.setup")
    setup_report = setup_module.configure_before_backend_init(
        native_setup_session=native_setup_session,
        diagnostic_sink=record_diagnostic_event,
    )
    PROCESS_RUNTIME_STATE.complete_jax_runtime_setup_session(
        native_requested_policy,
        native_setup_session,
    )
    return setup_report


def build_native_logging_runtime_policy(
    diagnostics_config: config.GDiagnosticsConfig,
    telemetry_paths: events.TelemetryPaths | None,
) -> _core.NativeLoggingRuntimePolicy:
    """Build the typed native process-global logging policy requested by a run."""
    telemetry_stream_file = None if telemetry_paths is None else telemetry_paths.stream_file
    return PROCESS_RUNTIME_STATE.build_logging_runtime_policy(
        diagnostics_config.log_filter,
        None if diagnostics_config.log_file is None else str(diagnostics_config.log_file),
        diagnostics_config.log_stderr,
        diagnostics_config.log_queue_size,
        diagnostics_config.log_lossy,
        diagnostics_config.include_source_location,
        diagnostics_config.include_span_events,
        None if diagnostics_config.trace_file is None else str(diagnostics_config.trace_file),
        diagnostics_config.trace_filter,
        diagnostics_config.trace_event_cap,
        diagnostics_config.telemetry.value,
        None if telemetry_stream_file is None else str(telemetry_stream_file),
    )


def build_logging_runtime_policy(
    diagnostics_config: config.GDiagnosticsConfig,
    telemetry_paths: events.TelemetryPaths | None,
) -> LoggingRuntimePolicy:
    """Build the process-global logging policy requested by a run."""
    return logging_runtime_policy_from_native_policy(
        build_native_logging_runtime_policy(diagnostics_config, telemetry_paths)
    )


def logging_runtime_policy_from_native_policy(native_policy: _core.NativeLoggingRuntimePolicy) -> LoggingRuntimePolicy:
    """Adapt a typed native logging-runtime policy to the Python dataclass."""
    return LoggingRuntimePolicy(
        log_filter=native_policy.log_filter,
        log_file=None if native_policy.log_file is None else Path(native_policy.log_file),
        log_stderr=native_policy.log_stderr,
        log_queue_size=native_policy.log_queue_size,
        log_lossy=native_policy.log_lossy,
        include_source_location=native_policy.include_source_location,
        include_span_events=native_policy.include_span_events,
        trace_file=None if native_policy.trace_file is None else Path(native_policy.trace_file),
        trace_filter=native_policy.trace_filter,
        trace_event_cap=native_policy.trace_event_cap,
    )


def logging_runtime_policy_to_native_policy(policy: LoggingRuntimePolicy) -> _core.NativeLoggingRuntimePolicy:
    """Adapt a Python logging runtime policy view to a typed native handle."""
    return PROCESS_RUNTIME_STATE.build_logging_runtime_policy_from_values(
        policy.log_filter,
        None if policy.log_file is None else str(policy.log_file),
        policy.log_stderr,
        policy.log_queue_size,
        policy.log_lossy,
        policy.include_source_location,
        policy.include_span_events,
        None if policy.trace_file is None else str(policy.trace_file),
        policy.trace_filter,
        policy.trace_event_cap,
    )


def build_runtime_policy(
    regenie_config: config.RegenieConfig,
    telemetry_paths: events.TelemetryPaths,
) -> RuntimePolicy:
    """Build the process-global runtime policy requested by a run."""
    native_logging_policy = build_native_logging_runtime_policy(regenie_config.g_diagnostics, telemetry_paths)
    jax_policy = jax_runtime_resolution.resolve_jax_runtime_policy(regenie_config.g_compute)
    return RuntimePolicy(
        native_policy=PROCESS_RUNTIME_STATE.build_runtime_policy_handle(
            native_logging_policy,
            regenie_config.trait.threads,
            jax_runtime_resolution.jax_runtime_policy_to_native_policy(jax_policy),
        )
    )


def require_compatible_logging_runtime_policy(logging_policy: LoggingRuntimePolicy) -> None:
    """Raise when a run requests incompatible process-global logging settings."""
    PROCESS_RUNTIME_STATE.require_compatible_logging_runtime_policy(
        logging_runtime_policy_to_native_policy(logging_policy)
    )


def require_compatible_rayon_thread_count(thread_count: int | None) -> None:
    """Raise when a run requests an incompatible process-global Rayon thread count."""
    PROCESS_RUNTIME_STATE.require_compatible_rayon_thread_count(thread_count)


def require_compatible_jax_runtime_policy(jax_policy: jax_runtime_models.JaxRuntimePolicy) -> None:
    """Raise when a run requests incompatible process-global JAX settings."""
    PROCESS_RUNTIME_STATE.require_compatible_jax_runtime_policy(
        jax_runtime_resolution.jax_runtime_policy_to_native_policy(jax_policy)
    )


def record_jax_runtime_policy(jax_policy: jax_runtime_models.JaxRuntimePolicy) -> None:
    """Record the process-global JAX runtime policy configured in this process."""
    PROCESS_RUNTIME_STATE.record_jax_runtime_policy(
        jax_runtime_resolution.jax_runtime_policy_to_native_policy(jax_policy)
    )


def require_compatible_runtime_policy(runtime_policy: RuntimePolicy) -> _core.NativeRuntimeCompatibilityToken:
    """Return a native token after process-global runtime checks pass."""
    return build_run_runtime(runtime_policy).runtime_compatibility_token


def build_run_runtime(runtime_policy: RuntimePolicy) -> RunRuntime:
    """Build a run-scoped native runtime handle after compatibility checks pass."""
    return RunRuntime(native_runtime=PROCESS_RUNTIME_STATE.build_run_runtime(runtime_policy.native_policy))


def describe_runtime_state() -> RuntimeState:
    """Return the process-global runtime state known to this Python process."""
    native_snapshot = PROCESS_RUNTIME_STATE.runtime_state()
    return RuntimeState(
        logging_policy=None
        if native_snapshot.logging_policy is None
        else logging_runtime_policy_from_native_policy(native_snapshot.logging_policy),
        rayon_thread_count=native_snapshot.rayon_thread_count,
        jax_policy=None
        if native_snapshot.jax_policy is None
        else jax_runtime_resolution.jax_runtime_policy_from_native_policy(native_snapshot.jax_policy),
    )


def run_regenie2_linear_bgen_pipeline(**kwargs: typing.Unpack[LinearSingleTraitPipelineKwargs]) -> Path | None:
    """Run the linear native pipeline after JAX runtime setup."""
    single_trait_pipeline_module = typing.cast(
        "SingleTraitPipelineModule",
        importlib.import_module("g.engine.regenie2_pipeline.single_trait"),
    )
    return single_trait_pipeline_module.run_regenie2_linear_bgen_pipeline(**kwargs)


def run_regenie2_binary_bgen_pipeline(**kwargs: typing.Unpack[BinarySingleTraitPipelineKwargs]) -> Path | None:
    """Run the binary native pipeline after JAX runtime setup."""
    single_trait_pipeline_module = typing.cast(
        "SingleTraitPipelineModule",
        importlib.import_module("g.engine.regenie2_pipeline.single_trait"),
    )
    return single_trait_pipeline_module.run_regenie2_binary_bgen_pipeline(**kwargs)


def run_regenie2_multi_phenotype_linear_bgen_pipeline(
    **kwargs: typing.Unpack[LinearMultiPhenotypePipelineKwargs],
) -> tuple[Path | None, ...]:
    """Run the multi-phenotype linear native pipeline after JAX runtime setup."""
    multi_trait_pipeline_module = typing.cast(
        "MultiPhenotypePipelineModule",
        importlib.import_module("g.engine.regenie2_pipeline.multi_trait"),
    )
    return multi_trait_pipeline_module.run_regenie2_multi_phenotype_linear_bgen_pipeline(**kwargs)


def run_regenie2_multi_phenotype_binary_bgen_pipeline(
    **kwargs: typing.Unpack[BinaryMultiPhenotypePipelineKwargs],
) -> tuple[Path | None, ...]:
    """Run the multi-phenotype binary native pipeline after JAX runtime setup."""
    multi_trait_pipeline_module = typing.cast(
        "MultiPhenotypePipelineModule",
        importlib.import_module("g.engine.regenie2_pipeline.multi_trait"),
    )
    return multi_trait_pipeline_module.run_regenie2_multi_phenotype_binary_bgen_pipeline(**kwargs)


def configure_rayon_thread_pool(thread_count: int) -> None:
    """Configure Rayon global thread count through the native process state."""
    PROCESS_RUNTIME_STATE.configure_rayon_thread_pool(thread_count)


def effective_rayon_thread_count(requested_thread_count: int | None) -> int | None:
    """Return the Rayon thread count known to be effective in this process."""
    return PROCESS_RUNTIME_STATE.effective_rayon_thread_count(requested_thread_count)


def configure_runtime(compute_config: config.GComputeConfig, trait_config: config.TraitConfig) -> None:
    """Apply native runtime knobs before engine execution."""
    PROCESS_RUNTIME_STATE.configure_runtime_knobs(
        compute_config.bgen_decode_tile_variant_count,
        trait_config.threads,
    )


def initialize_logging(
    diagnostics_config: config.GDiagnosticsConfig,
    telemetry_paths: events.TelemetryPaths | None,
) -> None:
    """Initialize unified Rust/Python logging before runtime setup."""
    native_logging_policy = build_native_logging_runtime_policy(diagnostics_config, telemetry_paths)
    PROCESS_RUNTIME_STATE.initialize_logging_runtime_policy(native_logging_policy)
