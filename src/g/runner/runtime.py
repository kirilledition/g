"""Runtime setup and process-global policy for REGENIE-compatible runs."""

from __future__ import annotations

import importlib
import typing
from dataclasses import dataclass
from pathlib import Path

from g import _core, types

if typing.TYPE_CHECKING:
    from g.engine import dispatch_requests
    from g.runner import events


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
class JaxRuntimePolicy:
    """Process-global JAX runtime settings selected by the first run.

    Attributes:
        device: Requested JAX platform.
        cache_directory: Persistent compilation cache directory.
        matmul_precision: Requested matmul precision.
        persistent_cache: Whether persistent compilation caching is enabled.
        persistent_cache_min_entry_size_bytes: Minimum cache entry size.
        persistent_cache_min_compile_time_seconds: Minimum compile time for cache entries.
        xla_autotune_cache: Whether XLA autotune caches are enabled.
        transfer_guard: Whether transfer guard diagnostics are enabled.

    """

    device: types.Device
    cache_directory: Path | None
    matmul_precision: types.JaxMatmulPrecision | None
    persistent_cache: bool
    persistent_cache_min_entry_size_bytes: int
    persistent_cache_min_compile_time_seconds: int
    xla_autotune_cache: bool
    transfer_guard: bool


@dataclass(frozen=True)
class RuntimePolicyRequest:
    """Narrow process-runtime policy request.

    Attributes:
        diagnostics_config: Logging, tracing, and telemetry runtime settings.
        compute_config: JAX runtime settings.
        rayon_thread_count: Requested Rayon worker thread count.
        telemetry_paths: Resolved telemetry output paths.

    """

    diagnostics_config: _core.GDiagnosticsConfig
    compute_config: _core.GComputeConfig
    rayon_thread_count: int | None
    telemetry_paths: events.TelemetryPaths | None


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
    def jax_policy(self) -> JaxRuntimePolicy:
        """Return the requested JAX runtime policy view."""
        return jax_runtime_policy_from_native_policy(self.native_policy.jax_runtime_policy())


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
    def jax_policy(self) -> JaxRuntimePolicy:
        """Return the checked JAX runtime policy view."""
        return jax_runtime_policy_from_native_policy(self.native_runtime.jax_runtime_policy())


PROCESS_RUNTIME_STATE: _core.NativeRuntimeState = _core.NativeRuntimeState.global_process_runtime_state()


def configure_runtime_before_jax_import(
    compute_config: _core.GComputeConfig,
    telemetry_session: events.TelemetrySession | None,
) -> _core.NativeJaxRuntimeSetupReport | None:
    """Configure JAX platform and runtime before compute modules are imported."""
    requested_policy = build_jax_runtime_policy(compute_config)
    native_setup_session = PROCESS_RUNTIME_STATE.build_jax_runtime_setup_session_resolving_cache_directory(
        requested_policy,
    )
    if not native_setup_session.should_configure:
        return None

    def record_diagnostic_event(diagnostic_event: _core.NativeJaxRuntimeDiagnosticEvent) -> None:
        _core.record_jax_runtime_diagnostic_event(diagnostic_event, telemetry_session)

    setup_module = importlib.import_module("g.jax_runtime")
    setup_report = setup_module.configure_before_backend_init(
        native_setup_session=native_setup_session,
        diagnostic_sink=record_diagnostic_event,
    )
    PROCESS_RUNTIME_STATE.complete_jax_runtime_setup_session(
        requested_policy,
        native_setup_session,
    )
    return setup_report


def build_logging_runtime_policy(
    diagnostics_config: _core.GDiagnosticsConfig,
    telemetry_paths: events.TelemetryPaths | None,
) -> LoggingRuntimePolicy:
    """Build the process-global logging policy requested by a run."""
    telemetry_stream_file = None if telemetry_paths is None else telemetry_paths.stream_file
    native_policy = PROCESS_RUNTIME_STATE.build_logging_runtime_policy(
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
    return logging_runtime_policy_from_native_policy(native_policy)


def logging_runtime_policy_from_native_policy(native_policy: _core.NativeLoggingRuntimePolicy) -> LoggingRuntimePolicy:
    """Adapt a native logging-runtime policy to the Python dataclass."""
    return LoggingRuntimePolicy(
        log_filter=native_policy.log_filter,
        log_file=optional_path_from_native_value(native_policy.log_file),
        log_stderr=native_policy.log_stderr,
        log_queue_size=native_policy.log_queue_size,
        log_lossy=native_policy.log_lossy,
        include_source_location=native_policy.include_source_location,
        include_span_events=native_policy.include_span_events,
        trace_file=optional_path_from_native_value(native_policy.trace_file),
        trace_filter=native_policy.trace_filter,
        trace_event_cap=native_policy.trace_event_cap,
    )


def logging_runtime_policy_to_native_policy(policy: LoggingRuntimePolicy) -> _core.NativeLoggingRuntimePolicy:
    """Adapt a Python logging runtime policy view to a native policy handle."""
    return PROCESS_RUNTIME_STATE.build_logging_runtime_policy(
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
        "trace",
        None,
    )


def build_jax_runtime_policy(compute_config: _core.GComputeConfig) -> _core.NativeJaxRuntimePolicy:
    """Build the process-global native JAX runtime policy requested by a run."""
    return PROCESS_RUNTIME_STATE.build_jax_runtime_policy(
        device=compute_config.device.value,
        cache_directory=(
            None if compute_config.jax_cache_dir is None else str(compute_config.jax_cache_dir.expanduser())
        ),
        matmul_precision=None
        if compute_config.jax_matmul_precision is None
        else compute_config.jax_matmul_precision.value,
        persistent_cache=compute_config.jax_persistent_cache,
        persistent_cache_min_entry_size_bytes=compute_config.jax_persistent_cache_min_entry_size_bytes,
        persistent_cache_min_compile_time_seconds=compute_config.jax_persistent_cache_min_compile_time_seconds,
        xla_autotune_cache=compute_config.jax_xla_autotune_cache,
        transfer_guard=compute_config.jax_transfer_guard,
    )


def jax_runtime_policy_from_native_policy(
    native_policy: _core.NativeJaxRuntimePolicy,
) -> JaxRuntimePolicy:
    """Adapt a native JAX runtime policy to the Python dataclass."""
    return JaxRuntimePolicy(
        device=types.Device(native_policy.device),
        cache_directory=optional_path_from_native_value(native_policy.cache_directory),
        matmul_precision=None
        if native_policy.matmul_precision is None
        else types.JaxMatmulPrecision(native_policy.matmul_precision),
        persistent_cache=native_policy.persistent_cache,
        persistent_cache_min_entry_size_bytes=native_policy.persistent_cache_min_entry_size_bytes,
        persistent_cache_min_compile_time_seconds=native_policy.persistent_cache_min_compile_time_seconds,
        xla_autotune_cache=native_policy.xla_autotune_cache,
        transfer_guard=native_policy.transfer_guard,
    )


def optional_path_from_native_value(path_value: str | None) -> Path | None:
    """Adapt a native optional path value to `Path`."""
    if path_value is None:
        return None
    return Path(path_value)


def build_runtime_policy(request: RuntimePolicyRequest) -> RuntimePolicy:
    """Build the process-global runtime policy requested by a run."""
    logging_policy = build_logging_runtime_policy(request.diagnostics_config, request.telemetry_paths)
    jax_policy = build_jax_runtime_policy(request.compute_config)
    return RuntimePolicy(
        native_policy=PROCESS_RUNTIME_STATE.build_runtime_policy_handle(
            logging_runtime_policy_to_native_policy(logging_policy),
            request.rayon_thread_count,
            jax_policy,
        )
    )


def require_compatible_runtime_policy(runtime_policy: RuntimePolicy) -> _core.NativeRuntimeCompatibilityToken:
    """Return a native token after process-global runtime checks pass."""
    return build_run_runtime(runtime_policy).runtime_compatibility_token


def build_run_runtime(runtime_policy: RuntimePolicy) -> RunRuntime:
    """Build a run-scoped native runtime handle after compatibility checks pass."""
    return RunRuntime(native_runtime=PROCESS_RUNTIME_STATE.build_run_runtime(runtime_policy.native_policy))


def run_regenie2_linear_bgen_pipeline(request: dispatch_requests.SingleTraitLinearPipelineRequest) -> Path | None:
    """Run the linear native pipeline after JAX runtime setup."""
    single_trait_pipeline_module = importlib.import_module("g.engine.regenie2_pipeline.single_trait")
    return single_trait_pipeline_module.run_regenie2_linear_bgen_pipeline(request)


def run_regenie2_binary_bgen_pipeline(request: dispatch_requests.SingleTraitBinaryPipelineRequest) -> Path | None:
    """Run the binary native pipeline after JAX runtime setup."""
    single_trait_pipeline_module = importlib.import_module("g.engine.regenie2_pipeline.single_trait")
    return single_trait_pipeline_module.run_regenie2_binary_bgen_pipeline(request)


def run_regenie2_multi_phenotype_linear_bgen_pipeline(
    request: dispatch_requests.MultiTraitLinearPipelineRequest,
) -> tuple[Path | None, ...]:
    """Run the multi-phenotype linear native pipeline after JAX runtime setup."""
    multi_trait_pipeline_module = importlib.import_module("g.engine.regenie2_pipeline.multi_trait")
    return multi_trait_pipeline_module.run_regenie2_multi_phenotype_bgen_pipeline(request)


def run_regenie2_multi_phenotype_binary_bgen_pipeline(
    request: dispatch_requests.MultiTraitBinaryPipelineRequest,
) -> tuple[Path | None, ...]:
    """Run the multi-phenotype binary native pipeline after JAX runtime setup."""
    multi_trait_pipeline_module = importlib.import_module("g.engine.regenie2_pipeline.multi_trait")
    return multi_trait_pipeline_module.run_regenie2_multi_phenotype_bgen_pipeline(request)


def initialize_logging(
    diagnostics_config: _core.GDiagnosticsConfig,
    telemetry_paths: events.TelemetryPaths | None,
) -> None:
    """Initialize unified Rust/Python logging before runtime setup."""
    logging_policy = build_logging_runtime_policy(diagnostics_config, telemetry_paths)
    PROCESS_RUNTIME_STATE.initialize_logging_runtime_policy(logging_runtime_policy_to_native_policy(logging_policy))
