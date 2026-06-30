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
    from g.engine import telemetry
    from g.interface import config


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
        return logging_runtime_policy_from_native_payload(self.native_policy.logging_runtime_policy_payload())

    @property
    def rayon_thread_count(self) -> int | None:
        """Return the requested Rayon thread count, or None when unset."""
        return self.native_policy.rayon_thread_count

    @property
    def jax_policy(self) -> jax_runtime_models.JaxRuntimePolicy:
        """Return the requested JAX runtime policy view."""
        return jax_runtime_policy_from_native_payload(self.native_policy.jax_runtime_policy_payload())


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
        return logging_runtime_policy_from_native_payload(self.native_runtime.logging_runtime_policy_payload())

    @property
    def rayon_thread_count(self) -> int | None:
        """Return the checked Rayon thread count, or None when unset."""
        return self.native_runtime.rayon_thread_count

    @property
    def jax_policy(self) -> jax_runtime_models.JaxRuntimePolicy:
        """Return the checked JAX runtime policy view."""
        return jax_runtime_policy_from_native_payload(self.native_runtime.jax_runtime_policy_payload())


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


def build_process_runtime_state(
    logging_policy: LoggingRuntimePolicy | None,
    rayon_thread_count: int | None,
    jax_policy: jax_runtime_models.JaxRuntimePolicy | None,
) -> _core.NativeRuntimeState:
    """Build a native process runtime state handle.

    Args:
        logging_policy: Optional configured logging policy to seed.
        rayon_thread_count: Optional configured Rayon thread count to seed.
        jax_policy: Optional configured JAX runtime policy to seed.

    Returns:
        Native process runtime state handle.

    """
    process_runtime_state = _core.NativeRuntimeState()
    if logging_policy is not None:
        process_runtime_state.record_logging_runtime_policy(logging_runtime_policy_to_native_payload(logging_policy))
    if rayon_thread_count is not None:
        process_runtime_state.record_rayon_thread_count(rayon_thread_count)
    if jax_policy is not None:
        process_runtime_state.record_jax_runtime_policy(jax_runtime_policy_to_native_payload(jax_policy))
    return process_runtime_state


PROCESS_RUNTIME_STATE: _core.NativeRuntimeState = build_process_runtime_state(None, None, None)


def record_jax_runtime_diagnostic_event(
    diagnostic_event: jax_runtime_models.JaxRuntimeDiagnosticEvent,
    *,
    telemetry_session: telemetry.TelemetrySession | None,
) -> None:
    """Record one structured JAX runtime diagnostic event.

    Args:
        diagnostic_event: Runtime diagnostic event to record.
        telemetry_session: Optional run telemetry session.

    """
    event_fields = {diagnostic_field.name: diagnostic_field.value for diagnostic_field in diagnostic_event.fields}
    record_plan = _core.plan_jax_runtime_diagnostic_record(
        diagnostic_level=diagnostic_event.level.value,
        has_telemetry_session=telemetry_session is not None,
    )
    _core.emit_diagnostic_event_fields(
        record_plan.logging_level_name.lower(),
        diagnostic_event.event_name,
        diagnostic_event.message,
        event_fields,
    )
    if not record_plan.should_emit_telemetry:
        return
    active_telemetry_session = typing.cast("telemetry.TelemetrySession", telemetry_session)
    active_telemetry_session.log_jax_runtime_diagnostic_event(
        diagnostic_event,
        telemetry_level=record_plan.telemetry_level,
    )


def configure_runtime_before_jax_import(
    compute_config: config.GComputeConfig,
    telemetry_session: telemetry.TelemetrySession | None,
) -> jax_runtime_models.JaxRuntimeSetupReport | None:
    """Configure JAX platform and runtime before compute modules are imported."""
    requested_policy = jax_runtime_resolution.resolve_jax_runtime_policy(compute_config)
    native_setup_session = PROCESS_RUNTIME_STATE.build_jax_runtime_setup_session(
        jax_runtime_policy_to_native_payload(requested_policy),
        str(jax_runtime_resolution.resolve_jax_runtime_cache_directory(requested_policy)),
    )
    if not native_setup_session.should_configure:
        return None

    def record_diagnostic_event(diagnostic_event: jax_runtime_models.JaxRuntimeDiagnosticEvent) -> None:
        record_jax_runtime_diagnostic_event(diagnostic_event, telemetry_session=telemetry_session)

    setup_module = importlib.import_module("g.jax_runtime.setup")
    setup_report = setup_module.configure_before_backend_init(
        requested_policy,
        native_setup_session=native_setup_session,
        diagnostic_sink=record_diagnostic_event,
    )
    PROCESS_RUNTIME_STATE.complete_jax_runtime_setup(jax_runtime_policy_to_native_payload(requested_policy))
    return setup_report


def build_logging_runtime_policy(
    diagnostics_config: config.GDiagnosticsConfig,
    telemetry_paths: telemetry.TelemetryPaths | None,
) -> LoggingRuntimePolicy:
    """Build the process-global logging policy requested by a run."""
    telemetry_stream_file = None if telemetry_paths is None else telemetry_paths.stream_file
    native_payload = _core.build_logging_runtime_policy_payload(
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
    return logging_runtime_policy_from_native_payload(native_payload)


def logging_runtime_policy_from_native_payload(payload: object) -> LoggingRuntimePolicy:
    """Adapt a native logging-runtime policy payload to the Python dataclass."""
    policy_payload = native_mapping_payload(payload)
    return LoggingRuntimePolicy(
        log_filter=str(policy_payload["log_filter"]),
        log_file=optional_path_from_native_payload(policy_payload["log_file"]),
        log_stderr=bool(policy_payload["log_stderr"]),
        log_queue_size=int(policy_payload["log_queue_size"]),
        log_lossy=bool(policy_payload["log_lossy"]),
        include_source_location=bool(policy_payload["include_source_location"]),
        include_span_events=bool(policy_payload["include_span_events"]),
        trace_file=optional_path_from_native_payload(policy_payload["trace_file"]),
        trace_filter=str(policy_payload["trace_filter"]),
        trace_event_cap=None if policy_payload["trace_event_cap"] is None else int(policy_payload["trace_event_cap"]),
    )


def logging_runtime_policy_to_native_payload(policy: LoggingRuntimePolicy) -> dict[str, object]:
    """Adapt a Python logging runtime policy view to the native payload shape."""
    return {
        "log_filter": policy.log_filter,
        "log_file": None if policy.log_file is None else str(policy.log_file),
        "log_stderr": policy.log_stderr,
        "log_queue_size": policy.log_queue_size,
        "log_lossy": policy.log_lossy,
        "include_source_location": policy.include_source_location,
        "include_span_events": policy.include_span_events,
        "trace_file": None if policy.trace_file is None else str(policy.trace_file),
        "trace_filter": policy.trace_filter,
        "trace_event_cap": policy.trace_event_cap,
    }


def jax_runtime_policy_to_native_payload(policy: jax_runtime_models.JaxRuntimePolicy) -> dict[str, object]:
    """Adapt a Python JAX runtime policy view to the native payload shape."""
    return {
        "device": policy.device.value,
        "cache_directory": None if policy.cache_directory is None else str(policy.cache_directory),
        "matmul_precision": None if policy.matmul_precision is None else policy.matmul_precision.value,
        "persistent_cache": policy.persistent_cache,
        "persistent_cache_min_entry_size_bytes": policy.persistent_cache_min_entry_size_bytes,
        "persistent_cache_min_compile_time_seconds": policy.persistent_cache_min_compile_time_seconds,
        "xla_autotune_cache": policy.xla_autotune_cache,
        "transfer_guard": policy.transfer_guard,
    }


def jax_runtime_policy_from_native_payload(payload: object) -> jax_runtime_models.JaxRuntimePolicy:
    """Adapt a native JAX runtime policy payload to the Python dataclass."""
    policy_payload = native_mapping_payload(payload)
    return jax_runtime_models.JaxRuntimePolicy(
        device=types.Device(str(policy_payload["device"])),
        cache_directory=optional_path_from_native_payload(policy_payload["cache_directory"]),
        matmul_precision=None
        if policy_payload["matmul_precision"] is None
        else types.JaxMatmulPrecision(str(policy_payload["matmul_precision"])),
        persistent_cache=bool(policy_payload["persistent_cache"]),
        persistent_cache_min_entry_size_bytes=int(policy_payload["persistent_cache_min_entry_size_bytes"]),
        persistent_cache_min_compile_time_seconds=int(policy_payload["persistent_cache_min_compile_time_seconds"]),
        xla_autotune_cache=bool(policy_payload["xla_autotune_cache"]),
        transfer_guard=bool(policy_payload["transfer_guard"]),
    )


def optional_path_from_native_payload(path_payload: object) -> Path | None:
    """Adapt a native optional path payload to `Path`."""
    if path_payload is None:
        return None
    return Path(str(path_payload))


def native_mapping_payload(payload: object) -> dict[str, typing.Any]:
    """Adapt a native mapping payload to a mutable Python dictionary."""
    return dict(typing.cast("typing.Mapping[str, typing.Any]", payload))


def build_runtime_policy(
    regenie_config: config.RegenieConfig,
    telemetry_paths: telemetry.TelemetryPaths,
) -> RuntimePolicy:
    """Build the process-global runtime policy requested by a run."""
    logging_policy = build_logging_runtime_policy(regenie_config.g_diagnostics, telemetry_paths)
    jax_policy = jax_runtime_resolution.resolve_jax_runtime_policy(regenie_config.g_compute)
    return RuntimePolicy(
        native_policy=_core.build_runtime_policy_handle(
            logging_runtime_policy_to_native_payload(logging_policy),
            regenie_config.trait.threads,
            jax_runtime_policy_to_native_payload(jax_policy),
        )
    )


def describe_logging_runtime_policy(policy: LoggingRuntimePolicy) -> str:
    """Format a logging runtime policy for concise errors."""
    return str(
        _core.describe_logging_runtime_policy_value(
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
    )


def require_compatible_logging_runtime_policy(logging_policy: LoggingRuntimePolicy) -> None:
    """Raise when a run requests incompatible process-global logging settings."""
    PROCESS_RUNTIME_STATE.require_compatible_logging_runtime_policy(
        logging_runtime_policy_to_native_payload(logging_policy)
    )


def require_compatible_rayon_thread_count(thread_count: int | None) -> None:
    """Raise when a run requests an incompatible process-global Rayon thread count."""
    PROCESS_RUNTIME_STATE.require_compatible_rayon_thread_count(thread_count)


def require_compatible_jax_runtime_policy(jax_policy: jax_runtime_models.JaxRuntimePolicy) -> None:
    """Raise when a run requests incompatible process-global JAX settings."""
    PROCESS_RUNTIME_STATE.require_compatible_jax_runtime_policy(jax_runtime_policy_to_native_payload(jax_policy))


def record_jax_runtime_policy(jax_policy: jax_runtime_models.JaxRuntimePolicy) -> None:
    """Record the process-global JAX runtime policy configured in this process."""
    PROCESS_RUNTIME_STATE.record_jax_runtime_policy(jax_runtime_policy_to_native_payload(jax_policy))


def configured_jax_runtime_policy() -> jax_runtime_models.JaxRuntimePolicy | None:
    """Return the process-global JAX runtime policy known to the native state handle."""
    policy_payload = PROCESS_RUNTIME_STATE.jax_runtime_policy_payload()
    if policy_payload is None:
        return None
    return jax_runtime_policy_from_native_payload(policy_payload)


def require_compatible_runtime_policy(runtime_policy: RuntimePolicy) -> _core.NativeRuntimeCompatibilityToken:
    """Return a native token after process-global runtime checks pass."""
    return build_run_runtime(runtime_policy).runtime_compatibility_token


def build_run_runtime(runtime_policy: RuntimePolicy) -> RunRuntime:
    """Build a run-scoped native runtime handle after compatibility checks pass."""
    return RunRuntime(native_runtime=PROCESS_RUNTIME_STATE.build_run_runtime(runtime_policy.native_policy))


def describe_runtime_state() -> RuntimeState:
    """Return the process-global runtime state known to this Python process."""
    logging_policy_payload = PROCESS_RUNTIME_STATE.logging_runtime_policy_payload()
    return RuntimeState(
        logging_policy=None
        if logging_policy_payload is None
        else logging_runtime_policy_from_native_payload(logging_policy_payload),
        rayon_thread_count=PROCESS_RUNTIME_STATE.rayon_thread_count,
        jax_policy=configured_jax_runtime_policy(),
    )


def run_regenie2_linear_bgen_pipeline(**kwargs: typing.Any) -> Path | None:
    """Run the linear native pipeline after JAX runtime setup."""
    single_trait_pipeline_module = importlib.import_module("g.engine.regenie2_pipeline.single_trait")
    return single_trait_pipeline_module.run_regenie2_linear_bgen_pipeline(**kwargs)


def run_regenie2_binary_bgen_pipeline(**kwargs: typing.Any) -> Path | None:
    """Run the binary native pipeline after JAX runtime setup."""
    single_trait_pipeline_module = importlib.import_module("g.engine.regenie2_pipeline.single_trait")
    return single_trait_pipeline_module.run_regenie2_binary_bgen_pipeline(**kwargs)


def run_regenie2_multi_phenotype_linear_bgen_pipeline(**kwargs: typing.Any) -> tuple[Path | None, ...]:
    """Run the multi-phenotype linear native pipeline after JAX runtime setup."""
    multi_trait_pipeline_module = importlib.import_module("g.engine.regenie2_pipeline.multi_trait")
    return multi_trait_pipeline_module.run_regenie2_multi_phenotype_linear_bgen_pipeline(**kwargs)


def run_regenie2_multi_phenotype_binary_bgen_pipeline(**kwargs: typing.Any) -> tuple[Path | None, ...]:
    """Run the multi-phenotype binary native pipeline after JAX runtime setup."""
    multi_trait_pipeline_module = importlib.import_module("g.engine.regenie2_pipeline.multi_trait")
    return multi_trait_pipeline_module.run_regenie2_multi_phenotype_binary_bgen_pipeline(**kwargs)


def configure_rayon_thread_pool(thread_count: int) -> None:
    """Configure Rayon global thread count through the native process state."""
    PROCESS_RUNTIME_STATE.configure_rayon_thread_pool(thread_count)


def effective_rayon_thread_count(requested_thread_count: int | None) -> int | None:
    """Return the Rayon thread count known to be effective in this process."""
    return PROCESS_RUNTIME_STATE.effective_rayon_thread_count(requested_thread_count)


def configure_runtime(compute_config: config.GComputeConfig, trait_config: config.TraitConfig) -> None:
    """Apply native runtime knobs before engine execution."""
    _core.emit_diagnostic_event_fields(
        "debug",
        "native_runtime_knobs_configured",
        "Configuring native runtime knobs.",
        {
            "bgen_decode_tile_variant_count": compute_config.bgen_decode_tile_variant_count,
            "threads": trait_config.threads,
        },
    )
    _core.configure_bgen_decode_tile_variant_count(compute_config.bgen_decode_tile_variant_count)
    if trait_config.threads is not None:
        configure_rayon_thread_pool(trait_config.threads)


def initialize_logging(
    diagnostics_config: config.GDiagnosticsConfig,
    telemetry_paths: telemetry.TelemetryPaths | None,
) -> None:
    """Initialize unified Rust/Python logging before runtime setup."""
    logging_policy = build_logging_runtime_policy(diagnostics_config, telemetry_paths)
    PROCESS_RUNTIME_STATE.initialize_logging_runtime_policy(logging_runtime_policy_to_native_payload(logging_policy))
