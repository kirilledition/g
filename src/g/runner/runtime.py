"""Runtime setup and process-global policy for REGENIE-compatible runs."""

from __future__ import annotations

import importlib
import logging
import typing
from dataclasses import dataclass
from pathlib import Path

from g import _core, types
from g.jax_runtime import models as jax_runtime_models
from g.jax_runtime import resolution as jax_runtime_resolution
from g.jax_runtime import state as jax_runtime_state

if typing.TYPE_CHECKING:
    from g.engine import telemetry
    from g.interface import config

logger = logging.getLogger(__name__)


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
        logging_policy: Requested logging and tracing sink policy.
        rayon_thread_count: Requested Rayon thread count, or None when unset.
        jax_policy: Requested JAX runtime policy.

    """

    logging_policy: LoggingRuntimePolicy
    rayon_thread_count: int | None
    jax_policy: jax_runtime_models.JaxRuntimePolicy


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


CONFIGURED_LOGGING_RUNTIME_POLICY: LoggingRuntimePolicy | None = None
CONFIGURED_RAYON_THREAD_COUNT: int | None = None


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
    logging_level = logging.INFO
    if diagnostic_event.level == jax_runtime_models.JaxRuntimeDiagnosticLevel.ERROR:
        logging_level = logging.ERROR
    logger.log(
        logging_level,
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
    telemetry_session: telemetry.TelemetrySession | None,
) -> jax_runtime_models.JaxRuntimeSetupReport | None:
    """Configure JAX platform and runtime before compute modules are imported."""
    requested_policy = jax_runtime_resolution.resolve_jax_runtime_policy(compute_config)
    jax_runtime_state.require_compatible_jax_runtime_policy(requested_policy)
    if requested_policy == jax_runtime_state.CONFIGURED_JAX_RUNTIME_POLICY:
        return None

    def record_diagnostic_event(diagnostic_event: jax_runtime_models.JaxRuntimeDiagnosticEvent) -> None:
        record_jax_runtime_diagnostic_event(diagnostic_event, telemetry_session=telemetry_session)

    setup_module = importlib.import_module("g.jax_runtime.setup")
    setup_report = setup_module.configure_before_backend_init(
        requested_policy,
        diagnostic_sink=record_diagnostic_event,
    )
    jax_runtime_state.CONFIGURED_JAX_RUNTIME_POLICY = requested_policy
    return setup_report


def build_logging_runtime_policy(
    diagnostics_config: config.GDiagnosticsConfig,
    telemetry_paths: telemetry.TelemetryPaths | None,
) -> LoggingRuntimePolicy:
    """Build the process-global logging policy requested by a run."""
    native_build_logging_runtime_policy = getattr(_core, "build_logging_runtime_policy_payload", None)
    if callable(native_build_logging_runtime_policy):
        telemetry_stream_file = None if telemetry_paths is None else telemetry_paths.stream_file
        native_payload = native_build_logging_runtime_policy(
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
    return build_logging_runtime_policy_with_python_fallback(diagnostics_config, telemetry_paths)


def build_logging_runtime_policy_with_python_fallback(
    diagnostics_config: config.GDiagnosticsConfig,
    telemetry_paths: telemetry.TelemetryPaths | None,
) -> LoggingRuntimePolicy:
    """Build logging policy when the native helper is unavailable."""
    telemetry_stream_file = None if telemetry_paths is None else telemetry_paths.stream_file
    log_file = diagnostics_config.log_file if telemetry_stream_file is None else None
    trace_file = diagnostics_config.trace_file if telemetry_stream_file is None else telemetry_stream_file
    trace_filter = diagnostics_config.trace_filter
    if telemetry_stream_file is not None and diagnostics_config.telemetry != types.TelemetryMode.TRACE:
        trace_filter = diagnostics_config.log_filter
    trace_event_cap = (
        diagnostics_config.trace_event_cap if diagnostics_config.telemetry == types.TelemetryMode.TRACE else None
    )
    return LoggingRuntimePolicy(
        log_filter=diagnostics_config.log_filter,
        log_file=log_file,
        log_stderr=diagnostics_config.log_stderr,
        log_queue_size=diagnostics_config.log_queue_size,
        log_lossy=diagnostics_config.log_lossy,
        include_source_location=diagnostics_config.include_source_location,
        include_span_events=diagnostics_config.include_span_events,
        trace_file=trace_file,
        trace_filter=trace_filter,
        trace_event_cap=trace_event_cap,
    )


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
    return RuntimePolicy(
        logging_policy=build_logging_runtime_policy(regenie_config.g_diagnostics, telemetry_paths),
        rayon_thread_count=regenie_config.trait.threads,
        jax_policy=jax_runtime_resolution.resolve_jax_runtime_policy(regenie_config.g_compute),
    )


def describe_logging_runtime_policy(policy: LoggingRuntimePolicy) -> str:
    """Format a logging runtime policy for concise errors."""
    native_describe_logging_runtime_policy = getattr(_core, "describe_logging_runtime_policy_value", None)
    if callable(native_describe_logging_runtime_policy):
        return str(
            native_describe_logging_runtime_policy(
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
    log_file = "<none>" if policy.log_file is None else str(policy.log_file)
    trace_file = "<none>" if policy.trace_file is None else str(policy.trace_file)
    trace_event_cap = "<none>" if policy.trace_event_cap is None else str(policy.trace_event_cap)
    return (
        f"log-filter={policy.log_filter}, "
        f"log-file={log_file}, "
        f"log-stderr={policy.log_stderr}, "
        f"log-queue-size={policy.log_queue_size}, "
        f"log-lossy={policy.log_lossy}, "
        f"include-source-location={policy.include_source_location}, "
        f"include-span-events={policy.include_span_events}, "
        f"trace-file={trace_file}, "
        f"trace-filter={policy.trace_filter}, "
        f"trace-event-cap={trace_event_cap}"
    )


def require_compatible_logging_runtime_policy(logging_policy: LoggingRuntimePolicy) -> None:
    """Raise when a run requests incompatible process-global logging settings."""
    if CONFIGURED_LOGGING_RUNTIME_POLICY is None or logging_policy == CONFIGURED_LOGGING_RUNTIME_POLICY:
        return
    message = (
        "Logging runtime policy is process-global for this Python process. "
        f"Configured policy: {describe_logging_runtime_policy(CONFIGURED_LOGGING_RUNTIME_POLICY)}. "
        f"Requested policy: {describe_logging_runtime_policy(logging_policy)}. "
        "Start a fresh Python process for incompatible logging settings."
    )
    raise RuntimeError(message)


def require_compatible_rayon_thread_count(thread_count: int | None) -> None:
    """Raise when a run requests an incompatible process-global Rayon thread count."""
    if thread_count is None or CONFIGURED_RAYON_THREAD_COUNT is None or thread_count == CONFIGURED_RAYON_THREAD_COUNT:
        return
    message = (
        "Rayon --threads is process-global for this Python process. "
        f"Configured thread count: {CONFIGURED_RAYON_THREAD_COUNT}. "
        f"Requested thread count: {thread_count}. "
        "Start a fresh Python process for incompatible Rayon settings."
    )
    raise RuntimeError(message)


def require_compatible_runtime_policy(runtime_policy: RuntimePolicy) -> None:
    """Raise when a run conflicts with process-global runtime state."""
    require_compatible_logging_runtime_policy(runtime_policy.logging_policy)
    require_compatible_rayon_thread_count(runtime_policy.rayon_thread_count)
    jax_runtime_state.require_compatible_jax_runtime_policy(runtime_policy.jax_policy)


def describe_runtime_state() -> RuntimeState:
    """Return the process-global runtime state known to this Python process."""
    return RuntimeState(
        logging_policy=CONFIGURED_LOGGING_RUNTIME_POLICY,
        rayon_thread_count=CONFIGURED_RAYON_THREAD_COUNT,
        jax_policy=jax_runtime_state.CONFIGURED_JAX_RUNTIME_POLICY,
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


def configure_rayon_thread_pool(core_module: typing.Any, thread_count: int) -> None:
    """Configure Rayon global thread count once and reject incompatible repeats."""
    global CONFIGURED_RAYON_THREAD_COUNT
    if thread_count == CONFIGURED_RAYON_THREAD_COUNT:
        return
    require_compatible_rayon_thread_count(thread_count)
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
    telemetry_paths: telemetry.TelemetryPaths | None,
) -> None:
    """Initialize unified Rust/Python logging before runtime setup."""
    global CONFIGURED_LOGGING_RUNTIME_POLICY
    logging_policy = build_logging_runtime_policy(diagnostics_config, telemetry_paths)
    require_compatible_logging_runtime_policy(logging_policy)
    initialized_logging = _core.initialize_logging(
        log_filter=logging_policy.log_filter,
        log_file=None if logging_policy.log_file is None else str(logging_policy.log_file),
        log_stderr=logging_policy.log_stderr,
        log_queue_size=logging_policy.log_queue_size,
        log_lossy=logging_policy.log_lossy,
        include_source_location=logging_policy.include_source_location,
        include_span_events=logging_policy.include_span_events,
        trace_file=None if logging_policy.trace_file is None else str(logging_policy.trace_file),
        trace_filter=logging_policy.trace_filter,
        trace_event_cap=logging_policy.trace_event_cap,
    )
    if initialized_logging is False:
        CONFIGURED_LOGGING_RUNTIME_POLICY = logging_policy
        return
    if initialized_logging is True:
        CONFIGURED_LOGGING_RUNTIME_POLICY = logging_policy
