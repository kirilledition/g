"""JAX runtime policy, setup reports, and diagnostics."""

from __future__ import annotations

import enum
import logging
import typing
from dataclasses import dataclass

if typing.TYPE_CHECKING:
    from pathlib import Path

    from g import types
    from g.interface import config


class JaxPlatform(enum.StrEnum):
    """JAX backend platform selector."""

    CPU = "cpu"
    CUDA = "cuda"


class XlaAuxiliaryCacheMode(enum.StrEnum):
    """XLA auxiliary persistent cache mode."""

    DISABLED = "none"
    PER_FUSION_AUTOTUNE = "xla_gpu_per_fusion_autotune_cache_dir"


class GpuValidationStatus(enum.StrEnum):
    """GPU validation state for a runtime setup attempt."""

    PENDING = "pending"
    SKIPPED = "skipped"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class JaxRuntimeDiagnosticLevel(enum.StrEnum):
    """Diagnostic event severity level."""

    INFO = "info"
    ERROR = "error"


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
class XlaAuxiliaryCacheResolution:
    """Resolved XLA auxiliary cache behavior.

    Attributes:
        mode: JAX config value for `jax_persistent_cache_enable_xla_caches`.
        reason: Human-readable reason for the selected mode.

    """

    mode: XlaAuxiliaryCacheMode
    reason: str

    @property
    def enabled(self) -> bool:
        """Return whether an auxiliary XLA cache is enabled."""
        return self.mode != XlaAuxiliaryCacheMode.DISABLED


@dataclass(frozen=True)
class GpuValidationResult:
    """Result of validating a requested GPU backend.

    Attributes:
        status: Validation status.
        message: Optional validation detail or failure message.

    """

    status: GpuValidationStatus
    message: str | None = None


@dataclass(frozen=True)
class JaxConfigUpdateOperation:
    """One ordered `jax.config.update(...)` operation.

    Attributes:
        setting_name: JAX config setting name.
        value: Config value to apply.

    """

    setting_name: str
    value: bool | int | str


@dataclass(frozen=True)
class JaxRuntimeSetupReport:
    """Resolved JAX runtime setup decisions.

    Attributes:
        requested_device: User-requested execution device.
        platform: JAX platform selected from the device.
        cache_directory: Resolved persistent compilation cache directory.
        matmul_precision: Resolved default matmul precision.
        persistent_cache_enabled: Whether JAX persistent compilation caching is enabled.
        persistent_cache_min_entry_size_bytes: Minimum persistent-cache entry size.
        persistent_cache_min_compile_time_seconds: Minimum persistent-cache compile time.
        xla_auxiliary_cache: Resolved XLA auxiliary cache behavior.
        transfer_guard_enabled: Whether transfer guard diagnostics are enabled.
        gpu_validation: GPU validation result.

    """

    requested_device: types.Device
    platform: JaxPlatform
    cache_directory: Path
    matmul_precision: types.JaxMatmulPrecision
    persistent_cache_enabled: bool
    persistent_cache_min_entry_size_bytes: int
    persistent_cache_min_compile_time_seconds: int
    xla_auxiliary_cache: XlaAuxiliaryCacheResolution
    transfer_guard_enabled: bool
    gpu_validation: GpuValidationResult


@dataclass(frozen=True)
class JaxRuntimeDiagnosticEvent:
    """Structured diagnostic event for JAX runtime setup.

    Attributes:
        event_name: Telemetry event name.
        level: Event severity.
        message: Human-readable log message.
        fields: Structured event fields.

    """

    event_name: str
    level: JaxRuntimeDiagnosticLevel
    message: str
    fields: tuple[tuple[str, object], ...]


CONFIGURED_JAX_RUNTIME_POLICY: JaxRuntimePolicy | None = None


def build_jax_runtime_policy(compute_config: config.GComputeConfig) -> JaxRuntimePolicy:
    """Build the process-global JAX runtime policy requested by a run.

    Args:
        compute_config: Normalized compute configuration.

    Returns:
        Requested JAX runtime policy.

    """
    cache_directory = None
    if compute_config.jax_cache_dir is not None:
        cache_directory = compute_config.jax_cache_dir.expanduser()
    return JaxRuntimePolicy(
        device=compute_config.device,
        cache_directory=cache_directory,
        matmul_precision=compute_config.jax_matmul_precision,
        persistent_cache=compute_config.jax_persistent_cache,
        persistent_cache_min_entry_size_bytes=compute_config.jax_persistent_cache_min_entry_size_bytes,
        persistent_cache_min_compile_time_seconds=compute_config.jax_persistent_cache_min_compile_time_seconds,
        xla_autotune_cache=compute_config.jax_xla_autotune_cache,
        transfer_guard=compute_config.jax_transfer_guard,
    )


def describe_jax_runtime_policy(policy: JaxRuntimePolicy) -> str:
    """Format a JAX runtime policy for diagnostics.

    Args:
        policy: Runtime policy to format.

    Returns:
        Stable human-readable policy description.

    """
    cache_directory = "<default>" if policy.cache_directory is None else str(policy.cache_directory)
    matmul_precision = "<default>" if policy.matmul_precision is None else policy.matmul_precision.value
    return (
        f"device={policy.device.value}, "
        f"jax-cache-dir={cache_directory}, "
        f"jax-matmul-precision={matmul_precision}, "
        f"jax-persistent-cache={policy.persistent_cache}, "
        f"jax-persistent-cache-min-entry-size-bytes={policy.persistent_cache_min_entry_size_bytes}, "
        f"jax-persistent-cache-min-compile-time-seconds={policy.persistent_cache_min_compile_time_seconds}, "
        f"jax-xla-autotune-cache={policy.xla_autotune_cache}, "
        f"jax-transfer-guard={policy.transfer_guard}"
    )


def require_compatible_jax_runtime_policy(compute_config: config.GComputeConfig) -> JaxRuntimePolicy:
    """Return the requested policy or raise when it conflicts with the first run.

    Args:
        compute_config: Normalized compute configuration.

    Returns:
        Requested JAX runtime policy.

    Raises:
        RuntimeError: If a previous run configured incompatible process-global JAX settings.

    """
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


def jax_runtime_policy_is_configured(policy: JaxRuntimePolicy) -> bool:
    """Return whether the requested policy is already configured.

    Args:
        policy: Policy to compare with the configured process-global policy.

    Returns:
        Whether the policy has already been configured.

    """
    return policy == CONFIGURED_JAX_RUNTIME_POLICY


def mark_jax_runtime_policy_configured(policy: JaxRuntimePolicy) -> None:
    """Record that JAX has been configured for this process.

    Args:
        policy: Policy that was successfully configured.

    """
    global CONFIGURED_JAX_RUNTIME_POLICY
    CONFIGURED_JAX_RUNTIME_POLICY = policy


def diagnostic_fields(**fields: object) -> tuple[tuple[str, object], ...]:
    """Build immutable diagnostic fields without `None` values.

    Args:
        fields: Candidate event fields.

    Returns:
        Event field tuple.

    """
    return tuple((key, value) for key, value in fields.items() if value is not None)


def diagnostic_event_fields(diagnostic_event: JaxRuntimeDiagnosticEvent) -> dict[str, object]:
    """Return diagnostic event fields as a dictionary.

    Args:
        diagnostic_event: Event to convert.

    Returns:
        Dictionary of event fields.

    """
    return dict(diagnostic_event.fields)


def diagnostic_logging_level(level: JaxRuntimeDiagnosticLevel) -> int:
    """Return a Python logging level for a runtime diagnostic level.

    Args:
        level: Runtime diagnostic severity.

    Returns:
        Python logging level constant.

    """
    if level == JaxRuntimeDiagnosticLevel.ERROR:
        return logging.ERROR
    return logging.INFO


def build_jax_runtime_diagnostic_events(
    setup_report: JaxRuntimeSetupReport,
) -> tuple[JaxRuntimeDiagnosticEvent, ...]:
    """Build structured diagnostic events from a setup report.

    Args:
        setup_report: Resolved setup report to describe.

    Returns:
        Ordered diagnostic events.

    """
    gpu_validation_level = JaxRuntimeDiagnosticLevel.INFO
    if setup_report.gpu_validation.status == GpuValidationStatus.FAILED:
        gpu_validation_level = JaxRuntimeDiagnosticLevel.ERROR
    return (
        JaxRuntimeDiagnosticEvent(
            event_name="jax_platform_selected",
            level=JaxRuntimeDiagnosticLevel.INFO,
            message=f"Selected JAX platform {setup_report.platform.value}.",
            fields=diagnostic_fields(
                requested_device=setup_report.requested_device.value,
                platform=setup_report.platform.value,
            ),
        ),
        JaxRuntimeDiagnosticEvent(
            event_name="jax_persistent_cache_configured",
            level=JaxRuntimeDiagnosticLevel.INFO,
            message=(
                "JAX persistent compilation cache enabled."
                if setup_report.persistent_cache_enabled
                else "JAX persistent compilation cache disabled."
            ),
            fields=diagnostic_fields(
                enabled=setup_report.persistent_cache_enabled,
                cache_directory=str(setup_report.cache_directory),
                min_entry_size_bytes=setup_report.persistent_cache_min_entry_size_bytes,
                min_compile_time_seconds=setup_report.persistent_cache_min_compile_time_seconds,
            ),
        ),
        JaxRuntimeDiagnosticEvent(
            event_name="jax_xla_auxiliary_cache_configured",
            level=JaxRuntimeDiagnosticLevel.INFO,
            message=(
                "XLA auxiliary persistent cache enabled."
                if setup_report.xla_auxiliary_cache.enabled
                else "XLA auxiliary persistent cache disabled."
            ),
            fields=diagnostic_fields(
                enabled=setup_report.xla_auxiliary_cache.enabled,
                mode=setup_report.xla_auxiliary_cache.mode.value,
                reason=setup_report.xla_auxiliary_cache.reason,
            ),
        ),
        JaxRuntimeDiagnosticEvent(
            event_name="jax_transfer_guard_configured",
            level=JaxRuntimeDiagnosticLevel.INFO,
            message=(
                "JAX transfer guard diagnostics enabled."
                if setup_report.transfer_guard_enabled
                else "JAX transfer guard diagnostics disabled."
            ),
            fields=diagnostic_fields(enabled=setup_report.transfer_guard_enabled),
        ),
        JaxRuntimeDiagnosticEvent(
            event_name="jax_gpu_validation",
            level=gpu_validation_level,
            message=f"JAX GPU validation {setup_report.gpu_validation.status.value}.",
            fields=diagnostic_fields(
                status=setup_report.gpu_validation.status.value,
                message=setup_report.gpu_validation.message,
            ),
        ),
    )


def emit_jax_runtime_setup_diagnostics(
    setup_report: JaxRuntimeSetupReport,
    diagnostic_sink: typing.Callable[[JaxRuntimeDiagnosticEvent], None] | None,
) -> None:
    """Emit setup report diagnostics through an optional sink.

    Args:
        setup_report: Setup report to describe.
        diagnostic_sink: Optional structured diagnostic event sink.

    """
    if diagnostic_sink is None:
        return
    for diagnostic_event in build_jax_runtime_diagnostic_events(setup_report):
        diagnostic_sink(diagnostic_event)
