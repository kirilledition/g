"""JAX runtime policy, setup report, and diagnostic models."""

from __future__ import annotations

import enum
import typing
from dataclasses import dataclass

if typing.TYPE_CHECKING:
    from pathlib import Path

    from g import types

JAX_ENABLE_X64 = True
JAX_CPU_PLATFORM_NAME = "cpu"
JAX_CUDA_PLATFORM_NAME = "cuda"


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
class JaxRuntimeSetupReport:
    """Resolved JAX runtime setup decisions.

    Attributes:
        requested_device: User-requested execution device.
        platform_name: JAX platform name selected from the device.
        cache_directory: Resolved persistent compilation cache directory.
        matmul_precision: Resolved default matmul precision.
        persistent_cache_enabled: Whether JAX persistent compilation caching is enabled.
        persistent_cache_min_entry_size_bytes: Minimum persistent-cache entry size.
        persistent_cache_min_compile_time_seconds: Minimum persistent-cache compile time.
        xla_auxiliary_cache_mode: JAX config value for `jax_persistent_cache_enable_xla_caches`.
        xla_auxiliary_cache_reason: Human-readable reason for the selected mode.
        transfer_guard_enabled: Whether transfer guard diagnostics are enabled.
        gpu_validation_status: GPU validation status.
        gpu_validation_message: Optional validation detail or failure message.

    """

    requested_device: types.Device
    platform_name: str
    cache_directory: Path
    matmul_precision: types.JaxMatmulPrecision
    persistent_cache_enabled: bool
    persistent_cache_min_entry_size_bytes: int
    persistent_cache_min_compile_time_seconds: int
    xla_auxiliary_cache_mode: XlaAuxiliaryCacheMode
    xla_auxiliary_cache_reason: str
    transfer_guard_enabled: bool
    gpu_validation_status: GpuValidationStatus
    gpu_validation_message: str | None


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
