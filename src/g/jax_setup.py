"""JAX runtime configuration for the GWAS engine."""

from __future__ import annotations

import dataclasses
import typing
from pathlib import Path

import jax

from g import jax_runtime, runtime_paths, runtime_policy, types

DEFAULT_CACHE_DIRECTORY_NAME = "g-jax-cache"


def default_node_local_jax_compilation_cache_directory() -> Path:
    """Build the default node-local JAX compilation cache directory."""
    return runtime_paths.default_node_local_cache_directory(DEFAULT_CACHE_DIRECTORY_NAME)


GPU_DEVICE_PLATFORM_NAME = "gpu"
NVIDIA_CONTROL_DEVICE_PATH = Path("/dev/nvidiactl")
NVIDIA_UVM_DEVICE_PATH = Path("/dev/nvidia-uvm")
NVIDIA_DRIVER_DIRECTORY_PATH = Path("/proc/driver/nvidia")


def resolve_jax_compilation_cache_directory(cache_directory: Path | None = None) -> Path:
    """Resolve the persistent JAX compilation cache directory."""
    if cache_directory is not None:
        return cache_directory.expanduser()
    return default_node_local_jax_compilation_cache_directory()


def resolve_jax_platform(device: types.Device) -> jax_runtime.JaxPlatform:
    """Resolve the JAX backend platform for a requested device.

    Args:
        device: Requested execution device.

    Returns:
        JAX platform selector.

    """
    if device == types.Device.GPU:
        return jax_runtime.JaxPlatform.CUDA
    return jax_runtime.JaxPlatform.CPU


def resolve_xla_auxiliary_cache(
    cache_directory: Path,
    *,
    persistent_cache: bool,
    enable_xla_autotune_cache: bool,
) -> jax_runtime.XlaAuxiliaryCacheResolution:
    """Resolve whether XLA auxiliary persistent caches should be enabled.

    Args:
        cache_directory: Resolved persistent compilation cache directory.
        persistent_cache: Whether the JAX persistent compilation cache is enabled.
        enable_xla_autotune_cache: Whether the user requested XLA autotune caches.

    Returns:
        XLA auxiliary cache mode and reason.

    """
    if not persistent_cache:
        return jax_runtime.XlaAuxiliaryCacheResolution(
            mode=jax_runtime.XlaAuxiliaryCacheMode.DISABLED,
            reason="persistent compilation cache is disabled",
        )
    if not enable_xla_autotune_cache:
        return jax_runtime.XlaAuxiliaryCacheResolution(
            mode=jax_runtime.XlaAuxiliaryCacheMode.DISABLED,
            reason="XLA auxiliary cache was not requested",
        )
    if runtime_paths.path_is_beegfs(cache_directory):
        return jax_runtime.XlaAuxiliaryCacheResolution(
            mode=jax_runtime.XlaAuxiliaryCacheMode.DISABLED,
            reason="cache directory is on BeeGFS",
        )
    if not runtime_paths.path_is_node_local(cache_directory):
        return jax_runtime.XlaAuxiliaryCacheResolution(
            mode=jax_runtime.XlaAuxiliaryCacheMode.DISABLED,
            reason="cache directory is not node-local",
        )
    return jax_runtime.XlaAuxiliaryCacheResolution(
        mode=jax_runtime.XlaAuxiliaryCacheMode.PER_FUSION_AUTOTUNE,
        reason="cache directory is node-local",
    )


def nvidia_driver_is_visible() -> bool:
    """Return whether the process can see a Linux NVIDIA driver/device mount."""
    return (
        NVIDIA_CONTROL_DEVICE_PATH.exists() or NVIDIA_UVM_DEVICE_PATH.exists() or NVIDIA_DRIVER_DIRECTORY_PATH.exists()
    )


def resolve_jax_runtime_setup(policy: jax_runtime.JaxRuntimePolicy) -> jax_runtime.JaxRuntimeSetupReport:
    """Resolve JAX setup decisions without mutating process-global state.

    Args:
        policy: Requested runtime policy.

    Returns:
        Setup report with pure resolution decisions.

    """
    resolved_cache_directory = resolve_jax_compilation_cache_directory(policy.cache_directory)
    gpu_validation = jax_runtime.GpuValidationResult(
        status=jax_runtime.GpuValidationStatus.SKIPPED,
        message="CPU runtime requested; GPU validation skipped.",
    )
    if policy.device == types.Device.GPU:
        gpu_validation = jax_runtime.GpuValidationResult(status=jax_runtime.GpuValidationStatus.PENDING)
    matmul_precision = types.JaxMatmulPrecision.FLOAT32
    if policy.matmul_precision is not None:
        matmul_precision = policy.matmul_precision
    return jax_runtime.JaxRuntimeSetupReport(
        requested_device=policy.device,
        platform=resolve_jax_platform(policy.device),
        cache_directory=resolved_cache_directory,
        matmul_precision=matmul_precision,
        persistent_cache_enabled=policy.persistent_cache,
        persistent_cache_min_entry_size_bytes=policy.persistent_cache_min_entry_size_bytes,
        persistent_cache_min_compile_time_seconds=policy.persistent_cache_min_compile_time_seconds,
        xla_auxiliary_cache=resolve_xla_auxiliary_cache(
            resolved_cache_directory,
            persistent_cache=policy.persistent_cache,
            enable_xla_autotune_cache=policy.xla_autotune_cache,
        ),
        transfer_guard_enabled=policy.transfer_guard,
        gpu_validation=gpu_validation,
    )


def build_jax_config_update_operations(
    setup_report: jax_runtime.JaxRuntimeSetupReport,
) -> tuple[jax_runtime.JaxConfigUpdateOperation, ...]:
    """Build ordered JAX config mutations from a setup report.

    Args:
        setup_report: Resolved setup decisions.

    Returns:
        Ordered JAX config update operations.

    """
    operations = [
        jax_runtime.JaxConfigUpdateOperation("jax_platforms", setup_report.platform.value),
        jax_runtime.JaxConfigUpdateOperation("jax_enable_x64", runtime_policy.JAX_ENABLE_X64),
        jax_runtime.JaxConfigUpdateOperation("jax_default_matmul_precision", setup_report.matmul_precision.value),
    ]
    if setup_report.persistent_cache_enabled:
        operations.extend(
            [
                jax_runtime.JaxConfigUpdateOperation(
                    "jax_compilation_cache_dir",
                    str(setup_report.cache_directory),
                ),
                jax_runtime.JaxConfigUpdateOperation(
                    "jax_persistent_cache_min_entry_size_bytes",
                    setup_report.persistent_cache_min_entry_size_bytes,
                ),
                jax_runtime.JaxConfigUpdateOperation(
                    "jax_persistent_cache_min_compile_time_secs",
                    setup_report.persistent_cache_min_compile_time_seconds,
                ),
                jax_runtime.JaxConfigUpdateOperation(
                    "jax_persistent_cache_enable_xla_caches",
                    setup_report.xla_auxiliary_cache.mode.value,
                ),
            ]
        )
    if setup_report.transfer_guard_enabled:
        operations.append(jax_runtime.JaxConfigUpdateOperation("jax_transfer_guard", "disallow"))
    return tuple(operations)


def apply_jax_config_update_operations(
    operations: tuple[jax_runtime.JaxConfigUpdateOperation, ...],
) -> None:
    """Apply ordered JAX config update operations.

    Args:
        operations: Config update operations to apply.

    """
    for operation in operations:
        jax.config.update(operation.setting_name, operation.value)


def configure_jax_runtime(
    policy: jax_runtime.JaxRuntimePolicy,
    *,
    diagnostic_sink: typing.Callable[[jax_runtime.JaxRuntimeDiagnosticEvent], None] | None = None,
) -> jax_runtime.JaxRuntimeSetupReport:
    """Configure JAX runtime knobs before engine modules are imported.

    Args:
        policy: Requested runtime policy.
        diagnostic_sink: Optional structured diagnostic event sink.

    Returns:
        Setup report after validation.

    """
    return configure_jax_runtime_before_backend_init(policy, diagnostic_sink=diagnostic_sink)


def configure_jax_runtime_before_backend_init(
    policy: jax_runtime.JaxRuntimePolicy,
    *,
    diagnostic_sink: typing.Callable[[jax_runtime.JaxRuntimeDiagnosticEvent], None] | None = None,
) -> jax_runtime.JaxRuntimeSetupReport:
    """Configure JAX platform and runtime knobs before backend initialization.

    Args:
        policy: Requested runtime policy.
        diagnostic_sink: Optional structured diagnostic event sink.

    Returns:
        Setup report after validation.

    Raises:
        RuntimeError: If GPU execution was requested but validation fails.

    """
    setup_report = resolve_jax_runtime_setup(policy)
    if setup_report.persistent_cache_enabled:
        setup_report.cache_directory.mkdir(parents=True, exist_ok=True)
    apply_jax_config_update_operations(build_jax_config_update_operations(setup_report))
    if policy.device != types.Device.GPU:
        jax_runtime.emit_jax_runtime_setup_diagnostics(setup_report, diagnostic_sink)
        return setup_report
    try:
        require_gpu_device()
    except RuntimeError as error:
        failed_report = dataclasses.replace(
            setup_report,
            gpu_validation=jax_runtime.GpuValidationResult(
                status=jax_runtime.GpuValidationStatus.FAILED,
                message=str(error),
            ),
        )
        jax_runtime.emit_jax_runtime_setup_diagnostics(failed_report, diagnostic_sink)
        raise
    validated_report = dataclasses.replace(
        setup_report,
        gpu_validation=jax_runtime.GpuValidationResult(
            status=jax_runtime.GpuValidationStatus.SUCCEEDED,
            message="JAX reported at least one GPU device.",
        ),
    )
    jax_runtime.emit_jax_runtime_setup_diagnostics(validated_report, diagnostic_sink)
    return validated_report


def require_gpu_device() -> None:
    """Raise when JAX cannot initialize a GPU backend."""
    if not nvidia_driver_is_visible():
        message = (
            "JAX GPU execution was requested, but this process cannot see the NVIDIA driver or device files. "
            "Observed no /dev/nvidiactl, no /dev/nvidia-uvm, and no /proc/driver/nvidia. "
            "Run on a GPU allocation/node or expose the NVIDIA devices to this container/session."
        )
        raise RuntimeError(message)
    try:
        devices = jax.devices()
    except Exception as error:
        message = (
            "JAX GPU execution was requested, but no CUDA-enabled JAX backend could be initialized. "
            "The JAX CUDA plugin failed while initializing the backend. Confirm that the process is running on a "
            "GPU node, the NVIDIA driver is loaded, CUDA device files are visible, and the installed JAX CUDA plugin "
            "matches the node driver/runtime. Install the GPU dependency group when needed, for example: "
            "`uv sync --python 3.14 --group dev --group gpu`."
        )
        raise RuntimeError(message) from error
    gpu_devices = [
        device
        for device in devices
        if getattr(typing.cast("typing.Any", device), "platform", None) == GPU_DEVICE_PLATFORM_NAME
    ]
    if not gpu_devices:
        observed_devices = ", ".join(str(device) for device in devices) or "none"
        message = (
            "JAX GPU execution was requested, but JAX did not report any GPU devices. "
            f"Observed devices: {observed_devices}."
        )
        raise RuntimeError(message)
