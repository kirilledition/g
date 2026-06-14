"""Order-sensitive JAX runtime configuration and GPU validation."""

from __future__ import annotations

import dataclasses
import typing
from pathlib import Path

import jax

from g import types
from g.jax_runtime import diagnostics, models, resolution

GPU_DEVICE_PLATFORM_NAME = "gpu"
NVIDIA_CONTROL_DEVICE_PATH = Path("/dev/nvidiactl")
NVIDIA_UVM_DEVICE_PATH = Path("/dev/nvidia-uvm")
NVIDIA_DRIVER_DIRECTORY_PATH = Path("/proc/driver/nvidia")


def configure_before_backend_init(
    policy: models.JaxRuntimePolicy,
    *,
    diagnostic_sink: typing.Callable[[models.JaxRuntimeDiagnosticEvent], None] | None,
) -> models.JaxRuntimeSetupReport:
    """Configure JAX platform and runtime knobs before backend initialization.

    Args:
        policy: Requested runtime policy.
        diagnostic_sink: Optional structured diagnostic event sink.

    Returns:
        Setup report after validation.

    Raises:
        RuntimeError: If GPU execution was requested but validation fails.

    """
    setup_report = resolution.resolve_jax_runtime_setup(policy)
    if setup_report.persistent_cache_enabled:
        setup_report.cache_directory.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_platforms", setup_report.platform_name)
    jax.config.update("jax_enable_x64", models.JAX_ENABLE_X64)
    jax.config.update("jax_default_matmul_precision", setup_report.matmul_precision.value)
    if setup_report.persistent_cache_enabled:
        jax.config.update("jax_compilation_cache_dir", str(setup_report.cache_directory))
        jax.config.update(
            "jax_persistent_cache_min_entry_size_bytes",
            setup_report.persistent_cache_min_entry_size_bytes,
        )
        jax.config.update(
            "jax_persistent_cache_min_compile_time_secs",
            setup_report.persistent_cache_min_compile_time_seconds,
        )
        jax.config.update(
            "jax_persistent_cache_enable_xla_caches",
            setup_report.xla_auxiliary_cache_mode.value,
        )
    if setup_report.transfer_guard_enabled:
        jax.config.update("jax_transfer_guard", "disallow")
    if policy.device != types.Device.GPU:
        if diagnostic_sink is not None:
            for diagnostic_event in diagnostics.diagnostic_events_from_setup_report(setup_report):
                diagnostic_sink(diagnostic_event)
        return setup_report
    try:
        require_gpu_device()
    except RuntimeError as error:
        failed_report = dataclasses.replace(
            setup_report,
            gpu_validation_status=models.GpuValidationStatus.FAILED,
            gpu_validation_message=str(error),
        )
        if diagnostic_sink is not None:
            for diagnostic_event in diagnostics.diagnostic_events_from_setup_report(failed_report):
                diagnostic_sink(diagnostic_event)
        raise
    validated_report = dataclasses.replace(
        setup_report,
        gpu_validation_status=models.GpuValidationStatus.SUCCEEDED,
        gpu_validation_message="JAX reported at least one GPU device.",
    )
    if diagnostic_sink is not None:
        for diagnostic_event in diagnostics.diagnostic_events_from_setup_report(validated_report):
            diagnostic_sink(diagnostic_event)
    return validated_report


def nvidia_driver_is_visible() -> bool:
    """Return whether the process can see a Linux NVIDIA driver/device mount.

    Returns:
        Whether NVIDIA driver files are visible.

    """
    return (
        NVIDIA_CONTROL_DEVICE_PATH.exists() or NVIDIA_UVM_DEVICE_PATH.exists() or NVIDIA_DRIVER_DIRECTORY_PATH.exists()
    )


def require_gpu_device() -> None:
    """Raise when JAX cannot initialize a GPU backend.

    Raises:
        RuntimeError: If no visible NVIDIA driver or JAX GPU device is available.

    """
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
