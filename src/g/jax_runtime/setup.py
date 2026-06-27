"""Order-sensitive JAX runtime configuration and GPU validation."""

from __future__ import annotations

import dataclasses
import typing
from pathlib import Path

import jax

from g import _core, types
from g.jax_runtime import diagnostics, models, resolution

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
        gpu_validation_report = validate_gpu_device()
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
        gpu_validation_status=gpu_validation_report.status,
        gpu_validation_message=gpu_validation_report.message,
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
    validate_gpu_device()


def validate_gpu_device() -> models.JaxGpuValidationReport:
    """Validate that JAX can report a GPU backend.

    Returns:
        Native validation report when at least one GPU is available.

    Raises:
        RuntimeError: If no visible NVIDIA driver or JAX GPU device is available.

    """
    if not nvidia_driver_is_visible():
        missing_driver_plan = jax_gpu_validation_report_from_native_payload(
            _core.plan_jax_gpu_validation_payload(
                nvidia_driver_visible=False,
                backend_initialization_failed=False,
                device_platforms=(),
                device_descriptions=(),
            )
        )
        raise RuntimeError(missing_driver_plan.message)
    try:
        devices = jax.devices()
    except Exception as error:
        backend_failure_plan = jax_gpu_validation_report_from_native_payload(
            _core.plan_jax_gpu_validation_payload(
                nvidia_driver_visible=True,
                backend_initialization_failed=True,
                device_platforms=(),
                device_descriptions=(),
            )
        )
        raise RuntimeError(backend_failure_plan.message) from error
    validation_report = jax_gpu_validation_report_from_native_payload(
        _core.plan_jax_gpu_validation_payload(
            nvidia_driver_visible=True,
            backend_initialization_failed=False,
            device_platforms=tuple(
                str(getattr(typing.cast("typing.Any", device), "platform", "")) for device in devices
            ),
            device_descriptions=tuple(str(device) for device in devices),
        )
    )
    if validation_report.status == models.GpuValidationStatus.FAILED:
        raise RuntimeError(validation_report.message)
    return validation_report


def jax_gpu_validation_report_from_native_payload(payload: object) -> models.JaxGpuValidationReport:
    """Adapt a native JAX GPU validation payload."""
    validation_payload = dict(typing.cast("typing.Mapping[str, object]", payload))
    return models.JaxGpuValidationReport(
        status=models.GpuValidationStatus(str(validation_payload["status"])),
        message=str(validation_payload["message"]),
    )
