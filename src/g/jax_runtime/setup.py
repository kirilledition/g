"""Order-sensitive JAX runtime configuration and GPU validation."""

from __future__ import annotations

import typing
from pathlib import Path

import jax

from g import _core
from g.jax_runtime import diagnostics, models, resolution

NVIDIA_CONTROL_DEVICE_PATH = Path("/dev/nvidiactl")
NVIDIA_UVM_DEVICE_PATH = Path("/dev/nvidia-uvm")
NVIDIA_DRIVER_DIRECTORY_PATH = Path("/proc/driver/nvidia")


def configure_before_backend_init(
    policy: models.JaxRuntimePolicy,
    *,
    native_setup_session: _core.NativeJaxRuntimeSetupSession | None,
    diagnostic_sink: typing.Callable[[models.JaxRuntimeDiagnosticEvent], None] | None,
) -> models.JaxRuntimeSetupReport:
    """Configure JAX platform and runtime knobs before backend initialization.

    Args:
        policy: Requested runtime policy.
        native_setup_session: Native setup session to own setup decisions.
        diagnostic_sink: Optional structured diagnostic event sink.

    Returns:
        Setup report after validation.

    Raises:
        RuntimeError: If GPU execution was requested but validation fails.

    """
    active_setup_session = resolve_active_setup_session(policy, native_setup_session)
    setup_report = resolution.jax_runtime_setup_report_from_native_payload(active_setup_session.setup_payload())
    side_effect_plan = active_setup_session.side_effect_plan_payload()
    active_setup_session.create_cache_directory_if_configured()
    apply_jax_runtime_config_updates(active_setup_session)
    if not typing.cast("bool", side_effect_plan["should_validate_gpu"]):
        if diagnostic_sink is not None:
            for diagnostic_event in diagnostics.diagnostic_events_from_native_setup_session(active_setup_session):
                diagnostic_sink(diagnostic_event)
        return setup_report
    try:
        gpu_validation_report = validate_gpu_device()
    except RuntimeError as error:
        complete_jax_runtime_setup_validation_report(
            active_setup_session,
            validation_status=models.GpuValidationStatus.FAILED,
            validation_message=str(error),
        )
        if diagnostic_sink is not None:
            for diagnostic_event in diagnostics.diagnostic_events_from_native_setup_session(active_setup_session):
                diagnostic_sink(diagnostic_event)
        raise
    validated_report = complete_jax_runtime_setup_validation_report(
        active_setup_session,
        validation_status=gpu_validation_report.status,
        validation_message=gpu_validation_report.message,
    )
    if diagnostic_sink is not None:
        for diagnostic_event in diagnostics.diagnostic_events_from_native_setup_session(active_setup_session):
            diagnostic_sink(diagnostic_event)
    return validated_report


def resolve_active_setup_session(
    policy: models.JaxRuntimePolicy,
    native_setup_session: _core.NativeJaxRuntimeSetupSession | None,
) -> _core.NativeJaxRuntimeSetupSession:
    """Return the caller-provided native setup session or build a direct one.

    Args:
        policy: Requested runtime policy.
        native_setup_session: Optional native setup session.

    Returns:
        Native setup session.

    """
    if native_setup_session is not None:
        return native_setup_session
    return resolution.build_native_jax_runtime_setup_session(policy)


def nvidia_driver_is_visible() -> bool:
    """Return whether the process can see a Linux NVIDIA driver/device mount.

    Returns:
        Whether NVIDIA driver files are visible.

    """
    return _core.nvidia_driver_files_are_visible_value(
        control_device_path=str(NVIDIA_CONTROL_DEVICE_PATH),
        uvm_device_path=str(NVIDIA_UVM_DEVICE_PATH),
        driver_directory_path=str(NVIDIA_DRIVER_DIRECTORY_PATH),
    )


def apply_jax_runtime_config_updates(native_setup_session: _core.NativeJaxRuntimeSetupSession) -> None:
    """Apply native-ordered JAX runtime config updates."""
    native_setup_session.apply_config_updates()


def complete_jax_runtime_setup_validation_report(
    native_setup_session: _core.NativeJaxRuntimeSetupSession,
    *,
    validation_status: models.GpuValidationStatus,
    validation_message: str | None,
) -> models.JaxRuntimeSetupReport:
    """Complete a setup report after the JAX GPU validation side effect.

    Args:
        native_setup_session: Native setup session before GPU validation has completed.
        validation_status: Final GPU validation status.
        validation_message: Optional validation detail.

    Returns:
        Completed setup report.

    """
    completed_payload = native_setup_session.complete_validation_payload(
        validation_status.value,
        validation_message,
    )
    return resolution.jax_runtime_setup_report_from_native_payload(completed_payload)


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
