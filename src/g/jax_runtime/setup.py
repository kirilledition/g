"""Order-sensitive JAX runtime configuration and GPU validation."""

from __future__ import annotations

import typing
from dataclasses import dataclass
from pathlib import Path

from g import _core
from g.jax_runtime import diagnostics, models, resolution


@dataclass(frozen=True)
class NvidiaDriverProbePaths:
    """Linux NVIDIA driver paths used for native GPU visibility checks.

    Attributes:
        control_device_path: NVIDIA control device path.
        uvm_device_path: NVIDIA unified-memory device path.
        driver_directory_path: NVIDIA procfs driver directory path.

    """

    control_device_path: Path
    uvm_device_path: Path
    driver_directory_path: Path


def configure_before_backend_init(
    native_setup_session: _core.NativeJaxRuntimeSetupSession,
    diagnostic_sink: typing.Callable[[models.JaxRuntimeDiagnosticEvent], None] | None,
) -> models.JaxRuntimeSetupReport:
    """Configure JAX platform and runtime knobs before backend initialization.

    Args:
        native_setup_session: Native setup session to own setup decisions.
        diagnostic_sink: Optional structured diagnostic event sink.

    Returns:
        Setup report after validation.

    Raises:
        RuntimeError: If GPU execution was requested but validation fails.

    """
    setup_report = resolution.jax_runtime_setup_report_from_native_payload(native_setup_session.setup_payload())
    native_setup_session.create_cache_directory_if_configured()
    apply_jax_runtime_config_updates(native_setup_session)
    if not native_setup_session.should_validate_gpu:
        if diagnostic_sink is not None:
            for diagnostic_event in diagnostics.diagnostic_events_from_native_setup_session(native_setup_session):
                diagnostic_sink(diagnostic_event)
        return setup_report
    try:
        validated_payload = validate_gpu_if_configured_with_default_probe_paths(native_setup_session)
    except RuntimeError:
        if diagnostic_sink is not None:
            for diagnostic_event in diagnostics.diagnostic_events_from_native_setup_session(native_setup_session):
                diagnostic_sink(diagnostic_event)
        raise
    validated_report = resolution.jax_runtime_setup_report_from_native_payload(validated_payload)
    if diagnostic_sink is not None:
        for diagnostic_event in diagnostics.diagnostic_events_from_native_setup_session(native_setup_session):
            diagnostic_sink(diagnostic_event)
    return validated_report


def default_nvidia_driver_probe_paths() -> NvidiaDriverProbePaths:
    """Return native-owned default NVIDIA driver probe paths."""
    paths_payload = typing.cast(
        "typing.Mapping[str, str]",
        _core.default_nvidia_driver_probe_paths_payload(),
    )
    return NvidiaDriverProbePaths(
        control_device_path=Path(paths_payload["control_device_path"]),
        uvm_device_path=Path(paths_payload["uvm_device_path"]),
        driver_directory_path=Path(paths_payload["driver_directory_path"]),
    )


def nvidia_driver_is_visible() -> bool:
    """Return whether the process can see a Linux NVIDIA driver/device mount.

    Returns:
        Whether NVIDIA driver files are visible.

    """
    probe_paths = default_nvidia_driver_probe_paths()
    return _core.nvidia_driver_files_are_visible_value(
        control_device_path=str(probe_paths.control_device_path),
        uvm_device_path=str(probe_paths.uvm_device_path),
        driver_directory_path=str(probe_paths.driver_directory_path),
    )


def apply_jax_runtime_config_updates(native_setup_session: _core.NativeJaxRuntimeSetupSession) -> None:
    """Apply native-ordered JAX runtime config updates."""
    native_setup_session.apply_config_updates()


def validate_gpu_if_configured(native_setup_session: _core.NativeJaxRuntimeSetupSession) -> dict[str, object]:
    """Validate GPU setup using Python-adapted default probe paths."""
    probe_paths = default_nvidia_driver_probe_paths()
    return native_setup_session.validate_gpu_if_configured(
        str(probe_paths.control_device_path),
        str(probe_paths.uvm_device_path),
        str(probe_paths.driver_directory_path),
    )


def validate_gpu_if_configured_with_default_probe_paths(
    native_setup_session: _core.NativeJaxRuntimeSetupSession,
) -> dict[str, object]:
    """Validate GPU setup using native-owned default probe paths."""
    return native_setup_session.validate_gpu_if_configured_with_default_probe_paths()


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
    native_validation_session = _core.NativeRuntimeState().build_jax_runtime_setup_session_resolving_cache_directory(
        _core.build_jax_runtime_policy_payload(
            device="gpu",
            cache_directory="",
            matmul_precision=None,
            persistent_cache=False,
            persistent_cache_min_entry_size_bytes=0,
            persistent_cache_min_compile_time_seconds=0,
            xla_autotune_cache=False,
            transfer_guard=False,
        )
    )
    validated_payload = validate_gpu_if_configured_with_default_probe_paths(native_validation_session)
    validated_report = resolution.jax_runtime_setup_report_from_native_payload(validated_payload)
    return models.JaxGpuValidationReport(
        status=validated_report.gpu_validation_status,
        message="" if validated_report.gpu_validation_message is None else validated_report.gpu_validation_message,
    )


def jax_gpu_validation_report_from_native_payload(payload: object) -> models.JaxGpuValidationReport:
    """Adapt a native JAX GPU validation payload."""
    validation_payload = dict(typing.cast("typing.Mapping[str, object]", payload))
    return models.JaxGpuValidationReport(
        status=models.GpuValidationStatus(str(validation_payload["status"])),
        message=str(validation_payload["message"]),
    )
