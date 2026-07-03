"""Order-sensitive JAX runtime configuration and GPU validation."""

from __future__ import annotations

import typing

from g import _core
from g.jax_runtime import diagnostics, models, resolution


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


def apply_jax_runtime_config_updates(native_setup_session: _core.NativeJaxRuntimeSetupSession) -> None:
    """Apply native-ordered JAX runtime config updates."""
    native_setup_session.apply_config_updates()


def validate_gpu_if_configured_with_default_probe_paths(
    native_setup_session: _core.NativeJaxRuntimeSetupSession,
) -> dict[str, object]:
    """Validate GPU setup using native-owned default probe paths."""
    return native_setup_session.validate_gpu_if_configured_with_default_probe_paths()


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
    native_validation_session = _build_gpu_validation_setup_session()
    validated_payload = validate_gpu_if_configured_with_default_probe_paths(native_validation_session)
    validated_report = resolution.jax_runtime_setup_report_from_native_payload(validated_payload)
    return models.JaxGpuValidationReport(
        status=validated_report.gpu_validation_status,
        message="" if validated_report.gpu_validation_message is None else validated_report.gpu_validation_message,
    )


def _build_gpu_validation_setup_session() -> _core.NativeJaxRuntimeSetupSession:
    """Build the native standalone GPU validation session."""
    return _core.NativeRuntimeState().build_jax_runtime_setup_session_resolving_cache_directory(
        resolution.build_native_jax_runtime_policy_payload(
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
