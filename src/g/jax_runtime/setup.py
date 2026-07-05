"""Order-sensitive JAX runtime configuration and GPU validation."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from g import _core


def configure_before_backend_init(
    native_setup_session: _core.NativeJaxRuntimeSetupSession,
    diagnostic_sink: typing.Callable[[_core.NativeJaxRuntimeDiagnosticEvent], None] | None,
) -> _core.NativeJaxRuntimeSetupReport:
    """Configure JAX platform and runtime knobs before backend initialization.

    Args:
        native_setup_session: Native setup session to own setup decisions.
        diagnostic_sink: Optional structured diagnostic event sink.

    Returns:
        Setup report after validation.

    Raises:
        RuntimeError: If GPU execution was requested but validation fails.

    """
    setup_report = native_setup_session.setup_report()
    native_setup_session.create_cache_directory_if_configured()
    native_setup_session.apply_config_updates()
    if not native_setup_session.should_validate_gpu:
        emit_jax_runtime_diagnostics(native_setup_session, diagnostic_sink)
        return setup_report
    try:
        validated_report = native_setup_session.validate_gpu_if_configured_with_default_probe_paths()
    except RuntimeError:
        emit_jax_runtime_diagnostics(native_setup_session, diagnostic_sink)
        raise
    emit_jax_runtime_diagnostics(native_setup_session, diagnostic_sink)
    return validated_report


def emit_jax_runtime_diagnostics(
    native_setup_session: _core.NativeJaxRuntimeSetupSession,
    diagnostic_sink: typing.Callable[[_core.NativeJaxRuntimeDiagnosticEvent], None] | None,
) -> None:
    """Emit native JAX runtime setup diagnostics through an optional sink.

    Args:
        native_setup_session: Native setup session with ordered diagnostics.
        diagnostic_sink: Optional structured diagnostic event sink.

    """
    if diagnostic_sink is None:
        return
    for diagnostic_event in native_setup_session.diagnostic_events():
        diagnostic_sink(diagnostic_event)
