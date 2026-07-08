"""Runner-local lifecycle helpers."""

from __future__ import annotations

import typing

from g import _core

if typing.TYPE_CHECKING:
    import signal
    import types as python_types


class GracefulShutdownRequested(Exception):  # noqa: N818
    """Raised after the first handled shutdown signal."""

    def __init__(self, shutdown_signal: _core.NativeShutdownSignal) -> None:
        """Initialize the graceful shutdown request."""
        self.shutdown_signal = shutdown_signal
        super().__init__(f"Graceful shutdown requested by {shutdown_signal.name}.")

    @property
    def signal_name(self) -> str:
        """Return the signal name that requested shutdown."""
        return self.shutdown_signal.name

    @property
    def exit_code(self) -> int:
        """Return the conventional process exit code for the signal."""
        return self.shutdown_signal.exit_code


class GracefulShutdownController:
    """Install process signal handlers that request one graceful drain."""

    def __init__(self, handled_signals: tuple[signal.Signals, ...] | None) -> None:
        """Initialize the controller."""
        self.native_controller = _core.NativeShutdownController(
            handled_signals,
        )

    @property
    def handlers_installed(self) -> bool:
        """Return whether native lifecycle state says handlers are installed."""
        return self.native_controller.handlers_installed

    @property
    def requested_signal(self) -> _core.NativeShutdownSignal | None:
        """Return the shutdown signal currently recorded by the native handle."""
        return self.native_controller.requested_signal()

    def __enter__(self) -> GracefulShutdownController:
        """Install signal handlers and return this controller."""
        self.native_controller.install_python_signal_handlers(self.handle_signal)
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception_value: BaseException | None,
        traceback: python_types.TracebackType | None,
    ) -> None:
        """Restore previous signal handlers."""
        del exception_type, exception_value, traceback
        self.native_controller.restore_python_signal_handlers_and_reset()

    def handle_signal(self, signal_number: int, frame: python_types.FrameType | None) -> None:
        """Request graceful shutdown on first signal and fast abort on the second."""
        del frame
        shutdown_signal = self.native_controller.request_shutdown_signal_or_raise_second_signal(signal_number)
        raise GracefulShutdownRequested(shutdown_signal)
