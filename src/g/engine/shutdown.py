"""Graceful shutdown handling for long-running engine commands."""

from __future__ import annotations

import signal
import typing
from dataclasses import dataclass

if typing.TYPE_CHECKING:
    import types as python_types


@dataclass(frozen=True)
class ShutdownSignal:
    """Signal metadata for an interrupted run.

    Attributes:
        number: Numeric signal value.
        name: POSIX signal name.
        exit_code: Conventional process exit code for the signal.

    """

    number: int
    name: str
    exit_code: int


class GracefulShutdownRequested(Exception):  # noqa: N818
    """Raised after the first handled shutdown signal."""

    def __init__(self, shutdown_signal: ShutdownSignal) -> None:
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

    def __init__(self, handled_signals: tuple[signal.Signals, ...] | None = None) -> None:
        """Initialize the controller."""
        self.handled_signals = handled_signals or (signal.SIGINT, signal.SIGTERM)
        self.previous_handlers: dict[signal.Signals, typing.Any] = {}
        self.requested_signal: ShutdownSignal | None = None
        self.handlers_installed = False

    def __enter__(self) -> GracefulShutdownController:
        """Install signal handlers and return this controller."""
        self.requested_signal = None
        self.previous_handlers = {}
        for handled_signal in self.handled_signals:
            self.previous_handlers[handled_signal] = signal.getsignal(handled_signal)
            signal.signal(handled_signal, self.handle_signal)
        self.handlers_installed = True
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception_value: BaseException | None,
        traceback: python_types.TracebackType | None,
    ) -> None:
        """Restore previous signal handlers."""
        del exception_type, exception_value, traceback
        self.restore_previous_handlers()
        self.requested_signal = None

    def handle_signal(self, signal_number: int, frame: python_types.FrameType | None) -> None:
        """Request graceful shutdown on first signal and fast abort on the second."""
        del frame
        shutdown_signal = build_shutdown_signal(signal_number)
        if self.requested_signal is None:
            self.requested_signal = shutdown_signal
            raise GracefulShutdownRequested(shutdown_signal)
        self.restore_previous_handlers()
        raise_second_signal_exception(shutdown_signal)

    def restore_previous_handlers(self) -> None:
        """Restore signal handlers captured when the controller was installed."""
        if not self.handlers_installed:
            return
        for handled_signal, previous_handler in self.previous_handlers.items():
            signal.signal(handled_signal, previous_handler)
        self.handlers_installed = False


def build_shutdown_signal(signal_number: int) -> ShutdownSignal:
    """Build shutdown metadata for a POSIX signal."""
    signal_value = signal.Signals(signal_number)
    return ShutdownSignal(
        number=signal_number,
        name=signal_value.name,
        exit_code=128 + signal_number,
    )


def raise_second_signal_exception(shutdown_signal: ShutdownSignal) -> typing.NoReturn:
    """Raise a hard-interrupt exception for a repeated shutdown signal."""
    if shutdown_signal.number == signal.SIGINT:
        raise KeyboardInterrupt
    raise SystemExit(shutdown_signal.exit_code)


def install_graceful_shutdown_handlers() -> GracefulShutdownController:
    """Create a controller that installs default graceful shutdown handlers."""
    return GracefulShutdownController()
