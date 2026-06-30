"""Graceful shutdown handling for long-running engine commands."""

from __future__ import annotations

import signal
import typing
from dataclasses import dataclass

import g._core

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

    def __init__(self, handled_signals: tuple[signal.Signals, ...] | None) -> None:
        """Initialize the controller."""
        resolved_handled_signals = handled_signals or (signal.SIGINT, signal.SIGTERM)
        self.native_controller = g._core.NativeShutdownController(
            [int(handled_signal) for handled_signal in resolved_handled_signals]
        )

    @property
    def handlers_installed(self) -> bool:
        """Return whether native lifecycle state says handlers are installed."""
        return self.native_controller.handlers_installed

    @property
    def requested_signal(self) -> ShutdownSignal | None:
        """Return the shutdown signal currently recorded by the native handle."""
        signal_payload = self.native_controller.requested_signal_payload()
        if signal_payload is None:
            return None
        return shutdown_signal_from_native_payload(signal_payload)

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
        shutdown_signal = shutdown_signal_from_native_payload(
            self.native_controller.request_shutdown_signal_or_raise_second_signal_payload(signal_number)
        )
        raise GracefulShutdownRequested(shutdown_signal)

    def restore_previous_handlers(self) -> None:
        """Restore signal handlers captured when the controller was installed."""
        self.native_controller.restore_python_signal_handlers()


def build_shutdown_signal(signal_number: int) -> ShutdownSignal:
    """Build shutdown metadata for a POSIX signal."""
    return shutdown_signal_from_native_payload(g._core.build_shutdown_signal_payload(signal_number))


def shutdown_signal_from_native_payload(payload: object) -> ShutdownSignal:
    """Adapt native shutdown signal metadata to the public Python dataclass."""
    signal_payload = native_mapping_payload(payload)
    return ShutdownSignal(
        number=int(signal_payload["number"]),
        name=str(signal_payload["name"]),
        exit_code=int(signal_payload["exit_code"]),
    )


def native_mapping_payload(payload: object) -> typing.Mapping[str, typing.Any]:
    """Adapt a native mapping payload to a Python mapping."""
    return typing.cast("typing.Mapping[str, typing.Any]", payload)


def raise_second_signal_exception(shutdown_signal: ShutdownSignal) -> typing.NoReturn:
    """Raise a hard-interrupt exception for a repeated shutdown signal."""
    g._core.raise_second_signal_exception(shutdown_signal.number)


def install_graceful_shutdown_handlers() -> GracefulShutdownController:
    """Create a controller that installs default graceful shutdown handlers."""
    return GracefulShutdownController(handled_signals=None)
