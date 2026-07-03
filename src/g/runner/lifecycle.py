"""Runner-owned lifecycle helpers."""

from __future__ import annotations

import typing
from dataclasses import dataclass

from g import _core

if typing.TYPE_CHECKING:
    import signal
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
        self.native_controller = _core.NativeShutdownController(
            handled_signals,
        )

    @property
    def handlers_installed(self) -> bool:
        """Return whether native lifecycle state says handlers are installed."""
        return self.native_controller.handlers_installed

    @property
    def requested_signal(self) -> ShutdownSignal | None:
        """Return the shutdown signal currently recorded by the native handle."""
        native_signal = self.native_controller.requested_signal()
        if native_signal is None:
            return None
        return shutdown_signal_from_native_signal(native_signal)

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
        shutdown_signal = shutdown_signal_from_native_signal(
            self.native_controller.request_shutdown_signal_or_raise_second_signal(signal_number)
        )
        raise GracefulShutdownRequested(shutdown_signal)

    def restore_previous_handlers(self) -> None:
        """Restore signal handlers captured when the controller was installed."""
        self.native_controller.restore_python_signal_handlers()


def shutdown_signal_from_native_signal(native_signal: _core.NativeShutdownSignal) -> ShutdownSignal:
    """Adapt native shutdown signal metadata to the public Python dataclass."""
    return ShutdownSignal(
        number=native_signal.number,
        name=native_signal.name,
        exit_code=native_signal.exit_code,
    )


def shutdown_signal_from_native_payload(payload: object) -> ShutdownSignal:
    """Adapt native shutdown signal metadata to the public Python dataclass."""
    signal_payload = native_mapping_payload(payload)
    return ShutdownSignal(
        number=native_int_payload(signal_payload["number"]),
        name=str(signal_payload["name"]),
        exit_code=native_int_payload(signal_payload["exit_code"]),
    )


def native_int_payload(payload: object) -> int:
    """Adapt a native integer-like payload to `int`."""
    return int(typing.cast("int | str", payload))


def native_mapping_payload(payload: object) -> typing.Mapping[str, object]:
    """Adapt a native mapping payload to a Python mapping."""
    return typing.cast("typing.Mapping[str, object]", payload)


def install_graceful_shutdown_handlers() -> GracefulShutdownController:
    """Create a controller that installs default graceful shutdown handlers."""
    return GracefulShutdownController(handled_signals=None)
