from __future__ import annotations

import signal
import unittest.mock

import pytest

from g.engine import shutdown


def test_build_shutdown_signal_uses_native_metadata_for_supported_linux_signals() -> None:
    for signal_name in ("SIGSTKFLT", "SIGPWR", "SIGRTMIN", "SIGRTMAX"):
        signal_member = getattr(signal, signal_name, None)
        if signal_member is None:
            continue

        signal_number = int(signal_member)
        assert shutdown.build_shutdown_signal(signal_number) == shutdown.ShutdownSignal(
            number=signal_number,
            name=signal_name,
            exit_code=128 + signal_number,
        )


def test_build_shutdown_signal_rejects_unknown_signal() -> None:
    with pytest.raises(ValueError, match="0 is not a valid Signals"):
        shutdown.build_shutdown_signal(0)


def test_shutdown_controller_records_first_signal_in_native_handle() -> None:
    controller = shutdown.GracefulShutdownController(handled_signals=(signal.SIGINT,))

    with pytest.raises(shutdown.GracefulShutdownRequested) as shutdown_request:
        controller.handle_signal(int(signal.SIGINT), None)

    assert shutdown_request.value.shutdown_signal == shutdown.ShutdownSignal(
        number=int(signal.SIGINT),
        name="SIGINT",
        exit_code=128 + int(signal.SIGINT),
    )
    assert controller.requested_signal == shutdown_request.value.shutdown_signal


def test_shutdown_controller_repeated_signal_restores_handlers_and_aborts() -> None:
    controller = shutdown.GracefulShutdownController(handled_signals=(signal.SIGINT,))
    previous_handler = object()
    controller.previous_handlers = {signal.SIGINT: previous_handler}
    controller.handlers_installed = True

    with pytest.raises(shutdown.GracefulShutdownRequested):
        controller.handle_signal(int(signal.SIGINT), None)
    with (
        unittest.mock.patch("g.engine.shutdown.signal.signal") as signal_mock,
        pytest.raises(KeyboardInterrupt),
    ):
        controller.handle_signal(int(signal.SIGINT), None)

    signal_mock.assert_called_once_with(signal.SIGINT, previous_handler)
    assert not controller.handlers_installed


def test_shutdown_controller_context_resets_native_requested_signal() -> None:
    controller = shutdown.GracefulShutdownController(handled_signals=(signal.SIGTERM,))
    previous_handler = object()

    with pytest.raises(shutdown.GracefulShutdownRequested):
        controller.handle_signal(int(signal.SIGTERM), None)
    assert controller.requested_signal is not None

    with (
        unittest.mock.patch("g.engine.shutdown.signal.getsignal", return_value=previous_handler),
        unittest.mock.patch("g.engine.shutdown.signal.signal") as signal_mock,
        controller as active_controller,
    ):
        assert active_controller.requested_signal is None
        assert active_controller.previous_handlers == {signal.SIGTERM: previous_handler}

    assert controller.requested_signal is None
    signal_mock.assert_any_call(signal.SIGTERM, controller.handle_signal)
    signal_mock.assert_any_call(signal.SIGTERM, previous_handler)
