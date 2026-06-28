from __future__ import annotations

import signal
import unittest.mock

import pytest

from g import _core
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


def test_native_second_signal_exception_plan() -> None:
    sigint_plan = _core.plan_second_signal_exception(int(signal.SIGINT))
    sigterm_plan = _core.plan_second_signal_exception(int(signal.SIGTERM))

    assert isinstance(sigint_plan, _core.NativeSecondSignalExceptionPlan)
    assert sigint_plan.raise_keyboard_interrupt is True
    assert sigint_plan.exit_code == 128 + int(signal.SIGINT)
    assert sigterm_plan.raise_keyboard_interrupt is False
    assert sigterm_plan.exit_code == 128 + int(signal.SIGTERM)
    with pytest.raises(ValueError, match="0 is not a valid Signals"):
        _core.plan_second_signal_exception(0)


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


def test_second_signal_exception_uses_native_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSecondSignalExceptionPlan:
        def __init__(self, *, raise_keyboard_interrupt: bool, exit_code: int) -> None:
            self.raise_keyboard_interrupt = raise_keyboard_interrupt
            self.exit_code = exit_code

    def plan_second_signal_exception(signal_number: int) -> FakeSecondSignalExceptionPlan:
        assert signal_number == int(signal.SIGINT)
        return FakeSecondSignalExceptionPlan(raise_keyboard_interrupt=False, exit_code=199)

    monkeypatch.setattr(shutdown.g._core, "plan_second_signal_exception", plan_second_signal_exception)

    with pytest.raises(SystemExit) as system_exit:
        shutdown.raise_second_signal_exception(
            shutdown.ShutdownSignal(number=int(signal.SIGINT), name="SIGINT", exit_code=128 + int(signal.SIGINT))
        )

    assert system_exit.value.code == 199


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
