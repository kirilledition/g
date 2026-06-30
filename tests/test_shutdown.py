from __future__ import annotations

import signal
import typing
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


def test_native_second_signal_exception_raiser() -> None:
    with pytest.raises(KeyboardInterrupt):
        _core.raise_second_signal_exception(int(signal.SIGINT))
    with pytest.raises(SystemExit) as system_exit:
        _core.raise_second_signal_exception(int(signal.SIGTERM))
    assert system_exit.value.code == 128 + int(signal.SIGTERM)
    with pytest.raises(ValueError, match="0 is not a valid Signals"):
        _core.raise_second_signal_exception(0)


def test_native_shutdown_controller_owns_handler_lifecycle() -> None:
    native_controller = _core.NativeShutdownController([int(signal.SIGINT), int(signal.SIGTERM)])

    install_plan = dict(native_controller.handler_install_plan_payload())
    handled_signal_payloads = tuple(typing.cast("typing.Iterable[object]", install_plan["handled_signals"]))

    assert native_controller.handlers_installed is False
    assert [typing.cast("typing.Mapping[str, object]", payload)["name"] for payload in handled_signal_payloads] == [
        "SIGINT",
        "SIGTERM",
    ]
    native_controller.mark_handlers_installed()
    assert native_controller.handlers_installed is True
    restore_plan = dict(native_controller.handler_restore_plan_payload())
    assert restore_plan["should_restore"] is True
    restore_signal_payloads = tuple(typing.cast("typing.Iterable[object]", restore_plan["handled_signals"]))
    assert [typing.cast("typing.Mapping[str, object]", payload)["name"] for payload in restore_signal_payloads] == [
        "SIGINT",
        "SIGTERM",
    ]
    native_controller.mark_handlers_restored()
    assert native_controller.handlers_installed is False
    assert dict(native_controller.handler_restore_plan_payload())["should_restore"] is False


def test_native_shutdown_controller_installs_and_restores_python_handlers() -> None:
    native_controller = _core.NativeShutdownController([int(signal.SIGINT)])
    previous_handler = object()
    installed_handler = object()

    with (
        unittest.mock.patch("g.engine.shutdown.signal.getsignal", return_value=previous_handler) as get_signal_mock,
        unittest.mock.patch("g.engine.shutdown.signal.signal") as signal_mock,
    ):
        native_controller.install_python_signal_handlers(installed_handler)
        assert native_controller.handlers_installed is True
        restored_handlers = native_controller.restore_python_signal_handlers()

    assert restored_handlers is True
    get_signal_mock.assert_called_once_with(signal.SIGINT)
    signal_mock.assert_any_call(signal.SIGINT, installed_handler)
    signal_mock.assert_any_call(signal.SIGINT, previous_handler)
    assert native_controller.handlers_installed is False


def test_native_shutdown_controller_restores_and_resets_handler_session() -> None:
    native_controller = _core.NativeShutdownController([int(signal.SIGINT)])
    previous_handler = object()
    installed_handler = object()

    with (
        unittest.mock.patch("g.engine.shutdown.signal.getsignal", return_value=previous_handler),
        unittest.mock.patch("g.engine.shutdown.signal.signal") as signal_mock,
    ):
        native_controller.install_python_signal_handlers(installed_handler)
        native_controller.request_shutdown_payload(int(signal.SIGINT))
        assert native_controller.requested_signal_payload() is not None
        restored_handlers = native_controller.restore_python_signal_handlers_and_reset()

    assert restored_handlers is True
    assert native_controller.requested_signal_payload() is None
    assert native_controller.handlers_installed is False
    signal_mock.assert_any_call(signal.SIGINT, installed_handler)
    signal_mock.assert_any_call(signal.SIGINT, previous_handler)


def test_native_shutdown_controller_aborts_repeated_signal() -> None:
    native_controller = _core.NativeShutdownController([int(signal.SIGINT)])
    previous_handler = object()
    installed_handler = object()

    with (
        unittest.mock.patch("g.engine.shutdown.signal.getsignal", return_value=previous_handler),
        unittest.mock.patch("g.engine.shutdown.signal.signal") as signal_mock,
    ):
        native_controller.install_python_signal_handlers(installed_handler)
        first_decision = dict(native_controller.request_shutdown_or_raise_second_signal_payload(int(signal.SIGINT)))
        with pytest.raises(KeyboardInterrupt):
            native_controller.request_shutdown_or_raise_second_signal_payload(int(signal.SIGINT))

    assert first_decision["action"] == "graceful"
    assert typing.cast("typing.Mapping[str, object]", first_decision["signal"])["name"] == "SIGINT"
    assert native_controller.handlers_installed is False
    signal_mock.assert_any_call(signal.SIGINT, installed_handler)
    signal_mock.assert_any_call(signal.SIGINT, previous_handler)


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

    with (
        unittest.mock.patch("g.engine.shutdown.signal.getsignal", return_value=previous_handler),
        unittest.mock.patch("g.engine.shutdown.signal.signal") as signal_mock,
        controller,
    ):
        with pytest.raises(shutdown.GracefulShutdownRequested):
            controller.handle_signal(int(signal.SIGINT), None)
        with pytest.raises(KeyboardInterrupt):
            controller.handle_signal(int(signal.SIGINT), None)
        assert not controller.handlers_installed

    signal_mock.assert_any_call(signal.SIGINT, controller.handle_signal)
    signal_mock.assert_any_call(signal.SIGINT, previous_handler)
    assert not controller.handlers_installed


def test_second_signal_exception_uses_native_raiser(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_second_signal_exception(signal_number: int) -> typing.NoReturn:
        assert signal_number == int(signal.SIGINT)
        raise SystemExit(199)

    monkeypatch.setattr(shutdown.g._core, "raise_second_signal_exception", raise_second_signal_exception)

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
        assert active_controller.handlers_installed is True

    assert controller.requested_signal is None
    signal_mock.assert_any_call(signal.SIGTERM, controller.handle_signal)
    signal_mock.assert_any_call(signal.SIGTERM, previous_handler)
