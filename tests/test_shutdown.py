from __future__ import annotations

import signal
import typing
import unittest.mock

import pytest

from g import _core
from g.runner import lifecycle as shutdown


def test_shutdown_controller_uses_native_metadata_for_supported_linux_signals() -> None:
    for signal_name in ("SIGSTKFLT", "SIGPWR", "SIGRTMIN", "SIGRTMAX"):
        signal_member = getattr(signal, signal_name, None)
        if signal_member is None:
            continue

        signal_number = int(signal_member)
        native_controller = _core.NativeShutdownController([signal_number])
        native_signal = native_controller.request_shutdown_signal_or_raise_second_signal(signal_number)

        assert shutdown.shutdown_signal_from_native_signal(native_signal) == shutdown.ShutdownSignal(
            number=signal_number,
            name=signal_name,
            exit_code=128 + signal_number,
        )


def test_native_shutdown_controller_returns_typed_signal() -> None:
    signal_number = int(signal.SIGTERM)
    native_controller = _core.NativeShutdownController([signal_number])

    native_signal = native_controller.request_shutdown_signal_or_raise_second_signal(signal_number)
    requested_signal = native_controller.requested_signal()

    assert isinstance(native_signal, _core.NativeShutdownSignal)
    assert native_signal.number == signal_number
    assert native_signal.name == "SIGTERM"
    assert native_signal.exit_code == 128 + signal_number
    assert requested_signal is not None
    assert requested_signal.name == native_signal.name


def test_shutdown_controller_rejects_unknown_signal() -> None:
    with pytest.raises(ValueError, match="0 is not a valid Signals"):
        _core.NativeShutdownController([0])


def test_detached_shutdown_helpers_are_not_exported() -> None:
    assert not hasattr(_core, "NativeSecondSignalExceptionPlan")
    assert not hasattr(_core, "build_shutdown_signal_payload")
    assert not hasattr(_core, "default_shutdown_signal_numbers")
    assert not hasattr(_core, "plan_second_signal_exception")
    assert not hasattr(_core, "raise_second_signal_exception")
    assert not hasattr(shutdown, "build_shutdown_signal")
    assert not hasattr(shutdown, "raise_second_signal_exception")


def test_native_shutdown_controller_repeated_sigterm_raises_system_exit() -> None:
    native_controller = _core.NativeShutdownController([int(signal.SIGTERM)])
    previous_handler = object()
    installed_handler = object()

    with (
        unittest.mock.patch("signal.getsignal", return_value=previous_handler),
        unittest.mock.patch("signal.signal") as signal_mock,
    ):
        native_controller.install_python_signal_handlers(installed_handler)
        first_signal = native_controller.request_shutdown_signal_or_raise_second_signal(int(signal.SIGTERM))
        with pytest.raises(SystemExit) as system_exit:
            native_controller.request_shutdown_signal_or_raise_second_signal(int(signal.SIGTERM))

    assert first_signal.name == "SIGTERM"
    assert system_exit.value.code == 128 + int(signal.SIGTERM)
    assert native_controller.handlers_installed is False
    signal_mock.assert_any_call(signal.SIGTERM, installed_handler)
    signal_mock.assert_any_call(signal.SIGTERM, previous_handler)


def test_native_shutdown_controller_owns_handler_lifecycle() -> None:
    native_controller = _core.NativeShutdownController([int(signal.SIGINT), int(signal.SIGTERM)])
    default_controller = _core.NativeShutdownController()

    install_plan = dict(native_controller.handler_install_plan_payload())
    handled_signal_payloads = tuple(typing.cast("typing.Iterable[object]", install_plan["handled_signals"]))
    default_install_plan = dict(default_controller.handler_install_plan_payload())
    default_signal_payloads = tuple(typing.cast("typing.Iterable[object]", default_install_plan["handled_signals"]))

    assert native_controller.handlers_installed is False
    assert [typing.cast("typing.Mapping[str, object]", payload)["name"] for payload in handled_signal_payloads] == [
        "SIGINT",
        "SIGTERM",
    ]
    assert [typing.cast("typing.Mapping[str, object]", payload)["name"] for payload in default_signal_payloads] == [
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
        unittest.mock.patch("signal.getsignal", return_value=previous_handler) as get_signal_mock,
        unittest.mock.patch("signal.signal") as signal_mock,
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
        unittest.mock.patch("signal.getsignal", return_value=previous_handler),
        unittest.mock.patch("signal.signal") as signal_mock,
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
        unittest.mock.patch("signal.getsignal", return_value=previous_handler),
        unittest.mock.patch("signal.signal") as signal_mock,
    ):
        native_controller.install_python_signal_handlers(installed_handler)
        first_signal_payload = dict(
            native_controller.request_shutdown_signal_or_raise_second_signal_payload(int(signal.SIGINT))
        )
        with pytest.raises(KeyboardInterrupt):
            native_controller.request_shutdown_signal_or_raise_second_signal_payload(int(signal.SIGINT))

    assert first_signal_payload["name"] == "SIGINT"
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
        unittest.mock.patch("signal.getsignal", return_value=previous_handler),
        unittest.mock.patch("signal.signal") as signal_mock,
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


def test_shutdown_controller_context_resets_native_requested_signal() -> None:
    controller = shutdown.GracefulShutdownController(handled_signals=(signal.SIGTERM,))
    previous_handler = object()

    with pytest.raises(shutdown.GracefulShutdownRequested):
        controller.handle_signal(int(signal.SIGTERM), None)
    assert controller.requested_signal is not None

    with (
        unittest.mock.patch("signal.getsignal", return_value=previous_handler),
        unittest.mock.patch("signal.signal") as signal_mock,
        controller as active_controller,
    ):
        assert active_controller.requested_signal is None
        assert active_controller.handlers_installed is True

    assert controller.requested_signal is None
    signal_mock.assert_any_call(signal.SIGTERM, controller.handle_signal)
    signal_mock.assert_any_call(signal.SIGTERM, previous_handler)


def test_shutdown_controller_passes_optional_signals_to_native_handle() -> None:
    default_controller = shutdown.GracefulShutdownController(handled_signals=None)
    explicit_controller = shutdown.GracefulShutdownController(handled_signals=(signal.SIGTERM,))

    default_install_plan = dict(default_controller.native_controller.handler_install_plan_payload())
    explicit_install_plan = dict(explicit_controller.native_controller.handler_install_plan_payload())
    default_signal_payloads = tuple(typing.cast("typing.Iterable[object]", default_install_plan["handled_signals"]))
    explicit_signal_payloads = tuple(typing.cast("typing.Iterable[object]", explicit_install_plan["handled_signals"]))

    assert [typing.cast("typing.Mapping[str, object]", payload)["name"] for payload in default_signal_payloads] == [
        "SIGINT",
        "SIGTERM",
    ]
    assert [typing.cast("typing.Mapping[str, object]", payload)["name"] for payload in explicit_signal_payloads] == [
        "SIGTERM"
    ]
