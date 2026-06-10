"""Tests for CLI and native frontend diagnostic bridging."""

from __future__ import annotations

import contextlib
import types as python_types
import typing
import unittest.mock
from pathlib import Path

from g import cli
from g.engine import run_events, shutdown

if typing.TYPE_CHECKING:
    import pytest


class FakeTelemetrySession:
    """Telemetry session test double for CLI ownership tests."""

    def __init__(self) -> None:
        self.paths = python_types.SimpleNamespace()
        self.logged_events: list[str] = []
        self.closed = False

    def log_event(self, event: str, level: str = "info", **fields: object) -> None:
        """Record a telemetry event name."""
        del level, fields
        self.logged_events.append(event)

    def writer_counters(self) -> dict[str, object]:
        """Return empty writer counters."""
        return {}

    def close(self) -> None:
        """Record session closure."""
        self.closed = True


def test_run_args_bridges_native_stdio_events(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Ensure native CLI stdout/stderr are mirrored through structured logs and output."""
    outcome = python_types.SimpleNamespace(stdout="native-stdout\n", stderr="native-stderr\n", exit_code=7, config=None)
    with (
        unittest.mock.patch("g.cli.g._core.dispatch_cli", return_value=outcome),
        unittest.mock.patch("g.cli.g._core.emit_diagnostic_event") as diagnostic_event_mock,
    ):
        exit_code = cli.run_args(["--help"])

    output = capsys.readouterr()
    assert exit_code == 7
    assert output.out == "native-stdout\n"
    assert output.err == "native-stderr\n"
    event_names = {call.args[1] for call in diagnostic_event_mock.call_args_list}
    assert "native_cli_stdout" in event_names
    assert "native_cli_stderr" in event_names


def test_run_args_bridges_completion_events(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Ensure successful completion lines are logged and printed."""
    run_config = python_types.SimpleNamespace(g_diagnostics=python_types.SimpleNamespace())
    telemetry_session = FakeTelemetrySession()
    run_artifacts = run_events.RunArtifacts(
        output_run_directory=Path("output.run"),
        final_dataset=Path("output.run/parts"),
        final_parquet=None,
        final_regenie=None,
        effective_config=None,
        phenotype_artifacts=(),
        phenotype_name=None,
        association_mode=None,
        phenotype_count=None,
        run_id=None,
    )
    outcome = python_types.SimpleNamespace(stdout="", stderr="", exit_code=0, config=run_config)
    with (
        unittest.mock.patch("g.cli.g._core.dispatch_cli", return_value=outcome),
        unittest.mock.patch("g.cli.telemetry.build_telemetry_session", return_value=telemetry_session),
        unittest.mock.patch("g.cli.runner.initialize_logging"),
        unittest.mock.patch("g.cli.shutdown.install_graceful_shutdown_handlers", return_value=contextlib.nullcontext()),
        unittest.mock.patch("g.cli.runner.regenie", return_value=run_artifacts) as regenie_mock,
        unittest.mock.patch("g.cli.g._core.emit_diagnostic_event") as diagnostic_event_mock,
    ):
        exit_code = cli.run_args(["regenie"])

    output = capsys.readouterr()
    assert exit_code == 0
    assert "Success. Chunked run saved to output.run" in output.out
    regenie_mock.assert_called_once_with(
        run_config,
        run_telemetry_session=telemetry_session,
        close_telemetry_session_on_exit=False,
    )
    assert telemetry_session.closed is True
    event_names = {call.args[1] for call in diagnostic_event_mock.call_args_list}
    assert "native_cli_completed_line" in event_names


def test_run_args_bridges_interruption_events(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Ensure graceful interruption is logged and printed as structured completion lines."""
    run_config = python_types.SimpleNamespace(g_diagnostics=python_types.SimpleNamespace())
    telemetry_session = FakeTelemetrySession()
    shutdown_request = shutdown.GracefulShutdownRequested(
        shutdown.ShutdownSignal(number=2, name="SIGINT", exit_code=130)
    )
    outcome = python_types.SimpleNamespace(stdout="", stderr="", exit_code=0, config=run_config)
    with (
        unittest.mock.patch("g.cli.g._core.dispatch_cli", return_value=outcome),
        unittest.mock.patch("g.cli.telemetry.build_telemetry_session", return_value=telemetry_session),
        unittest.mock.patch("g.cli.runner.initialize_logging"),
        unittest.mock.patch("g.cli.shutdown.install_graceful_shutdown_handlers", return_value=contextlib.nullcontext()),
        unittest.mock.patch("g.cli.runner.regenie", side_effect=shutdown_request) as regenie_mock,
        unittest.mock.patch("g.cli.g._core.emit_diagnostic_event") as diagnostic_event_mock,
    ):
        exit_code = cli.run_args(["regenie"])

    output = capsys.readouterr()
    assert exit_code == 130
    assert "Interrupted by SIGINT. Flushed queued chunks and saved committed output for --resume." in output.err
    regenie_mock.assert_called_once_with(
        run_config,
        run_telemetry_session=telemetry_session,
        close_telemetry_session_on_exit=False,
    )
    assert telemetry_session.closed is True
    event_names = {call.args[1] for call in diagnostic_event_mock.call_args_list}
    assert "native_cli_interrupted_line" in event_names
