"""Tests for CLI and native frontend diagnostic bridging."""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import textwrap
import types as python_types
import unittest.mock
from pathlib import Path

import pytest

from g import cli


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


def test_run_args_configless_paths_print_without_runtime_imports() -> None:
    """Ensure help and parse errors only parse and print native output."""
    script = textwrap.dedent(
        """
        import contextlib
        import io
        import sys
        import types
        import unittest.mock

        import g.cli

        forbidden_modules = (
            "g.runner",
            "g.engine.run_events",
            "g.engine.shutdown",
            "g.engine.telemetry",
            "g.jax_setup",
            "jax",
            "jax.numpy",
        )

        cases = (
            (["--help"], "native-help\\n", "", 0),
            (["regenie", "--help"], "native-regenie-help\\n", "", 0),
            (["regenie", "--bad-option"], "", "native-error\\n", 2),
        )
        for arguments, expected_stdout, expected_stderr, expected_exit_code in cases:
            outcome = types.SimpleNamespace(
                stdout=expected_stdout,
                stderr=expected_stderr,
                exit_code=expected_exit_code,
                config=None,
            )
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            with (
                unittest.mock.patch("g.cli.g._core.dispatch_cli", return_value=outcome),
                unittest.mock.patch("g.cli.g._core.emit_diagnostic_event") as diagnostic_event_mock,
                contextlib.redirect_stdout(stdout_buffer),
                contextlib.redirect_stderr(stderr_buffer),
            ):
                actual_exit_code = g.cli.run_args(arguments)
            if actual_exit_code != expected_exit_code:
                raise AssertionError((actual_exit_code, expected_exit_code))
            if stdout_buffer.getvalue() != expected_stdout:
                raise AssertionError((stdout_buffer.getvalue(), expected_stdout))
            if stderr_buffer.getvalue() != expected_stderr:
                raise AssertionError((stderr_buffer.getvalue(), expected_stderr))
            diagnostic_event_mock.assert_not_called()

        imported_modules = [module_name for module_name in forbidden_modules if module_name in sys.modules]
        if imported_modules:
            raise AssertionError(f"unexpected runtime imports: {imported_modules}")
        """
    )

    completed_process = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed_process.returncode == 0, completed_process.stderr


def test_log_native_cli_output_bounds_payloads() -> None:
    """Ensure native output diagnostics carry bounded previews, not full payloads."""
    long_stdout = "x" * (cli.NATIVE_CLI_OUTPUT_LOG_LIMIT + 3)
    outcome = python_types.SimpleNamespace(stdout=long_stdout, stderr="", exit_code=0, config=None)

    with unittest.mock.patch("g.cli.g._core.emit_diagnostic_event") as diagnostic_event_mock:
        cli.log_native_cli_output(outcome)

    diagnostic_event_mock.assert_called_once()
    payload = json.loads(diagnostic_event_mock.call_args.args[3])
    assert payload["stdout_character_count"] == len(long_stdout)
    assert payload["stdout_preview"] == long_stdout[: cli.NATIVE_CLI_OUTPUT_LOG_LIMIT]
    assert payload["stdout_preview"] != long_stdout
    assert payload["stdout_truncated"] is True
    assert payload["stdout_omitted_character_count"] == 3
    assert "outcome_stdout" not in payload


def test_run_args_bridges_completion_events(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Ensure successful completion lines are logged and printed."""
    from g import runner as runner_module
    from g.engine import run_events, shutdown
    from g.engine import telemetry as telemetry_module

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
        unittest.mock.patch.object(telemetry_module, "build_telemetry_session", return_value=telemetry_session),
        unittest.mock.patch.object(runner_module, "initialize_logging") as initialize_logging_mock,
        unittest.mock.patch.object(
            shutdown, "install_graceful_shutdown_handlers", return_value=contextlib.nullcontext()
        ),
        unittest.mock.patch.object(runner_module, "regenie", return_value=run_artifacts) as regenie_mock,
        unittest.mock.patch("g.cli.g._core.emit_diagnostic_event") as diagnostic_event_mock,
    ):
        exit_code = cli.run_args(["regenie"])

    output = capsys.readouterr()
    assert exit_code == 0
    assert "Success. Chunked run saved to output.run" in output.out
    initialize_logging_mock.assert_called_once_with(run_config.g_diagnostics, telemetry_session.paths)
    regenie_mock.assert_called_once_with(
        run_config,
        run_telemetry_session=telemetry_session,
        close_telemetry_session_on_exit=False,
        initialize_logging_on_entry=False,
    )
    assert telemetry_session.closed is True
    event_names = {call.args[1] for call in diagnostic_event_mock.call_args_list}
    assert "native_cli_completed_line" in event_names


def test_run_args_bridges_interruption_events(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Ensure graceful interruption is logged and printed as structured completion lines."""
    from g import runner as runner_module
    from g.engine import shutdown
    from g.engine import telemetry as telemetry_module

    run_config = python_types.SimpleNamespace(g_diagnostics=python_types.SimpleNamespace())
    telemetry_session = FakeTelemetrySession()
    shutdown_request = shutdown.GracefulShutdownRequested(
        shutdown.ShutdownSignal(number=2, name="SIGINT", exit_code=130)
    )
    outcome = python_types.SimpleNamespace(stdout="", stderr="", exit_code=0, config=run_config)
    with (
        unittest.mock.patch("g.cli.g._core.dispatch_cli", return_value=outcome),
        unittest.mock.patch.object(telemetry_module, "build_telemetry_session", return_value=telemetry_session),
        unittest.mock.patch.object(runner_module, "initialize_logging") as initialize_logging_mock,
        unittest.mock.patch.object(
            shutdown, "install_graceful_shutdown_handlers", return_value=contextlib.nullcontext()
        ),
        unittest.mock.patch.object(runner_module, "regenie", side_effect=shutdown_request) as regenie_mock,
        unittest.mock.patch("g.cli.g._core.emit_diagnostic_event") as diagnostic_event_mock,
    ):
        exit_code = cli.run_args(["regenie"])

    output = capsys.readouterr()
    assert exit_code == 130
    assert "Interrupted by SIGINT. Flushed queued chunks and saved committed output for --resume." in output.err
    initialize_logging_mock.assert_called_once_with(run_config.g_diagnostics, telemetry_session.paths)
    regenie_mock.assert_called_once_with(
        run_config,
        run_telemetry_session=telemetry_session,
        close_telemetry_session_on_exit=False,
        initialize_logging_on_entry=False,
    )
    assert telemetry_session.closed is True
    event_names = {call.args[1] for call in diagnostic_event_mock.call_args_list}
    assert "native_cli_interrupted_line" in event_names


def test_run_args_closes_telemetry_when_logging_initialization_fails() -> None:
    """Ensure telemetry cleanup starts immediately after session creation."""
    from g import runner as runner_module
    from g.engine import telemetry as telemetry_module

    run_config = python_types.SimpleNamespace(g_diagnostics=python_types.SimpleNamespace())
    telemetry_session = FakeTelemetrySession()
    outcome = python_types.SimpleNamespace(stdout="", stderr="", exit_code=0, config=run_config)

    with (
        unittest.mock.patch("g.cli.g._core.dispatch_cli", return_value=outcome),
        unittest.mock.patch.object(telemetry_module, "build_telemetry_session", return_value=telemetry_session),
        unittest.mock.patch.object(runner_module, "initialize_logging", side_effect=RuntimeError("logging failed")),
        unittest.mock.patch.object(runner_module, "regenie") as regenie_mock,
        pytest.raises(RuntimeError, match="logging failed"),
    ):
        cli.run_args(["regenie"])

    regenie_mock.assert_not_called()
    assert telemetry_session.logged_events == ["telemetry_session_closed"]
    assert telemetry_session.closed is True
