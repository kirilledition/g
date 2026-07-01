"""Tests for CLI and native frontend diagnostic bridging."""

from __future__ import annotations

import contextlib
import subprocess
import sys
import textwrap
import types as python_types
import typing
import unittest.mock
from pathlib import Path

from g import cli

if typing.TYPE_CHECKING:
    import pytest

    import g._core


class FakeTelemetrySession:
    """Telemetry session test double for CLI ownership tests."""

    def __init__(self, close_error: Exception | None = None) -> None:
        self.paths = python_types.SimpleNamespace()
        self.logged_events: list[str] = []
        self.logged_payloads: list[dict[str, object]] = []
        self.closed = False
        self.close_error = close_error

    def log_event(self, event: str, level: str = "info", **fields: object) -> None:
        """Record a telemetry event name."""
        self.logged_events.append(event)
        self.logged_payloads.append({"event": event, "level": level, **fields})

    def log_run_failed(self, event: typing.Any) -> None:
        """Record a native run-failed telemetry event."""
        self.logged_events.append("run_failed")
        self.logged_payloads.append(
            {
                "event": "run_failed",
                "level": "error",
                "failure_kind": "exception",
                "error_type": event.error_type,
                "error_message": event.error_message,
            }
        )

    def writer_counters(self) -> dict[str, object]:
        """Return empty writer counters."""
        return {}

    def close(self) -> None:
        """Record session closure."""
        if self.close_error is not None:
            raise self.close_error
        self.closed = True

    def close_with_event(self) -> None:
        """Record native-style session closure."""
        self.log_event("telemetry_session_closed", level="debug", writer_counters=self.writer_counters())
        self.close()


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
            "g.jax_runtime.setup",
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
                unittest.mock.patch("g.cli.g._core.record_native_cli_stdout_diagnostic_event") as stdout_record_mock,
                unittest.mock.patch("g.cli.g._core.record_native_cli_stderr_diagnostic_event") as stderr_record_mock,
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
            stdout_record_mock.assert_not_called()
            stderr_record_mock.assert_not_called()

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


def test_log_native_cli_output_uses_native_recorders() -> None:
    """Ensure native output diagnostics are recorded by native adapters."""
    long_stdout = "x" * (cli.NATIVE_CLI_OUTPUT_LOG_LIMIT + 3)
    outcome = typing.cast(
        "g._core.CliOutcome",
        python_types.SimpleNamespace(stdout=long_stdout, stderr="", exit_code=0, config=None),
    )

    with (
        unittest.mock.patch("g.cli.g._core.record_native_cli_stdout_diagnostic_event") as stdout_record_mock,
        unittest.mock.patch("g.cli.g._core.record_native_cli_stderr_diagnostic_event") as stderr_record_mock,
    ):
        cli.log_native_cli_output(outcome, max_payload_chars=cli.NATIVE_CLI_OUTPUT_LOG_LIMIT)

    stdout_record_mock.assert_called_once_with(
        output_text=long_stdout,
        max_payload_chars=cli.NATIVE_CLI_OUTPUT_LOG_LIMIT,
    )
    stderr_record_mock.assert_not_called()


def test_run_args_bridges_completion_events(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Ensure successful completion lines are logged and printed."""
    from g.engine import run_events, shutdown
    from g.engine import telemetry as telemetry_module
    from g.runner import execution as runner_execution
    from g.runner import runtime as runner_runtime

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
    runtime_policy = python_types.SimpleNamespace()
    with (
        unittest.mock.patch("g.cli.g._core.dispatch_cli", return_value=outcome),
        unittest.mock.patch.object(telemetry_module, "build_telemetry_session", return_value=telemetry_session),
        unittest.mock.patch.object(
            runner_runtime, "build_runtime_policy", return_value=runtime_policy
        ) as build_runtime_policy_mock,
        unittest.mock.patch.object(runner_runtime, "require_compatible_runtime_policy") as runtime_preflight_mock,
        unittest.mock.patch.object(runner_runtime, "initialize_logging") as initialize_logging_mock,
        unittest.mock.patch.object(
            shutdown, "install_graceful_shutdown_handlers", return_value=contextlib.nullcontext()
        ),
        unittest.mock.patch.object(runner_execution, "regenie", return_value=run_artifacts) as regenie_mock,
        unittest.mock.patch("g.cli.g._core.record_native_cli_completed_line_diagnostic_event") as completed_record_mock,
    ):
        exit_code = cli.run_args(["regenie"])

    output = capsys.readouterr()
    assert exit_code == 0
    assert "Success. Chunked run saved to output.run" in output.out
    build_runtime_policy_mock.assert_called_once_with(run_config, telemetry_session.paths)
    runtime_preflight_mock.assert_called_once_with(runtime_policy)
    initialize_logging_mock.assert_called_once_with(run_config.g_diagnostics, telemetry_session.paths)
    regenie_mock.assert_called_once_with(
        run_config,
        run_telemetry_session=telemetry_session,
        close_telemetry_session_on_exit=False,
        initialize_logging_on_entry=False,
    )
    assert telemetry_session.closed is True
    completed_lines = [call.kwargs["line"] for call in completed_record_mock.call_args_list]
    assert "Success. Chunked run saved to output.run" in completed_lines


def test_run_args_bridges_interruption_events(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Ensure graceful interruption is logged and printed as structured completion lines."""
    from g.engine import shutdown
    from g.engine import telemetry as telemetry_module
    from g.runner import execution as runner_execution
    from g.runner import runtime as runner_runtime

    run_config = python_types.SimpleNamespace(g_diagnostics=python_types.SimpleNamespace())
    telemetry_session = FakeTelemetrySession()
    shutdown_request = shutdown.GracefulShutdownRequested(
        shutdown.ShutdownSignal(number=2, name="SIGINT", exit_code=130)
    )
    outcome = python_types.SimpleNamespace(stdout="", stderr="", exit_code=0, config=run_config)
    runtime_policy = python_types.SimpleNamespace()
    with (
        unittest.mock.patch("g.cli.g._core.dispatch_cli", return_value=outcome),
        unittest.mock.patch.object(telemetry_module, "build_telemetry_session", return_value=telemetry_session),
        unittest.mock.patch.object(
            runner_runtime, "build_runtime_policy", return_value=runtime_policy
        ) as build_runtime_policy_mock,
        unittest.mock.patch.object(runner_runtime, "require_compatible_runtime_policy") as runtime_preflight_mock,
        unittest.mock.patch.object(runner_runtime, "initialize_logging") as initialize_logging_mock,
        unittest.mock.patch.object(
            shutdown, "install_graceful_shutdown_handlers", return_value=contextlib.nullcontext()
        ),
        unittest.mock.patch.object(runner_execution, "regenie", side_effect=shutdown_request) as regenie_mock,
        unittest.mock.patch(
            "g.cli.g._core.record_native_cli_interrupted_line_diagnostic_event"
        ) as interrupted_record_mock,
    ):
        exit_code = cli.run_args(["regenie"])

    output = capsys.readouterr()
    assert exit_code == 130
    assert "Interrupted by SIGINT. Flushed queued chunks and saved committed output for --resume." in output.err
    build_runtime_policy_mock.assert_called_once_with(run_config, telemetry_session.paths)
    runtime_preflight_mock.assert_called_once_with(runtime_policy)
    initialize_logging_mock.assert_called_once_with(run_config.g_diagnostics, telemetry_session.paths)
    regenie_mock.assert_called_once_with(
        run_config,
        run_telemetry_session=telemetry_session,
        close_telemetry_session_on_exit=False,
        initialize_logging_on_entry=False,
    )
    assert telemetry_session.closed is True
    interrupted_lines = [call.kwargs["line"] for call in interrupted_record_mock.call_args_list]
    assert "Interrupted by SIGINT. Flushed queued chunks and saved committed output for --resume." in interrupted_lines


def test_run_args_reports_runtime_initialization_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Ensure runtime initialization failures are rendered without tracebacks."""
    from g.engine import telemetry as telemetry_module
    from g.runner import execution as runner_execution
    from g.runner import runtime as runner_runtime

    run_config = python_types.SimpleNamespace(g_diagnostics=python_types.SimpleNamespace())
    telemetry_session = FakeTelemetrySession()
    outcome = python_types.SimpleNamespace(stdout="", stderr="", exit_code=0, config=run_config)
    runtime_policy = python_types.SimpleNamespace()

    with (
        unittest.mock.patch("g.cli.g._core.dispatch_cli", return_value=outcome),
        unittest.mock.patch.object(telemetry_module, "build_telemetry_session", return_value=telemetry_session),
        unittest.mock.patch.object(
            runner_runtime, "build_runtime_policy", return_value=runtime_policy
        ) as build_runtime_policy_mock,
        unittest.mock.patch.object(runner_runtime, "require_compatible_runtime_policy") as runtime_preflight_mock,
        unittest.mock.patch.object(runner_runtime, "initialize_logging", side_effect=RuntimeError("logging failed")),
        unittest.mock.patch.object(runner_execution, "regenie") as regenie_mock,
        unittest.mock.patch("g.cli.g._core.record_native_cli_failed_line_diagnostic_event") as failed_record_mock,
    ):
        exit_code = cli.run_args(["regenie"])

    output = capsys.readouterr()
    assert exit_code == 1
    assert output.out == ""
    assert output.err == "Error: logging failed\n"
    assert "Traceback" not in output.err
    regenie_mock.assert_not_called()
    build_runtime_policy_mock.assert_called_once_with(run_config, telemetry_session.paths)
    runtime_preflight_mock.assert_called_once_with(runtime_policy)
    assert telemetry_session.logged_events == ["run_failed", "telemetry_session_closed"]
    assert telemetry_session.logged_payloads[0]["error_type"] == "RuntimeError"
    assert telemetry_session.logged_payloads[0]["error_message"] == "logging failed"
    assert telemetry_session.closed is True
    failed_lines = [call.kwargs["line"] for call in failed_record_mock.call_args_list]
    assert "Error: logging failed" in failed_lines


def test_run_args_reports_runner_failure_without_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Ensure runner failures return a concise CLI error instead of raising."""
    from g.engine import shutdown
    from g.engine import telemetry as telemetry_module
    from g.runner import execution as runner_execution
    from g.runner import runtime as runner_runtime

    run_config = python_types.SimpleNamespace(g_diagnostics=python_types.SimpleNamespace())
    telemetry_session = FakeTelemetrySession()
    outcome = python_types.SimpleNamespace(stdout="", stderr="", exit_code=0, config=run_config)
    runtime_policy = python_types.SimpleNamespace()
    with (
        unittest.mock.patch("g.cli.g._core.dispatch_cli", return_value=outcome),
        unittest.mock.patch.object(telemetry_module, "build_telemetry_session", return_value=telemetry_session),
        unittest.mock.patch.object(runner_runtime, "build_runtime_policy", return_value=runtime_policy),
        unittest.mock.patch.object(runner_runtime, "require_compatible_runtime_policy"),
        unittest.mock.patch.object(runner_runtime, "initialize_logging"),
        unittest.mock.patch.object(
            shutdown, "install_graceful_shutdown_handlers", return_value=contextlib.nullcontext()
        ),
        unittest.mock.patch.object(runner_execution, "regenie", side_effect=RuntimeError("pipeline failed")),
        unittest.mock.patch("g.cli.g._core.record_native_cli_failed_line_diagnostic_event") as failed_record_mock,
    ):
        exit_code = cli.run_args(["regenie"])

    output = capsys.readouterr()
    assert exit_code == 1
    assert output.out == ""
    assert output.err == "Error: pipeline failed\n"
    assert "Traceback" not in output.err
    assert telemetry_session.logged_events == ["telemetry_session_closed"]
    assert telemetry_session.closed is True
    failed_lines = [call.kwargs["line"] for call in failed_record_mock.call_args_list]
    assert "Error: pipeline failed" in failed_lines


def test_run_args_reports_telemetry_close_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Ensure telemetry close failures are rendered without tracebacks."""
    from g.engine import run_events, shutdown
    from g.engine import telemetry as telemetry_module
    from g.runner import execution as runner_execution
    from g.runner import runtime as runner_runtime

    run_config = python_types.SimpleNamespace(g_diagnostics=python_types.SimpleNamespace())
    telemetry_session = FakeTelemetrySession(close_error=RuntimeError("telemetry close failed"))
    run_artifacts = run_events.RunArtifacts(
        output_run_directory=Path("output.run"),
        final_dataset=None,
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
    runtime_policy = python_types.SimpleNamespace()
    with (
        unittest.mock.patch("g.cli.g._core.dispatch_cli", return_value=outcome),
        unittest.mock.patch.object(telemetry_module, "build_telemetry_session", return_value=telemetry_session),
        unittest.mock.patch.object(runner_runtime, "build_runtime_policy", return_value=runtime_policy),
        unittest.mock.patch.object(runner_runtime, "require_compatible_runtime_policy"),
        unittest.mock.patch.object(runner_runtime, "initialize_logging"),
        unittest.mock.patch.object(
            shutdown, "install_graceful_shutdown_handlers", return_value=contextlib.nullcontext()
        ),
        unittest.mock.patch.object(runner_execution, "regenie", return_value=run_artifacts),
        unittest.mock.patch("g.cli.g._core.record_native_cli_completed_line_diagnostic_event") as completed_record_mock,
        unittest.mock.patch("g.cli.g._core.record_native_cli_failed_line_diagnostic_event") as failed_record_mock,
    ):
        exit_code = cli.run_args(["regenie"])

    output = capsys.readouterr()
    assert exit_code == 1
    assert output.out == "Success. Chunked run saved to output.run\n"
    assert output.err == "Error: telemetry close failed\n"
    assert "Traceback" not in output.err
    assert telemetry_session.logged_events == ["telemetry_session_closed"]
    assert telemetry_session.closed is False
    completed_lines = [call.kwargs["line"] for call in completed_record_mock.call_args_list]
    failed_lines = [call.kwargs["line"] for call in failed_record_mock.call_args_list]
    assert "Success. Chunked run saved to output.run" in completed_lines
    assert "Error: telemetry close failed" in failed_lines


def test_run_args_preserves_runner_failure_when_telemetry_close_fails(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Ensure close failures do not replace an existing runner failure."""
    from g.engine import shutdown
    from g.engine import telemetry as telemetry_module
    from g.runner import execution as runner_execution
    from g.runner import runtime as runner_runtime

    run_config = python_types.SimpleNamespace(g_diagnostics=python_types.SimpleNamespace())
    telemetry_session = FakeTelemetrySession(close_error=RuntimeError("telemetry close failed"))
    outcome = python_types.SimpleNamespace(stdout="", stderr="", exit_code=0, config=run_config)
    runtime_policy = python_types.SimpleNamespace()
    with (
        unittest.mock.patch("g.cli.g._core.dispatch_cli", return_value=outcome),
        unittest.mock.patch.object(telemetry_module, "build_telemetry_session", return_value=telemetry_session),
        unittest.mock.patch.object(runner_runtime, "build_runtime_policy", return_value=runtime_policy),
        unittest.mock.patch.object(runner_runtime, "require_compatible_runtime_policy"),
        unittest.mock.patch.object(runner_runtime, "initialize_logging"),
        unittest.mock.patch.object(
            shutdown, "install_graceful_shutdown_handlers", return_value=contextlib.nullcontext()
        ),
        unittest.mock.patch.object(runner_execution, "regenie", side_effect=RuntimeError("pipeline failed")),
        unittest.mock.patch("g.cli.g._core.record_native_cli_failed_line_diagnostic_event") as failed_record_mock,
    ):
        exit_code = cli.run_args(["regenie"])

    output = capsys.readouterr()
    assert exit_code == 1
    assert output.out == ""
    assert output.err == "Error: pipeline failed\n"
    assert telemetry_session.logged_events == ["telemetry_session_closed"]
    assert telemetry_session.closed is False
    failed_lines = [call.kwargs["line"] for call in failed_record_mock.call_args_list]
    assert "Error: pipeline failed" in failed_lines
    assert "Error: telemetry close failed" not in failed_lines
