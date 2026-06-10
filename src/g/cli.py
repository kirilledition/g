"""Command-line dispatcher for the GWAS engine."""

from __future__ import annotations

import json
import sys
import typing

import g._core
from g import runner
from g.engine import run_events, shutdown, telemetry


def run_args(arguments: typing.Sequence[str], *, direct_regenie: bool = False) -> int:
    """Run CLI arguments through the Rust frontend."""
    outcome = g._core.dispatch_cli(list(arguments), direct_regenie)
    if outcome.config is None:
        emit_native_cli_output(outcome)
        return outcome.exit_code

    run_telemetry_session = telemetry.build_telemetry_session(outcome.config)
    runner.initialize_logging(outcome.config.g_diagnostics, run_telemetry_session.paths)
    try:
        emit_native_cli_output(outcome)
        try:
            with shutdown.install_graceful_shutdown_handlers():
                artifacts = runner.regenie(
                    outcome.config,
                    run_telemetry_session=run_telemetry_session,
                    close_telemetry_session_on_exit=False,
                )
        except shutdown.GracefulShutdownRequested as shutdown_request:
            emit_interrupted_lines(run_events.build_run_interrupted_event(shutdown_request))
            return shutdown_request.exit_code

        emit_completed_lines(run_events.build_run_completed_event(artifacts))
        return 0
    finally:
        telemetry.close_telemetry_session(run_telemetry_session)


def emit_native_cli_output(outcome: g._core.CliOutcome) -> None:
    """Mirror native CLI stdout and stderr to tracing before printing."""
    if outcome.stdout:
        emit_diagnostic_event(
            "info",
            "native_cli_stdout",
            "Native CLI emitted stdout output.",
            {
                "outcome_stdout": outcome.stdout,
            },
        )
        print(outcome.stdout, end="")
    if outcome.stderr:
        emit_diagnostic_event(
            "warn",
            "native_cli_stderr",
            "Native CLI emitted stderr output.",
            {
                "outcome_stderr": outcome.stderr,
            },
        )
        print(outcome.stderr, end="", file=sys.stderr)


def emit_diagnostic_event(level: str, event: str, message: str, fields: dict[str, object]) -> None:
    """Emit a structured CLI diagnostic through native tracing."""
    g._core.emit_diagnostic_event(level, event, message, json.dumps(fields, sort_keys=True, default=str))


def emit_interrupted_lines(interrupted_event: run_events.RunInterruptedEvent) -> None:
    """Mirror graceful interruption details to tracing before printing."""
    interrupted_lines = run_events.render_run_interrupted_lines(interrupted_event)
    for line in interrupted_lines:
        emit_diagnostic_event(
            "warn",
            "native_cli_interrupted_line",
            "Native CLI interruption detail.",
            {
                "line": line,
            },
        )
        print(line, file=sys.stderr)


def emit_completed_lines(completed_event: run_events.RunCompletedEvent) -> None:
    """Mirror completion details to tracing before printing."""
    completed_lines = run_events.render_run_completed_lines(completed_event)
    for line in completed_lines:
        emit_diagnostic_event(
            "info",
            "native_cli_completed_line",
            "Native CLI completion detail.",
            {
                "line": line,
            },
        )
        print(line)


def regenie_main() -> None:
    """Run the direct g-regenie executable."""
    raise SystemExit(run_args(sys.argv[1:], direct_regenie=True))


def main() -> None:
    """Run the GWAS CLI."""
    raise SystemExit(run_args(sys.argv[1:]))
