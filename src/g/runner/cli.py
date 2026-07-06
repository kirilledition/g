"""Runner-owned CLI backend lifecycle."""

from __future__ import annotations

import sys
import typing

import g._core
from g.runner import events, execution, lifecycle, runtime


def run(arguments: typing.Sequence[str]) -> int:
    """Run CLI arguments through the Rust frontend and Python backend."""
    outcome = g._core.dispatch_cli(list(arguments))
    cli_output_streams = ((outcome.stdout, sys.stdout), (outcome.stderr, sys.stderr))
    regenie_config = outcome.config
    if regenie_config is None:
        for output_text, output_stream in cli_output_streams:
            if output_text:
                print(output_text, end="", file=output_stream)
        return outcome.exit_code

    cli_lifecycle_state = g._core.NativeCliRunLifecycleState()
    run_telemetry_session: events.TelemetrySession | None = None
    exit_code: int | None = None
    try:
        run_telemetry_session = events.build_telemetry_session(regenie_config)
        runtime.initialize_logging(regenie_config.g_diagnostics, run_telemetry_session.paths)
        for output_text, output_stream in cli_output_streams:
            if output_text:
                print(output_text, end="", file=output_stream)
        cli_lifecycle_state.record_frontend_output(outcome.stdout, outcome.stderr)
        with lifecycle.GracefulShutdownController(handled_signals=None):
            cli_lifecycle_state.mark_runner_started()
            artifacts = execution.regenie(
                regenie_config,
                run_telemetry_session=run_telemetry_session,
                close_telemetry_session_on_exit=False,
                initialize_logging_on_entry=False,
            )
        terminal_result = cli_lifecycle_state.completed_result(artifacts)
        for line in terminal_result.stdout_lines:
            print(line)
        for line in terminal_result.stderr_lines:
            print(line, file=sys.stderr)
        exit_code = terminal_result.exit_code
    except lifecycle.GracefulShutdownRequested as shutdown_request:
        terminal_result = cli_lifecycle_state.interrupted_result(shutdown_request)
        for line in terminal_result.stdout_lines:
            print(line)
        for line in terminal_result.stderr_lines:
            print(line, file=sys.stderr)
        exit_code = terminal_result.exit_code
    except Exception as error:  # noqa: BLE001
        terminal_result = cli_lifecycle_state.failed_result(error, run_telemetry_session)
        for line in terminal_result.stdout_lines:
            print(line)
        for line in terminal_result.stderr_lines:
            print(line, file=sys.stderr)
        exit_code = terminal_result.exit_code
    finally:
        if run_telemetry_session is not None:
            close_result = cli_lifecycle_state.finish_telemetry_result(
                1 if exit_code is None else exit_code,
                run_telemetry_session,
            )
            for line in close_result.stdout_lines:
                print(line)
            for line in close_result.stderr_lines:
                print(line, file=sys.stderr)
            exit_code = close_result.exit_code
    return 1 if exit_code is None else exit_code


def main() -> None:
    """Run the GWAS CLI."""
    raise SystemExit(run(sys.argv[1:]))
