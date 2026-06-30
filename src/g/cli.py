"""Command-line dispatcher for the GWAS engine."""

from __future__ import annotations

import contextlib
import sys
import typing

import g._core

if typing.TYPE_CHECKING:
    from g.engine import run_events


NATIVE_CLI_OUTPUT_LOG_LIMIT = 4096
RUNTIME_FAILURE_EXIT_CODE = 1


def run_args(arguments: typing.Sequence[str]) -> int:
    """Run CLI arguments through the Rust frontend."""
    outcome = g._core.dispatch_cli(list(arguments))
    if outcome.config is None:
        print_native_cli_output(outcome)
        return outcome.exit_code

    # Runtime imports intentionally stay past Rust parsing so help and parser
    # error paths avoid importing JAX-facing runner and engine modules.
    from g.engine import run_events, shutdown, telemetry
    from g.runner import execution as runner_execution
    from g.runner import runtime as runner_runtime

    run_telemetry_session = None
    exit_code = RUNTIME_FAILURE_EXIT_CODE
    runner_started = False
    try:
        run_telemetry_session = telemetry.build_telemetry_session(outcome.config)
        runtime_policy = runner_runtime.build_runtime_policy(outcome.config, run_telemetry_session.paths)
        runner_runtime.require_compatible_runtime_policy(runtime_policy)
        runner_runtime.initialize_logging(outcome.config.g_diagnostics, run_telemetry_session.paths)
        print_native_cli_output(outcome)
        log_native_cli_output(outcome, max_payload_chars=NATIVE_CLI_OUTPUT_LOG_LIMIT)
        try:
            with shutdown.install_graceful_shutdown_handlers():
                runner_started = True
                artifacts = runner_execution.regenie(
                    outcome.config,
                    run_telemetry_session=run_telemetry_session,
                    close_telemetry_session_on_exit=False,
                    initialize_logging_on_entry=False,
                )
        except shutdown.GracefulShutdownRequested as shutdown_request:
            interrupted_event = run_events.build_run_interrupted_event(shutdown_request)
            print_interrupted_lines(run_events, interrupted_event)
            log_interrupted_lines(run_events, interrupted_event)
            exit_code = shutdown_request.exit_code
        else:
            completed_event = run_events.build_run_completed_event(artifacts)
            print_completed_lines(run_events, completed_event)
            log_completed_lines(run_events, completed_event)
            exit_code = 0
    except Exception as error:  # noqa: BLE001
        exit_code = print_and_log_failed_event(
            run_events,
            error,
            telemetry_session=run_telemetry_session,
            log_run_failed_to_telemetry=not runner_started,
        )
    finally:
        if run_telemetry_session is not None:
            try:
                telemetry.close_telemetry_session(run_telemetry_session)
            except Exception as error:  # noqa: BLE001
                if exit_code == 0:
                    print_and_log_failed_event(
                        run_events,
                        error,
                        telemetry_session=None,
                        log_run_failed_to_telemetry=False,
                    )
                    exit_code = RUNTIME_FAILURE_EXIT_CODE
    return exit_code


def print_native_cli_output(outcome: g._core.CliOutcome) -> None:
    """Print native CLI stdout and stderr exactly as emitted."""
    if outcome.stdout:
        print(outcome.stdout, end="")
    if outcome.stderr:
        print(outcome.stderr, end="", file=sys.stderr)


def log_native_cli_output(outcome: g._core.CliOutcome, *, max_payload_chars: int) -> None:
    """Emit bounded diagnostics for native CLI stdout and stderr."""
    if outcome.stdout:
        emit_diagnostic_event(
            "info",
            "native_cli_stdout",
            "Native CLI emitted stdout output.",
            bounded_output_fields("stdout", outcome.stdout, max_payload_chars=max_payload_chars),
        )
    if outcome.stderr:
        emit_diagnostic_event(
            "warn",
            "native_cli_stderr",
            "Native CLI emitted stderr output.",
            bounded_output_fields("stderr", outcome.stderr, max_payload_chars=max_payload_chars),
        )


def bounded_output_fields(stream_name: str, output_text: str, *, max_payload_chars: int) -> dict[str, object]:
    """Build bounded diagnostic fields for native CLI output text."""
    preview_character_count = max(0, max_payload_chars)
    preview = output_text[:preview_character_count]
    truncated = len(output_text) > preview_character_count
    fields: dict[str, object] = {
        f"{stream_name}_character_count": len(output_text),
        f"{stream_name}_byte_count": len(output_text.encode("utf-8")),
        f"{stream_name}_preview": preview,
        f"{stream_name}_truncated": truncated,
    }
    if truncated:
        fields[f"{stream_name}_omitted_character_count"] = len(output_text) - preview_character_count
    return fields


def emit_diagnostic_event(level: str, event: str, message: str, fields: dict[str, object]) -> None:
    """Emit a structured CLI diagnostic through native tracing."""
    g._core.emit_diagnostic_event_fields(level, event, message, fields)


def print_interrupted_lines(run_events_module: typing.Any, interrupted_event: run_events.RunInterruptedEvent) -> None:
    """Print graceful interruption details."""
    interrupted_lines = run_events_module.render_run_interrupted_lines(interrupted_event)
    for line in interrupted_lines:
        print(line, file=sys.stderr)


def log_interrupted_lines(run_events_module: typing.Any, interrupted_event: run_events.RunInterruptedEvent) -> None:
    """Emit graceful interruption diagnostics."""
    interrupted_lines = run_events_module.render_run_interrupted_lines(interrupted_event)
    for line in interrupted_lines:
        emit_diagnostic_event(
            "warn",
            "native_cli_interrupted_line",
            "Native CLI interruption detail.",
            {
                "line": line,
            },
        )


def print_and_log_failed_event(
    run_events_module: typing.Any,
    error: Exception,
    *,
    telemetry_session: typing.Any,
    log_run_failed_to_telemetry: bool,
) -> int:
    """Print and log a concise runtime failure."""
    failed_event = run_events_module.build_run_failed_event(error)
    if log_run_failed_to_telemetry:
        log_run_failed_telemetry_event(failed_event, telemetry_session=telemetry_session)
    print_failed_lines(run_events_module, failed_event)
    log_failed_lines(run_events_module, failed_event)
    return RUNTIME_FAILURE_EXIT_CODE


def log_run_failed_telemetry_event(
    failed_event: run_events.RunFailedEvent,
    *,
    telemetry_session: typing.Any,
) -> None:
    """Write a run failure event to telemetry when available."""
    if telemetry_session is None:
        return
    with contextlib.suppress(Exception):
        telemetry_session.log_run_failed(failed_event)


def print_failed_lines(run_events_module: typing.Any, failed_event: run_events.RunFailedEvent) -> None:
    """Print failure details."""
    failed_lines = run_events_module.render_run_failed_lines(failed_event)
    for line in failed_lines:
        print(line, file=sys.stderr)


def log_failed_lines(run_events_module: typing.Any, failed_event: run_events.RunFailedEvent) -> None:
    """Emit failure diagnostics."""
    failed_lines = run_events_module.render_run_failed_lines(failed_event)
    for line in failed_lines:
        with contextlib.suppress(Exception):
            emit_diagnostic_event(
                "error",
                "native_cli_failed_line",
                "Native CLI failure detail.",
                {
                    "line": line,
                },
            )


def print_completed_lines(run_events_module: typing.Any, completed_event: run_events.RunCompletedEvent) -> None:
    """Print completion details."""
    completed_lines = run_events_module.render_run_completed_lines(completed_event)
    for line in completed_lines:
        print(line)


def log_completed_lines(run_events_module: typing.Any, completed_event: run_events.RunCompletedEvent) -> None:
    """Emit completion diagnostics."""
    completed_lines = run_events_module.render_run_completed_lines(completed_event)
    for line in completed_lines:
        emit_diagnostic_event(
            "info",
            "native_cli_completed_line",
            "Native CLI completion detail.",
            {
                "line": line,
            },
        )


def main() -> None:
    """Run the GWAS CLI."""
    raise SystemExit(run_args(sys.argv[1:]))
