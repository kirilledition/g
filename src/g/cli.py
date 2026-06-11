"""Command-line dispatcher for the GWAS engine."""

from __future__ import annotations

import json
import sys
import typing

import g._core

if typing.TYPE_CHECKING:
    from g.engine import run_events


NATIVE_CLI_OUTPUT_LOG_LIMIT = 4096


def run_args(arguments: typing.Sequence[str], *, direct_regenie: bool = False) -> int:
    """Run CLI arguments through the Rust frontend."""
    outcome = g._core.dispatch_cli(list(arguments), direct_regenie)
    if outcome.config is None:
        print_native_cli_output(outcome)
        return outcome.exit_code

    # Runtime imports intentionally stay past Rust parsing so help and parser
    # error paths avoid importing JAX-facing runner and engine modules.
    from g import runner
    from g.engine import run_events, shutdown, telemetry

    run_telemetry_session = telemetry.build_telemetry_session(outcome.config)
    try:
        runner.initialize_logging(outcome.config.g_diagnostics, run_telemetry_session.paths)
        print_native_cli_output(outcome)
        log_native_cli_output(outcome)
        try:
            with shutdown.install_graceful_shutdown_handlers():
                artifacts = runner.regenie(
                    outcome.config,
                    run_telemetry_session=run_telemetry_session,
                    close_telemetry_session_on_exit=False,
                    initialize_logging_on_entry=False,
                )
        except shutdown.GracefulShutdownRequested as shutdown_request:
            interrupted_event = run_events.build_run_interrupted_event(shutdown_request)
            print_interrupted_lines(run_events, interrupted_event)
            log_interrupted_lines(run_events, interrupted_event)
            return shutdown_request.exit_code

        completed_event = run_events.build_run_completed_event(artifacts)
        print_completed_lines(run_events, completed_event)
        log_completed_lines(run_events, completed_event)
        return 0
    finally:
        telemetry.close_telemetry_session(run_telemetry_session)


def print_native_cli_output(outcome: g._core.CliOutcome) -> None:
    """Print native CLI stdout and stderr exactly as emitted."""
    if outcome.stdout:
        print(outcome.stdout, end="")
    if outcome.stderr:
        print(outcome.stderr, end="", file=sys.stderr)


def log_native_cli_output(outcome: g._core.CliOutcome, *, max_payload_chars: int = NATIVE_CLI_OUTPUT_LOG_LIMIT) -> None:
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
    g._core.emit_diagnostic_event(level, event, message, json.dumps(fields, sort_keys=True, default=str))


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


def regenie_main() -> None:
    """Run the direct g-regenie executable."""
    raise SystemExit(run_args(sys.argv[1:], direct_regenie=True))


def main() -> None:
    """Run the GWAS CLI."""
    raise SystemExit(run_args(sys.argv[1:]))
