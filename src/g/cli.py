"""Command-line dispatcher for the GWAS engine."""

from __future__ import annotations

import contextlib
import os
import sys
import typing

import g._core

if typing.TYPE_CHECKING:
    from g.engine import run_events


NATIVE_CLI_OUTPUT_LOG_LIMIT = 4096
NATIVE_CLI_PYTHON_BRIDGE_SENTINEL_ENVIRONMENT_VARIABLE = "G_NATIVE_CLI_PYTHON_BRIDGE_SENTINEL"
RUNTIME_FAILURE_EXIT_CODE = 1


def run_args(arguments: typing.Sequence[str]) -> int:
    """Run CLI arguments through the native PyO3 frontend."""
    if os.environ.get(NATIVE_CLI_PYTHON_BRIDGE_SENTINEL_ENVIRONMENT_VARIABLE) == "1":
        return run_args_legacy(arguments)

    outcome = g._core.run_native_cli_python_bridge(
        list(arguments),
        sys.executable,
        NATIVE_CLI_PYTHON_BRIDGE_SENTINEL_ENVIRONMENT_VARIABLE,
    )
    print_native_cli_output(outcome)
    return outcome.exit_code


def run_args_legacy(arguments: typing.Sequence[str]) -> int:
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

    cli_lifecycle_state = g._core.NativeCliRunLifecycleState()
    run_telemetry_session = None
    exit_code = RUNTIME_FAILURE_EXIT_CODE
    try:
        run_telemetry_session = telemetry.build_telemetry_session(outcome.config)
        runtime_policy = runner_runtime.build_runtime_policy(outcome.config, run_telemetry_session.paths)
        runner_runtime.require_compatible_runtime_policy(runtime_policy)
        runner_runtime.initialize_logging(outcome.config.g_diagnostics, run_telemetry_session.paths)
        print_native_cli_output(outcome)
        log_native_cli_output(outcome, max_payload_chars=NATIVE_CLI_OUTPUT_LOG_LIMIT)
        try:
            with shutdown.install_graceful_shutdown_handlers():
                cli_lifecycle_state.mark_runner_started()
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
            cli_lifecycle_state=cli_lifecycle_state,
            telemetry_session=run_telemetry_session,
        )
    finally:
        if run_telemetry_session is not None:
            try:
                telemetry.close_telemetry_session(run_telemetry_session)
            except Exception as error:  # noqa: BLE001
                close_failure_plan = cli_lifecycle_state.plan_telemetry_close_failure(
                    exit_code,
                    RUNTIME_FAILURE_EXIT_CODE,
                )
                if close_failure_plan.should_report_failure:
                    print_and_log_failed_event(
                        run_events,
                        error,
                        cli_lifecycle_state=cli_lifecycle_state,
                        telemetry_session=None,
                    )
                exit_code = close_failure_plan.exit_code
    return exit_code


def print_native_cli_output(outcome: g._core.CliOutcome) -> None:
    """Print native CLI stdout and stderr exactly as emitted."""
    if outcome.stdout:
        print(outcome.stdout, end="")
    if outcome.stderr:
        print(outcome.stderr, end="", file=sys.stderr)


def log_native_cli_output(outcome: g._core.CliOutcome, *, max_payload_chars: int) -> None:
    """Emit bounded diagnostics for native CLI stdout and stderr."""
    if not outcome.stdout and not outcome.stderr:
        return

    native_diagnostic_policy = native_cli_diagnostic_policy()
    if outcome.stdout:
        native_diagnostic_policy.record_native_cli_stdout_diagnostic_event(
            output_text=outcome.stdout,
            max_payload_chars=max_payload_chars,
        )
    if outcome.stderr:
        native_diagnostic_policy.record_native_cli_stderr_diagnostic_event(
            output_text=outcome.stderr,
            max_payload_chars=max_payload_chars,
        )


def native_cli_diagnostic_policy() -> g._core.NativeCliDiagnosticPolicy:
    """Build the native CLI diagnostic policy handle."""
    return g._core.NativeCliDiagnosticPolicy()


def print_interrupted_lines(run_events_module: typing.Any, interrupted_event: run_events.RunInterruptedEvent) -> None:
    """Print graceful interruption details."""
    interrupted_lines = run_events_module.render_run_interrupted_lines(interrupted_event)
    for line in interrupted_lines:
        print(line, file=sys.stderr)


def log_interrupted_lines(run_events_module: typing.Any, interrupted_event: run_events.RunInterruptedEvent) -> None:
    """Emit graceful interruption diagnostics."""
    interrupted_lines = run_events_module.render_run_interrupted_lines(interrupted_event)
    native_diagnostic_policy = native_cli_diagnostic_policy()
    for line in interrupted_lines:
        native_diagnostic_policy.record_native_cli_interrupted_line_diagnostic_event(line=line)


def print_and_log_failed_event(
    run_events_module: typing.Any,
    error: Exception,
    *,
    cli_lifecycle_state: g._core.NativeCliRunLifecycleState,
    telemetry_session: typing.Any,
) -> int:
    """Print and log a concise runtime failure."""
    failed_event = run_events_module.build_run_failed_event(error)
    cli_lifecycle_state.emit_run_failed_telemetry_event(
        telemetry_session,
        failed_event,
    )
    print_failed_lines(run_events_module, failed_event)
    log_failed_lines(run_events_module, failed_event)
    return RUNTIME_FAILURE_EXIT_CODE


def print_failed_lines(run_events_module: typing.Any, failed_event: run_events.RunFailedEvent) -> None:
    """Print failure details."""
    failed_lines = run_events_module.render_run_failed_lines(failed_event)
    for line in failed_lines:
        print(line, file=sys.stderr)


def log_failed_lines(run_events_module: typing.Any, failed_event: run_events.RunFailedEvent) -> None:
    """Emit failure diagnostics."""
    failed_lines = run_events_module.render_run_failed_lines(failed_event)
    native_diagnostic_policy = native_cli_diagnostic_policy()
    for line in failed_lines:
        with contextlib.suppress(Exception):
            native_diagnostic_policy.record_native_cli_failed_line_diagnostic_event(line=line)


def print_completed_lines(run_events_module: typing.Any, completed_event: run_events.RunCompletedEvent) -> None:
    """Print completion details."""
    completed_lines = run_events_module.render_run_completed_lines(completed_event)
    for line in completed_lines:
        print(line)


def log_completed_lines(run_events_module: typing.Any, completed_event: run_events.RunCompletedEvent) -> None:
    """Emit completion diagnostics."""
    completed_lines = run_events_module.render_run_completed_lines(completed_event)
    native_diagnostic_policy = native_cli_diagnostic_policy()
    for line in completed_lines:
        native_diagnostic_policy.record_native_cli_completed_line_diagnostic_event(line=line)


def main() -> None:
    """Run the GWAS CLI."""
    raise SystemExit(run_args(sys.argv[1:]))
