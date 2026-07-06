"""Command-line dispatcher for the GWAS engine."""

from __future__ import annotations

import os
import sys
import typing

import g._core

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
    from g.runner import events as runner_events
    from g.runner import execution as runner_execution
    from g.runner import lifecycle as runner_lifecycle
    from g.runner import runtime as runner_runtime

    cli_lifecycle_state = g._core.NativeCliRunLifecycleState()
    run_telemetry_session = None
    exit_code = RUNTIME_FAILURE_EXIT_CODE
    try:
        run_telemetry_session = runner_events.build_telemetry_session(outcome.config)
        runtime_policy = runner_runtime.build_runtime_policy(outcome.config, run_telemetry_session.paths)
        runner_runtime.require_compatible_runtime_policy(runtime_policy)
        runner_runtime.initialize_logging(outcome.config.g_diagnostics, run_telemetry_session.paths)
        print_native_cli_output(outcome)
        if outcome.stdout or outcome.stderr:
            g._core.record_native_cli_output_diagnostic_events(
                outcome.stdout,
                outcome.stderr,
                NATIVE_CLI_OUTPUT_LOG_LIMIT,
            )
        try:
            with runner_lifecycle.GracefulShutdownController(handled_signals=None):
                cli_lifecycle_state.mark_runner_started()
                artifacts = runner_execution.regenie(
                    outcome.config,
                    run_telemetry_session=run_telemetry_session,
                    close_telemetry_session_on_exit=False,
                    initialize_logging_on_entry=False,
                )
        except runner_lifecycle.GracefulShutdownRequested as shutdown_request:
            interrupted_event = g._core.build_run_interrupted_event(shutdown_request)
            for line in g._core.render_and_record_run_interrupted_lines(interrupted_event):
                print(line, file=sys.stderr)
            exit_code = shutdown_request.exit_code
        else:
            completed_event = g._core.build_run_completed_event(artifacts)
            for line in g._core.render_and_record_run_completed_lines(completed_event):
                print(line)
            exit_code = 0
    except Exception as error:  # noqa: BLE001
        exit_code = print_and_log_failed_event(
            error,
            cli_lifecycle_state=cli_lifecycle_state,
            telemetry_session=run_telemetry_session,
        )
    finally:
        if run_telemetry_session is not None:
            try:
                g._core.close_telemetry_session_with_event(run_telemetry_session)
            except Exception as error:  # noqa: BLE001
                close_failure_plan = cli_lifecycle_state.plan_telemetry_close_failure(
                    exit_code,
                    RUNTIME_FAILURE_EXIT_CODE,
                )
                if close_failure_plan.should_report_failure:
                    print_and_log_failed_event(
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


def print_and_log_failed_event(
    error: Exception,
    *,
    cli_lifecycle_state: g._core.NativeCliRunLifecycleState,
    telemetry_session: typing.Any,
) -> int:
    """Print and log a concise runtime failure."""
    failed_event = g._core.build_run_failed_event(error)
    cli_lifecycle_state.emit_run_failed_telemetry_event(
        telemetry_session,
        failed_event,
    )
    for line in g._core.render_and_record_run_failed_lines(failed_event):
        print(line, file=sys.stderr)
    return RUNTIME_FAILURE_EXIT_CODE


def main() -> None:
    """Run the GWAS CLI."""
    raise SystemExit(run_args(sys.argv[1:]))
