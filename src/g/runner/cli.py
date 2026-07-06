"""Runner-owned CLI backend lifecycle."""

from __future__ import annotations

import sys

import g._core
from g.runner import events, execution, lifecycle, runtime

NATIVE_CLI_OUTPUT_LOG_LIMIT = 4096
RUNTIME_FAILURE_EXIT_CODE = 1


def run_validated_cli_outcome(outcome: g._core.CliOutcome) -> int:
    """Run a validated native CLI outcome through the Python backend."""
    regenie_config = outcome.config
    if regenie_config is None:
        message = "validated CLI outcome must contain a config"
        raise ValueError(message)

    cli_lifecycle_state = g._core.NativeCliRunLifecycleState()
    run_telemetry_session: events.TelemetrySession | None = None
    exit_code = RUNTIME_FAILURE_EXIT_CODE
    try:
        run_telemetry_session = events.build_telemetry_session(regenie_config)
        runtime_policy = runtime.build_runtime_policy(
            runtime.RuntimePolicyRequest(
                diagnostics_config=regenie_config.g_diagnostics,
                compute_config=regenie_config.g_compute,
                rayon_thread_count=regenie_config.trait.threads,
                telemetry_paths=run_telemetry_session.paths,
            )
        )
        runtime.require_compatible_runtime_policy(runtime_policy)
        runtime.initialize_logging(regenie_config.g_diagnostics, run_telemetry_session.paths)
        if outcome.stdout:
            print(outcome.stdout, end="")
        if outcome.stderr:
            print(outcome.stderr, end="", file=sys.stderr)
        if outcome.stdout or outcome.stderr:
            g._core.record_native_cli_output_diagnostic_events(
                outcome.stdout,
                outcome.stderr,
                NATIVE_CLI_OUTPUT_LOG_LIMIT,
            )
        try:
            with lifecycle.GracefulShutdownController(handled_signals=None):
                cli_lifecycle_state.mark_runner_started()
                artifacts = execution.regenie(
                    regenie_config,
                    run_telemetry_session=run_telemetry_session,
                    close_telemetry_session_on_exit=False,
                    initialize_logging_on_entry=False,
                )
        except lifecycle.GracefulShutdownRequested as shutdown_request:
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
                native_telemetry_session = events.native_telemetry_session_handle(run_telemetry_session)
                if native_telemetry_session is not None:
                    native_telemetry_session.finish_with_current_close_event_metadata()
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


def print_and_log_failed_event(
    error: Exception,
    *,
    cli_lifecycle_state: g._core.NativeCliRunLifecycleState,
    telemetry_session: events.TelemetrySession | None,
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
