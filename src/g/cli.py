"""Command-line dispatcher for the GWAS engine."""

from __future__ import annotations

import sys
import typing

import g._core
from g import api
from g.engine import run_events, shutdown


def run_args(arguments: typing.Sequence[str], *, direct_regenie: bool = False) -> int:
    """Run CLI arguments through the Rust frontend."""
    outcome = g._core.dispatch_cli(list(arguments), direct_regenie)
    if outcome.stdout:
        print(outcome.stdout, end="")
    if outcome.stderr:
        print(outcome.stderr, end="", file=sys.stderr)
    if outcome.config is None:
        return outcome.exit_code

    try:
        with shutdown.install_graceful_shutdown_handlers():
            artifacts = api.regenie(outcome.config)
    except shutdown.GracefulShutdownRequested as shutdown_request:
        interrupted_event = run_events.build_run_interrupted_event(shutdown_request)
        for line in run_events.render_run_interrupted_lines(interrupted_event):
            print(line, file=sys.stderr)
        return shutdown_request.exit_code

    completed_event = run_events.build_run_completed_event(artifacts)
    for line in run_events.render_run_completed_lines(completed_event):
        print(line)
    return 0


def regenie_main() -> None:
    """Run the direct g-regenie executable."""
    raise SystemExit(run_args(sys.argv[1:], direct_regenie=True))


def main() -> None:
    """Run the GWAS CLI."""
    raise SystemExit(run_args(sys.argv[1:]))
