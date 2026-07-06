"""Command-line dispatcher for the GWAS engine."""

from __future__ import annotations

import sys
import typing

import g._core


def run_args(arguments: typing.Sequence[str]) -> int:
    """Run CLI arguments through the Rust frontend."""
    outcome = g._core.dispatch_cli(list(arguments))
    if outcome.config is None:
        if outcome.stdout:
            print(outcome.stdout, end="")
        if outcome.stderr:
            print(outcome.stderr, end="", file=sys.stderr)
        return outcome.exit_code

    # Keep runtime imports behind Rust parsing so help and parser errors stay light.
    from g.runner import cli as runner_cli

    return runner_cli.run_validated_cli_outcome(outcome)


def main() -> None:
    """Run the GWAS CLI."""
    raise SystemExit(run_args(sys.argv[1:]))
