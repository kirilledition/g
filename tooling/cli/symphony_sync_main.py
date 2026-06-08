#!/usr/bin/env python3
"""CLI for safe local main synchronization after Symphony direct merges."""

from __future__ import annotations

import argparse
import sys
import typing
from pathlib import Path

import tooling.symphony_sync_main as symphony_sync_main
from tooling.common import paths as tooling_paths


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        Configured argument parser.

    """
    parser = argparse.ArgumentParser(
        description="Safely fast-forward the local main worktree to origin/main.",
    )
    parser.add_argument(
        "--repository",
        type=Path,
        default=None,
        help=(
            "Repository or worktree path used to locate git metadata. Defaults to "
            "the current repository. The main worktree is located automatically when possible."
        ),
    )
    parser.add_argument(
        "--branch",
        default=symphony_sync_main.DEFAULT_BRANCH_NAME,
        help="Local branch to synchronize. Defaults to main.",
    )
    parser.add_argument(
        "--remote",
        default=symphony_sync_main.DEFAULT_REMOTE_NAME,
        help="Remote name to fetch from. Defaults to origin.",
    )
    return parser


def resolve_repository(argument_repository: Path | None) -> Path:
    """Resolve the repository path used for git metadata lookup.

    Args:
        argument_repository: Optional repository argument.

    Returns:
        Repository or worktree path.

    """
    if argument_repository is not None:
        return argument_repository
    return tooling_paths.find_repository_root(Path.cwd())


def main(argument_vector: typing.Sequence[str] | None = None) -> int:
    """Run safe local main synchronization.

    Args:
        argument_vector: Optional argument vector for tests.

    Returns:
        Process exit code.

    """
    parser = build_argument_parser()
    arguments = parser.parse_args(argument_vector)
    result = symphony_sync_main.sync_local_main(
        resolve_repository(arguments.repository),
        branch_name=arguments.branch,
        remote_name=arguments.remote,
    )
    sys.stdout.write(symphony_sync_main.render_sync_main_result(result))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
