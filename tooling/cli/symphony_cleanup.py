#!/usr/bin/env python3
"""CLI for safe Symphony stale worktree and branch cleanup."""

from __future__ import annotations

import argparse
import os
import sys
import typing
from pathlib import Path

import tooling.symphony_cleanup as symphony_cleanup
from tooling.common import paths as tooling_paths


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        Configured argument parser.

    """
    parser = argparse.ArgumentParser(
        description="Safely inspect and clean stale Symphony worktrees and branches.",
    )
    parser.add_argument(
        "--repository",
        type=Path,
        default=None,
        help="Repository root used for git commands. Defaults to the current repository.",
    )
    parser.add_argument(
        "--worktree-root",
        type=Path,
        default=None,
        help=(
            "Configured Symphony worktree root. Defaults to SYMPHONY_WORKTREE_ROOT "
            "or /mnt/beegfs/kirill/Projects/g-worktrees/symphony."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply candidate worktree removals. Omit for dry-run.",
    )
    parser.add_argument(
        "--include-unknown",
        action="store_true",
        help="Allow unknown Linear issue states to be selected. Active states are still retained.",
    )
    parser.add_argument(
        "--delete-local-branches",
        action="store_true",
        help="With --apply, delete selected local branches using git branch -d.",
    )
    parser.add_argument(
        "--delete-remote-branches",
        action="store_true",
        help="With --apply, delete selected origin branches using git push origin --delete.",
    )
    return parser


def resolve_repository(argument_repository: Path | None) -> Path:
    """Resolve the repository root for cleanup.

    Args:
        argument_repository: Optional repository argument.

    Returns:
        Repository root.

    """
    if argument_repository is not None:
        return argument_repository
    return tooling_paths.find_repository_root(Path.cwd())


def resolve_worktree_root(argument_worktree_root: Path | None) -> Path:
    """Resolve the configured Symphony worktree root.

    Args:
        argument_worktree_root: Optional worktree-root argument.

    Returns:
        Worktree root.

    """
    if argument_worktree_root is not None:
        return argument_worktree_root
    return symphony_cleanup.configured_worktree_root(os.environ)


def main(argument_vector: typing.Sequence[str] | None = None) -> int:
    """Run safe Symphony cleanup.

    Args:
        argument_vector: Optional argument vector for tests.

    Returns:
        Process exit code.

    """
    parser = build_argument_parser()
    arguments = parser.parse_args(argument_vector)
    repository_root = resolve_repository(arguments.repository)
    worktree_root = resolve_worktree_root(arguments.worktree_root)
    try:
        cleanup_plan = symphony_cleanup.build_cleanup_plan(
            repository_root,
            worktree_root,
            include_unknown=arguments.include_unknown,
            environment=os.environ,
        )
    except symphony_cleanup.CleanupError as error:
        parser.exit(2, f"error: {error}\n")
    sys.stdout.write(
        symphony_cleanup.render_cleanup_plan(
            cleanup_plan,
            apply_changes=arguments.apply,
            include_unknown=arguments.include_unknown,
            delete_local_branches=arguments.delete_local_branches,
            delete_remote_branches=arguments.delete_remote_branches,
        )
    )
    if not arguments.apply:
        return 0
    execution_result = symphony_cleanup.apply_cleanup_plan(
        cleanup_plan,
        delete_local_branches=arguments.delete_local_branches,
        delete_remote_branches=arguments.delete_remote_branches,
    )
    sys.stdout.write(symphony_cleanup.render_execution_result(execution_result))
    return execution_result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
