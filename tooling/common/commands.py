"""Subprocess helpers for development tooling."""

from __future__ import annotations

import os
import subprocess
import typing
from dataclasses import dataclass

if typing.TYPE_CHECKING:
    import collections.abc
    from pathlib import Path


@dataclass(frozen=True)
class CommandOutput:
    """Captured subprocess output.

    Attributes:
        command_arguments: Command argument vector.
        return_code: Process return code, or None when the executable is missing.
        stdout: Captured standard output.
        stderr: Captured standard error.

    """

    command_arguments: list[str]
    return_code: int | None
    stdout: str
    stderr: str


def run_captured_command(
    command_arguments: list[str],
    *,
    environment_overrides: collections.abc.Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> CommandOutput:
    """Run a command and return captured output without raising.

    Args:
        command_arguments: Command argument vector.
        environment_overrides: Environment variables to add or override.
        cwd: Optional working directory.

    Returns:
        Captured command output.

    """
    environment = dict(os.environ)
    if environment_overrides is not None:
        environment.update(environment_overrides)
    try:
        completed_process = subprocess.run(
            command_arguments,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            cwd=cwd,
        )
    except FileNotFoundError as error:
        return CommandOutput(
            command_arguments=command_arguments,
            return_code=None,
            stdout="",
            stderr=str(error),
        )
    return CommandOutput(
        command_arguments=command_arguments,
        return_code=completed_process.returncode,
        stdout=completed_process.stdout,
        stderr=completed_process.stderr,
    )
