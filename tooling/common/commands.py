"""Subprocess helpers for development tooling."""

from __future__ import annotations

import dataclasses
import os
import select
import subprocess
import time
import typing
from dataclasses import dataclass

from tooling.common import artifact_format as tooling_artifact_format

if typing.TYPE_CHECKING:
    import collections.abc
    from pathlib import Path


REDACTED_ENVIRONMENT_VALUE = "<redacted>"


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


@dataclass(frozen=True)
class CommandSpec:
    """Complete command execution request.

    Attributes:
        args: Shell-free command argument vector.
        cwd: Optional working directory.
        env: Environment overrides.
        timeout_seconds: Optional timeout.
        stdout_path: Optional standard-output log path.
        stderr_path: Optional standard-error log path.
        stream: Whether output should be streamed while captured.
        sensitive_env_keys: Environment keys that must be redacted in reports.

    """

    args: tuple[str, ...]
    cwd: Path | None
    env: dict[str, str]
    timeout_seconds: float | None
    stdout_path: Path | None
    stderr_path: Path | None
    stream: bool
    sensitive_env_keys: tuple[str, ...]


@dataclass(frozen=True)
class CommandResult:
    """Structured command execution result.

    Attributes:
        args: Command argument vector.
        return_code: Process return code, or ``None`` when execution did not start.
        stdout: Captured standard output.
        stderr: Captured standard error.
        timed_out: Whether a timeout stopped the command.
        missing_executable: Whether the executable was missing.
        cwd: Working directory used for execution.
        environment_overrides: Redacted environment overrides.

    """

    args: tuple[str, ...]
    return_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    missing_executable: bool
    cwd: str | None
    environment_overrides: dict[str, str]


def build_command_spec(
    args: collections.abc.Sequence[str],
    *,
    cwd: Path | None = None,
    env: collections.abc.Mapping[str, str] | None = None,
    timeout_seconds: float | None = None,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
    stream: bool = False,
    sensitive_env_keys: collections.abc.Sequence[str] = (),
) -> CommandSpec:
    """Build a command spec with normalized immutable arguments.

    Args:
        args: Shell-free command argument vector.
        cwd: Optional working directory.
        env: Environment overrides.
        timeout_seconds: Optional timeout.
        stdout_path: Optional standard-output log path.
        stderr_path: Optional standard-error log path.
        stream: Whether output should be streamed while captured.
        sensitive_env_keys: Environment keys that must be redacted in reports.

    Returns:
        Normalized command spec.

    """
    return CommandSpec(
        args=tuple(str(argument) for argument in args),
        cwd=cwd,
        env=dict(env or {}),
        timeout_seconds=timeout_seconds,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        stream=stream,
        sensitive_env_keys=tuple(str(key) for key in sensitive_env_keys),
    )


def redacted_environment_overrides(spec: CommandSpec) -> dict[str, str]:
    """Return environment overrides with sensitive values redacted.

    Args:
        spec: Command specification.

    Returns:
        Redacted environment override mapping.

    """
    sensitive_keys = set(spec.sensitive_env_keys)
    return {
        key: REDACTED_ENVIRONMENT_VALUE if key in sensitive_keys else value
        for key, value in sorted(spec.env.items(), key=lambda item: item[0])
    }


def build_process_environment(spec: CommandSpec) -> dict[str, str]:
    """Build the child-process environment.

    Args:
        spec: Command specification.

    Returns:
        Process environment.

    """
    environment = dict(os.environ)
    environment.update(spec.env)
    return environment


def write_command_logs(spec: CommandSpec, stdout: str, stderr: str) -> None:
    """Write configured command log files.

    Args:
        spec: Command specification.
        stdout: Captured standard output.
        stderr: Captured standard error.

    """
    if spec.stdout_path is not None:
        spec.stdout_path.parent.mkdir(parents=True, exist_ok=True)
        spec.stdout_path.write_text(stdout, encoding="utf-8")
    if spec.stderr_path is not None:
        spec.stderr_path.parent.mkdir(parents=True, exist_ok=True)
        spec.stderr_path.write_text(stderr, encoding="utf-8")


def run_command(spec: CommandSpec) -> CommandResult:
    """Run a command and return a structured result without raising.

    Args:
        spec: Command specification.

    Returns:
        Command result.

    """
    if spec.stream:
        return run_streaming_command(spec)
    try:
        completed_process = subprocess.run(
            list(spec.args),
            check=False,
            capture_output=True,
            text=True,
            env=build_process_environment(spec),
            cwd=spec.cwd,
            timeout=spec.timeout_seconds,
        )
    except FileNotFoundError as error:
        result = build_exception_result(spec, error, timed_out=False, missing_executable=True)
    except subprocess.TimeoutExpired as error:
        stdout = subprocess_output_text(error.stdout)
        stderr = subprocess_output_text(error.stderr)
        result = CommandResult(
            args=spec.args,
            return_code=None,
            stdout=stdout,
            stderr=stderr,
            timed_out=True,
            missing_executable=False,
            cwd=str(spec.cwd) if spec.cwd is not None else None,
            environment_overrides=redacted_environment_overrides(spec),
        )
    else:
        result = CommandResult(
            args=spec.args,
            return_code=completed_process.returncode,
            stdout=completed_process.stdout,
            stderr=completed_process.stderr,
            timed_out=False,
            missing_executable=False,
            cwd=str(spec.cwd) if spec.cwd is not None else None,
            environment_overrides=redacted_environment_overrides(spec),
        )
    write_command_logs(spec, result.stdout, result.stderr)
    return result


def run_streaming_command(spec: CommandSpec) -> CommandResult:
    """Run a command while streaming combined output to stdout.

    Args:
        spec: Command specification.

    Returns:
        Command result.

    """
    stdout_parts: list[str] = []
    try:
        process = subprocess.Popen(
            list(spec.args),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=build_process_environment(spec),
            cwd=spec.cwd,
        )
    except FileNotFoundError as error:
        result = build_exception_result(spec, error, timed_out=False, missing_executable=True)
        write_command_logs(spec, result.stdout, result.stderr)
        return result
    deadline = time.monotonic() + spec.timeout_seconds if spec.timeout_seconds is not None else None
    try:
        if process.stdout is not None:
            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(spec.args, typing.cast("float", spec.timeout_seconds))
                remaining_timeout = 0.1
                if deadline is not None:
                    remaining_timeout = max(0.0, min(remaining_timeout, deadline - time.monotonic()))
                readable_streams, _, _ = select.select([process.stdout], [], [], remaining_timeout)
                if readable_streams:
                    raw_chunk = os.read(process.stdout.fileno(), 4096)
                    if raw_chunk:
                        chunk = subprocess_output_text(raw_chunk)
                        print(chunk, end="")
                        stdout_parts.append(chunk)
                        continue
                if process.poll() is not None:
                    remaining_output = subprocess_output_text(process.stdout.read())
                    if remaining_output:
                        print(remaining_output, end="")
                        stdout_parts.append(remaining_output)
                    break
        return_code = process.wait(timeout=0)
        result = CommandResult(
            args=spec.args,
            return_code=return_code,
            stdout="".join(stdout_parts),
            stderr="",
            timed_out=False,
            missing_executable=False,
            cwd=str(spec.cwd) if spec.cwd is not None else None,
            environment_overrides=redacted_environment_overrides(spec),
        )
    except subprocess.TimeoutExpired:
        process.kill()
        remaining_stdout, _ = process.communicate()
        if remaining_stdout:
            decoded_stdout = subprocess_output_text(remaining_stdout)
            print(decoded_stdout, end="")
            stdout_parts.append(decoded_stdout)
        result = CommandResult(
            args=spec.args,
            return_code=None,
            stdout="".join(stdout_parts),
            stderr=f"Command timed out after {spec.timeout_seconds} seconds.",
            timed_out=True,
            missing_executable=False,
            cwd=str(spec.cwd) if spec.cwd is not None else None,
            environment_overrides=redacted_environment_overrides(spec),
        )
    write_command_logs(spec, result.stdout, result.stderr)
    return result


def build_exception_result(
    spec: CommandSpec,
    error: Exception,
    *,
    timed_out: bool,
    missing_executable: bool,
) -> CommandResult:
    """Build a result for a command that did not complete normally.

    Args:
        spec: Command specification.
        error: Execution exception.
        timed_out: Whether the command timed out.
        missing_executable: Whether the executable was missing.

    Returns:
        Command result.

    """
    return CommandResult(
        args=spec.args,
        return_code=None,
        stdout="",
        stderr=str(error),
        timed_out=timed_out,
        missing_executable=missing_executable,
        cwd=str(spec.cwd) if spec.cwd is not None else None,
        environment_overrides=redacted_environment_overrides(spec),
    )


def subprocess_output_text(output: str | bytes | None) -> str:
    """Return subprocess output as text.

    Args:
        output: Raw subprocess output.

    Returns:
        Text output.

    """
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


def command_result_to_output(result: CommandResult) -> CommandOutput:
    """Adapt a command result to the legacy captured-output shape.

    Args:
        result: Command result.

    Returns:
        Legacy command output.

    """
    return CommandOutput(
        command_arguments=list(result.args),
        return_code=result.return_code,
        stdout=result.stdout,
        stderr=result.stderr,
    )


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
    spec = build_command_spec(
        command_arguments,
        cwd=cwd,
        env=environment_overrides,
    )
    return command_result_to_output(run_command(spec))


def command_result_to_json_dict(result: CommandResult) -> dict[str, object]:
    """Convert a command result to a JSON-ready dictionary.

    Args:
        result: Command result.

    Returns:
        JSON-ready result.

    """
    return dataclasses.asdict(result)


def command_record_from_result(
    *,
    command_id: str,
    tool_name: str,
    run_id: str,
    phase: str,
    spec: CommandSpec,
    result: CommandResult,
    output_directory: Path,
    started_at: str | None = None,
    finished_at: str | None = None,
    wall_time_seconds: float | None = None,
) -> tooling_artifact_format.CommandRecord:
    """Convert a command result into a Tooling Artifact Format command record.

    Args:
        command_id: Stable command identifier.
        tool_name: Tool name.
        run_id: Tool run identifier.
        phase: Tool phase.
        spec: Command specification.
        result: Command result.
        output_directory: Artifact output directory.
        started_at: Optional start timestamp.
        finished_at: Optional finish timestamp.
        wall_time_seconds: Optional wall time.

    Returns:
        Command ledger record.

    """
    status = tooling_artifact_format.ToolArtifactStatus.SUCCESS
    if result.timed_out:
        status = tooling_artifact_format.ToolArtifactStatus.TIMED_OUT
    elif result.missing_executable or result.return_code not in (0, None):
        status = tooling_artifact_format.ToolArtifactStatus.FAILED
    return tooling_artifact_format.build_command_record(
        command_id=command_id,
        tool_name=tool_name,
        run_id=run_id,
        phase=phase,
        args=result.args,
        output_directory=output_directory,
        cwd=spec.cwd,
        environment_overrides=result.environment_overrides,
        redacted_environment_keys=spec.sensitive_env_keys,
        stdout_log=spec.stdout_path,
        stderr_log=spec.stderr_path,
        status=status,
        return_code=result.return_code,
        started_at=started_at,
        finished_at=finished_at,
        wall_time_seconds=wall_time_seconds,
    )
