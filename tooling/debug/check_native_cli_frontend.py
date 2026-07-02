#!/usr/bin/env python3
"""Check native CLI frontend parity and startup timing."""

from __future__ import annotations

import statistics
import subprocess
import sys
import time
import typing
from dataclasses import dataclass
from pathlib import Path

import hydra

from tooling.common import hydra_compat as tooling_hydra_compat

if typing.TYPE_CHECKING:
    import omegaconf

PYTHON_BRIDGE_SCRIPT = "import g.cli, sys; raise SystemExit(g.cli.run_args(sys.argv[1:]))"
TEXT_SAMPLE_LIMIT = 500


@dataclass(frozen=True)
class CommandSpec:
    """One configless CLI command to compare.

    Attributes:
        name: Stable case name.
        arguments: Command-line arguments after the executable name.

    """

    name: str
    arguments: tuple[str, ...]


@dataclass(frozen=True)
class CommandResult:
    """Observed result for one CLI subprocess invocation.

    Attributes:
        exit_code: Process exit code.
        stdout: Captured stdout text.
        stderr: Captured stderr text.
        duration_seconds: Wall-clock duration.

    """

    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float


@dataclass(frozen=True)
class CaseComparison:
    """Parity and timing result for one CLI command case.

    Attributes:
        command: Command specification.
        native_results: Native binary subprocess results.
        python_bridge_results: Python bridge subprocess results.
        mismatch_messages: Output or exit-code mismatch diagnostics.

    """

    command: CommandSpec
    native_results: tuple[CommandResult, ...]
    python_bridge_results: tuple[CommandResult, ...]
    mismatch_messages: tuple[str, ...]


@dataclass(frozen=True)
class NativeCliFrontendCheckArguments:
    """Resolved arguments for the native CLI frontend check.

    Attributes:
        native_binary_path: Path to the compiled native `g` binary.
        python_executable_path: Python executable used for the bridge command.
        trial_count: Number of measured subprocess trials per case.
        warmup_count: Number of warmup subprocess trials per case.

    """

    native_binary_path: Path
    python_executable_path: Path
    trial_count: int
    warmup_count: int


CONFIGLESS_COMMANDS = (
    CommandSpec(name="root_help", arguments=("--help",)),
    CommandSpec(name="regenie_help", arguments=("regenie", "--help")),
    CommandSpec(name="parse_error", arguments=("regenie", "--bad-option")),
    CommandSpec(name="unknown_command", arguments=("unknown",)),
)


def build_arguments_from_config(config: omegaconf.DictConfig) -> NativeCliFrontendCheckArguments:
    """Build resolved native CLI frontend check arguments."""
    python_executable_value = config.tool.python_executable_path
    python_executable_path = (
        Path(sys.executable) if python_executable_value is None else Path(str(python_executable_value))
    )
    return NativeCliFrontendCheckArguments(
        native_binary_path=Path(str(config.tool.native_binary_path)),
        python_executable_path=python_executable_path,
        trial_count=int(config.tool.trial_count),
        warmup_count=int(config.tool.warmup_count),
    )


def run_native_command(native_binary_path: Path, command: CommandSpec) -> CommandResult:
    """Run one native CLI command."""
    return run_subprocess((str(native_binary_path), *command.arguments))


def run_python_bridge_command(python_executable_path: Path, command: CommandSpec) -> CommandResult:
    """Run one Python bridge CLI command."""
    return run_subprocess((str(python_executable_path), "-c", PYTHON_BRIDGE_SCRIPT, *command.arguments))


def run_subprocess(command_arguments: tuple[str, ...]) -> CommandResult:
    """Run one subprocess and capture output plus wall-clock duration."""
    start_time = time.perf_counter()
    completed_process = subprocess.run(
        command_arguments,
        check=False,
        capture_output=True,
        text=True,
    )
    duration_seconds = time.perf_counter() - start_time
    return CommandResult(
        exit_code=completed_process.returncode,
        stdout=completed_process.stdout,
        stderr=completed_process.stderr,
        duration_seconds=duration_seconds,
    )


def compare_command_results(native_result: CommandResult, python_bridge_result: CommandResult) -> tuple[str, ...]:
    """Return parity mismatch messages for one native/Python result pair."""
    mismatch_messages: list[str] = []
    if native_result.exit_code != python_bridge_result.exit_code:
        mismatch_messages.append(
            f"exit code mismatch: native={native_result.exit_code}, python={python_bridge_result.exit_code}"
        )
    if native_result.stdout != python_bridge_result.stdout:
        mismatch_messages.append(
            "stdout mismatch:\n"
            f"native={format_text_sample(native_result.stdout)}\n"
            f"python={format_text_sample(python_bridge_result.stdout)}"
        )
    if native_result.stderr != python_bridge_result.stderr:
        mismatch_messages.append(
            "stderr mismatch:\n"
            f"native={format_text_sample(native_result.stderr)}\n"
            f"python={format_text_sample(python_bridge_result.stderr)}"
        )
    return tuple(mismatch_messages)


def format_text_sample(value: str) -> str:
    """Render a bounded representation of subprocess output."""
    if len(value) <= TEXT_SAMPLE_LIMIT:
        return repr(value)
    return repr(f"{value[:TEXT_SAMPLE_LIMIT]}...<truncated {len(value) - TEXT_SAMPLE_LIMIT} chars>")


def compare_case(arguments: NativeCliFrontendCheckArguments, command: CommandSpec) -> CaseComparison:
    """Run warmup and measured trials for one command case."""
    for _ in range(arguments.warmup_count):
        run_native_command(arguments.native_binary_path, command)
        run_python_bridge_command(arguments.python_executable_path, command)

    native_results: list[CommandResult] = []
    python_bridge_results: list[CommandResult] = []
    mismatch_messages: list[str] = []
    for trial_index in range(arguments.trial_count):
        native_result = run_native_command(arguments.native_binary_path, command)
        python_bridge_result = run_python_bridge_command(arguments.python_executable_path, command)
        native_results.append(native_result)
        python_bridge_results.append(python_bridge_result)
        for message in compare_command_results(native_result, python_bridge_result):
            mismatch_messages.append(f"trial {trial_index + 1}: {message}")

    return CaseComparison(
        command=command,
        native_results=tuple(native_results),
        python_bridge_results=tuple(python_bridge_results),
        mismatch_messages=tuple(mismatch_messages),
    )


def median_duration_seconds(results: tuple[CommandResult, ...]) -> float:
    """Return the median subprocess duration."""
    return float(statistics.median(result.duration_seconds for result in results))


def validate_arguments(arguments: NativeCliFrontendCheckArguments) -> tuple[str, ...]:
    """Validate paths and trial counts before running subprocesses."""
    errors: list[str] = []
    if not arguments.native_binary_path.is_file():
        errors.append(f"native binary does not exist: {arguments.native_binary_path}")
    if not arguments.python_executable_path.is_file():
        errors.append(f"python executable does not exist: {arguments.python_executable_path}")
    if arguments.trial_count < 1:
        errors.append(f"trial_count must be at least 1, got {arguments.trial_count}")
    if arguments.warmup_count < 0:
        errors.append(f"warmup_count must be non-negative, got {arguments.warmup_count}")
    return tuple(errors)


def run_tool(arguments: NativeCliFrontendCheckArguments) -> int:
    """Run the native CLI frontend parity and startup check."""
    validation_errors = validate_arguments(arguments)
    if validation_errors:
        print("Native CLI frontend check configuration is invalid:")
        for validation_error in validation_errors:
            print(f"  {validation_error}")
        return 1

    comparisons = tuple(compare_case(arguments, command) for command in CONFIGLESS_COMMANDS)
    print(
        "Native CLI frontend process checkpoint "
        f"using {arguments.native_binary_path} and {arguments.python_executable_path}:"
    )
    for comparison in comparisons:
        native_median_seconds = median_duration_seconds(comparison.native_results)
        python_median_seconds = median_duration_seconds(comparison.python_bridge_results)
        print(
            f"  {comparison.command.name}: "
            f"native_median={native_median_seconds:.6f}s, "
            f"python_bridge_median={python_median_seconds:.6f}s"
        )
        for mismatch_message in comparison.mismatch_messages:
            print(f"    mismatch: {mismatch_message}")

    if any(comparison.mismatch_messages for comparison in comparisons):
        return 1
    print("Native CLI frontend check passed.")
    return 0


@hydra.main(version_base=None, config_path="../configs", config_name="debug_check_native_cli_frontend")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Run the native CLI frontend check from Hydra configuration."""
    exit_code = run_tool(build_arguments_from_config(config))
    if exit_code:
        raise SystemExit(exit_code)


def main() -> int:
    """Run the native CLI frontend check from default Hydra configuration."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
