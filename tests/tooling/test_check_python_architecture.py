"""Tests for Python package ownership architecture checks."""

from __future__ import annotations

import typing

from tooling.debug import check_python_architecture

if typing.TYPE_CHECKING:
    from pathlib import Path


CURRENT_CLI_SOURCE = """
import sys
import typing

import g._core


def run(arguments: typing.Sequence[str]) -> int:
    result = g._core.cli.run(arguments)
    for output_text in result.stdout_chunks:
        print(output_text, end="")
    for output_text in result.stderr_chunks:
        print(output_text, end="", file=sys.stderr)
    return result.exit_code


def main() -> None:
    raise SystemExit(run(sys.argv[1:]))
"""


def write_cli(package_root: Path, source: str) -> None:
    """Write one CLI fixture under a temporary package root."""
    package_root.mkdir(parents=True)
    (package_root / "cli.py").write_text(source, encoding="utf-8")


def test_direct_native_cli_shim_passes(tmp_path: Path) -> None:
    """The supported Python launcher delegates directly to the native CLI."""
    package_root = tmp_path / "src/g"
    write_cli(package_root, CURRENT_CLI_SOURCE)

    violations = check_python_architecture.collect_python_cli_shim_violations(package_root)

    assert violations == ()


def test_removed_python_runner_import_is_rejected(tmp_path: Path) -> None:
    """The CLI cannot restore the removed Python runner lifecycle."""
    package_root = tmp_path / "src/g"
    write_cli(package_root, f"import g.runner.cli\n{CURRENT_CLI_SOURCE}")

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    runner_violations = [violation for violation in violations if violation.forbidden_import == "g.runner"]
    assert len(runner_violations) == 1
    assert runner_violations[0].import_name == "g.runner.cli"


def test_legacy_run_args_shim_is_rejected(tmp_path: Path) -> None:
    """The former Python parser and validated-run bridge cannot return."""
    package_root = tmp_path / "src/g"
    write_cli(
        package_root,
        """
def run_args(arguments: list[str]) -> int:
    parsed = dispatch_cli(arguments)
    return run_validated_cli_outcome(parsed)


def main() -> None:
    raise SystemExit(run_args([]))
""",
    )

    violations = check_python_architecture.collect_python_cli_shim_violations(package_root)

    assert {violation.subject for violation in violations} == {"run", "run_args", "main"}


def test_indirect_native_cli_call_is_rejected(tmp_path: Path) -> None:
    """The public run function cannot delegate through another Python owner."""
    package_root = tmp_path / "src/g"
    write_cli(
        package_root,
        """
def run(arguments: list[str]) -> int:
    return python_runner(arguments)


def main() -> None:
    raise SystemExit(run([]))
""",
    )

    violations = check_python_architecture.collect_python_cli_shim_violations(package_root)

    assert len(violations) == 1
    assert violations[0].subject == "run"
