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


def write_numerical_source(package_root: Path, relative_path: str, source: str) -> None:
    """Write a kernel or backend fixture with its owning directories."""
    source_path = package_root / relative_path
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(source, encoding="utf-8")


def test_backend_cannot_invoke_native_cli(tmp_path: Path) -> None:
    """The backend cannot recursively take ownership of the CLI lifecycle."""
    package_root = tmp_path / "src/g"
    write_cli(package_root, CURRENT_CLI_SOURCE)
    write_numerical_source(package_root, "jax_backend.py", "import g._core as native\nnative.cli.run([])\n")

    imports = check_python_architecture.collect_python_import_policy_violations(package_root)
    calls = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [violation.import_name for violation in imports] == ["g._core"]
    assert [violation.call_name for violation in calls] == ["g._core.cli.run"]


def test_backend_helper_cannot_import_python_orchestration(tmp_path: Path) -> None:
    """The extracted backend package inherits the same numerical boundaries."""
    package_root = tmp_path / "src/g"
    write_numerical_source(package_root, "backend/transport.py", "from g import cli\n")

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert [violation.import_name for violation in violations] == ["g.cli"]


def test_backend_file_io_alias_is_rejected(tmp_path: Path) -> None:
    """Module aliases cannot hide host file loading in transport helpers."""
    package_root = tmp_path / "src/g"
    write_numerical_source(package_root, "backend/transport.py", "import numpy as arrays\narrays.load('input.npy')\n")

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [violation.call_name for violation in violations] == ["numpy.load"]


def test_directly_imported_runtime_setup_call_is_rejected(tmp_path: Path) -> None:
    """Direct imports and aliases cannot bypass Rust runtime setup ownership."""
    package_root = tmp_path / "src/g"
    write_numerical_source(
        package_root,
        "jax_backend.py",
        "from jax.config import update as configure\nconfigure('jax_enable_x64', True)\n",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [violation.call_name for violation in violations] == ["jax.config.update"]


def test_compute_cannot_materialize_or_import_transport(tmp_path: Path) -> None:
    """Kernels cannot depend on their callers or materialize host output."""
    package_root = tmp_path / "src/g"
    write_numerical_source(
        package_root,
        "compute/score.py",
        "import g.backend.transport\nimport jax as numerical\nnumerical.device_get(None)\n",
    )

    imports = check_python_architecture.collect_python_import_policy_violations(package_root)
    calls = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [violation.import_name for violation in imports] == ["g.backend.transport"]
    assert [violation.call_name for violation in calls] == ["jax.device_get"]


def test_backend_materialization_remains_supported(tmp_path: Path) -> None:
    """The backend may transfer and materialize arrays while calling kernels."""
    package_root = tmp_path / "src/g"
    write_numerical_source(
        package_root,
        "backend/materialization.py",
        "import jax\nfrom g.compute.common import result\njax.device_get(result)\n",
    )

    assert check_python_architecture.collect_python_import_policy_violations(package_root) == ()
    assert check_python_architecture.collect_python_call_policy_violations(package_root) == ()


def test_architecture_gate_rejects_missing_current_sources(tmp_path: Path) -> None:
    """A CLI-only or empty snapshot cannot report a passing ownership check."""
    package_root = tmp_path / "src/g"
    write_cli(package_root, CURRENT_CLI_SOURCE)

    assert check_python_architecture.run_tool(package_root) == 1


def test_numerical_file_and_worker_boundaries_are_rejected(tmp_path: Path) -> None:
    """Kernels and transport cannot restore file handling or Python worker pools."""
    package_root = tmp_path / "src/g"
    write_numerical_source(
        package_root,
        "compute/score.py",
        "with open('input.txt') as stream:\n    values = stream.read()\n",
    )
    write_numerical_source(
        package_root,
        "backend/transport.py",
        "from concurrent import futures\nfrom pathlib import Path\nPath('input.txt').read_text()\n",
    )

    imports = check_python_architecture.collect_python_import_policy_violations(package_root)
    calls = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert {violation.import_name for violation in imports} == {"concurrent.futures", "pathlib.Path"}
    assert {violation.call_name for violation in calls} == {"open", "read_text"}


def test_nested_import_alias_cannot_hide_module_call(tmp_path: Path) -> None:
    """A local helper import cannot change the enclosing module's bindings."""
    package_root = tmp_path / "src/g"
    write_numerical_source(
        package_root,
        "jax_backend.py",
        "import numpy as arrays\narrays.load('input.npy')\ndef helper():\n    import math as arrays\n",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [violation.call_name for violation in violations] == ["numpy.load"]


def test_sibling_function_import_aliases_are_isolated(tmp_path: Path) -> None:
    """A sibling's local import cannot hide kernel host materialization."""
    package_root = tmp_path / "src/g"
    write_numerical_source(
        package_root,
        "compute/score.py",
        "def invalid():\n    from jax import device_get as materialize\n    materialize(None)\n"
        "def valid():\n    from numpy import array as materialize\n    materialize([1])\n",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [violation.call_name for violation in violations] == ["jax.device_get"]


def test_class_import_does_not_shadow_method_global_alias(tmp_path: Path) -> None:
    """Method bodies use enclosing lexical bindings rather than class imports."""
    package_root = tmp_path / "src/g"
    write_numerical_source(
        package_root,
        "jax_backend.py",
        "import numpy as arrays\nclass Backend:\n    import math as arrays\n"
        "    def prepare(self):\n        arrays.load('input.npy')\n",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [violation.call_name for violation in violations] == ["numpy.load"]


def test_same_scope_import_rebinding_is_checked_conservatively(tmp_path: Path) -> None:
    """Repeated imports cannot hide a forbidden target behind an earlier alias."""
    package_root = tmp_path / "src/g"
    write_numerical_source(
        package_root,
        "compute/score.py",
        "import math as numerical\nimport jax as numerical\nnumerical.device_get(None)\n",
    )

    violations = check_python_architecture.collect_python_call_policy_violations(package_root)

    assert [violation.call_name for violation in violations] == ["jax.device_get"]


def test_function_parameter_and_assignment_shadow_imports(tmp_path: Path) -> None:
    """Explicit numerical operand bindings do not inherit a module's alias."""
    package_root = tmp_path / "src/g"
    write_numerical_source(
        package_root,
        "jax_backend.py",
        "import numpy as arrays\ndef parameter(arrays):\n    arrays.load(None)\n"
        "def assignment(operand):\n    arrays = operand\n    arrays.load(None)\n",
    )

    assert check_python_architecture.collect_python_call_policy_violations(package_root) == ()


def test_lambda_and_comprehension_bindings_shadow_imports(tmp_path: Path) -> None:
    """Expression-local operands do not resolve to unrelated imported aliases."""
    package_root = tmp_path / "src/g"
    write_numerical_source(
        package_root,
        "jax_backend.py",
        "import numpy as arrays\noperation = lambda arrays: arrays.load(None)\n"
        "values = [arrays.load(None) for arrays in operands]\n",
    )

    assert check_python_architecture.collect_python_call_policy_violations(package_root) == ()
