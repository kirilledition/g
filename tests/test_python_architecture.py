"""Architecture tests for Python package ownership boundaries."""

from __future__ import annotations

from pathlib import Path

from tooling.debug import check_python_architecture

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
PRODUCTION_PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "g"


def test_python_import_policy_allows_current_production_tree() -> None:
    assert check_python_architecture.collect_python_import_policy_violations(PRODUCTION_PACKAGE_ROOT) == ()


def test_compute_import_policy_rejects_cli_output_and_config_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    compute_directory = package_root / "compute"
    compute_directory.mkdir(parents=True)
    (compute_directory / "kernel.py").write_text(
        "\n".join(
            (
                "from g import cli",
                "from g.io import output",
                "import g.interface.config",
                "from ..io import source",
                "from g.io import *",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.import_name, violation.forbidden_import)
        for violation in violations
    ] == [
        (Path("g/compute/kernel.py"), 1, "g.cli", "g.cli"),
        (Path("g/compute/kernel.py"), 2, "g.io.output", "g.io"),
        (Path("g/compute/kernel.py"), 3, "g.interface.config", "g.interface"),
        (Path("g/compute/kernel.py"), 4, "g.io.source", "g.io"),
        (Path("g/compute/kernel.py"), 5, "g.io", "g.io"),
    ]


def test_jax_runtime_import_policy_rejects_runner_orchestration_imports(tmp_path: Path) -> None:
    package_root = tmp_path / "g"
    jax_runtime_directory = package_root / "jax_runtime"
    jax_runtime_directory.mkdir(parents=True)
    (jax_runtime_directory / "setup.py").write_text(
        "\n".join(
            (
                "from g.runner import runtime",
                "import g.runner.cli",
            )
        ),
        encoding="utf-8",
    )

    violations = check_python_architecture.collect_python_import_policy_violations(package_root)

    assert [
        (violation.path, violation.line_number, violation.import_name, violation.forbidden_import)
        for violation in violations
    ] == [
        (Path("g/jax_runtime/setup.py"), 1, "g.runner.runtime", "g.runner"),
        (Path("g/jax_runtime/setup.py"), 2, "g.runner.cli", "g.runner"),
    ]
