"""Tests for maintained Python module reachability and source coverage."""

from __future__ import annotations

import typing

from tooling.debug import check_dead_code

if typing.TYPE_CHECKING:
    from pathlib import Path


def write_source(repository_root: Path, relative_path: str, source: str) -> None:
    """Write one bounded source or configuration fixture."""
    source_path = repository_root / relative_path
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(source, encoding="utf-8")


def declare_console(repository_root: Path) -> None:
    """Declare the same explicit console root used by the installed package."""
    write_source(repository_root, "pyproject.toml", '[project.scripts]\ng = "g.cli:main"\n')
    write_source(repository_root, "src/g/cli.py", "def main() -> None:\n    pass\n")


def test_orphan_cycle_and_self_declared_main_are_unreachable(tmp_path: Path) -> None:
    """Mutual imports or a main guard do not invent an external entrypoint."""
    declare_console(tmp_path)
    write_source(tmp_path, "tooling/orphan.py", "import tooling.other\nif __name__ == '__main__':\n    pass\n")
    write_source(tmp_path, "tooling/other.py", "import tooling.orphan\n")

    scan = check_dead_code.collect_module_scan(tmp_path)

    assert [str(path) for path in scan.unreachable_paths] == ["tooling/orphan.py", "tooling/other.py"]


def test_rust_dispatched_backend_and_jax_imports_are_live(tmp_path: Path) -> None:
    """Native import strings root numerical modules and their decorated kernels."""
    declare_console(tmp_path)
    write_source(tmp_path, "src/binding/engine.rs", 'PyModule::import(py, "g.jax_backend")?;\n')
    write_source(tmp_path, "src/g/jax_backend.py", "from g.compute import score\n")
    write_source(tmp_path, "src/g/compute/score.py", "import jax\n@jax.jit\ndef score():\n    pass\n")

    scan = check_dead_code.collect_module_scan(tmp_path)

    assert "g.jax_backend" in scan.root_modules
    assert scan.unreachable_paths == ()


def test_hydra_registry_and_literal_subprocess_modules_are_live(tmp_path: Path) -> None:
    """The grouped CLI reaches registered owners and literal module commands."""
    declare_console(tmp_path)
    write_source(tmp_path, "Justfile", "check:\n    python -m tooling.cli.debug --config-name debug\n")
    write_source(tmp_path, "tooling/cli/debug.py", "from tooling.debug import guard\nTOOLS = {'guard': guard.run}\n")
    write_source(tmp_path, "tooling/debug/guard.py", "def run():\n    return ['python', '-m', 'tooling.worker']\n")
    write_source(tmp_path, "tooling/worker.py", "def main():\n    pass\n")

    scan = check_dead_code.collect_module_scan(tmp_path)

    assert "tooling.cli.debug" in scan.root_modules
    assert scan.unreachable_paths == ()


def test_configured_target_and_importlib_module_are_live(tmp_path: Path) -> None:
    """Explicit Hydra targets and literal importlib calls preserve dynamic paths."""
    declare_console(tmp_path)
    write_source(tmp_path, "tooling/configs/worker.yaml", "_target_: tooling.worker.run\n")
    write_source(tmp_path, "tooling/worker.py", "import importlib\nimportlib.import_module('tooling.support')\n")
    write_source(tmp_path, "tooling/support.py", "VALUE = 1\n")

    assert check_dead_code.collect_module_scan(tmp_path).unreachable_paths == ()


def test_diagnostic_strings_do_not_make_orphans_live(tmp_path: Path) -> None:
    """An arbitrary string mentioning a module is not an execution edge."""
    declare_console(tmp_path)
    write_source(tmp_path, "src/g/cli.py", "MESSAGE = 'tooling.orphan'\n")
    write_source(tmp_path, "tooling/orphan.py", "VALUE = 1\n")

    scan = check_dead_code.collect_module_scan(tmp_path)

    assert [str(path) for path in scan.unreachable_paths] == ["tooling/orphan.py"]


def test_package_initializers_are_followed_but_empty_markers_are_allowed(tmp_path: Path) -> None:
    """Package imports keep initializer dependencies without flagging markers."""
    declare_console(tmp_path)
    write_source(tmp_path, "src/g/__init__.py", "import tooling.support\n")
    write_source(tmp_path, "tooling/support.py", "VALUE = 1\n")
    write_source(tmp_path, "tooling/markers/__init__.py", '"""Package marker."""\n')

    assert check_dead_code.collect_module_scan(tmp_path).unreachable_paths == ()


def test_supported_standalone_data_modules_are_live(tmp_path: Path) -> None:
    """The reviewed direct data interfaces remain roots without grouped callers."""
    declare_console(tmp_path)
    write_source(tmp_path, "tooling/data/fetch.py", "def main():\n    pass\n")
    write_source(tmp_path, "tooling/data/simulate.py", "def main():\n    pass\n")

    scan = check_dead_code.collect_module_scan(tmp_path)

    assert {"tooling.data.fetch", "tooling.data.simulate"} <= scan.root_modules
    assert scan.unreachable_paths == ()


def test_missing_or_narrow_source_scope_fails_closed(tmp_path: Path) -> None:
    """A successful parser with no real repository coverage is not a passing scan."""
    declare_console(tmp_path)

    scan = check_dead_code.collect_module_scan(tmp_path)
    errors = check_dead_code.source_coverage_errors(tmp_path, scan)

    assert any("missing maintained source anchor" in error for error in errors)
    assert any("production files" in error for error in errors)
    assert any("tooling files" in error for error in errors)
    assert check_dead_code.run_tool(tmp_path) == 1


def test_orphan_initializer_with_behavior_is_unreachable(tmp_path: Path) -> None:
    """Only empty package markers receive the initializer exemption."""
    declare_console(tmp_path)
    write_source(tmp_path, "tooling/orphan/__init__.py", "VALUE = 1\n")

    scan = check_dead_code.collect_module_scan(tmp_path)

    assert [str(path) for path in scan.unreachable_paths] == ["tooling/orphan/__init__.py"]
