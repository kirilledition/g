from __future__ import annotations

import textwrap
import typing

from tooling.debug import check_internal_init_exports

if typing.TYPE_CHECKING:
    from pathlib import Path

    import pytest


def write_module(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")


def test_internal_init_export_checker_reports_internal_exports(tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "g"
    write_module(
        source_root / "__init__.py",
        """
        import importlib

        def __getattr__(name: str) -> object:
            return importlib.import_module(name)
        """,
    )
    write_module(
        source_root / "engine" / "__init__.py",
        """
        \"\"\"Engine package.\"\"\"

        from g.engine import callbacks

        RuntimeState = object
        """,
    )
    write_module(
        source_root / "jax_runtime" / "__init__.py",
        """
        \"\"\"JAX runtime package.\"\"\"

        __all__ = ("models",)
        """,
    )

    violations = check_internal_init_exports.collect_package_init_violations(source_root)

    observed = {(violation.kind, violation.statement_summary) for violation in violations}
    assert observed == {
        (
            check_internal_init_exports.InternalInitExportViolationKind.INTERNAL_IMPORT,
            "g.engine: callbacks",
        ),
        (
            check_internal_init_exports.InternalInitExportViolationKind.INTERNAL_ASSIGNMENT,
            "RuntimeState",
        ),
        (
            check_internal_init_exports.InternalInitExportViolationKind.ALL_DECLARATION,
            "__all__",
        ),
    }


def test_internal_init_export_checker_rejects_top_level_all(tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "g"
    write_module(
        source_root / "__init__.py",
        """
        __all__ = ("regenie",)
        """,
    )

    violations = check_internal_init_exports.collect_package_init_violations(source_root)

    assert len(violations) == 1
    assert violations[0].kind == check_internal_init_exports.InternalInitExportViolationKind.ALL_DECLARATION


def test_internal_init_export_checker_allows_package_markers(tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "g"
    write_module(
        source_root / "__init__.py",
        """
        import importlib

        def __getattr__(name: str) -> object:
            return importlib.import_module(name)
        """,
    )
    write_module(
        source_root / "engine" / "__init__.py",
        """
        \"\"\"Engine package marker.\"\"\"
        """,
    )

    assert check_internal_init_exports.collect_package_init_violations(source_root) == ()


def test_internal_init_export_checker_run_tool_returns_failure_for_violations(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_root = tmp_path / "src" / "g"
    write_module(
        source_root / "engine" / "__init__.py",
        """
        from g.engine import callbacks
        """,
    )

    exit_code = check_internal_init_exports.run_tool(source_root)
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "internal package initializer imports `g.engine: callbacks`" in captured.out
