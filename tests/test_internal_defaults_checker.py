from __future__ import annotations

import textwrap
import typing

from tooling.debug import check_internal_defaults

if typing.TYPE_CHECKING:
    from pathlib import Path

    import pytest


def write_module(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")


def test_internal_default_checker_reports_function_and_dataclass_defaults(tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "g"
    write_module(
        source_root / "example.py",
        """
        import dataclasses
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class Config:
            value: int = 1
            generated: list[int] = dataclasses.field(default_factory=list)
            metadata_only: int = dataclasses.field(metadata={"column": "id"})

        def build(value: int = 1, *, name: str | None = None) -> None:
            pass

        class Container:
            def method(self, enabled: bool = False) -> None:
                pass
        """,
    )

    violations = check_internal_defaults.collect_internal_default_violations(source_root)

    observed = {(violation.kind, violation.qualified_name, violation.subject_name) for violation in violations}
    assert observed == {
        (
            check_internal_defaults.InternalDefaultViolationKind.DATACLASS_FIELD,
            "Config",
            "value",
        ),
        (
            check_internal_defaults.InternalDefaultViolationKind.DATACLASS_FIELD,
            "Config",
            "generated",
        ),
        (
            check_internal_defaults.InternalDefaultViolationKind.FUNCTION_PARAMETER,
            "build",
            "value",
        ),
        (
            check_internal_defaults.InternalDefaultViolationKind.FUNCTION_PARAMETER,
            "build",
            "name",
        ),
        (
            check_internal_defaults.InternalDefaultViolationKind.FUNCTION_PARAMETER,
            "Container.method",
            "enabled",
        ),
    }


def test_internal_default_checker_allows_explicit_signatures_and_metadata_only_fields(tmp_path: Path) -> None:
    source_root = tmp_path / "src" / "g"
    write_module(
        source_root / "example.py",
        """
        import dataclasses

        @dataclasses.dataclass(frozen=True)
        class Config:
            value: int
            metadata_only: int = dataclasses.field(metadata={"column": "id"})

        def build(*, value: int, name: str | None) -> None:
            pass
        """,
    )

    assert check_internal_defaults.collect_internal_default_violations(source_root) == ()


def test_internal_default_checker_run_tool_returns_failure_for_violations(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_root = tmp_path / "src" / "g"
    write_module(
        source_root / "example.py",
        """
        def build(value: int = 1) -> int:
            return value
        """,
    )

    exit_code = check_internal_defaults.run_tool(source_root)
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "function `build` parameter `value` has a default value" in captured.out
