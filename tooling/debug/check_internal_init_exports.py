#!/usr/bin/env python3
"""Verify that internal package initializers do not re-export symbols."""

from __future__ import annotations

import ast
import dataclasses
import enum
import sys
import typing
from pathlib import Path

import hydra

from tooling.common import hydra_compat as tooling_hydra_compat

if typing.TYPE_CHECKING:
    import omegaconf

PRODUCTION_PACKAGE_ROOT = Path("src/g")


class InternalInitExportViolationKind(enum.StrEnum):
    """Kinds of package initializer exports rejected by the checker."""

    ALL_DECLARATION = "all_declaration"
    INTERNAL_IMPORT = "internal_import"
    INTERNAL_ASSIGNMENT = "internal_assignment"
    INTERNAL_STATEMENT = "internal_statement"


@dataclasses.dataclass(frozen=True)
class InternalInitExportViolation:
    """A package initializer statement that exposes internal aliases."""

    path: Path
    line_number: int
    column_offset: int
    kind: InternalInitExportViolationKind
    statement_summary: str


def is_docstring_statement(statement: ast.stmt) -> bool:
    """Return whether a statement is the module docstring expression."""
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def assignment_target_names(target: ast.expr) -> tuple[str, ...]:
    """Return simple names assigned by one target expression."""
    if isinstance(target, ast.Name):
        return (target.id,)
    if isinstance(target, ast.Tuple | ast.List):
        names: list[str] = []
        for element in target.elts:
            names.extend(assignment_target_names(element))
        return tuple(names)
    return ()


def statement_assigns_all(statement: ast.stmt) -> bool:
    """Return whether a statement assigns the module `__all__` symbol."""
    if isinstance(statement, ast.Assign):
        return any("__all__" in assignment_target_names(target) for target in statement.targets)
    if isinstance(statement, ast.AnnAssign):
        return "__all__" in assignment_target_names(statement.target)
    if isinstance(statement, ast.AugAssign):
        return "__all__" in assignment_target_names(statement.target)
    return False


def classify_internal_statement(statement: ast.stmt) -> InternalInitExportViolationKind:
    """Classify a non-docstring statement in an internal package initializer."""
    if statement_assigns_all(statement):
        return InternalInitExportViolationKind.ALL_DECLARATION
    if isinstance(statement, ast.Import | ast.ImportFrom):
        return InternalInitExportViolationKind.INTERNAL_IMPORT
    if isinstance(statement, ast.Assign | ast.AnnAssign | ast.AugAssign):
        return InternalInitExportViolationKind.INTERNAL_ASSIGNMENT
    return InternalInitExportViolationKind.INTERNAL_STATEMENT


def summarize_statement(statement: ast.stmt) -> str:
    """Render a short statement summary for diagnostics."""
    if statement_assigns_all(statement):
        return "__all__"
    if isinstance(statement, ast.Import):
        return ", ".join(alias.name for alias in statement.names)
    if isinstance(statement, ast.ImportFrom):
        module_name = "" if statement.module is None else statement.module
        imported_names = ", ".join(alias.name for alias in statement.names)
        return f"{module_name}: {imported_names}"
    if isinstance(statement, ast.Assign):
        assigned_names: list[str] = []
        for target in statement.targets:
            assigned_names.extend(assignment_target_names(target))
        return ", ".join(assigned_names) or type(statement).__name__
    if isinstance(statement, ast.AnnAssign):
        assigned_names = assignment_target_names(statement.target)
        return ", ".join(assigned_names) or type(statement).__name__
    if isinstance(statement, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
        return statement.name
    return type(statement).__name__


def collect_statement_violation(path: Path, statement: ast.stmt) -> InternalInitExportViolation:
    """Build one initializer violation from an AST statement."""
    return InternalInitExportViolation(
        path=path,
        line_number=statement.lineno,
        column_offset=statement.col_offset,
        kind=classify_internal_statement(statement),
        statement_summary=summarize_statement(statement),
    )


def collect_package_init_violations(source_root: Path) -> tuple[InternalInitExportViolation, ...]:
    """Collect package initializer export violations under a production source root."""
    violations: list[InternalInitExportViolation] = []
    public_boundary_path = source_root / "__init__.py"
    for path in sorted(source_root.rglob("__init__.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for statement in tree.body:
            if is_docstring_statement(statement):
                continue
            if path == public_boundary_path:
                if statement_assigns_all(statement):
                    violations.append(collect_statement_violation(path, statement))
                continue
            violations.append(collect_statement_violation(path, statement))
    return tuple(violations)


def render_violation(violation: InternalInitExportViolation) -> str:
    """Render a package initializer violation for command-line output."""
    location = f"{violation.path}:{violation.line_number}:{violation.column_offset + 1}"
    if violation.kind == InternalInitExportViolationKind.ALL_DECLARATION:
        return f"{location}: package initializer defines `__all__`"
    if violation.kind == InternalInitExportViolationKind.INTERNAL_IMPORT:
        return f"{location}: internal package initializer imports `{violation.statement_summary}`"
    if violation.kind == InternalInitExportViolationKind.INTERNAL_ASSIGNMENT:
        return f"{location}: internal package initializer assigns `{violation.statement_summary}`"
    if violation.kind == InternalInitExportViolationKind.INTERNAL_STATEMENT:
        return f"{location}: internal package initializer contains `{violation.statement_summary}`"
    typing.assert_never(violation.kind)


def run_tool(source_root: Path) -> int:
    """Verify that internal package initializers are package markers only."""
    violations = collect_package_init_violations(source_root)
    if violations:
        print(f"Internal package initializers must not export aliases under `{source_root}`:")
        for violation in violations:
            print(f"  {render_violation(violation)}")
        return 1

    print(f"No internal package initializer exports found under `{source_root}`.")
    return 0


@hydra.main(version_base=None, config_path="../configs", config_name="debug_check_internal_init_exports")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Run the internal initializer export checker from Hydra configuration."""
    del config
    exit_code = run_tool(PRODUCTION_PACKAGE_ROOT)
    if exit_code:
        raise SystemExit(exit_code)


def main() -> int:
    """Run the internal initializer export checker from default Hydra configuration."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
