#!/usr/bin/env python3
"""Verify that production Python code avoids implicit defaults."""

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

PRODUCTION_SOURCE_ROOT = Path("src/g")


class InternalDefaultViolationKind(enum.StrEnum):
    """Kinds of implicit defaults rejected in production Python."""

    FUNCTION_PARAMETER = "function_parameter"
    DATACLASS_FIELD = "dataclass_field"


@dataclasses.dataclass(frozen=True)
class InternalDefaultViolation:
    """An implicit default found in production Python source."""

    path: Path
    line_number: int
    column_offset: int
    kind: InternalDefaultViolationKind
    qualified_name: str
    subject_name: str


def is_named_expression(expression: ast.AST, name: str) -> bool:
    """Return whether an AST expression refers to a named symbol or attribute."""
    if isinstance(expression, ast.Name):
        return expression.id == name
    if isinstance(expression, ast.Attribute):
        return expression.attr == name
    return False


def is_named_decorator(decorator: ast.expr, name: str) -> bool:
    """Return whether a decorator applies a named decorator function."""
    expression = decorator.func if isinstance(decorator, ast.Call) else decorator
    return is_named_expression(expression, name)


def is_dataclass_node(class_node: ast.ClassDef) -> bool:
    """Return whether a class is decorated as a dataclass."""
    return any(is_named_decorator(decorator, "dataclass") for decorator in class_node.decorator_list)


def is_dataclass_field_call(expression: ast.AST) -> bool:
    """Return whether an expression calls dataclasses.field."""
    if not isinstance(expression, ast.Call):
        return False
    return is_named_expression(expression.func, "field")


def dataclass_field_call_has_default(expression: ast.AST) -> bool:
    """Return whether a dataclasses.field call provides a default value."""
    if not isinstance(expression, ast.Call):
        return False
    return any(keyword.arg in {"default", "default_factory"} for keyword in expression.keywords)


def annassign_target_name(statement: ast.AnnAssign) -> str | None:
    """Return the simple field name from an annotated assignment."""
    if isinstance(statement.target, ast.Name):
        return statement.target.id
    return None


def format_qualified_name(scope: tuple[str, ...], name: str) -> str:
    """Format a scoped function, class, or field name."""
    return ".".join((*scope, name))


def collect_function_default_violations(
    path: Path,
    function_node: ast.FunctionDef | ast.AsyncFunctionDef,
    scope: tuple[str, ...],
) -> list[InternalDefaultViolation]:
    """Collect implicit default parameter violations from one function."""
    violations: list[InternalDefaultViolation] = []
    qualified_name = format_qualified_name(scope, function_node.name)
    positional_arguments = (*function_node.args.posonlyargs, *function_node.args.args)
    first_default_index = len(positional_arguments) - len(function_node.args.defaults)

    for argument_index, argument in enumerate(positional_arguments):
        if argument_index >= first_default_index:
            violations.append(
                InternalDefaultViolation(
                    path=path,
                    line_number=argument.lineno,
                    column_offset=argument.col_offset,
                    kind=InternalDefaultViolationKind.FUNCTION_PARAMETER,
                    qualified_name=qualified_name,
                    subject_name=argument.arg,
                )
            )

    for argument, default_value in zip(function_node.args.kwonlyargs, function_node.args.kw_defaults, strict=True):
        if default_value is not None:
            violations.append(
                InternalDefaultViolation(
                    path=path,
                    line_number=argument.lineno,
                    column_offset=argument.col_offset,
                    kind=InternalDefaultViolationKind.FUNCTION_PARAMETER,
                    qualified_name=qualified_name,
                    subject_name=argument.arg,
                )
            )

    return violations


def collect_dataclass_default_violations(
    path: Path,
    class_node: ast.ClassDef,
    scope: tuple[str, ...],
) -> list[InternalDefaultViolation]:
    """Collect implicit default field violations from one dataclass."""
    if not is_dataclass_node(class_node):
        return []

    violations: list[InternalDefaultViolation] = []
    qualified_name = format_qualified_name(scope, class_node.name)
    for statement in class_node.body:
        if not isinstance(statement, ast.AnnAssign) or statement.value is None:
            continue
        field_name = annassign_target_name(statement)
        if field_name is None:
            continue
        if is_dataclass_field_call(statement.value) and not dataclass_field_call_has_default(statement.value):
            continue
        violations.append(
            InternalDefaultViolation(
                path=path,
                line_number=statement.lineno,
                column_offset=statement.col_offset,
                kind=InternalDefaultViolationKind.DATACLASS_FIELD,
                qualified_name=qualified_name,
                subject_name=field_name,
            )
        )
    return violations


def collect_nested_violations(path: Path, node: ast.AST, scope: tuple[str, ...]) -> list[InternalDefaultViolation]:
    """Collect implicit defaults from nested classes and functions."""
    violations: list[InternalDefaultViolation] = []
    for child_node in ast.iter_child_nodes(node):
        if isinstance(child_node, ast.FunctionDef | ast.AsyncFunctionDef):
            function_scope = (*scope, child_node.name)
            violations.extend(collect_function_default_violations(path, child_node, scope))
            violations.extend(collect_nested_violations(path, child_node, function_scope))
            continue
        if isinstance(child_node, ast.ClassDef):
            class_scope = (*scope, child_node.name)
            violations.extend(collect_dataclass_default_violations(path, child_node, scope))
            violations.extend(collect_nested_violations(path, child_node, class_scope))
            continue
        violations.extend(collect_nested_violations(path, child_node, scope))
    return violations


def collect_internal_default_violations(source_root: Path) -> tuple[InternalDefaultViolation, ...]:
    """Collect implicit default violations under a production source root."""
    violations: list[InternalDefaultViolation] = []
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        violations.extend(collect_nested_violations(path, tree, ()))
    return tuple(violations)


def render_violation(violation: InternalDefaultViolation) -> str:
    """Render a violation for command-line output."""
    location = f"{violation.path}:{violation.line_number}:{violation.column_offset + 1}"
    if violation.kind == InternalDefaultViolationKind.FUNCTION_PARAMETER:
        return (
            f"{location}: function `{violation.qualified_name}` parameter "
            f"`{violation.subject_name}` has a default value"
        )
    if violation.kind == InternalDefaultViolationKind.DATACLASS_FIELD:
        return (
            f"{location}: dataclass `{violation.qualified_name}` field `{violation.subject_name}` has a default value"
        )
    typing.assert_never(violation.kind)


def run_tool(source_root: Path) -> int:
    """Verify that production Python source contains no implicit defaults."""
    violations = collect_internal_default_violations(source_root)
    if violations:
        print(f"Implicit defaults are not allowed under `{source_root}`:")
        for violation in violations:
            print(f"  {render_violation(violation)}")
        return 1

    print(f"No implicit defaults found under `{source_root}`.")
    return 0


@hydra.main(version_base=None, config_path="../configs", config_name="debug_check_internal_defaults")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Run the internal-default checker from Hydra configuration."""
    del config
    exit_code = run_tool(PRODUCTION_SOURCE_ROOT)
    if exit_code:
        raise SystemExit(exit_code)


def main() -> int:
    """Run the internal-default checker from default Hydra configuration."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
