#!/usr/bin/env python3
"""Verify that the Python stub mirrors `_core` registrations."""

from __future__ import annotations

import ast
import re
import sys
import typing
from dataclasses import dataclass
from pathlib import Path

import hydra

from tooling.common import hydra_compat as tooling_hydra_compat

if typing.TYPE_CHECKING:
    import collections.abc

    import omegaconf

BINDING_SOURCE_DIRECTORY = Path("src/binding")
BINDING_REGISTRATION_FILE = BINDING_SOURCE_DIRECTORY / "mod.rs"
STUB_FILE = Path("src/g/_core.pyi")

CLASS_PATTERN = re.compile(r"add_class::\s*<\s*(?P<export_name>[A-Za-z_][A-Za-z0-9_]*)\s*>")
FUNCTION_PATTERN = re.compile(r"add_function\s*\(\s*wrap_pyfunction!\(\s*(?P<export_name>[A-Za-z_][A-Za-z0-9_]*)")
SUBMODULE_REGISTRATION_PATTERN = re.compile(
    r"let\s+full_name\s*=\s*format!\(\s*\"\{\}\.(?P<python_module>[A-Za-z_][A-Za-z0-9_]*)\".*?"
    r"(?P<rust_module>[A-Za-z_][A-Za-z0-9_]*)::register_module\(\s*&submodule\s*\)\?;",
    re.DOTALL,
)


@dataclass(frozen=True)
class ExportSurface:
    """Classes, functions, and module namespaces on a native Python surface.

    Attributes:
        classes: Qualified exported class names.
        functions: Qualified exported function names.
        module_names: Registered Python submodule names.

    """

    classes: frozenset[str]
    functions: frozenset[str]
    module_names: frozenset[str]


def rust_module_source_path(registration_path: Path, rust_module: str) -> Path:
    """Resolve a Rust module registered by the binding root."""
    flat_module_path = registration_path.parent / f"{rust_module}.rs"
    if flat_module_path.is_file():
        return flat_module_path
    return registration_path.parent / rust_module / "mod.rs"


def qualified_export_name(module_name: str, export_name: str) -> str:
    """Qualify an export with its Python module when one is present."""
    if not module_name:
        return export_name
    return f"{module_name}.{export_name}"


def read_rust_exports(registration_path: Path) -> ExportSurface:
    """Read qualified PyO3 exports reachable from the Rust binding root."""
    rust_classes: set[str] = set()
    rust_functions: set[str] = set()
    module_names: set[str] = set()

    registration_text = registration_path.read_text(encoding="utf-8")
    rust_classes.update(CLASS_PATTERN.findall(registration_text))
    rust_functions.update(FUNCTION_PATTERN.findall(registration_text))

    for registration_match in SUBMODULE_REGISTRATION_PATTERN.finditer(registration_text):
        python_module = registration_match.group("python_module")
        rust_module = registration_match.group("rust_module")
        module_names.add(python_module)
        module_text = rust_module_source_path(registration_path, rust_module).read_text(encoding="utf-8")
        rust_classes.update(
            qualified_export_name(python_module, export_name) for export_name in CLASS_PATTERN.findall(module_text)
        )
        rust_functions.update(
            qualified_export_name(python_module, export_name) for export_name in FUNCTION_PATTERN.findall(module_text)
        )

    return ExportSurface(
        classes=frozenset(rust_classes),
        functions=frozenset(rust_functions),
        module_names=frozenset(module_names),
    )


def read_stub_exports(stub_path: Path, module_names: frozenset[str]) -> ExportSurface:
    """Read qualified classes and functions from the `_core` Python stub."""
    stub_classes: set[str] = set()
    stub_functions: set[str] = set()

    syntax_tree = ast.parse(stub_path.read_text(encoding="utf-8"), filename=str(stub_path))
    for statement in syntax_tree.body:
        if isinstance(statement, ast.ClassDef) and statement.name in module_names:
            for module_statement in statement.body:
                if isinstance(module_statement, ast.ClassDef):
                    stub_classes.add(qualified_export_name(statement.name, module_statement.name))
                elif isinstance(module_statement, ast.FunctionDef | ast.AsyncFunctionDef):
                    stub_functions.add(qualified_export_name(statement.name, module_statement.name))
        elif isinstance(statement, ast.ClassDef):
            stub_classes.add(statement.name)
        elif isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
            stub_functions.add(statement.name)

    return ExportSurface(
        classes=frozenset(stub_classes),
        functions=frozenset(stub_functions),
        module_names=module_names,
    )


def format_list(values: collections.abc.Set[str]) -> str:
    """Format a set of names as a human-readable indented list."""
    if not values:
        return "none"
    return "\n".join(f"  - {value}" for value in sorted(values))


def run_tool() -> int:
    """Verify that Rust `_core` registrations match the Python stub."""
    rust_exports = read_rust_exports(BINDING_REGISTRATION_FILE)
    stub_exports = read_stub_exports(STUB_FILE, rust_exports.module_names)

    missing_classes = rust_exports.classes.difference(stub_exports.classes)
    extra_classes = stub_exports.classes.difference(rust_exports.classes)
    missing_functions = rust_exports.functions.difference(stub_exports.functions)
    extra_functions = stub_exports.functions.difference(rust_exports.functions)

    if missing_classes or extra_classes or missing_functions or extra_functions:
        print("Mismatch between Rust `_core` registrations and `src/g/_core.pyi`:")
        print(f"  Rust-only classes ({len(missing_classes)}):")
        print(format_list(missing_classes))
        print(f"  Stub-only classes ({len(extra_classes)}):")
        print(format_list(extra_classes))
        print(f"  Rust-only functions ({len(missing_functions)}):")
        print(format_list(missing_functions))
        print(f"  Stub-only functions ({len(extra_functions)}):")
        print(format_list(extra_functions))
        return 1

    print("`src/g/_core.pyi` matches Rust `_core` registrations.")
    return 0


@hydra.main(version_base=None, config_path="../configs", config_name="debug_check_pyo3_stub")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Run the stub checker from Hydra configuration."""
    del config
    exit_code = run_tool()
    if exit_code:
        raise SystemExit(exit_code)


def main() -> int:
    """Run the stub checker from default Hydra configuration."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
