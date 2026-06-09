#!/usr/bin/env python3
"""Verify that the Python stub mirrors top-level `_core` registrations."""

from __future__ import annotations

import re
import sys
from pathlib import Path


RUST_EXPORT_FILES = (
    Path("src/python/mod.rs"),
    Path("src/python/config/mod.rs"),
)
STUB_FILE = Path("src/g/_core.pyi")

CLASS_PATTERN = re.compile(r"add_class::<([A-Za-z_][A-Za-z0-9_]*)>")
FUNC_PATTERN = re.compile(r"add_function\(wrap_pyfunction!\(([A-Za-z_][A-Za-z0-9_]*)")
ALLOWED_STUB_ONLY_CLASSES = {"ChunkStatsComputeArrays"}


def read_rust_exports(paths: tuple[Path, ...]) -> tuple[set[str], set[str]]:
    rust_classes: set[str] = set()
    rust_functions: set[str] = set()

    for path in paths:
        text = path.read_text()
        rust_classes.update(CLASS_PATTERN.findall(text))
        rust_functions.update(FUNC_PATTERN.findall(text))

    return rust_classes, rust_functions


def read_stub_exports(stub_path: Path) -> tuple[set[str], set[str]]:
    stub_classes: set[str] = set()
    stub_functions: set[str] = set()

    for line in stub_path.read_text().splitlines():
        if line.startswith("class "):
            stub_classes.add(line.removeprefix("class ").split("(", maxsplit=1)[0].removesuffix(":"))
        if line.startswith("def "):
            stub_functions.add(line.removeprefix("def ").split("(", maxsplit=1)[0])

    return stub_classes, stub_functions


def format_list(values: set[str]) -> str:
    if not values:
        return "none"
    return "\n".join(f"  - {value}" for value in sorted(values))


def main() -> int:
    rust_classes, rust_functions = read_rust_exports(RUST_EXPORT_FILES)
    stub_classes, stub_functions = read_stub_exports(STUB_FILE)

    missing_classes = sorted(rust_classes.difference(stub_classes))
    extra_classes = sorted(stub_classes.difference(rust_classes).difference(ALLOWED_STUB_ONLY_CLASSES))
    missing_functions = sorted(rust_functions.difference(stub_functions))
    extra_functions = sorted(stub_functions.difference(rust_functions))

    if missing_classes or extra_classes or missing_functions or extra_functions:
        print("Mismatch between Rust `_core` registrations and `src/g/_core.pyi`:")
        print(f"  Rust-only classes ({len(missing_classes)}):")
        print(format_list(set(missing_classes)))
        print(f"  Stub-only classes ({len(extra_classes)}):")
        print(format_list(set(extra_classes)))
        print(f"  Rust-only functions ({len(missing_functions)}):")
        print(format_list(set(missing_functions)))
        print(f"  Stub-only functions ({len(extra_functions)}):")
        print(format_list(set(extra_functions)))
        return 1

    print("`src/g/_core.pyi` matches top-level Rust `_core` registrations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
