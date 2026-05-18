"""Architecture tests for Rust core and Python binding boundaries."""

from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
RUST_SOURCE_DIRECTORY = REPOSITORY_ROOT / "src"
PYTHON_BINDING_DIRECTORY = RUST_SOURCE_DIRECTORY / "python"
PYTHON_BINDING_MARKERS = (
    "pyo3",
    "numpy::",
    "#[pyclass",
    "#[pymethods",
    "#[pyfunction",
    "PyArray",
    "PyReadonly",
    "PyReadwrite",
    "PyResult",
    "PyRef",
)


def test_python_binding_markers_are_isolated_to_python_modules() -> None:
    violations: list[str] = []
    for rust_source_path in sorted(RUST_SOURCE_DIRECTORY.rglob("*.rs")):
        if rust_source_path == RUST_SOURCE_DIRECTORY / "lib.rs":
            continue
        if rust_source_path.is_relative_to(PYTHON_BINDING_DIRECTORY):
            continue
        rust_source_text = rust_source_path.read_text(encoding="utf-8")
        observed_markers = [marker for marker in PYTHON_BINDING_MARKERS if marker in rust_source_text]
        if observed_markers:
            relative_source_path = rust_source_path.relative_to(REPOSITORY_ROOT)
            violations.append(f"{relative_source_path}: {', '.join(observed_markers)}")

    assert violations == []
