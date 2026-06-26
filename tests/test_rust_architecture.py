"""Architecture tests for Rust core and Python binding boundaries."""

from __future__ import annotations

import typing
from pathlib import Path

from tooling.debug import check_rust_architecture

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
RUST_SOURCE_DIRECTORY = REPOSITORY_ROOT / "src"
RUST_CRATE_DIRECTORY = REPOSITORY_ROOT / "crates"
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


def iter_rust_source_paths() -> typing.Iterator[Path]:
    """Yield Rust source files that should satisfy binding-isolation policy."""
    for rust_source_directory in (RUST_SOURCE_DIRECTORY, RUST_CRATE_DIRECTORY):
        if not rust_source_directory.exists():
            continue
        yield from sorted(rust_source_directory.rglob("*.rs"))


def test_python_binding_markers_are_isolated_to_python_modules() -> None:
    violations: list[str] = []
    for rust_source_path in iter_rust_source_paths():
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


def build_package_payload(package_name: str, dependency_names: tuple[str, ...]) -> dict[str, typing.Any]:
    """Build a small Cargo metadata package payload for architecture tests."""
    return {
        "id": f"path+file:///test/{package_name}#0.1.0",
        "name": package_name,
        "dependencies": [{"name": dependency_name} for dependency_name in dependency_names],
    }


def build_metadata_payload(packages: tuple[dict[str, typing.Any], ...]) -> dict[str, typing.Any]:
    """Build a small Cargo metadata payload for architecture tests."""
    return {
        "packages": list(packages),
        "workspace_members": [str(package_payload["id"]) for package_payload in packages],
    }


def test_rust_architecture_policy_allows_current_single_package_workspace() -> None:
    metadata_payload = build_metadata_payload((build_package_payload("g", ("pyo3", "numpy", "arrow")),))

    assert check_rust_architecture.collect_rust_architecture_violations(metadata_payload) == ()


def test_rust_architecture_policy_rejects_python_binding_dependencies_outside_root() -> None:
    metadata_payload = build_metadata_payload(
        (
            build_package_payload("g", ("g-genotype", "pyo3")),
            build_package_payload("g-genotype", ("pyo3",)),
        )
    )

    violations = check_rust_architecture.collect_rust_architecture_violations(metadata_payload)

    assert violations == (
        check_rust_architecture.RustArchitectureViolation(
            package_name="g-genotype",
            dependency_name="pyo3",
            message="only the root `g` package may depend on PyO3 or NumPy crates",
        ),
    )


def test_rust_architecture_policy_rejects_forbidden_internal_dependencies() -> None:
    metadata_payload = build_metadata_payload(
        (
            build_package_payload("g", ("g-interface",)),
            build_package_payload("g-interface", ("g-genotype", "g-plan")),
            build_package_payload("g-genotype", ()),
            build_package_payload("g-plan", ()),
        )
    )

    violations = check_rust_architecture.collect_rust_architecture_violations(metadata_payload)

    assert violations == (
        check_rust_architecture.RustArchitectureViolation(
            package_name="g-interface",
            dependency_name="g-genotype",
            message="workspace package depends on a forbidden internal crate",
        ),
    )


def test_rust_architecture_policy_requires_new_internal_crates_to_declare_policy() -> None:
    metadata_payload = build_metadata_payload(
        (
            build_package_payload("g", ("g-surprise",)),
            build_package_payload("g-surprise", ()),
        )
    )

    violations = check_rust_architecture.collect_rust_architecture_violations(metadata_payload)

    assert violations == (
        check_rust_architecture.RustArchitectureViolation(
            package_name="g-surprise",
            dependency_name="*",
            message="workspace package has no declared Rust architecture policy",
        ),
    )
