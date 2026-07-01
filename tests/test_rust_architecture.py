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


def test_root_crate_boundary_policy_allows_current_private_adapter() -> None:
    assert check_rust_architecture.collect_root_crate_boundary_violations(REPOSITORY_ROOT) == ()


def test_python_telemetry_fallback_policy_allows_current_adapter() -> None:
    assert check_rust_architecture.collect_python_telemetry_fallback_violations(REPOSITORY_ROOT) == ()


def test_root_crate_boundary_policy_rejects_public_domain_reexports(tmp_path: Path) -> None:
    root_source_directory = tmp_path / "src"
    python_source_directory = root_source_directory / "python"
    python_source_directory.mkdir(parents=True)
    (root_source_directory / "lib.rs").write_text(
        "\n".join(
            (
                "pub use g_engine as engine;",
                "pub mod python;",
                "fn _core() {}",
            )
        ),
        encoding="utf-8",
    )
    (python_source_directory / "mod.rs").write_text("pub fn register_module() {}\n", encoding="utf-8")

    violations = check_rust_architecture.collect_root_crate_boundary_violations(tmp_path)

    assert violations == (
        check_rust_architecture.RootCrateBoundaryViolation(
            source_path=Path("src/lib.rs"),
            marker="pub use g_",
            message="root crate must not re-export internal domain crates as public Rust aliases",
        ),
        check_rust_architecture.RootCrateBoundaryViolation(
            source_path=Path("src/lib.rs"),
            marker="pub mod python;",
            message="root crate must keep its internal PyO3 adapter module private",
        ),
        check_rust_architecture.RootCrateBoundaryViolation(
            source_path=Path("src/lib.rs"),
            marker="mod python;",
            message="root crate must declare the internal PyO3 adapter module privately",
        ),
        check_rust_architecture.RootCrateBoundaryViolation(
            source_path=Path("src/python/mod.rs"),
            marker="pub(crate) fn register_module",
            message="root PyO3 adapter registration must be crate-private",
        ),
    )


def test_python_telemetry_fallback_policy_rejects_rust_to_python_dispatch(tmp_path: Path) -> None:
    python_source_directory = tmp_path / "src" / "python"
    python_source_directory.mkdir(parents=True)
    (python_source_directory / "telemetry.rs").write_text(
        "\n".join(
            (
                'session.call_method1("log_run_failed", ());',
                'session.call_method0("close_with_event");',
                'session.call_method("log_jax_runtime_diagnostic_event", ());',
            )
        ),
        encoding="utf-8",
    )

    violations = check_rust_architecture.collect_python_telemetry_fallback_violations(tmp_path)

    assert violations == (
        check_rust_architecture.PythonTelemetryFallbackViolation(
            source_path=Path("src/python/telemetry.rs"),
            method_name="log_run_failed",
            line_number=1,
            message=check_rust_architecture.PYTHON_TELEMETRY_FALLBACK_MESSAGE,
        ),
        check_rust_architecture.PythonTelemetryFallbackViolation(
            source_path=Path("src/python/telemetry.rs"),
            method_name="close_with_event",
            line_number=2,
            message=check_rust_architecture.PYTHON_TELEMETRY_FALLBACK_MESSAGE,
        ),
        check_rust_architecture.PythonTelemetryFallbackViolation(
            source_path=Path("src/python/telemetry.rs"),
            method_name="log_jax_runtime_diagnostic_event",
            line_number=3,
            message=check_rust_architecture.PYTHON_TELEMETRY_FALLBACK_MESSAGE,
        ),
    )


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
