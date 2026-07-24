"""Tests for Rust workspace dependency architecture checks."""

from __future__ import annotations

import typing

from tooling.debug import check_rust_architecture


def dependency_payload(name: str, kind: str | None) -> dict[str, typing.Any]:
    """Build a Cargo metadata dependency fixture."""
    return {"name": name, "kind": kind}


def package_payload(
    name: str,
    production_dependencies: tuple[str, ...],
    development_dependencies: tuple[str, ...],
) -> dict[str, typing.Any]:
    """Build a Cargo metadata package fixture."""
    dependencies = [dependency_payload(dependency_name, None) for dependency_name in production_dependencies]
    dependencies.extend(dependency_payload(dependency_name, "dev") for dependency_name in development_dependencies)
    return {"name": name, "dependencies": dependencies}


def current_workspace_metadata() -> dict[str, typing.Any]:
    """Build the documented production dependency graph."""
    return {
        "packages": [
            package_payload("g", (), ()),
            package_payload("g-compute-cuda", (), ()),
            package_payload("g-genotype-contracts", (), ()),
            package_payload("g-genotype-cuda", ("g-genotype-contracts",), ()),
            package_payload("g-genotype", ("g-genotype-contracts",), ()),
            package_payload("g-input", ("g-plan",), ()),
            package_payload("g-interface", ("g-genotype-contracts", "g-plan"), ()),
            package_payload("g-output", ("g-genotype-contracts", "g-plan"), ()),
            package_payload("g-plan", ("g-genotype-contracts",), ()),
            package_payload("g-runtime", (), ()),
            package_payload(
                "g-engine",
                ("g-genotype", "g-genotype-contracts", "g-input", "g-output", "g-plan", "g-runtime"),
                (),
            ),
            package_payload(
                "g-runner",
                ("g-engine", "g-interface", "g-plan", "g-runtime"),
                ("g-genotype", "g-input"),
            ),
        ]
    }


def test_current_workspace_dependency_graph_passes() -> None:
    """Current owner crates and test-only runner fixtures satisfy policy."""
    violations = check_rust_architecture.collect_rust_architecture_violations(current_workspace_metadata())

    assert violations == ()


def test_new_cuda_cross_dependency_is_rejected() -> None:
    """The independent CUDA crates cannot depend on each other."""
    metadata_payload = current_workspace_metadata()
    packages = typing.cast("list[dict[str, typing.Any]]", metadata_payload["packages"])
    genotype_cuda_package = next(package for package in packages if package["name"] == "g-genotype-cuda")
    dependencies = typing.cast("list[dict[str, typing.Any]]", genotype_cuda_package["dependencies"])
    dependencies.append(dependency_payload("g-compute-cuda", None))

    violations = check_rust_architecture.collect_rust_architecture_violations(metadata_payload)

    assert len(violations) == 1
    assert violations[0].package_name == "g-genotype-cuda"
    assert violations[0].dependency_name == "g-compute-cuda"
    assert violations[0].message == "workspace package depends on a forbidden internal crate"


def test_development_pyo3_dependency_remains_rejected() -> None:
    """Test-only dependencies cannot bypass the native binding boundary."""
    metadata_payload = current_workspace_metadata()
    packages = typing.cast("list[dict[str, typing.Any]]", metadata_payload["packages"])
    runner_package = next(package for package in packages if package["name"] == "g-runner")
    dependencies = typing.cast("list[dict[str, typing.Any]]", runner_package["dependencies"])
    dependencies.append(dependency_payload("pyo3", "dev"))

    violations = check_rust_architecture.collect_rust_architecture_violations(metadata_payload)

    assert len(violations) == 1
    assert violations[0].package_name == "g-runner"
    assert violations[0].dependency_name == "pyo3"
    assert violations[0].message == "only the root `g` package may depend on PyO3 or NumPy crates"
