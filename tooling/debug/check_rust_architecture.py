#!/usr/bin/env python3
"""Verify Rust workspace dependency boundaries."""

from __future__ import annotations

import json
import subprocess
import sys
import typing
from dataclasses import dataclass
from pathlib import Path

ROOT_PACKAGE_NAME = "g"
RESTRICTED_PYTHON_NATIVE_DEPENDENCIES = {"numpy", "pyo3"}
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

ALLOWED_INTERNAL_DEPENDENCIES_BY_PACKAGE: dict[str, set[str]] = {
    "g-plan": set(),
    "g-interface": {"g-plan"},
    "g-genotype": set(),
    "g-input": {"g-genotype", "g-plan"},
    "g-output": {"g-plan"},
    "g-runtime": {"g-plan"},
    "g-engine": {"g-genotype", "g-input", "g-output", "g-plan", "g-runtime"},
}


@dataclass(frozen=True)
class RustArchitectureViolation:
    """A Rust workspace dependency-policy violation.

    Attributes:
        package_name: Workspace package containing the violation.
        dependency_name: Dependency that violates the policy.
        message: Human-readable violation description.

    """

    package_name: str
    dependency_name: str
    message: str


def workspace_packages(metadata_payload: dict[str, typing.Any]) -> tuple[dict[str, typing.Any], ...]:
    """Return Cargo metadata package payloads for workspace members."""
    raw_packages = typing.cast("list[dict[str, typing.Any]]", metadata_payload.get("packages", []))
    raw_workspace_members = typing.cast("list[str]", metadata_payload.get("workspace_members", []))
    if not raw_workspace_members:
        return tuple(raw_packages)

    workspace_member_identifiers = set(raw_workspace_members)
    return tuple(
        package_payload
        for package_payload in raw_packages
        if str(package_payload.get("id", "")) in workspace_member_identifiers
    )


def dependency_names(package_payload: dict[str, typing.Any]) -> tuple[str, ...]:
    """Return dependency package names from one Cargo metadata package payload."""
    raw_dependencies = typing.cast("list[dict[str, typing.Any]]", package_payload.get("dependencies", []))
    return tuple(str(dependency_payload.get("name", "")) for dependency_payload in raw_dependencies)


def collect_rust_architecture_violations(
    metadata_payload: dict[str, typing.Any],
) -> tuple[RustArchitectureViolation, ...]:
    """Collect Rust workspace dependency-policy violations."""
    package_payloads = workspace_packages(metadata_payload)
    workspace_package_names = {
        str(package_payload.get("name", ""))
        for package_payload in package_payloads
        if isinstance(package_payload.get("name"), str)
    }
    violations: list[RustArchitectureViolation] = []

    for package_payload in package_payloads:
        package_name = str(package_payload.get("name", ""))
        package_dependency_names = set(dependency_names(package_payload))

        if package_name != ROOT_PACKAGE_NAME:
            for dependency_name in sorted(package_dependency_names.intersection(RESTRICTED_PYTHON_NATIVE_DEPENDENCIES)):
                violations.append(
                    RustArchitectureViolation(
                        package_name=package_name,
                        dependency_name=dependency_name,
                        message="only the root `g` package may depend on PyO3 or NumPy crates",
                    )
                )

        internal_dependency_names = package_dependency_names.intersection(workspace_package_names)
        if package_name == ROOT_PACKAGE_NAME:
            continue

        allowed_dependency_names = ALLOWED_INTERNAL_DEPENDENCIES_BY_PACKAGE.get(package_name)
        if allowed_dependency_names is None:
            violations.append(
                RustArchitectureViolation(
                    package_name=package_name,
                    dependency_name="*",
                    message="workspace package has no declared Rust architecture policy",
                )
            )
            continue

        for dependency_name in sorted(internal_dependency_names.difference(allowed_dependency_names)):
            violations.append(
                RustArchitectureViolation(
                    package_name=package_name,
                    dependency_name=dependency_name,
                    message="workspace package depends on a forbidden internal crate",
                )
            )

    return tuple(violations)


def load_cargo_metadata(repository_root: Path) -> dict[str, typing.Any]:
    """Load workspace Cargo metadata from the repository root."""
    completed_process = subprocess.run(
        ["cargo", "metadata", "--format-version=1", "--no-deps"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed_process.returncode != 0:
        message = (
            "cargo metadata failed while checking Rust architecture.\n"
            f"stdout:\n{completed_process.stdout}\n"
            f"stderr:\n{completed_process.stderr}"
        )
        raise RuntimeError(message)

    return typing.cast("dict[str, typing.Any]", json.loads(completed_process.stdout))


def format_violations(violations: tuple[RustArchitectureViolation, ...]) -> str:
    """Format architecture violations for command-line output."""
    return "\n".join(
        f"- {violation.package_name} -> {violation.dependency_name}: {violation.message}" for violation in violations
    )


def run_tool(repository_root: Path) -> int:
    """Verify Rust workspace architecture policy."""
    try:
        metadata_payload = load_cargo_metadata(repository_root)
    except (FileNotFoundError, RuntimeError, json.JSONDecodeError) as error:
        print(error)
        return 1

    violations = collect_rust_architecture_violations(metadata_payload)
    if violations:
        print("Rust workspace architecture violations:")
        print(format_violations(violations))
        return 1

    print("Rust workspace architecture policy passed.")
    return 0


def main() -> int:
    """Run the Rust architecture checker from the repository root."""
    return run_tool(REPOSITORY_ROOT)


if __name__ == "__main__":
    sys.exit(main())
