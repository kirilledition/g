#!/usr/bin/env python3
"""Verify Rust workspace dependency boundaries."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import typing
from dataclasses import dataclass
from pathlib import Path

ROOT_PACKAGE_NAME = "g"
RESTRICTED_PYTHON_NATIVE_DEPENDENCIES = {"numpy", "pyo3"}
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ROOT_CRATE_LIB_PATH = Path("src/lib.rs")
ROOT_CRATE_PYTHON_SOURCE_DIRECTORY = Path("src/python")
ROOT_CRATE_PYTHON_MODULE_PATH = Path("src/python/mod.rs")
DISALLOWED_ROOT_PYO3_EXPORT_NAMES = frozenset(
    (
        "NativeSecondSignalExceptionPlan",
        "NativeStageTimingRecorderPlan",
        "NativeTelemetryClosePlan",
        "NativeTelemetryEventEmissionPlan",
        "NativeTelemetryProgressEmissionPlan",
        "NativeTimingFileWritePlan",
        "build_current_run_manifest_header_json_from_input_json",
        "build_default_local_cache_directory_value",
        "build_file_content_sha256_value",
        "build_final_timing_outputs_write_started_diagnostic_payload",
        "build_jax_runtime_setup_diagnostic_payloads",
        "build_manifest_file_fingerprint_mapping_payload",
        "build_manifest_file_fingerprint_payload",
        "build_multi_run_artifacts_payload",
        "build_phenotype_run_artifacts_payload",
        "build_prediction_loco_file_fingerprints_json",
        "build_prepared_run_manifest_header_json",
        "build_prepared_run_plan_json",
        "build_run_manifest_extension_payload",
        "build_shutdown_signal_payload",
        "build_trusted_bgen_validation_cache_path_value",
        "build_trusted_bgen_validation_cache_payload",
        "build_trusted_bgen_validation_fingerprint_value",
        "complete_jax_runtime_setup_validation_payload",
        "configure_bgen_decode_tile_variant_count",
        "configure_rayon_global_thread_pool",
        "default_local_temporary_root_value",
        "default_shutdown_signal_numbers",
        "default_trusted_bgen_validation_cache_directory_value",
        "emit_diagnostic_event",
        "emit_diagnostic_event_fields",
        "format_rayon_thread_pool_configuration_error_value",
        "plan_jax_gpu_validation_payload",
        "plan_jax_runtime_config_update_payloads",
        "plan_jax_runtime_diagnostic_record",
        "plan_jax_runtime_diagnostic_record_payload",
        "plan_jax_runtime_setup_side_effects_payload",
        "plan_second_signal_exception",
        "plan_stage_timing_recorder",
        "plan_telemetry_close",
        "plan_telemetry_event_emission",
        "plan_telemetry_progress_emission",
        "plan_timing_file_write",
        "plan_trusted_bgen_validation_cache_lookup",
        "raise_second_signal_exception",
        "record_jax_runtime_diagnostic_log_event",
        "resolve_jax_runtime_setup_payload",
        "validate_binary_phenotype_case_control_counts",
        "validate_binary_phenotype_coding",
        "validate_finite_array",
        "write_trusted_bgen_validation_cache_payload",
    )
)
ROOT_PYO3_PYFUNCTION_EXPORT_PATTERN = re.compile(r"wrap_pyfunction!\s*\(\s*(?P<export_name>[A-Za-z0-9_]+)")
ROOT_PYO3_PYCLASS_EXPORT_PATTERN = re.compile(r"add_class::\s*<\s*(?P<export_name>[A-Za-z0-9_]+)\s*>")
ROOT_PYO3_REMOVED_EXPORT_MESSAGE = "root PyO3 adapter must not re-export removed raw helper surface"
PYTHON_TELEMETRY_FALLBACK_METHOD_NAMES = (
    "close_with_event",
    "log_binary_correction_summary",
    "log_callback_progress_event",
    "log_jax_runtime_diagnostic_event",
    "log_progress",
    "log_run_failed",
)
PYTHON_TELEMETRY_FALLBACK_MESSAGE = (
    "root PyO3 telemetry dispatch must use native telemetry handles, not Python fallback methods"
)
PYTHON_TELEMETRY_FALLBACK_CALL_PATTERN = re.compile(
    r"call_method(?:0|1)?\s*\(\s*\"(?P<method_name>"
    + "|".join(re.escape(method_name) for method_name in PYTHON_TELEMETRY_FALLBACK_METHOD_NAMES)
    + r")\""
)

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


@dataclass(frozen=True)
class RootCrateBoundaryViolation:
    """A root crate PyO3 adapter boundary violation.

    Attributes:
        source_path: Source file containing the violation.
        marker: Source marker that violates or proves the policy.
        message: Human-readable violation description.

    """

    source_path: Path
    marker: str
    message: str


@dataclass(frozen=True)
class PythonTelemetryFallbackViolation:
    """A root PyO3 adapter call back into Python telemetry fallback methods.

    Attributes:
        source_path: Source file containing the violation.
        method_name: Python telemetry fallback method called from Rust.
        line_number: One-based source line number containing the call.
        message: Human-readable violation description.

    """

    source_path: Path
    method_name: str
    line_number: int
    message: str


@dataclass(frozen=True)
class RootPyO3ExportViolation:
    """A removed root PyO3 export that was reintroduced.

    Attributes:
        source_path: Source file containing the violation.
        export_name: PyO3 export name that violates the policy.
        line_number: One-based source line number containing the export.
        message: Human-readable violation description.

    """

    source_path: Path
    export_name: str
    line_number: int
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


def collect_root_crate_boundary_violations(repository_root: Path) -> tuple[RootCrateBoundaryViolation, ...]:
    """Collect root crate PyO3 adapter boundary violations."""
    root_lib_path = repository_root / ROOT_CRATE_LIB_PATH
    root_python_module_path = repository_root / ROOT_CRATE_PYTHON_MODULE_PATH
    violations: list[RootCrateBoundaryViolation] = []

    try:
        root_lib_text = root_lib_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return (
            RootCrateBoundaryViolation(
                source_path=ROOT_CRATE_LIB_PATH,
                marker="missing",
                message="root crate library entrypoint is missing",
            ),
        )

    root_lib_lines = {source_line.strip() for source_line in root_lib_text.splitlines()}
    if any(source_line.startswith("pub use g_") for source_line in root_lib_lines):
        violations.append(
            RootCrateBoundaryViolation(
                source_path=ROOT_CRATE_LIB_PATH,
                marker="pub use g_",
                message="root crate must not re-export internal domain crates as public Rust aliases",
            )
        )
    if "pub mod python;" in root_lib_lines:
        violations.append(
            RootCrateBoundaryViolation(
                source_path=ROOT_CRATE_LIB_PATH,
                marker="pub mod python;",
                message="root crate must keep its internal PyO3 adapter module private",
            )
        )
    if "mod python;" not in root_lib_lines:
        violations.append(
            RootCrateBoundaryViolation(
                source_path=ROOT_CRATE_LIB_PATH,
                marker="mod python;",
                message="root crate must declare the internal PyO3 adapter module privately",
            )
        )

    try:
        root_python_module_text = root_python_module_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        violations.append(
            RootCrateBoundaryViolation(
                source_path=ROOT_CRATE_PYTHON_MODULE_PATH,
                marker="missing",
                message="root PyO3 adapter registration module is missing",
            )
        )
        return tuple(violations)

    if "pub(crate) fn register_module" not in root_python_module_text:
        violations.append(
            RootCrateBoundaryViolation(
                source_path=ROOT_CRATE_PYTHON_MODULE_PATH,
                marker="pub(crate) fn register_module",
                message="root PyO3 adapter registration must be crate-private",
            )
        )

    return tuple(violations)


def collect_python_telemetry_fallback_violations(repository_root: Path) -> tuple[PythonTelemetryFallbackViolation, ...]:
    """Collect root PyO3 adapter telemetry fallback dispatch violations."""
    root_python_source_directory = repository_root / ROOT_CRATE_PYTHON_SOURCE_DIRECTORY
    if not root_python_source_directory.exists():
        return ()

    violations: list[PythonTelemetryFallbackViolation] = []
    for rust_source_path in sorted(root_python_source_directory.rglob("*.rs")):
        source_text = rust_source_path.read_text(encoding="utf-8")
        relative_source_path = rust_source_path.relative_to(repository_root)
        for fallback_call_match in PYTHON_TELEMETRY_FALLBACK_CALL_PATTERN.finditer(source_text):
            line_number = source_text.count("\n", 0, fallback_call_match.start()) + 1
            method_name = fallback_call_match.group("method_name")
            violations.append(
                PythonTelemetryFallbackViolation(
                    source_path=relative_source_path,
                    method_name=method_name,
                    line_number=line_number,
                    message=PYTHON_TELEMETRY_FALLBACK_MESSAGE,
                )
            )

    return tuple(violations)


def collect_root_pyo3_export_violations(repository_root: Path) -> tuple[RootPyO3ExportViolation, ...]:
    """Collect removed raw helper exports from root PyO3 adapter modules."""
    root_python_source_directory = repository_root / ROOT_CRATE_PYTHON_SOURCE_DIRECTORY
    if not root_python_source_directory.exists():
        return ()

    violations: list[RootPyO3ExportViolation] = []
    for rust_source_path in sorted(root_python_source_directory.rglob("*.rs")):
        source_text = rust_source_path.read_text(encoding="utf-8")
        relative_source_path = rust_source_path.relative_to(repository_root)
        for export_pattern in (ROOT_PYO3_PYFUNCTION_EXPORT_PATTERN, ROOT_PYO3_PYCLASS_EXPORT_PATTERN):
            for export_match in export_pattern.finditer(source_text):
                export_name = export_match.group("export_name")
                if export_name not in DISALLOWED_ROOT_PYO3_EXPORT_NAMES:
                    continue
                line_number = source_text.count("\n", 0, export_match.start()) + 1
                violations.append(
                    RootPyO3ExportViolation(
                        source_path=relative_source_path,
                        export_name=export_name,
                        line_number=line_number,
                        message=ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
                    )
                )
    return tuple(sorted(violations, key=lambda violation: (violation.source_path, violation.line_number)))


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


def format_root_crate_boundary_violations(violations: tuple[RootCrateBoundaryViolation, ...]) -> str:
    """Format root crate boundary violations for command-line output."""
    return "\n".join(f"- {violation.source_path} [{violation.marker}]: {violation.message}" for violation in violations)


def format_python_telemetry_fallback_violations(violations: tuple[PythonTelemetryFallbackViolation, ...]) -> str:
    """Format root PyO3 telemetry fallback violations for command-line output."""
    return "\n".join(
        f"- {violation.source_path}:{violation.line_number} [{violation.method_name}]: {violation.message}"
        for violation in violations
    )


def format_root_pyo3_export_violations(violations: tuple[RootPyO3ExportViolation, ...]) -> str:
    """Format removed root PyO3 export violations for command-line output."""
    return "\n".join(
        f"- {violation.source_path}:{violation.line_number} [{violation.export_name}]: {violation.message}"
        for violation in violations
    )


def run_tool(repository_root: Path) -> int:
    """Verify Rust workspace architecture policy."""
    try:
        metadata_payload = load_cargo_metadata(repository_root)
    except (FileNotFoundError, RuntimeError, json.JSONDecodeError) as error:
        print(error)
        return 1

    dependency_violations = collect_rust_architecture_violations(metadata_payload)
    root_crate_violations = collect_root_crate_boundary_violations(repository_root)
    telemetry_fallback_violations = collect_python_telemetry_fallback_violations(repository_root)
    root_pyo3_export_violations = collect_root_pyo3_export_violations(repository_root)
    if dependency_violations or root_crate_violations or telemetry_fallback_violations or root_pyo3_export_violations:
        print("Rust workspace architecture violations:")
        if dependency_violations:
            print(format_violations(dependency_violations))
        if root_crate_violations:
            print(format_root_crate_boundary_violations(root_crate_violations))
        if telemetry_fallback_violations:
            print(format_python_telemetry_fallback_violations(telemetry_fallback_violations))
        if root_pyo3_export_violations:
            print(format_root_pyo3_export_violations(root_pyo3_export_violations))
        return 1

    print("Rust workspace architecture policy passed.")
    return 0


def main() -> int:
    """Run the Rust architecture checker from the repository root."""
    return run_tool(REPOSITORY_ROOT)


if __name__ == "__main__":
    sys.exit(main())
