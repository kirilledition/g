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
        "NativeCliRunFailureTelemetryPlan",
        "build_current_run_manifest_header_json_from_input_json",
        "build_default_local_cache_directory_value",
        "build_file_content_sha256_value",
        "build_final_timing_outputs_write_started_diagnostic_payload",
        "build_jax_runtime_policy_payload",
        "build_jax_runtime_setup_diagnostic_payloads",
        "build_logging_runtime_policy_payload",
        "build_manifest_json_sha256_from_value",
        "build_manifest_file_fingerprint_mapping_payload",
        "build_manifest_file_fingerprint_payload",
        "build_multi_run_artifacts_payload",
        "build_phenotype_compute_group_id_value",
        "build_phenotype_compute_groups_payload",
        "build_phenotype_output_directory_name",
        "build_pipeline_output_preparation_batch_from_values",
        "build_phenotype_run_artifacts_payload",
        "build_prediction_loco_file_fingerprints_json",
        "build_prepared_run_manifest_header_json",
        "build_prepared_run_plan_json_from_current_header",
        "build_prepared_run_plan_json",
        "build_process_runtime_state_handle",
        "build_callback_chunk_identity",
        "build_execution_run_artifacts_payload",
        "build_run_completed_event_payload",
        "build_run_failed_event_payload",
        "build_run_interrupted_event_payload",
        "build_run_manifest_extension_payload",
        "build_runtime_policy_handle",
        "build_shutdown_signal_payload",
        "build_trusted_bgen_validation_cache_path_value",
        "build_trusted_bgen_validation_cache_payload",
        "build_trusted_bgen_validation_fingerprint_value",
        "complete_jax_runtime_setup_validation_payload",
        "compile_run_request_json",
        "configure_bgen_decode_tile_variant_count",
        "configure_rayon_global_thread_pool",
        "default_local_cache_directory_value",
        "default_local_temporary_root_value",
        "default_nvidia_driver_probe_paths_payload",
        "default_shutdown_signal_numbers",
        "default_trusted_bgen_validation_cache_directory_value",
        "describe_logging_runtime_policy_value",
        "emit_cli_run_failed_telemetry_event",
        "emit_binary_correction_summary_telemetry",
        "emit_callback_progress_completion_telemetry",
        "emit_callback_progress_event_telemetry",
        "emit_callback_progress_update_telemetry",
        "emit_diagnostic_event",
        "emit_diagnostic_event_fields",
        "existing_manifest_json",
        "extend_run_manifest_metadata",
        "finalize_output_run_chunks",
        "format_rayon_thread_pool_configuration_error_value",
        "format_dosage_callback_worker_error_message",
        "format_result_callback_worker_error_message",
        "initialize_output_run_from_values",
        "initialize_pipeline_output_run_batch",
        "initialize_pipeline_output_runs",
        "load_run_manifest_payload",
        "nvidia_driver_files_are_visible_value",
        "plan_cli_telemetry_close_failure",
        "normalize_binary_correction_payload",
        "plan_association_backend_payload",
        "plan_jax_gpu_validation_payload",
        "plan_jax_runtime_config_update_payloads",
        "plan_jax_runtime_diagnostic_record",
        "plan_jax_runtime_diagnostic_record_payload",
        "plan_jax_runtime_setup_side_effects_payload",
        "plan_callback_queue_backpressure_observation",
        "plan_callback_queue_operation_observation",
        "plan_callback_queue_stage_backpressure_observation",
        "plan_callback_queue_stage_observation",
        "plan_callback_worker_abort",
        "plan_callback_worker_finish",
        "plan_callback_worker_start",
        "plan_callback_worker_stop_poll",
        "plan_dosage_buffer_reuse",
        "plan_dosage_callback_worker_join",
        "plan_dosage_callback_worker_stop",
        "plan_dosage_work_handoff",
        "plan_dosage_work_item_dispatch",
        "plan_dosage_work_item_stage_duration",
        "plan_auto_gpu_genotype_format_after_trusted_validation",
        "plan_bgen_delivery_cleanup",
        "plan_bgen_delivery_invocation",
        "plan_gpu_genotype_format_auto_to_dosage",
        "plan_multi_trait_chunk_write",
        "plan_multi_trait_output_write",
        "plan_null_logistic_nonconvergence",
        "plan_null_logistic_nonconvergence_from_array",
        "plan_result_callback_worker_join",
        "plan_result_callback_worker_stop",
        "plan_result_write_handoff",
        "plan_result_write_item_dispatch",
        "plan_single_trait_binary_gpu_genotype_format_resolution",
        "plan_single_trait_output_write",
        "plan_second_signal_exception",
        "plan_stage_timing_recorder",
        "plan_telemetry_close",
        "plan_telemetry_event_emission",
        "plan_telemetry_progress_emission",
        "plan_timing_file_write",
        "plan_trusted_bgen_validation_cache_lookup",
        "plan_variant_major_dosage_batch_handoff",
        "plan_writer_finish_execution",
        "prepare_output_run",
        "raise_second_signal_exception",
        "attach_run_metadata_payload",
        "record_association_backend_selected_telemetry_event",
        "record_bgen_engine_opened_telemetry_event",
        "record_effective_config_written_telemetry_event",
        "record_execution_plan_prepared_telemetry_event",
        "record_gpu_genotype_format_resolved_telemetry_event",
        "record_multi_phenotype_preflight_completed_telemetry_event",
        "record_multi_phenotype_sample_summary_telemetry_event",
        "record_multi_writer_finished_telemetry_event",
        "record_native_cli_completed_line_diagnostic_event",
        "record_native_cli_failed_line_diagnostic_event",
        "record_native_cli_interrupted_line_diagnostic_event",
        "record_native_cli_stderr_diagnostic_event",
        "record_native_cli_stdout_diagnostic_event",
        "record_native_runtime_knobs_configured_diagnostic_event",
        "record_final_timing_outputs_write_started_diagnostic_event",
        "record_jax_runtime_diagnostic_event",
        "record_callback_null_logistic_nonconvergence_warning_diagnostic_event",
        "record_io_output_resume_committed_chunks_diagnostic_event",
        "record_native_dispatch_bgen_engine_constructing_diagnostic_event",
        "record_native_dispatch_callback_drain_started_diagnostic_event",
        "record_native_dispatch_delivery_failed_diagnostic_event",
        "record_native_dispatch_delivery_finished_diagnostic_event",
        "record_native_dispatch_delivery_interrupted_diagnostic_event",
        "record_native_dispatch_delivery_started_diagnostic_event",
        "record_native_dispatch_pipeline_finished_diagnostic_event",
        "record_native_dispatch_trusted_bgen_validation_started_diagnostic_event",
        "record_native_dispatch_writer_session_finish_started_diagnostic_event",
        "record_native_dispatch_writer_session_interrupted_flush_started_diagnostic_event",
        "record_native_dispatch_writer_sessions_finish_started_diagnostic_event",
        "record_native_dispatch_writer_sessions_interrupted_flush_started_diagnostic_event",
        "record_pipeline_bgen_engine_open_started_diagnostic_event",
        "record_pipeline_bgen_engine_opened_diagnostic_event",
        "record_pipeline_gpu_genotype_format_resolved_diagnostic_event",
        "record_pipeline_grouped_per_phenotype_groups_prepared_diagnostic_event",
        "record_pipeline_grouped_per_phenotype_started_diagnostic_event",
        "record_pipeline_grouped_union_delivery_selected_diagnostic_event",
        "record_pipeline_multi_group_preflight_completed_diagnostic_event",
        "record_pipeline_multi_group_preflight_started_diagnostic_event",
        "record_pipeline_multi_phenotype_sample_summary_diagnostic_event",
        "record_pipeline_multi_trait_input_aligned_diagnostic_event",
        "record_pipeline_multi_trait_input_load_started_diagnostic_event",
        "record_pipeline_multi_trait_prediction_source_load_started_diagnostic_event",
        "record_pipeline_multi_trait_started_diagnostic_event",
        "record_pipeline_output_resume_committed_chunks_diagnostic_event",
        "record_pipeline_output_writer_sessions_create_started_diagnostic_event",
        "record_pipeline_prevalidated_bgen_engine_used_diagnostic_event",
        "record_pipeline_single_trait_input_aligned_diagnostic_event",
        "record_pipeline_single_trait_input_load_started_diagnostic_event",
        "record_pipeline_single_trait_prediction_source_load_started_diagnostic_event",
        "record_pipeline_single_trait_preflight_completed_diagnostic_event",
        "record_pipeline_single_trait_preflight_started_diagnostic_event",
        "record_pipeline_single_trait_started_diagnostic_event",
        "record_preflight_warning_diagnostic_event",
        "record_prediction_source_loaded_telemetry_event",
        "record_runner_binary_engine_dispatch_started_diagnostic_event",
        "record_runner_execution_plan_build_started_diagnostic_event",
        "record_runner_execution_plan_dispatch_started_diagnostic_event",
        "record_runner_execution_plan_finalization_started_diagnostic_event",
        "record_runner_execution_plan_prepared_diagnostic_event",
        "record_runner_jax_runtime_configuration_started_diagnostic_event",
        "record_runner_linear_engine_dispatch_started_diagnostic_event",
        "record_runner_metadata_artifacts_finalized_diagnostic_event",
        "record_runner_multi_phenotype_binary_engine_dispatch_started_diagnostic_event",
        "record_runner_multi_phenotype_dispatch_started_diagnostic_event",
        "record_runner_multi_phenotype_linear_engine_dispatch_started_diagnostic_event",
        "record_runner_run_completed_diagnostic_event",
        "record_runner_run_failed_diagnostic_event",
        "record_runner_run_interrupted_diagnostic_event",
        "record_runner_run_started_diagnostic_event",
        "record_runner_single_phenotype_dispatch_started_diagnostic_event",
        "read_manifest_committed_chunk_identifiers_from_value",
        "repair_strict_manifest_chunk_commits_from_value",
        "resolve_final_timing_output_context",
        "resolve_output_run_paths",
        "scan_committed_chunk_identifiers",
        "validate_run_manifest_compatibility_from_values",
        "validate_strict_manifest_chunks_from_value",
        "write_run_manifest",
        "record_runner_run_completed_telemetry_event",
        "record_runner_run_failed_telemetry_event",
        "record_runner_run_interrupted_telemetry_event",
        "record_runner_run_started_telemetry_event",
        "record_sample_alignment_completed_telemetry_event",
        "record_single_trait_preflight_completed_telemetry_event",
        "record_writer_finished_telemetry_event",
        "record_jax_runtime_diagnostic_log_event",
        "resolve_bgen_delivery_method_value",
        "resolve_association_mode_value",
        "resolve_callback_worker_backpressure_poll_timeout_seconds",
        "resolve_callback_worker_stop_poll_timeout_seconds",
        "resolve_delivery_callback_batch_size",
        "resolve_effective_trusted_no_missing_diploid",
        "resolve_grouped_union_callback_batch_size",
        "resolve_jax_runtime_setup_payload",
        "resolve_manifest_gpu_genotype_format",
        "resolve_native_callback_queue_limits",
        "resolve_native_callback_worker_shutdown_timeouts",
        "resolve_telemetry_output_run_root_value",
        "resolve_telemetry_paths_payload",
        "resolve_writer_finish_thread_count",
        "render_run_completed_lines",
        "render_run_failed_lines",
        "render_run_interrupted_lines",
        "should_attempt_callback_worker_stop",
        "build_preflight_report_payload",
        "intersect_committed_chunk_identifier_sets",
        "resolve_preflight_variant_count",
        "validate_binary_phenotype_array",
        "validate_binary_phenotype_case_control_counts",
        "validate_binary_phenotype_coding",
        "validate_covariate_matrix_rank",
        "validate_covariate_matrix_rank_array",
        "validate_finite_array_values",
        "validate_finite_array",
        "validate_multi_prediction_preflight_shape",
        "validate_multi_trait_preflight_shape_payload",
        "validate_pipeline_resume_compatibility",
        "validate_single_prediction_preflight_shape",
        "validate_single_trait_preflight_shape_payload",
        "write_trusted_bgen_validation_cache_payload",
    )
)
DISALLOWED_ROOT_PYO3_CLASS_MEMBER_NAMES = frozenset(
    (
        "build_current_run_manifest_header_json_from_input_json",
        "build_prediction_loco_file_fingerprints_json",
        "existing_manifest_json",
    )
)
ROOT_PYO3_PYFUNCTION_EXPORT_PATTERN = re.compile(r"wrap_pyfunction!\s*\(\s*(?P<export_name>[A-Za-z0-9_]+)")
ROOT_PYO3_PYCLASS_EXPORT_PATTERN = re.compile(r"add_class::\s*<\s*(?P<export_name>[A-Za-z0-9_]+)\s*>")
ROOT_PYO3_PYMETHOD_EXPORT_PATTERN = re.compile(
    r"(?m)^\s*(?:pub(?:\(crate\))?\s+)?fn\s+(?P<export_name>[A-Za-z0-9_]+)\s*\("
)
ROOT_PYO3_FIELD_GETTER_EXPORT_PATTERN = re.compile(
    r"(?m)^\s*#\[pyo3\(get\)\]\s*\n\s*(?P<export_name>[A-Za-z0-9_]+)\s*:"
)
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


def collect_root_pyo3_registered_export_violations(
    source_text: str,
    relative_source_path: Path,
) -> tuple[RootPyO3ExportViolation, ...]:
    """Collect removed free-function and class registrations from one source file."""
    violations: list[RootPyO3ExportViolation] = []
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
    return tuple(violations)


def collect_root_pyo3_class_member_export_violations(
    source_text: str,
    relative_source_path: Path,
) -> tuple[RootPyO3ExportViolation, ...]:
    """Collect removed class methods and getters from one source file."""
    violations: list[RootPyO3ExportViolation] = []
    for export_pattern in (ROOT_PYO3_PYMETHOD_EXPORT_PATTERN, ROOT_PYO3_FIELD_GETTER_EXPORT_PATTERN):
        for export_match in export_pattern.finditer(source_text):
            export_name = export_match.group("export_name")
            if export_name not in DISALLOWED_ROOT_PYO3_CLASS_MEMBER_NAMES:
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
        violations.extend(collect_root_pyo3_registered_export_violations(source_text, relative_source_path))
        violations.extend(collect_root_pyo3_class_member_export_violations(source_text, relative_source_path))
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
