#!/usr/bin/env python3
"""Verify Python package ownership boundaries for the Rust migration."""

from __future__ import annotations

import ast
import dataclasses
import fnmatch
import sys
import typing
from pathlib import Path

import hydra

from tooling.common import hydra_compat as tooling_hydra_compat

if typing.TYPE_CHECKING:
    import omegaconf

PRODUCTION_PACKAGE_ROOT = Path("src/g")


@dataclasses.dataclass(frozen=True)
class PythonImportPolicy:
    """A Python package import-boundary policy.

    Attributes:
        name: Stable policy name for diagnostics.
        source_directory: Package directory, relative to the production package root.
        forbidden_imports: Absolute import prefixes rejected under the source directory.
        message: Human-readable policy description.

    """

    name: str
    source_directory: Path
    forbidden_imports: tuple[str, ...]
    message: str


@dataclasses.dataclass(frozen=True)
class PythonImportViolation:
    """A Python import that crosses an ownership boundary.

    Attributes:
        path: Source file containing the violation.
        line_number: One-based source line number containing the import.
        column_offset: Zero-based source column containing the import.
        policy_name: Import policy that rejected the import.
        import_name: Absolute import name observed in source.
        forbidden_import: Forbidden import prefix that matched the observed import.
        message: Human-readable policy description.

    """

    path: Path
    line_number: int
    column_offset: int
    policy_name: str
    import_name: str
    forbidden_import: str
    message: str


@dataclasses.dataclass(frozen=True)
class PythonCallPolicy:
    """A Python call-boundary policy.

    Attributes:
        name: Stable policy name for diagnostics.
        source_directory: Package directory, relative to the production package root.
        forbidden_calls: Dotted call names rejected under the source directory.
        allowed_paths: Source paths, relative to the production package root, excluded from this policy.
        message: Human-readable policy description.

    """

    name: str
    source_directory: Path
    forbidden_calls: tuple[str, ...]
    allowed_paths: tuple[Path, ...]
    message: str


@dataclasses.dataclass(frozen=True)
class PythonCallViolation:
    """A Python call that crosses an ownership boundary.

    Attributes:
        path: Source file containing the violation.
        line_number: One-based source line number containing the call.
        column_offset: Zero-based source column containing the call.
        policy_name: Call policy that rejected the call.
        call_name: Dotted call name observed in source.
        forbidden_call: Forbidden call pattern that matched the observed call.
        message: Human-readable policy description.

    """

    path: Path
    line_number: int
    column_offset: int
    policy_name: str
    call_name: str
    forbidden_call: str
    message: str


@dataclasses.dataclass(frozen=True)
class PythonDefinitionPolicy:
    """A Python definition-boundary policy.

    Attributes:
        name: Stable policy name for diagnostics.
        source_directory: Package directory, relative to the production package root.
        forbidden_function_names: Function or method names rejected under the source directory.
        allowed_paths: Source paths, relative to the production package root, excluded from this policy.
        message: Human-readable policy description.

    """

    name: str
    source_directory: Path
    forbidden_function_names: tuple[str, ...]
    allowed_paths: tuple[Path, ...]
    message: str


@dataclasses.dataclass(frozen=True)
class PythonDefinitionViolation:
    """A Python function or method definition that crosses an ownership boundary.

    Attributes:
        path: Source file containing the violation.
        line_number: One-based source line number containing the definition.
        column_offset: Zero-based source column containing the definition.
        policy_name: Definition policy that rejected the definition.
        function_name: Function or method name observed in source.
        message: Human-readable policy description.

    """

    path: Path
    line_number: int
    column_offset: int
    policy_name: str
    function_name: str
    message: str


@dataclasses.dataclass(frozen=True)
class PythonCliShimViolation:
    """A Python CLI shim contract violation.

    Attributes:
        path: Source file containing the violation.
        line_number: One-based source line number for the relevant statement.
        column_offset: Zero-based source column for the relevant statement.
        policy_name: Stable policy name that rejected the CLI shim shape.
        subject: Function, constant, or file subject that violated the policy.
        message: Human-readable policy description.

    """

    path: Path
    line_number: int
    column_offset: int
    policy_name: str
    subject: str
    message: str


@dataclasses.dataclass(frozen=True)
class ModuleStringAssignment:
    """Top-level string assignment metadata.

    Attributes:
        value: Assigned string value.
        line_number: One-based assignment line number.
        column_offset: Zero-based assignment column offset.

    """

    value: str
    line_number: int
    column_offset: int


CLI_SHIM_PATH = Path("cli.py")
CLI_SHIM_POLICY_NAME = "native_cli_shim_process_owner"
CLI_SHIM_SENTINEL_CONSTANT_NAME = "NATIVE_CLI_PYTHON_BRIDGE_SENTINEL_ENVIRONMENT_VARIABLE"
CLI_SHIM_SENTINEL_ENVIRONMENT_VARIABLE = "G_NATIVE_CLI_PYTHON_BRIDGE_SENTINEL"
CLI_SHIM_MESSAGE = (
    "the public Python CLI entry point must remain compatibility glue into the native CLI runner; "
    "only the sentinel-protected legacy backend may call dispatch_cli"
)

PYTHON_IMPORT_POLICIES = (
    PythonImportPolicy(
        name="compute_kernel_isolation",
        source_directory=Path("compute"),
        forbidden_imports=("g.cli", "g.interface", "g.io"),
        message="JAX compute kernels must not import CLI, config, output, or file-parser packages",
    ),
    PythonImportPolicy(
        name="jax_runtime_orchestration_isolation",
        source_directory=Path("jax_runtime"),
        forbidden_imports=("g.runner",),
        message="JAX runtime helpers must not import runner orchestration packages",
    ),
    PythonImportPolicy(
        name="runner_jax_import_boundary",
        source_directory=Path("runner"),
        forbidden_imports=("g.engine.regenie2_pipeline", "g.engine.callbacks", "g.compute", "jax", "jaxlib"),
        message="runner modules must not import JAX-facing modules before runtime setup",
    ),
)

PYTHON_CALL_POLICIES = (
    PythonCallPolicy(
        name="production_manifest_write_isolation",
        source_directory=Path(),
        forbidden_calls=(
            "output.write_run_manifest",
            "_core.write_run_manifest",
            "_core.write_run_manifest_json",
            "write_run_manifest",
        ),
        allowed_paths=(Path("io/output.py"),),
        message="production Python must not write run manifests outside the output adapter helper",
    ),
    PythonCallPolicy(
        name="native_output_lifecycle_adapter_isolation",
        source_directory=Path(),
        forbidden_calls=(
            "_core.prepare_output_run",
            "_core.initialize_output_run",
            "_core.initialize_output_run_from_values",
            "_core.load_run_manifest_json",
            "_core.load_run_manifest_payload",
            "_core.NativeOutputLifecyclePolicy",
            "_core.validate_run_manifest_compatibility",
            "_core.validate_run_manifest_compatibility_from_values",
            "_core.read_manifest_committed_chunk_identifiers",
            "_core.read_manifest_committed_chunk_identifiers_from_value",
            "_core.validate_strict_manifest_chunks",
            "_core.validate_strict_manifest_chunks_from_value",
            "_core.repair_strict_manifest_chunk_commits",
            "_core.repair_strict_manifest_chunk_commits_from_value",
            "_core.scan_committed_chunk_identifiers",
            "_core.finalize_output_run_chunks",
            "_core.resolve_output_run_paths",
            "_core.build_pipeline_output_preparation_batch_from_values",
            "_core.NativePipelineOutputPreparationBatch",
            "_core.NativePipelineOutputPreparationPolicy",
            "_core.initialize_pipeline_output_run_batch",
            "_core.initialize_pipeline_output_runs",
        ),
        allowed_paths=(Path("io/output.py"),),
        message="production Python must route native output lifecycle calls through the output adapter helper",
    ),
    PythonCallPolicy(
        name="native_output_writer_lifecycle_adapter_isolation",
        source_directory=Path(),
        forbidden_calls=(
            "_core.finish_output_writer_session",
            "_core.finish_output_writer_session_interrupted",
            "_core.abort_output_writer_session",
        ),
        allowed_paths=(Path("engine/native_dispatch/writers.py"),),
        message="production Python must route native writer lifecycle calls through the native-dispatch adapter",
    ),
    PythonCallPolicy(
        name="native_output_chunk_write_adapter_isolation",
        source_directory=Path(),
        forbidden_calls=(
            "_core.write_regenie2_multi_native_chunk",
            "_core.write_regenie2_multi_native_chunk_f64",
            "_core.NativeOutputChunkWritePolicy",
        ),
        allowed_paths=(Path("engine/callbacks/writers.py"),),
        message="production Python must route native output chunk writes through the callback writer adapter",
    ),
    PythonCallPolicy(
        name="native_output_manifest_helper_adapter_isolation",
        source_directory=Path(),
        forbidden_calls=(
            "_core.NativeManifestFileFingerprintCache",
            "_core.build_current_run_manifest_header_json_from_input_json",
            "_core.build_current_run_manifest_header_payload_from_input",
            "_core.build_file_content_sha256_value",
            "_core.build_file_fingerprint_payload",
            "_core.build_manifest_file_fingerprint_payload",
            "_core.build_manifest_json_sha256",
            "_core.build_manifest_json_sha256_from_value",
            "_core.build_prediction_loco_file_fingerprints_json",
            "_core.build_prediction_loco_file_fingerprints_payload",
            "_core.build_prepared_run_manifest_header_json",
            "_core.build_prepared_run_manifest_header_json_from_current_header_json",
            "_core.build_prepared_run_plan_json",
            "_core.build_prepared_run_plan_json_from_current_header",
            "_core.build_prepared_run_plan_json_from_current_header_json",
            "build_current_run_manifest_header_json_from_input_json",
            "build_current_run_manifest_header_payload_from_input",
            "build_file_fingerprint_payload",
            "build_prediction_loco_file_fingerprints_json",
            "build_prediction_loco_file_fingerprints_payload",
        ),
        allowed_paths=(Path("io/output.py"),),
        message="production Python must route native output manifest helper calls through the output adapter helper",
    ),
    PythonCallPolicy(
        name="native_run_metadata_adapter_isolation",
        source_directory=Path(),
        forbidden_calls=(
            "_core.NativeRunMetadataBuilder",
            "_core.build_execution_run_artifacts_payload",
            "_core.extend_run_manifest_metadata",
        ),
        allowed_paths=(Path("runner/metadata.py"),),
        message="production Python must route native run-metadata helpers through the runner metadata adapter",
    ),
    PythonCallPolicy(
        name="callback_worker_queue_isolation",
        source_directory=Path("engine/callbacks"),
        forbidden_calls=(
            "queue.Queue",
            "threading.Thread",
            "threading.BoundedSemaphore",
            "BoundedSemaphore",
            "NativeCallbackObjectQueue",
            "NativeCallbackWaitSignal",
            "NativeCallbackWorkerThread",
            "NativeCallbackSchedulerState",
            "NativeCallbackProgressState",
            "NativeBinaryCorrectionSummary",
            "NativeDosageBufferPoolState",
            "NativeResultInFlightSlotState",
        ),
        allowed_paths=(),
        message="production callback code must use native worker queues and worker-thread handles",
    ),
    PythonCallPolicy(
        name="prepared_plan_reconstruction_isolation",
        source_directory=Path(),
        forbidden_calls=("_core.build_prepared_run_plan_json", "build_native_prepared_run_plan_input_mapping"),
        allowed_paths=(),
        message="production Python must not reconstruct canonical prepared-run plans",
    ),
    PythonCallPolicy(
        name="native_run_request_adapter_isolation",
        source_directory=Path(),
        forbidden_calls=("_core.compile_run_request_json", "_core.compile_run_request_payload"),
        allowed_paths=(Path("execution_plan.py"),),
        message="production Python must route native run-request compilation through the execution-plan adapter",
    ),
    PythonCallPolicy(
        name="native_diagnostic_payload_adapter_isolation",
        source_directory=Path(),
        forbidden_calls=("_core.build_*_diagnostic_payload", "_core.build_*_diagnostic_payloads"),
        allowed_paths=(Path("engine/run_events.py"), Path("engine/timing.py"), Path("jax_runtime/diagnostics.py")),
        message=(
            "production Python must use native diagnostic recorders; payload builders are compatibility adapters only"
        ),
    ),
    PythonCallPolicy(
        name="native_diagnostic_emitter_isolation",
        source_directory=Path(),
        forbidden_calls=("_core.emit_diagnostic_event", "_core.emit_diagnostic_event_fields"),
        allowed_paths=(),
        message="production Python must route diagnostics through typed native recorder helpers",
    ),
    PythonCallPolicy(
        name="native_telemetry_dispatch_isolation",
        source_directory=Path(),
        forbidden_calls=(
            "close_with_event",
            "log_jax_runtime_diagnostic_event",
            "log_callback_progress_event",
            "log_binary_correction_summary",
            "log_run_failed",
            "log_progress",
        ),
        allowed_paths=(),
        message="production Python must dispatch telemetry through native PyO3 helpers, not fallback methods",
    ),
    PythonCallPolicy(
        name="native_telemetry_handle_dispatch_isolation",
        source_directory=Path(),
        forbidden_calls=(
            "native_session_handle.emit_*",
            "native_telemetry_session.emit_*",
            "emit_current_event",
            "emit_payload",
            "emit_progress",
            "emit_run_completed_event",
            "emit_run_interrupted_event",
            "emit_run_failed_event",
            "emit_run_started_event",
            "emit_execution_plan_prepared_event",
            "emit_effective_config_written_event",
            "emit_phenotype_writer_finished_event",
            "emit_multi_phenotype_writer_finished_event",
            "emit_single_trait_preflight_completed_event",
            "emit_multi_phenotype_preflight_completed_event",
            "emit_sample_alignment_completed_event",
            "emit_prediction_source_loaded_event",
            "emit_multi_phenotype_sample_summary_event",
            "emit_gpu_genotype_format_resolved_event",
            "emit_association_backend_selected_event",
            "emit_bgen_engine_opened_event",
            "emit_callback_progress_event",
            "emit_binary_correction_summary_event",
            "emit_jax_runtime_diagnostic_event",
        ),
        allowed_paths=(Path("engine/telemetry.py"),),
        message="production Python telemetry event emission must go through typed native PyO3 dispatch helpers",
    ),
    PythonCallPolicy(
        name="native_telemetry_wrapper_dispatch_isolation",
        source_directory=Path(),
        forbidden_calls=(
            "log_event",
            "should_emit_progress",
            "build_event_payload",
            "write_json_line",
            "writer_counters",
        ),
        allowed_paths=(),
        message="production Python must not use compatibility telemetry wrapper methods for event dispatch",
    ),
    PythonCallPolicy(
        name="native_jax_cache_resolution_isolation",
        source_directory=Path(),
        forbidden_calls=("resolve_jax_runtime_cache_directory",),
        allowed_paths=(),
        message="production Python must resolve JAX setup cache directories through native runtime state",
    ),
    PythonCallPolicy(
        name="native_jax_setup_session_construction_isolation",
        source_directory=Path(),
        forbidden_calls=("_core.resolve_jax_runtime_setup_payload", "_core.NativeJaxRuntimeSetupSession"),
        allowed_paths=(),
        message="production Python must construct JAX setup sessions through native runtime state",
    ),
    PythonCallPolicy(
        name="native_jax_setup_side_effect_isolation",
        source_directory=Path(),
        forbidden_calls=("jax.config.update", "jax.devices", "side_effect_plan_payload"),
        allowed_paths=(),
        message="production Python must execute JAX setup side effects through typed native setup sessions",
    ),
    PythonCallPolicy(
        name="native_preflight_numeric_scan_isolation",
        source_directory=Path("engine/preflight.py"),
        forbidden_calls=(
            "np.isfinite",
            "numpy.isfinite",
            "np.unique",
            "numpy.unique",
            "np.count_nonzero",
            "numpy.count_nonzero",
        ),
        allowed_paths=(),
        message="production preflight must use native PyO3 array checks for finite and binary scans",
    ),
    PythonCallPolicy(
        name="native_preflight_required_chromosome_isolation",
        source_directory=Path("engine/preflight.py"),
        forbidden_calls=("getattr",),
        allowed_paths=(),
        message="production preflight must call the typed native required-chromosome API directly",
    ),
    PythonCallPolicy(
        name="native_covariate_rank_scan_isolation",
        source_directory=Path("engine"),
        forbidden_calls=("np.linalg.matrix_rank", "numpy.linalg.matrix_rank", "matrix_rank"),
        allowed_paths=(),
        message="production covariate rank scans must use the native PyO3 SVD-backed rank validator",
    ),
    PythonCallPolicy(
        name="native_callback_convergence_scan_isolation",
        source_directory=Path("engine/callbacks/diagnostics.py"),
        forbidden_calls=("np.ravel", "numpy.ravel", "np.count_nonzero", "numpy.count_nonzero"),
        allowed_paths=(),
        message="production callback diagnostics must use native PyO3 array checks for convergence scans",
    ),
    PythonCallPolicy(
        name="binary_diagnostics_result_contract_isolation",
        source_directory=Path("compute/regenie2_binary/diagnostics.py"),
        forbidden_calls=("getattr",),
        allowed_paths=(),
        message="production binary diagnostics must normalize typed results instead of probing optional fields",
    ),
    PythonCallPolicy(
        name="binary_diagnostics_host_materialization_isolation",
        source_directory=Path("compute/regenie2_binary/diagnostics.py"),
        forbidden_calls=("jax.device_get", "device_get"),
        allowed_paths=(),
        message="production binary diagnostics must leave host materialization to callback adapters",
    ),
    PythonCallPolicy(
        name="callback_readiness_blocker_contract_isolation",
        source_directory=Path("engine/callbacks/diagnostics.py"),
        forbidden_calls=("getattr",),
        allowed_paths=(),
        message="production callback readiness blocking must use the typed JAX readiness API directly",
    ),
    PythonCallPolicy(
        name="binary_callback_chromosome_state_contract_isolation",
        source_directory=Path("engine/callbacks/binary.py"),
        forbidden_calls=("getattr",),
        allowed_paths=(),
        message="production binary callbacks must require typed chromosome-state readiness fields",
    ),
    PythonCallPolicy(
        name="linear_callback_chromosome_state_contract_isolation",
        source_directory=Path("engine/callbacks/linear.py"),
        forbidden_calls=("getattr",),
        allowed_paths=(),
        message="production linear callbacks must require typed chromosome-state readiness fields",
    ),
    PythonCallPolicy(
        name="callback_transfer_contract_isolation",
        source_directory=Path("engine/callbacks/transfers.py"),
        forbidden_calls=("getattr",),
        allowed_paths=(),
        message="production callback transfer helpers must use typed array and chunk-stat contracts",
    ),
    PythonCallPolicy(
        name="callback_writer_contract_isolation",
        source_directory=Path("engine/callbacks/writers.py"),
        forbidden_calls=("getattr",),
        allowed_paths=(),
        message="production callback writers must use typed writer contracts, not method-name probing",
    ),
    PythonCallPolicy(
        name="native_compute_group_resolution_isolation",
        source_directory=Path("engine/native_dispatch/groups.py"),
        forbidden_calls=("getattr",),
        allowed_paths=(),
        message="production native-dispatch compute-group resolution must call native resolvers directly",
    ),
    PythonCallPolicy(
        name="native_delivery_callback_contract_isolation",
        source_directory=Path("engine/native_dispatch/delivery.py"),
        forbidden_calls=("getattr",),
        allowed_paths=(),
        message="production native BGEN delivery must use typed callback contracts, not optional callback probing",
    ),
    PythonCallPolicy(
        name="native_dispatch_callback_lifecycle_isolation",
        source_directory=Path("engine/native_dispatch/writers.py"),
        forbidden_calls=("getattr",),
        allowed_paths=(),
        message="production native-dispatch callback lifecycle must call typed callback methods directly",
    ),
    PythonCallPolicy(
        name="grouped_callback_fanout_contract_isolation",
        source_directory=Path("engine/callbacks/grouped.py"),
        forbidden_calls=("getattr",),
        allowed_paths=(),
        message="production grouped callback fanout must use typed callback lifecycle contracts",
    ),
    PythonCallPolicy(
        name="callback_metadata_chromosome_contract_isolation",
        source_directory=Path("engine/callbacks/shared.py"),
        forbidden_calls=("getattr",),
        allowed_paths=(),
        message="production callback metadata helpers must use native scalar chromosome labels directly",
    ),
    PythonCallPolicy(
        name="timing_snapshot_serialization_contract_isolation",
        source_directory=Path("engine/timing.py"),
        forbidden_calls=("getattr",),
        allowed_paths=(),
        message="production timing snapshot serialization must use typed dataclass mappings, not reflective probing",
    ),
    PythonCallPolicy(
        name="jax_host_materialization_isolation",
        source_directory=Path("engine"),
        forbidden_calls=("jax.device_get", "device_get"),
        allowed_paths=(Path("engine/callbacks/diagnostics.py"), Path("engine/callbacks/writers.py")),
        message="production JAX host materialization must stay isolated to callback diagnostic and writer adapters",
    ),
    PythonCallPolicy(
        name="compute_kernel_file_io_isolation",
        source_directory=Path("compute"),
        forbidden_calls=(
            "open",
            "Path.open",
            "read_text",
            "write_text",
            "read_bytes",
            "write_bytes",
            "np.load",
            "numpy.load",
            "jnp.load",
            "np.loadtxt",
            "numpy.loadtxt",
            "np.genfromtxt",
            "numpy.genfromtxt",
            "pandas.read_csv",
            "pd.read_csv",
            "pandas.read_parquet",
            "pd.read_parquet",
        ),
        allowed_paths=(),
        message="JAX compute kernels must not read or write files directly",
    ),
)

PYTHON_DEFINITION_POLICIES = (
    PythonDefinitionPolicy(
        name="native_telemetry_fallback_definition_isolation",
        source_directory=Path(),
        forbidden_function_names=(
            "close_with_event",
            "log_jax_runtime_diagnostic_event",
            "log_callback_progress_event",
            "log_binary_correction_summary",
            "log_run_failed",
            "log_progress",
        ),
        allowed_paths=(),
        message="production Python must not define old telemetry fallback methods",
    ),
    PythonDefinitionPolicy(
        name="telemetry_session_wrapper_definition_isolation",
        source_directory=Path("engine/telemetry.py"),
        forbidden_function_names=(
            "log_event",
            "log_run_completed",
            "log_run_interrupted",
            "log_run_started",
            "log_execution_plan_prepared",
            "log_effective_config_written",
            "log_writer_finished",
            "log_multi_writer_finished",
            "log_single_trait_preflight_completed",
            "log_multi_phenotype_preflight_completed",
            "log_sample_alignment_completed",
            "log_prediction_source_loaded",
            "log_multi_phenotype_sample_summary",
            "log_gpu_genotype_format_resolved",
            "log_association_backend_selected",
            "log_bgen_engine_opened",
            "should_emit_progress",
            "build_event_payload",
            "write_json_line",
            "writer_counters",
            "native_session_policy",
            "native_progress_throttle",
        ),
        allowed_paths=(),
        message="the real telemetry session must not define compatibility dispatch wrappers",
    ),
    PythonDefinitionPolicy(
        name="native_compute_group_fallback_definition_isolation",
        source_directory=Path("engine/native_dispatch/groups.py"),
        forbidden_function_names=(
            "fingerprint_sample_set",
            "fingerprint_covariate_design",
            "fingerprint_prediction_alignment",
            "update_array_fingerprint",
            "update_string_sequence_fingerprint",
            "update_fingerprint",
        ),
        allowed_paths=(),
        message="production native dispatch must not define Python compute-group fingerprint fallbacks",
    ),
)


def module_parts_for_source_path(path: Path, package_root: Path) -> tuple[str, ...]:
    """Return absolute module parts for a Python source file under a package root."""
    relative_path = path.relative_to(package_root.parent).with_suffix("")
    parts = relative_path.parts
    if parts[-1] == "__init__":
        return parts[:-1]
    return parts


def package_parts_for_source_path(path: Path, package_root: Path) -> tuple[str, ...]:
    """Return absolute package parts for a Python source file under a package root."""
    module_parts = module_parts_for_source_path(path, package_root)
    if path.name == "__init__.py":
        return module_parts
    return module_parts[:-1]


def import_from_base_module(path: Path, package_root: Path, statement: ast.ImportFrom) -> str:
    """Resolve the absolute base module for an import-from statement."""
    if statement.level == 0:
        return statement.module or ""

    package_parts = package_parts_for_source_path(path, package_root)
    base_parts = package_parts[: len(package_parts) - statement.level + 1]
    module_parts = () if statement.module is None else tuple(statement.module.split("."))
    return ".".join((*base_parts, *module_parts))


def import_names_from_statement(path: Path, package_root: Path, statement: ast.stmt) -> tuple[str, ...]:
    """Return absolute import names referenced by one import statement."""
    if isinstance(statement, ast.Import):
        return tuple(alias.name for alias in statement.names)

    if isinstance(statement, ast.ImportFrom):
        base_module = import_from_base_module(path, package_root, statement)
        import_names: list[str] = []
        for alias in statement.names:
            if alias.name == "*" and base_module:
                import_names.append(base_module)
                continue
            import_name = f"{base_module}.{alias.name}" if base_module else alias.name
            import_names.append(import_name)
        return tuple(import_names)

    return ()


def import_matches_forbidden_prefix(import_name: str, forbidden_import: str) -> bool:
    """Return whether an import name violates a forbidden import prefix."""
    return import_name == forbidden_import or import_name.startswith(f"{forbidden_import}.")


def call_name_from_expression(expression: ast.expr) -> str | None:
    """Return a dotted call name from an AST call expression."""
    if isinstance(expression, ast.Name):
        return expression.id
    if isinstance(expression, ast.Attribute):
        parent_name = call_name_from_expression(expression.value)
        if parent_name is None:
            return expression.attr
        return f"{parent_name}.{expression.attr}"
    return None


def call_matches_forbidden_name(call_name: str, forbidden_call: str) -> bool:
    """Return whether a call name violates a forbidden call pattern."""
    if "*" in forbidden_call:
        return fnmatch.fnmatchcase(call_name, forbidden_call) or fnmatch.fnmatchcase(
            call_name,
            f"*.{forbidden_call}",
        )
    return call_name == forbidden_call or call_name.endswith(f".{forbidden_call}")


def collect_import_violations_for_statement(
    path: Path,
    relative_path: Path,
    package_root: Path,
    policy: PythonImportPolicy,
    statement: ast.stmt,
) -> tuple[PythonImportViolation, ...]:
    """Collect import-policy violations from one AST statement."""
    import_names = import_names_from_statement(path, package_root, statement)
    violations: list[PythonImportViolation] = []
    for import_name in import_names:
        for forbidden_import in policy.forbidden_imports:
            if not import_matches_forbidden_prefix(import_name, forbidden_import):
                continue
            violations.append(
                PythonImportViolation(
                    path=relative_path,
                    line_number=statement.lineno,
                    column_offset=statement.col_offset,
                    policy_name=policy.name,
                    import_name=import_name,
                    forbidden_import=forbidden_import,
                    message=policy.message,
                )
            )
    return tuple(violations)


def collect_call_violations_for_statement(
    relative_path: Path,
    policy: PythonCallPolicy,
    statement: ast.Call,
) -> tuple[PythonCallViolation, ...]:
    """Collect call-policy violations from one AST call statement."""
    call_name = call_name_from_expression(statement.func)
    if call_name is None:
        return ()

    violations: list[PythonCallViolation] = []
    for forbidden_call in policy.forbidden_calls:
        if not call_matches_forbidden_name(call_name, forbidden_call):
            continue
        violations.append(
            PythonCallViolation(
                path=relative_path,
                line_number=statement.lineno,
                column_offset=statement.col_offset,
                policy_name=policy.name,
                call_name=call_name,
                forbidden_call=forbidden_call,
                message=policy.message,
            )
        )
        break
    return tuple(violations)


def collect_python_import_policy_violations(
    package_root: Path,
    policies: tuple[PythonImportPolicy, ...] = PYTHON_IMPORT_POLICIES,
) -> tuple[PythonImportViolation, ...]:
    """Collect Python import-boundary violations under a production package root."""
    violations: list[PythonImportViolation] = []
    for policy in policies:
        source_directory = package_root / policy.source_directory
        if not source_directory.exists():
            continue
        for path in python_source_paths_for_policy(source_directory):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            relative_path = path.relative_to(package_root.parent)
            for statement in ast.walk(tree):
                if not isinstance(statement, ast.Import | ast.ImportFrom):
                    continue
                violations.extend(
                    collect_import_violations_for_statement(path, relative_path, package_root, policy, statement)
                )
    return tuple(violations)


def collect_python_call_policy_violations(
    package_root: Path,
    policies: tuple[PythonCallPolicy, ...] = PYTHON_CALL_POLICIES,
) -> tuple[PythonCallViolation, ...]:
    """Collect Python call-boundary violations under a production package root."""
    violations: list[PythonCallViolation] = []
    for policy in policies:
        source_directory = package_root / policy.source_directory
        if not source_directory.exists():
            continue
        for path in python_source_paths_for_policy(source_directory):
            relative_path = path.relative_to(package_root.parent)
            package_relative_path = path.relative_to(package_root)
            if package_relative_path in policy.allowed_paths:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for statement in ast.walk(tree):
                if not isinstance(statement, ast.Call):
                    continue
                violations.extend(collect_call_violations_for_statement(relative_path, policy, statement))
    return tuple(violations)


def collect_definition_violations_for_statement(
    relative_path: Path,
    policy: PythonDefinitionPolicy,
    statement: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[PythonDefinitionViolation, ...]:
    """Collect definition-policy violations from one AST function statement."""
    if statement.name not in policy.forbidden_function_names:
        return ()
    return (
        PythonDefinitionViolation(
            path=relative_path,
            line_number=statement.lineno,
            column_offset=statement.col_offset,
            policy_name=policy.name,
            function_name=statement.name,
            message=policy.message,
        ),
    )


def collect_python_definition_policy_violations(
    package_root: Path,
    policies: tuple[PythonDefinitionPolicy, ...] = PYTHON_DEFINITION_POLICIES,
) -> tuple[PythonDefinitionViolation, ...]:
    """Collect Python definition-boundary violations under a production package root."""
    violations: list[PythonDefinitionViolation] = []
    for policy in policies:
        source_directory = package_root / policy.source_directory
        if not source_directory.exists():
            continue
        for path in python_source_paths_for_policy(source_directory):
            relative_path = path.relative_to(package_root.parent)
            package_relative_path = path.relative_to(package_root)
            if package_relative_path in policy.allowed_paths:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for statement in ast.walk(tree):
                if not isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                violations.extend(collect_definition_violations_for_statement(relative_path, policy, statement))
    return tuple(violations)


def top_level_function_definitions(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """Return top-level function definitions by name."""
    function_definitions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for statement in tree.body:
        if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
            function_definitions[statement.name] = statement
    return function_definitions


def top_level_string_assignments(tree: ast.Module) -> dict[str, ModuleStringAssignment]:
    """Return top-level string assignments by target name."""
    assignments: dict[str, ModuleStringAssignment] = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if (
                    isinstance(target, ast.Name)
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)
                ):
                    assignments[target.id] = ModuleStringAssignment(
                        value=statement.value.value,
                        line_number=statement.lineno,
                        column_offset=statement.col_offset,
                    )
            continue
        if (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            assignments[statement.target.id] = ModuleStringAssignment(
                value=statement.value.value,
                line_number=statement.lineno,
                column_offset=statement.col_offset,
            )
    return assignments


def call_names_under_node(node: ast.AST) -> frozenset[str]:
    """Return dotted call names under an AST node."""
    call_names: set[str] = set()
    for child_node in ast.walk(node):
        if not isinstance(child_node, ast.Call):
            continue
        call_name = call_name_from_expression(child_node.func)
        if call_name is not None:
            call_names.add(call_name)
    return frozenset(call_names)


def node_references_name(node: ast.AST, name: str) -> bool:
    """Return whether an AST node references a name."""
    return any(isinstance(child_node, ast.Name) and child_node.id == name for child_node in ast.walk(node))


def call_names_include(call_names: frozenset[str], expected_call_name: str) -> bool:
    """Return whether collected call names include the expected call name."""
    return expected_call_name in call_names or any(
        call_name.endswith(f".{expected_call_name}") for call_name in call_names
    )


def function_contains_call(
    function_definition: ast.FunctionDef | ast.AsyncFunctionDef, expected_call_name: str
) -> bool:
    """Return whether a function contains a call."""
    return call_names_include(call_names_under_node(function_definition), expected_call_name)


def function_has_sentinel_legacy_branch(function_definition: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether a function has a sentinel-gated legacy backend branch."""
    for statement in function_definition.body:
        if not isinstance(statement, ast.If):
            continue
        if not node_references_name(statement.test, CLI_SHIM_SENTINEL_CONSTANT_NAME):
            continue
        if any(
            call_names_include(call_names_under_node(body_statement), "run_args_legacy")
            for body_statement in statement.body
        ):
            return True
    return False


def cli_shim_violation(
    relative_path: Path,
    line_number: int,
    column_offset: int,
    subject: str,
) -> PythonCliShimViolation:
    """Build a CLI shim violation."""
    return PythonCliShimViolation(
        path=relative_path,
        line_number=line_number,
        column_offset=column_offset,
        policy_name=CLI_SHIM_POLICY_NAME,
        subject=subject,
        message=CLI_SHIM_MESSAGE,
    )


def collect_python_cli_shim_violations(package_root: Path) -> tuple[PythonCliShimViolation, ...]:
    """Collect Python CLI shim ownership violations."""
    cli_path = package_root / CLI_SHIM_PATH
    relative_path = cli_path.relative_to(package_root.parent)
    if not cli_path.is_file():
        return (cli_shim_violation(relative_path, 1, 0, str(CLI_SHIM_PATH)),)

    tree = ast.parse(cli_path.read_text(encoding="utf-8"), filename=str(cli_path))
    violations: list[PythonCliShimViolation] = []
    string_assignments = top_level_string_assignments(tree)
    sentinel_assignment = string_assignments.get(CLI_SHIM_SENTINEL_CONSTANT_NAME)
    if sentinel_assignment is None:
        violations.append(cli_shim_violation(relative_path, 1, 0, CLI_SHIM_SENTINEL_CONSTANT_NAME))
    elif sentinel_assignment.value != CLI_SHIM_SENTINEL_ENVIRONMENT_VARIABLE:
        violations.append(
            cli_shim_violation(
                relative_path,
                sentinel_assignment.line_number,
                sentinel_assignment.column_offset,
                CLI_SHIM_SENTINEL_CONSTANT_NAME,
            )
        )

    function_definitions = top_level_function_definitions(tree)
    run_args_definition = function_definitions.get("run_args")
    if run_args_definition is None:
        violations.append(cli_shim_violation(relative_path, 1, 0, "run_args"))
    else:
        if not function_contains_call(run_args_definition, "run_native_cli_python_bridge"):
            violations.append(
                cli_shim_violation(
                    relative_path,
                    run_args_definition.lineno,
                    run_args_definition.col_offset,
                    "run_args",
                )
            )
        if function_contains_call(run_args_definition, "dispatch_cli"):
            violations.append(
                cli_shim_violation(
                    relative_path,
                    run_args_definition.lineno,
                    run_args_definition.col_offset,
                    "run_args",
                )
            )
        if not function_has_sentinel_legacy_branch(run_args_definition):
            violations.append(
                cli_shim_violation(
                    relative_path,
                    run_args_definition.lineno,
                    run_args_definition.col_offset,
                    "run_args",
                )
            )

    run_args_legacy_definition = function_definitions.get("run_args_legacy")
    if run_args_legacy_definition is None:
        violations.append(cli_shim_violation(relative_path, 1, 0, "run_args_legacy"))
    else:
        if not function_contains_call(run_args_legacy_definition, "dispatch_cli"):
            violations.append(
                cli_shim_violation(
                    relative_path,
                    run_args_legacy_definition.lineno,
                    run_args_legacy_definition.col_offset,
                    "run_args_legacy",
                )
            )
        if function_contains_call(run_args_legacy_definition, "run_native_cli_python_bridge"):
            violations.append(
                cli_shim_violation(
                    relative_path,
                    run_args_legacy_definition.lineno,
                    run_args_legacy_definition.col_offset,
                    "run_args_legacy",
                )
            )

    main_definition = function_definitions.get("main")
    if main_definition is None:
        violations.append(cli_shim_violation(relative_path, 1, 0, "main"))
    elif not function_contains_call(main_definition, "run_args"):
        violations.append(cli_shim_violation(relative_path, main_definition.lineno, main_definition.col_offset, "main"))

    return tuple(violations)


def python_source_paths_for_policy(source_path: Path) -> tuple[Path, ...]:
    """Return Python source paths covered by one architecture policy."""
    if source_path.is_file():
        if source_path.suffix == ".py":
            return (source_path,)
        return ()
    return tuple(sorted(source_path.rglob("*.py")))


def render_violation(violation: PythonImportViolation) -> str:
    """Render an import-policy violation for command-line output."""
    location = f"{violation.path}:{violation.line_number}:{violation.column_offset + 1}"
    return (
        f"{location}: {violation.policy_name} rejects `{violation.import_name}` "
        f"via `{violation.forbidden_import}`: {violation.message}"
    )


def render_call_violation(violation: PythonCallViolation) -> str:
    """Render a call-policy violation for command-line output."""
    location = f"{violation.path}:{violation.line_number}:{violation.column_offset + 1}"
    return (
        f"{location}: {violation.policy_name} rejects `{violation.call_name}` "
        f"via `{violation.forbidden_call}`: {violation.message}"
    )


def render_definition_violation(violation: PythonDefinitionViolation) -> str:
    """Render a definition-policy violation for command-line output."""
    location = f"{violation.path}:{violation.line_number}:{violation.column_offset + 1}"
    return f"{location}: {violation.policy_name} rejects definition `{violation.function_name}`: {violation.message}"


def render_cli_shim_violation(violation: PythonCliShimViolation) -> str:
    """Render a CLI shim-policy violation for command-line output."""
    location = f"{violation.path}:{violation.line_number}:{violation.column_offset + 1}"
    return f"{location}: {violation.policy_name} rejects `{violation.subject}`: {violation.message}"


def run_tool(package_root: Path) -> int:
    """Verify Python package ownership boundaries."""
    import_violations = collect_python_import_policy_violations(package_root)
    call_violations = collect_python_call_policy_violations(package_root)
    definition_violations = collect_python_definition_policy_violations(package_root)
    cli_shim_violations = collect_python_cli_shim_violations(package_root)
    if import_violations or call_violations or definition_violations or cli_shim_violations:
        print(f"Python architecture violations under `{package_root}`:")
        for violation in import_violations:
            print(f"  {render_violation(violation)}")
        for violation in call_violations:
            print(f"  {render_call_violation(violation)}")
        for violation in definition_violations:
            print(f"  {render_definition_violation(violation)}")
        for violation in cli_shim_violations:
            print(f"  {render_cli_shim_violation(violation)}")
        return 1

    print(f"Python architecture policy passed for `{package_root}`.")
    return 0


@hydra.main(version_base=None, config_path="../configs", config_name="debug_check_python_architecture")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Run the Python architecture checker from Hydra configuration."""
    del config
    exit_code = run_tool(PRODUCTION_PACKAGE_ROOT)
    if exit_code:
        raise SystemExit(exit_code)


def main() -> int:
    """Run the Python architecture checker from default Hydra configuration."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
