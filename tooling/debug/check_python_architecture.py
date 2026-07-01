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
        forbidden_calls=("output.write_run_manifest", "_core.write_run_manifest_json", "write_run_manifest"),
        allowed_paths=(Path("io/output.py"),),
        message="production Python must not write run manifests outside the output adapter helper",
    ),
    PythonCallPolicy(
        name="native_output_lifecycle_adapter_isolation",
        source_directory=Path(),
        forbidden_calls=(
            "_core.prepare_output_run",
            "_core.initialize_output_run",
            "_core.load_run_manifest_json",
            "_core.validate_run_manifest_compatibility",
            "_core.read_manifest_committed_chunk_identifiers",
            "_core.validate_strict_manifest_chunks",
            "_core.repair_strict_manifest_chunk_commits",
            "_core.scan_committed_chunk_identifiers",
            "_core.finalize_output_run_chunks",
            "_core.resolve_output_run_paths",
            "_core.NativePipelineOutputPreparationBatch",
            "_core.initialize_pipeline_output_run_batch",
            "_core.initialize_pipeline_output_runs",
        ),
        allowed_paths=(Path("io/output.py"),),
        message="production Python must route native output lifecycle calls through the output adapter helper",
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
        for path in sorted(source_directory.rglob("*.py")):
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
        for path in sorted(source_directory.rglob("*.py")):
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
        for path in sorted(source_directory.rglob("*.py")):
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


def run_tool(package_root: Path) -> int:
    """Verify Python package ownership boundaries."""
    import_violations = collect_python_import_policy_violations(package_root)
    call_violations = collect_python_call_policy_violations(package_root)
    definition_violations = collect_python_definition_policy_violations(package_root)
    if import_violations or call_violations or definition_violations:
        print(f"Python architecture violations under `{package_root}`:")
        for violation in import_violations:
            print(f"  {render_violation(violation)}")
        for violation in call_violations:
            print(f"  {render_call_violation(violation)}")
        for violation in definition_violations:
            print(f"  {render_definition_violation(violation)}")
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
