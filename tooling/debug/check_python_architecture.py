#!/usr/bin/env python3
"""Verify current Python numerical and Rust orchestration ownership boundaries."""

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
        allowed_paths: Source paths, relative to the production package root, excluded from this policy.

    """

    name: str
    source_directory: Path
    forbidden_imports: tuple[str, ...]
    message: str
    allowed_paths: tuple[Path, ...] = ()


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


CLI_SHIM_PATH = Path("cli.py")
CLI_SHIM_POLICY_NAME = "native_cli_shim_process_owner"
CLI_SHIM_MESSAGE = (
    "the public Python CLI entry point must stay thin: call g._core.cli.run directly, forward its console output, "
    "and avoid removed Python orchestration or sentinel paths"
)

REMOVED_ORCHESTRATION_IMPORTS = (
    "g.api",
    "g.engine",
    "g.execution_plan",
    "g.interface",
    "g.io",
    "g.jax_runtime",
    "g.runner",
)
FILE_AND_PROCESS_IMPORTS = (
    "asyncio",
    "bgen_reader",
    "concurrent.futures",
    "csv",
    "h5py",
    "json",
    "multiprocessing",
    "pandas",
    "pathlib",
    "polars",
    "pyarrow",
    "queue",
    "subprocess",
    "threading",
)
FILE_IO_CALLS = (
    "open",
    "read_text",
    "write_text",
    "read_bytes",
    "write_bytes",
    "numpy.load",
    "numpy.loadtxt",
    "numpy.genfromtxt",
    "numpy.save",
    "numpy.savetxt",
    "numpy.savez",
    "numpy.savez_compressed",
    "jax.numpy.load",
    "jax.numpy.save",
)

PYTHON_IMPORT_POLICIES = (
    PythonImportPolicy(
        name="rust_orchestration_ownership",
        source_directory=Path(),
        forbidden_imports=REMOVED_ORCHESTRATION_IMPORTS,
        message="planning, input, scheduling, lifecycle and output belong to Rust",
    ),
    PythonImportPolicy(
        name="native_binding_entrypoint_ownership",
        source_directory=Path(),
        forbidden_imports=("g._core",),
        allowed_paths=(CLI_SHIM_PATH,),
        message="only the console forwarding shim imports native orchestration bindings",
    ),
    PythonImportPolicy(
        name="compute_kernel_isolation",
        source_directory=Path("compute"),
        forbidden_imports=("g.cli", "g.jax_backend", "g.backend", *FILE_AND_PROCESS_IMPORTS),
        message="compute kernels consume numerical operands without transport, orchestration or file I/O",
    ),
    *(
        PythonImportPolicy(
            name="numerical_backend_isolation",
            source_directory=source_path,
            forbidden_imports=("g.cli", *FILE_AND_PROCESS_IMPORTS),
            message="the numerical backend and its helpers must not own orchestration, files or worker queues",
        )
        for source_path in (Path("jax_backend.py"), Path("backend"))
    ),
    PythonImportPolicy(
        name="cli_numerical_import_isolation",
        source_directory=CLI_SHIM_PATH,
        forbidden_imports=("g.compute", "g.jax_backend", "g.backend", "jax", "jaxlib"),
        message="the console shim must leave numerical backend initialization to the native runtime",
    ),
)

PYTHON_CALL_POLICIES = (
    PythonCallPolicy(
        name="native_orchestration_call_ownership",
        source_directory=Path(),
        forbidden_calls=("g._core.*",),
        allowed_paths=(CLI_SHIM_PATH,),
        message="only the console shim may invoke native CLI orchestration",
    ),
    PythonCallPolicy(
        name="rust_runtime_setup_ownership",
        source_directory=Path(),
        forbidden_calls=("jax.config.update", "jax.devices"),
        allowed_paths=(),
        message="Rust initializes the runtime before importing numerical Python code",
    ),
    *(
        PythonCallPolicy(
            name="numerical_file_io_isolation",
            source_directory=source_path,
            forbidden_calls=FILE_IO_CALLS,
            allowed_paths=(),
            message="numerical kernels, transport and materialization must not read or write files",
        )
        for source_path in (Path("compute"), Path("jax_backend.py"), Path("backend"))
    ),
    PythonCallPolicy(
        name="compute_host_materialization_isolation",
        source_directory=Path("compute"),
        forbidden_calls=("jax.device_get",),
        allowed_paths=(),
        message="the backend owns host materialization after compute completes",
    ),
)

PYTHON_DEFINITION_POLICIES = (
    PythonDefinitionPolicy(
        name="removed_orchestration_definition_isolation",
        source_directory=Path(),
        forbidden_function_names=(
            "dispatch_cli",
            "run_validated_cli_outcome",
            "write_run_manifest",
            "prepare_output_run",
            "log_callback_progress_event",
            "log_binary_correction_summary",
        ),
        allowed_paths=(),
        message="Python must not restore removed planning, output or event-dispatch owners",
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


def collect_import_aliases(
    path: Path,
    package_root: Path,
    tree: ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> dict[str, frozenset[str]]:
    """Resolve all imported targets in one scope without absorbing nested imports."""
    aliases: dict[str, frozenset[str]] = {}
    pending_statements: list[ast.AST] = list(tree.body)
    while pending_statements:
        statement = pending_statements.pop()
        if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda):
            continue
        pending_statements.extend(ast.iter_child_nodes(statement))
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                local_name = alias.asname or alias.name.split(".")[0]
                imported_name = alias.name if alias.asname else alias.name.split(".")[0]
                aliases[local_name] = aliases.get(local_name, frozenset()) | {imported_name}
        elif isinstance(statement, ast.ImportFrom):
            base_module = import_from_base_module(path, package_root, statement)
            for alias in statement.names:
                if alias.name != "*":
                    local_name = alias.asname or alias.name
                    aliases[local_name] = aliases.get(local_name, frozenset()) | {f"{base_module}.{alias.name}"}
    return aliases


def collect_call_violations_for_statement(
    relative_path: Path,
    policy: PythonCallPolicy,
    statement: ast.Call,
    import_aliases: dict[str, frozenset[str]],
) -> tuple[PythonCallViolation, ...]:
    """Collect call-policy violations from one AST call statement."""
    call_name = call_name_from_expression(statement.func)
    if call_name is None:
        return ()
    first_component, separator, remainder = call_name.partition(".")
    call_names = (
        sorted(f"{import_name}{separator}{remainder}" for import_name in import_aliases[first_component])
        if first_component in import_aliases
        else [call_name]
    )
    for resolved_call_name in call_names:
        for forbidden_call in policy.forbidden_calls:
            if call_matches_forbidden_name(resolved_call_name, forbidden_call):
                return (
                    PythonCallViolation(
                        path=relative_path,
                        line_number=statement.lineno,
                        column_offset=statement.col_offset,
                        policy_name=policy.name,
                        call_name=resolved_call_name,
                        forbidden_call=forbidden_call,
                        message=policy.message,
                    ),
                )
    return ()


def function_parameter_names(arguments: ast.arguments) -> frozenset[str]:
    """Return parameter bindings that shadow names from an enclosing scope."""
    names = {parameter.arg for parameter in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)}
    if arguments.vararg is not None:
        names.add(arguments.vararg.arg)
    if arguments.kwarg is not None:
        names.add(arguments.kwarg.arg)
    return frozenset(names)


def local_assignment_names(statement: ast.FunctionDef | ast.AsyncFunctionDef) -> frozenset[str]:
    """Collect local assignments without absorbing nested lexical scopes."""
    names = set(function_parameter_names(statement.args))
    external_names: set[str] = set()
    pending_nodes: list[ast.AST] = list(statement.body)
    while pending_nodes:
        node = pending_nodes.pop()
        if isinstance(
            node,
            ast.FunctionDef
            | ast.AsyncFunctionDef
            | ast.ClassDef
            | ast.Lambda
            | ast.ListComp
            | ast.SetComp
            | ast.DictComp
            | ast.GeneratorExp,
        ):
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.Global | ast.Nonlocal):
            external_names.update(node.names)
        pending_nodes.extend(ast.iter_child_nodes(node))
    return frozenset(names - external_names)


class PythonCallPolicyVisitor(ast.NodeVisitor):
    """Resolve call aliases within each module, function and class scope."""

    def __init__(
        self,
        path: Path,
        relative_path: Path,
        package_root: Path,
        policy: PythonCallPolicy,
        tree: ast.Module,
    ) -> None:
        """Initialize the policy visitor with the module's own import bindings."""
        self.path = path
        self.relative_path = relative_path
        self.package_root = package_root
        self.policy = policy
        self.import_aliases = collect_import_aliases(path, package_root, tree)
        self.class_parent_aliases: dict[str, frozenset[str]] | None = None
        self.violations: list[PythonCallViolation] = []

    def visit_Call(self, node: ast.Call) -> None:
        """Check one call against aliases from its owning lexical scope."""
        self.violations.extend(
            collect_call_violations_for_statement(self.relative_path, self.policy, node, self.import_aliases)
        )
        self.generic_visit(node)

    def visit_scope_body(
        self,
        statement: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        *,
        function_scope: bool,
    ) -> None:
        """Visit a scope while isolating locally imported bindings from siblings."""
        saved_aliases = self.import_aliases
        saved_class_parent_aliases = self.class_parent_aliases
        enclosing_aliases = (
            self.class_parent_aliases
            if function_scope and self.class_parent_aliases is not None
            else self.import_aliases
        )
        if function_scope:
            # Class namespaces do not form lexical closures for their methods.
            self.class_parent_aliases = None
        shadowed_names = (
            local_assignment_names(statement)
            if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
            else frozenset()
        )
        self.import_aliases = {
            **{name: targets for name, targets in enclosing_aliases.items() if name not in shadowed_names},
            **collect_import_aliases(self.path, self.package_root, statement),
        }
        for body_statement in statement.body:
            self.visit(body_statement)
        self.import_aliases = saved_aliases
        self.class_parent_aliases = saved_class_parent_aliases

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Evaluate function annotations and decorators before entering its body."""
        for decorator in node.decorator_list:
            self.visit(decorator)
        self.visit(node.args)
        if node.returns is not None:
            self.visit(node.returns)
        self.visit_scope_body(node, function_scope=True)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Check asynchronous definitions with the same lexical-scope rules."""
        for decorator in node.decorator_list:
            self.visit(decorator)
        self.visit(node.args)
        if node.returns is not None:
            self.visit(node.returns)
        self.visit_scope_body(node, function_scope=True)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Evaluate class construction in the enclosing scope, then its body."""
        for expression in (*node.decorator_list, *node.bases):
            self.visit(expression)
        for keyword in node.keywords:
            self.visit(keyword)
        enclosing_class_parent_aliases = self.class_parent_aliases
        if self.class_parent_aliases is None:
            self.class_parent_aliases = self.import_aliases
        self.visit_scope_body(node, function_scope=False)
        self.class_parent_aliases = enclosing_class_parent_aliases

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Keep lambda parameters from resolving as imported module aliases."""
        self.visit(node.args)
        saved_aliases = self.import_aliases
        enclosing_aliases = self.class_parent_aliases if self.class_parent_aliases is not None else saved_aliases
        shadowed_names = function_parameter_names(node.args)
        self.import_aliases = {
            name: targets for name, targets in enclosing_aliases.items() if name not in shadowed_names
        }
        self.visit(node.body)
        self.import_aliases = saved_aliases

    def visit_comprehension_scope(
        self,
        node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
    ) -> None:
        """Keep comprehension targets local while checking the outer iterable."""
        self.visit(node.generators[0].iter)
        saved_aliases = self.import_aliases
        enclosing_aliases = self.class_parent_aliases if self.class_parent_aliases is not None else saved_aliases
        shadowed_names = {
            child.id
            for generator in node.generators
            for child in ast.walk(generator.target)
            if isinstance(child, ast.Name)
        }
        self.import_aliases = {
            name: targets for name, targets in enclosing_aliases.items() if name not in shadowed_names
        }
        for generator in node.generators[1:]:
            self.visit(generator.iter)
        for generator in node.generators:
            for condition in generator.ifs:
                self.visit(condition)
        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        else:
            self.visit(node.elt)
        self.import_aliases = saved_aliases

    def visit_ListComp(self, node: ast.ListComp) -> None:
        """Check a list comprehension in its own lexical scope."""
        self.visit_comprehension_scope(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        """Check a set comprehension in its own lexical scope."""
        self.visit_comprehension_scope(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        """Check a dictionary comprehension in its own lexical scope."""
        self.visit_comprehension_scope(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        """Check a generator expression in its own lexical scope."""
        self.visit_comprehension_scope(node)


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
            relative_path = path.relative_to(package_root.parent)
            package_relative_path = path.relative_to(package_root)
            if package_relative_path in policy.allowed_paths:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
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
            visitor = PythonCallPolicyVisitor(path, relative_path, package_root, policy, tree)
            visitor.visit(tree)
            violations.extend(visitor.violations)
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
    forbidden_names = frozenset({"NATIVE_CLI_PYTHON_BRIDGE_SENTINEL_ENVIRONMENT_VARIABLE"})
    for child_node in ast.walk(tree):
        if not isinstance(child_node, ast.Name):
            continue
        if child_node.id in forbidden_names:
            violations.append(
                cli_shim_violation(relative_path, child_node.lineno, child_node.col_offset, child_node.id)
            )
            break

    function_definitions = top_level_function_definitions(tree)
    run_definition = function_definitions.get("run")
    if run_definition is None:
        violations.append(cli_shim_violation(relative_path, 1, 0, "run"))
    else:
        if not function_contains_call(run_definition, "_core.cli.run"):
            violations.append(
                cli_shim_violation(
                    relative_path,
                    run_definition.lineno,
                    run_definition.col_offset,
                    "run",
                )
            )

    for removed_function_name in ("run_args", "run_args_legacy", "run_native_cli_python_bridge"):
        removed_function_definition = function_definitions.get(removed_function_name)
        if removed_function_definition is not None:
            violations.append(
                cli_shim_violation(
                    relative_path,
                    removed_function_definition.lineno,
                    removed_function_definition.col_offset,
                    removed_function_name,
                )
            )

    main_definition = function_definitions.get("main")
    if main_definition is None:
        violations.append(cli_shim_violation(relative_path, 1, 0, "main"))
    elif not function_contains_call(main_definition, "run"):
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
    required_sources = (
        CLI_SHIM_PATH,
        Path("jax_backend.py"),
        Path("compute/common/genotype.py"),
        Path("compute/regenie2_linear/state.py"),
        Path("compute/regenie2_binary/state.py"),
    )
    missing_sources = [source_path for source_path in required_sources if not (package_root / source_path).exists()]
    if missing_sources:
        print(f"Python architecture source coverage is incomplete under `{package_root}`: {missing_sources}")
        return 1
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
