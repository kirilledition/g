"""Reject unreachable maintained Python modules using reviewed repository roots.

This module-level gate intentionally does not infer function or method liveness.
Native dispatch, JAX decorators and registry callbacks need symbol-level review.
"""

from __future__ import annotations

import ast
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path()
SOURCE_ROOTS = (Path("src/g"), Path("tooling"))
SOURCE_ANCHORS = (
    Path("pyproject.toml"),
    Path("Justfile"),
    Path("src/binding/engine.rs"),
    Path("src/g/cli.py"),
    Path("src/g/jax_backend.py"),
    Path("src/g/compute/common/genotype.py"),
    Path("src/g/compute/regenie2_linear/state.py"),
    Path("src/g/compute/regenie2_binary/state.py"),
    Path("tooling/cli/debug.py"),
    Path("tooling/cli/benchmark.py"),
    Path("tooling/data/fetch.py"),
)
# Conservative floors reject an accidentally truncated snapshot. Update these
# alongside a deliberate reduction of the maintained source scope.
MINIMUM_PRODUCTION_SOURCE_COUNT = 30
MINIMUM_TOOLING_SOURCE_COUNT = 60
# These direct module CLIs remain supported independently of the grouped CLI.
STANDALONE_MODULE_ROOTS = frozenset({"tooling.data.fetch", "tooling.data.simulate"})
MODULE_COMMAND_PATTERN = re.compile(r"(?:^|\s)-m\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)")
NATIVE_IMPORT_PATTERN = re.compile(r'PyModule::import\s*\([^,]+,\s*"([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)"')


@dataclass(frozen=True)
class PythonModuleScan:
    """Reachability and source coverage from one repository snapshot.

    Attributes:
        production_source_count: Number of scanned production Python files.
        tooling_source_count: Number of scanned tooling Python files.
        root_modules: Maintained entrypoints and externally referenced modules.
        unreachable_paths: Non-marker Python modules without a path from a root.

    """

    production_source_count: int
    tooling_source_count: int
    root_modules: frozenset[str]
    unreachable_paths: tuple[Path, ...]


def module_name_for_path(path: Path) -> str:
    """Return a module name for a repository-relative maintained Python path."""
    module_path = path.with_suffix("")
    if module_path.parts[0] == "src":
        module_path = Path(*module_path.parts[1:])
    module_parts = module_path.parts
    if module_parts[-1] == "__init__":
        module_parts = module_parts[:-1]
    return ".".join(module_parts)


def imported_module_names(tree: ast.Module, module_name: str, *, package_marker: bool) -> frozenset[str]:
    """Collect imports and literal dynamic module references without executing code."""
    references: set[str] = set()
    package_parts = module_name.split(".") if package_marker else module_name.split(".")[:-1]
    for statement in ast.walk(tree):
        if isinstance(statement, ast.Import):
            references.update(alias.name for alias in statement.names)
        elif isinstance(statement, ast.ImportFrom):
            base_name = statement.module or ""
            if statement.level:
                retained_count = len(package_parts) - statement.level + 1
                base_name = ".".join((*package_parts[:retained_count], *base_name.split("."))).rstrip(".")
            if base_name:
                references.add(base_name)
                references.update(f"{base_name}.{alias.name}" for alias in statement.names if alias.name != "*")
        elif isinstance(statement, ast.Call) and statement.args:
            function = statement.func
            function_name = (
                function.id
                if isinstance(function, ast.Name)
                else function.attr
                if isinstance(function, ast.Attribute)
                else ""
            )
            target = statement.args[0]
            if (
                function_name in {"import_module", "__import__", "find_spec"}
                and isinstance(target, ast.Constant)
                and isinstance(target.value, str)
            ):
                references.add(target.value)
        elif isinstance(statement, ast.List | ast.Tuple):
            for preceding, following in zip(statement.elts, statement.elts[1:], strict=False):
                if (
                    isinstance(preceding, ast.Constant)
                    and preceding.value == "-m"
                    and isinstance(following, ast.Constant)
                    and isinstance(following.value, str)
                ):
                    references.add(following.value)
    return frozenset(references)


def maintained_modules(repository_root: Path) -> dict[str, Path]:
    """Index bounded maintained source directories instead of the working tree."""
    return {
        module_name_for_path(path.relative_to(repository_root)): path.relative_to(repository_root)
        for source_root in SOURCE_ROOTS
        for path in sorted((repository_root / source_root).rglob("*.py"))
    }


def console_module_roots(repository_root: Path) -> frozenset[str]:
    """Read declared console entrypoints from project metadata."""
    metadata_path = repository_root / "pyproject.toml"
    if not metadata_path.is_file():
        return frozenset()
    with metadata_path.open("rb") as metadata_file:
        metadata = tomllib.load(metadata_file)
    script_values = metadata.get("project", {}).get("scripts", {}).values()
    return frozenset(value.partition(":")[0] for value in script_values if isinstance(value, str))


def external_module_roots(repository_root: Path) -> frozenset[str]:
    """Read test, script, workflow, Hydra and Rust references as explicit roots."""
    roots = set(console_module_roots(repository_root)) | set(STANDALONE_MODULE_ROOTS)
    for source_directory in (Path("tests"), Path("scripts")):
        for path in sorted((repository_root / source_directory).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            roots.update(imported_module_names(tree, "", package_marker=False))
    command_paths = [repository_root / "Justfile", *sorted((repository_root / ".github/workflows").glob("*.yml"))]
    for path in command_paths:
        if path.is_file():
            roots.update(MODULE_COMMAND_PATTERN.findall(path.read_text(encoding="utf-8")))
    for path in sorted((repository_root / "src/binding").rglob("*.rs")):
        roots.update(NATIVE_IMPORT_PATTERN.findall(path.read_text(encoding="utf-8")))
    # Hydra targets, if configured, are explicit entrypoints. Ordinary config
    # values and diagnostic strings are not interpreted as module roots.
    for path in sorted((repository_root / "tooling/configs").rglob("*.yaml")):
        for target in re.findall(r"^\s*_target_:\s*['\"]?([\w.]+)", path.read_text(encoding="utf-8"), re.MULTILINE):
            roots.add(target.rpartition(".")[0])
    return frozenset(roots)


def reachable_module_names(graph: dict[str, frozenset[str]], roots: frozenset[str]) -> frozenset[str]:
    """Follow imports, dynamic literal edges and containing package markers."""
    pending_modules = list(roots & graph.keys())
    reachable_modules: set[str] = set()
    while pending_modules:
        module_name = pending_modules.pop()
        if module_name in reachable_modules:
            continue
        reachable_modules.add(module_name)
        pending_modules.extend(graph[module_name] - reachable_modules)
        package_name = module_name.rpartition(".")[0]
        if package_name in graph and package_name not in reachable_modules:
            pending_modules.append(package_name)
    return frozenset(reachable_modules)


def collect_module_scan(repository_root: Path) -> PythonModuleScan:
    """Scan all maintained modules and identify modules outside reviewed roots."""
    modules = maintained_modules(repository_root)
    graph: dict[str, frozenset[str]] = {}
    empty_package_markers: set[str] = set()
    for module_name, relative_path in modules.items():
        source_path = repository_root / relative_path
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        if relative_path.name == "__init__.py" and all(
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
            for statement in tree.body
        ):
            empty_package_markers.add(module_name)
        references = imported_module_names(tree, module_name, package_marker=relative_path.name == "__init__.py")
        graph[module_name] = references & modules.keys()
    roots = external_module_roots(repository_root) & modules.keys()
    reachable_modules = reachable_module_names(graph, roots)
    return PythonModuleScan(
        production_source_count=sum(path.parts[0] == "src" for path in modules.values()),
        tooling_source_count=sum(path.parts[0] == "tooling" for path in modules.values()),
        root_modules=roots,
        unreachable_paths=tuple(
            sorted(
                path
                for module_name, path in modules.items()
                if module_name not in reachable_modules and module_name not in empty_package_markers
            )
        ),
    )


def source_coverage_errors(repository_root: Path, scan: PythonModuleScan) -> tuple[str, ...]:
    """Reject missing anchors and unexpectedly narrow or empty source scans."""
    errors = [
        f"missing maintained source anchor: {path}" for path in SOURCE_ANCHORS if not (repository_root / path).is_file()
    ]
    if not (repository_root / "src/binding").is_dir():
        errors.append("missing native binding roots: src/binding")
    if not (repository_root / "tests/tooling").is_dir():
        errors.append("missing maintained test roots: tests/tooling")
    if scan.production_source_count < MINIMUM_PRODUCTION_SOURCE_COUNT:
        errors.append(
            f"scanned only {scan.production_source_count} production files; "
            f"expected at least {MINIMUM_PRODUCTION_SOURCE_COUNT}"
        )
    if scan.tooling_source_count < MINIMUM_TOOLING_SOURCE_COUNT:
        errors.append(
            f"scanned only {scan.tooling_source_count} tooling files; expected at least {MINIMUM_TOOLING_SOURCE_COUNT}"
        )
    if not scan.root_modules:
        errors.append("no maintained entrypoint roots were resolved")
    return tuple(errors)


def run_tool(repository_root: Path) -> int:
    """Report module reachability and fail closed on incomplete source coverage."""
    scan = collect_module_scan(repository_root)
    coverage_errors = source_coverage_errors(repository_root, scan)
    print(
        f"Python module reachability: {scan.production_source_count} production files, "
        f"{scan.tooling_source_count} tooling files, {len(scan.root_modules)} resolved roots."
    )
    for error in coverage_errors:
        print(f"Source coverage error: {error}")
    for path in scan.unreachable_paths:
        print(f"Unreachable maintained module: {path}")
    return int(bool(coverage_errors or scan.unreachable_paths))


def main() -> None:
    """Run the maintained Python module reachability gate."""
    raise SystemExit(run_tool(REPOSITORY_ROOT))


if __name__ == "__main__":
    main()
