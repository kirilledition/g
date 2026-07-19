#!/usr/bin/env python3
"""Verify that the Justfile remains a thin workflow entrypoint layer."""

from __future__ import annotations

import enum
import re
import sys
import typing
from dataclasses import dataclass
from pathlib import Path

import hydra

from tooling.common import hydra_arguments as tooling_hydra_arguments
from tooling.common import hydra_compat as tooling_hydra_compat

if typing.TYPE_CHECKING:
    import omegaconf

DEFAULT_JUSTFILE_PATH = Path("Justfile")
DEFAULT_DOCUMENTATION_PATHS = (
    Path("documentation/development/justfile.md"),
    Path("documentation/development/tooling.md"),
    Path("documentation/development/server-gauss-slurm.md"),
    Path("documentation/development/symphony.md"),
    Path("WORKFLOW.md"),
)
ALLOWED_RECIPE_PREFIXES = {
    "bench",
    "check",
    "ci",
    "coverage",
    "cuda",
    "data",
    "dev",
    "docs",
    "doctor",
    "format",
    "help",
    "legacy",
    "lint",
    "matrix",
    "perf",
    "profile",
    "rust",
    "server",
    "slurm",
    "symphony",
    "test",
    "typecheck",
    "upgrade",
    "workspace",
}
EXACT_RECIPE_NAMES = {"default"}
LEGACY_RECIPE_NAMES = {
    "benchmark-baselines",
    "benchmark-baselines-full",
    "benchmark-bgen-reader",
    "benchmark-callback-overhead",
    "benchmark-callback-overhead-gpu",
    "benchmark-output-stages-gpu",
    "benchmark-regenie-comparison",
    "benchmark-regenie-comparison-cpu",
    "benchmark-regenie-comparison-gpu",
    "benchmark-regenie2-binary-hot-gpu",
    "benchmark-regenie2-binary-hot-gpu-smoke",
    "benchmark-regenie2-linear-fresh-gpu",
    "benchmark-regenie2-linear-fresh-gpu-parquet",
    "install-perf-extension",
    "profile-app-full-dry-run",
    "profile-app-full-landau",
    "profile-chr10-gpu-binary-deep",
    "profile-chr10-gpu-binary-deep-dry-run",
    "profile-chr10-gpu-binary-deep-landau",
    "profile-chr10-gpu-binary-deep-smoke",
    "profile-regenie-comparison",
    "profile-regenie-comparison-cpu",
    "profile-regenie-comparison-gpu",
    "profile-regenie2-deep-dry-run",
    "profile-regenie2-deep-landau",
    "profile-regenie2-deep-smoke",
    "regenie-linear",
    "regenie2-binary-gpu",
    "regenie2-binary-gpu-smoke",
    "regenie2-chr10-matrix",
    "regenie2-chr10-matrix-dry-run",
    "regenie2-chr22-matrix",
    "regenie2-chr22-matrix-dry-run",
    "setup-binary-baseline",
    "setup-data",
    "setup-regenie2-binary-gpu-inputs",
    "slurm-benchmark-regenie2-binary-hot-gpu",
    "slurm-regenie2-binary-gpu",
    "slurm-regenie2-binary-gpu-smoke",
    "slurm-regenie2-chr10-matrix",
    "slurm-regenie2-chr22-matrix",
    "verify-regenie2-binary-gpu-output",
    "verify-regenie2-binary-gpu-smoke-output",
}
HYDRA_TOOLING_MODULES = (
    "tooling.cli.benchmark ",
    "tooling.cli.benchmark_bgen_reader ",
    "tooling.cli.benchmark_callback_overhead ",
    "tooling.cli.benchmark_output_stages ",
    "tooling.cli.benchmark_regenie2_binary_hot ",
    "tooling.cli.data ",
    "tooling.cli.debug ",
    "tooling.cli.performance ",
    "tooling.cli.profile_regenie2_deep ",
    "tooling.cli.run_regenie2_matrix ",
    "tooling.cli.rust_build_profiles ",
    "tooling.cli.schema_check ",
    "tooling.cli.server ",
    "tooling.cli.tune_regenie2_gpu ",
)
HYDRA_OVERRIDE_PATTERN = re.compile(r"(?:^|\s)(?:dataset|machine|sweep|telemetry|tool|workload)(?:\.[A-Za-z0-9_]+)?=")
RECIPE_PATTERN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_-]*)(?:\s+[^:=].*)?:")
DOCUMENTED_RECIPE_PATTERN = re.compile(r"(?:\bjust\s+|^### `)([A-Za-z0-9][A-Za-z0-9_-]*)", re.MULTILINE)
INLINE_OVERRIDE_LIMIT = 3


class JustfileViolationKind(enum.StrEnum):
    """Kinds of Justfile guardrail violations."""

    DEPRECATED_RECIPE = "deprecated_recipe"
    DEPRECATED_REFERENCE = "deprecated_reference"
    DOCUMENTED_RECIPE_MISSING = "documented_recipe_missing"
    INLINE_HYDRA_OVERRIDES = "inline_hydra_overrides"
    INVALID_RECIPE_PREFIX = "invalid_recipe_prefix"
    MISSING_CONFIG_NAME = "missing_config_name"
    STALE_G_FLAG = "stale_g_flag"


@dataclass(frozen=True)
class JustRecipe:
    """Parsed Justfile recipe."""

    name: str
    line_number: int
    body_lines: tuple[str, ...]


@dataclass(frozen=True)
class JustfileViolation:
    """One Justfile guardrail violation."""

    kind: JustfileViolationKind
    path: Path
    line_number: int
    message: str


@dataclass(frozen=True)
class JustfileCheckArguments:
    """Resolved arguments for the Justfile guardrail."""

    justfile_path: Path
    documentation_paths: tuple[Path, ...]


def parse_just_recipes(justfile_path: Path) -> tuple[JustRecipe, ...]:
    """Parse recipe names and body lines from a Justfile."""
    lines = justfile_path.read_text(encoding="utf-8").splitlines()
    recipes: list[JustRecipe] = []
    current_name: str | None = None
    current_line_number = 0
    current_body_lines: list[str] = []

    for line_number, line in enumerate(lines, start=1):
        if line.startswith("set ") or ":=" in line:
            continue
        recipe_match = RECIPE_PATTERN.match(line)
        if recipe_match and not line.startswith((" ", "\t")):
            if current_name is not None:
                recipes.append(
                    JustRecipe(
                        name=current_name,
                        line_number=current_line_number,
                        body_lines=tuple(current_body_lines),
                    )
                )
            current_name = recipe_match.group(1)
            current_line_number = line_number
            current_body_lines = []
            continue
        if current_name is not None and (line.startswith((" ", "\t")) or line == ""):
            current_body_lines.append(line)

    if current_name is not None:
        recipes.append(
            JustRecipe(
                name=current_name,
                line_number=current_line_number,
                body_lines=tuple(current_body_lines),
            )
        )
    return tuple(recipes)


def recipe_prefix(recipe_name: str) -> str:
    """Return the naming-domain prefix for a recipe."""
    return recipe_name.split("-", maxsplit=1)[0]


def recipe_uses_hydra_tooling(recipe: JustRecipe) -> bool:
    """Return whether a recipe invokes a migrated Hydra tooling module."""
    body_text = "\n".join(recipe.body_lines)
    return any(module_name in body_text for module_name in HYDRA_TOOLING_MODULES)


def count_hydra_overrides(recipe: JustRecipe) -> int:
    """Count inline Hydra override tokens in a recipe body."""
    body_text = "\n".join(recipe.body_lines)
    return len(HYDRA_OVERRIDE_PATTERN.findall(body_text))


def collect_documented_recipe_references(documentation_paths: tuple[Path, ...]) -> tuple[tuple[Path, int, str], ...]:
    """Collect recipe references from maintained command documentation."""
    references: list[tuple[Path, int, str]] = []
    for documentation_path in documentation_paths:
        if not documentation_path.exists():
            continue
        lines = documentation_path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            for match in DOCUMENTED_RECIPE_PATTERN.finditer(line):
                recipe_name = match.group(1)
                if recipe_name.endswith("-"):
                    continue
                references.append((documentation_path, line_number, recipe_name))
    return tuple(references)


def collect_recipe_violations(justfile_path: Path, recipes: tuple[JustRecipe, ...]) -> list[JustfileViolation]:
    """Collect guardrail violations from parsed Justfile recipes."""
    violations: list[JustfileViolation] = []
    for recipe in recipes:
        if recipe.name in LEGACY_RECIPE_NAMES:
            violations.append(
                JustfileViolation(
                    kind=JustfileViolationKind.DEPRECATED_RECIPE,
                    path=justfile_path,
                    line_number=recipe.line_number,
                    message=f"deprecated recipe `{recipe.name}` remains in the Justfile",
                )
            )
        if recipe.name not in EXACT_RECIPE_NAMES and recipe_prefix(recipe.name) not in ALLOWED_RECIPE_PREFIXES:
            violations.append(
                JustfileViolation(
                    kind=JustfileViolationKind.INVALID_RECIPE_PREFIX,
                    path=justfile_path,
                    line_number=recipe.line_number,
                    message=f"recipe `{recipe.name}` does not use an approved naming-domain prefix",
                )
            )
        override_count = count_hydra_overrides(recipe)
        if override_count > INLINE_OVERRIDE_LIMIT:
            violations.append(
                JustfileViolation(
                    kind=JustfileViolationKind.INLINE_HYDRA_OVERRIDES,
                    path=justfile_path,
                    line_number=recipe.line_number,
                    message=(
                        f"recipe `{recipe.name}` contains {override_count} inline Hydra overrides; "
                        "move workflow truth into tooling/configs"
                    ),
                )
            )
        body_text = "\n".join(recipe.body_lines)
        if recipe_uses_hydra_tooling(recipe) and "--config-name" not in body_text:
            violations.append(
                JustfileViolation(
                    kind=JustfileViolationKind.MISSING_CONFIG_NAME,
                    path=justfile_path,
                    line_number=recipe.line_number,
                    message=f"recipe `{recipe.name}` invokes tooling.cli without a saved --config-name",
                )
            )
        if "--g-" in body_text:
            violations.append(
                JustfileViolation(
                    kind=JustfileViolationKind.STALE_G_FLAG,
                    path=justfile_path,
                    line_number=recipe.line_number,
                    message=f"recipe `{recipe.name}` contains stale --g-* CLI flags",
                )
            )
    return violations


def collect_documentation_violations(
    recipes: tuple[JustRecipe, ...],
    documentation_paths: tuple[Path, ...],
) -> list[JustfileViolation]:
    """Collect stale and missing recipe references from documentation."""
    recipe_names = {recipe.name for recipe in recipes}
    violations: list[JustfileViolation] = []
    for documentation_path, line_number, recipe_name in collect_documented_recipe_references(documentation_paths):
        if recipe_name in LEGACY_RECIPE_NAMES:
            violations.append(
                JustfileViolation(
                    kind=JustfileViolationKind.DEPRECATED_REFERENCE,
                    path=documentation_path,
                    line_number=line_number,
                    message=f"documentation references deprecated recipe `{recipe_name}`",
                )
            )
            continue
        if recipe_name not in recipe_names:
            violations.append(
                JustfileViolation(
                    kind=JustfileViolationKind.DOCUMENTED_RECIPE_MISSING,
                    path=documentation_path,
                    line_number=line_number,
                    message=f"documentation references missing recipe `{recipe_name}`",
                )
            )
    return violations


def collect_justfile_violations(arguments: JustfileCheckArguments) -> tuple[JustfileViolation, ...]:
    """Collect all Justfile guardrail violations."""
    recipes = parse_just_recipes(arguments.justfile_path)
    return (
        *collect_recipe_violations(arguments.justfile_path, recipes),
        *collect_documentation_violations(recipes, arguments.documentation_paths),
    )


def render_violation(violation: JustfileViolation) -> str:
    """Render a guardrail violation for command-line output."""
    return f"{violation.path}:{violation.line_number}: {violation.message}"


def run_tool(arguments: JustfileCheckArguments) -> int:
    """Verify the repository Justfile command surface."""
    violations = collect_justfile_violations(arguments)
    if violations:
        print("Justfile guardrail violations:")
        for violation in violations:
            print(f"  {render_violation(violation)}")
        return 1

    print(f"Justfile guardrails passed for `{arguments.justfile_path}`.")
    return 0


def build_arguments_from_config(config: omegaconf.DictConfig) -> JustfileCheckArguments:
    """Resolve Justfile check arguments from Hydra config."""
    tool_values = tooling_hydra_arguments.tool_config_to_dictionary(config)
    raw_documentation_paths = typing.cast("list[typing.Any]", tool_values["documentation_paths"])
    return JustfileCheckArguments(
        justfile_path=tooling_hydra_arguments.path_or_none(tool_values["justfile_path"]) or DEFAULT_JUSTFILE_PATH,
        documentation_paths=tuple(Path(str(path)) for path in raw_documentation_paths),
    )


@hydra.main(version_base=None, config_path="../configs", config_name="debug_check_justfile")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Run the Justfile guardrail from Hydra configuration."""
    exit_code = run_tool(build_arguments_from_config(config))
    if exit_code:
        raise SystemExit(exit_code)


def main() -> int:
    """Run the Justfile guardrail from default Hydra configuration."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
