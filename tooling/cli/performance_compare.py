#!/usr/bin/env python3
"""CLI wrapper for benchmark JSON summary comparisons."""

from __future__ import annotations

import sys
import typing
from dataclasses import dataclass

import hydra

import tooling.performance_compare as performance_compare
from tooling.common import hydra_arguments as tooling_hydra_arguments
from tooling.common import hydra_compat as tooling_hydra_compat

if typing.TYPE_CHECKING:
    from pathlib import Path

    import omegaconf


@dataclass(frozen=True)
class PerformanceCompareArguments:
    """Resolved parameters for benchmark summary comparison.

    Attributes:
        baseline_json: Baseline benchmark JSON summary.
        new_json: New benchmark JSON summary.

    """

    baseline_json: Path
    new_json: Path


def run_tool(arguments: PerformanceCompareArguments) -> None:
    """Run the benchmark summary comparison CLI."""
    try:
        report = performance_compare.compare_summary_paths(arguments.baseline_json, arguments.new_json)
    except performance_compare.PerformanceComparisonError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(performance_compare.render_comparison_report(report), end="")


def required_path(tool_values: dict[str, typing.Any], key: str) -> Path:
    """Return a required path from a Hydra tool config."""
    path = tooling_hydra_arguments.path_or_none(tool_values[key])
    if path is None:
        message = f"tool.{key} is required."
        raise ValueError(message)
    return path


def build_arguments_from_config(config: omegaconf.DictConfig) -> PerformanceCompareArguments:
    """Resolve benchmark comparison parameters from Hydra config."""
    tool_values = tooling_hydra_arguments.tool_config_to_dictionary(config)
    return PerformanceCompareArguments(
        baseline_json=required_path(tool_values, "baseline_json"),
        new_json=required_path(tool_values, "new_json"),
    )


@hydra.main(version_base=None, config_path="../configs", config_name="performance_compare")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Run benchmark summary comparison from Hydra configuration."""
    run_tool(build_arguments_from_config(config))


def main() -> None:
    """Run benchmark summary comparison from default Hydra configuration."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()


if __name__ == "__main__":
    main()
