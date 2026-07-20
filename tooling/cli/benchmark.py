"""Grouped Hydra CLI for benchmark tooling."""

from __future__ import annotations

import enum
import typing
from pathlib import Path

import hydra

from tooling.benchmark import benchmark as baseline_benchmark
from tooling.benchmark import linear_startup
from tooling.common import hydra_arguments as tooling_hydra_arguments
from tooling.common import hydra_compat as tooling_hydra_compat
from tooling.common import registry as tooling_registry

if typing.TYPE_CHECKING:
    import omegaconf


class BenchmarkToolName(enum.StrEnum):
    """Available grouped benchmark tools."""

    BASELINES = "baselines"
    LINEAR_STARTUP = "linear_startup"


def build_baseline_arguments(config: omegaconf.DictConfig) -> baseline_benchmark.BaselineBenchmarkArguments:
    """Build baseline benchmark arguments from a grouped config."""
    tool_values = tooling_hydra_arguments.tool_config_to_dictionary(config)
    return baseline_benchmark.BaselineBenchmarkArguments(
        data_directory=tooling_hydra_arguments.path_or_none(tool_values["data_directory"]) or Path("data"),
        include_hail=tooling_hydra_arguments.boolean_value(tool_values["include_hail"]),
    )


def build_linear_startup_arguments(config: omegaconf.DictConfig) -> linear_startup.LinearStartupArguments:
    """Build fresh-process startup arguments from a grouped config."""
    return linear_startup.build_arguments_from_config(config)


TOOLS: dict[str, tooling_registry.ToolSpec[typing.Any]] = {
    BenchmarkToolName.BASELINES.value: tooling_registry.ToolSpec(
        build_arguments=build_baseline_arguments,
        run=baseline_benchmark.run_tool,
    ),
    BenchmarkToolName.LINEAR_STARTUP.value: tooling_registry.ToolSpec(
        build_arguments=build_linear_startup_arguments,
        run=linear_startup.run_tool,
    ),
}


@hydra.main(version_base=None, config_path="../configs", config_name="benchmark")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Dispatch a benchmark tool from Hydra configuration."""
    tooling_registry.dispatch_tool(config, TOOLS)


def main() -> None:
    """Run the grouped benchmark CLI from default Hydra configuration."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()


if __name__ == "__main__":
    main()
