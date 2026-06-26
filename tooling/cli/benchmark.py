"""Grouped Hydra CLI for benchmark tooling."""

from __future__ import annotations

import enum
import typing
from pathlib import Path

import hydra

from tooling.benchmark import benchmark as baseline_benchmark
from tooling.benchmark import comparison as regenie_comparison
from tooling.benchmark import linear_startup, profile_comparison
from tooling.common import hydra_arguments as tooling_hydra_arguments
from tooling.common import hydra_compat as tooling_hydra_compat
from tooling.common import registry as tooling_registry

if typing.TYPE_CHECKING:
    import omegaconf


class BenchmarkToolName(enum.StrEnum):
    """Available grouped benchmark tools."""

    BASELINES = "baselines"
    REGENIE_COMPARISON = "regenie_comparison"
    PROFILE_COMPARISON = "profile_comparison"
    LINEAR_STARTUP = "linear_startup"


def build_baseline_arguments(config: omegaconf.DictConfig) -> baseline_benchmark.BaselineBenchmarkArguments:
    """Build baseline benchmark arguments from a grouped config."""
    tool_values = tooling_hydra_arguments.tool_config_to_dictionary(config)
    return baseline_benchmark.BaselineBenchmarkArguments(
        data_directory=tooling_hydra_arguments.path_or_none(tool_values["data_directory"]) or Path("data"),
        include_hail=tooling_hydra_arguments.boolean_value(tool_values["include_hail"]),
    )


def build_comparison_arguments(config: omegaconf.DictConfig) -> regenie_comparison.ComparisonArguments:
    """Build REGENIE comparison arguments from a grouped config."""
    tool_values = tooling_hydra_arguments.tool_config_to_dictionary(config)
    return regenie_comparison.ComparisonArguments(
        include_gpu=tooling_hydra_arguments.boolean_value(tool_values["include_gpu"]),
        cpu_only=tooling_hydra_arguments.boolean_value(tool_values["cpu_only"]),
        variant_limit=tooling_hydra_arguments.integer_or_none(tool_values["variant_limit"]),
        chunk_size=int(tool_values["chunk_size"]),
        only_quantitative_step2=tooling_hydra_arguments.boolean_value(tool_values["only_quantitative_step2"]),
        only_binary_step2=tooling_hydra_arguments.boolean_value(tool_values["only_binary_step2"]),
        output_dir=tooling_hydra_arguments.path_or_none(tool_values["output_dir"])
        or Path("data/benchmarks/regenie_comparison"),
    )


def build_profile_comparison_arguments(config: omegaconf.DictConfig) -> profile_comparison.ProfileComparisonArguments:
    """Build profiling comparison arguments from a grouped config."""
    tool_values = tooling_hydra_arguments.tool_config_to_dictionary(config)
    return profile_comparison.ProfileComparisonArguments(
        include_gpu=tooling_hydra_arguments.boolean_value(tool_values["include_gpu"]),
        cpu_only=tooling_hydra_arguments.boolean_value(tool_values["cpu_only"]),
        output_dir=tooling_hydra_arguments.path_or_none(tool_values["output_dir"])
        or Path("data/profiles/regenie_comparison"),
        sample_interval_seconds=float(tool_values["sample_interval_seconds"]),
        variant_limit=tooling_hydra_arguments.integer_or_none(tool_values["g_variant_limit"]),
        chunk_size=int(tool_values["g_chunk_size"]),
        enable_jax_trace=tooling_hydra_arguments.boolean_value(tool_values["enable_jax_trace"]),
        enable_memory_profile=tooling_hydra_arguments.boolean_value(tool_values["enable_memory_profile"]),
    )


def build_linear_startup_arguments(config: omegaconf.DictConfig) -> linear_startup.LinearStartupArguments:
    """Build fresh-process startup arguments from a grouped config."""
    return linear_startup.build_arguments_from_config(config)


TOOLS: dict[str, tooling_registry.ToolSpec[typing.Any]] = {
    BenchmarkToolName.BASELINES.value: tooling_registry.ToolSpec(
        name=BenchmarkToolName.BASELINES.value,
        config_name="benchmark_baselines",
        build_arguments=build_baseline_arguments,
        run=baseline_benchmark.run_tool,
    ),
    BenchmarkToolName.REGENIE_COMPARISON.value: tooling_registry.ToolSpec(
        name=BenchmarkToolName.REGENIE_COMPARISON.value,
        config_name="benchmark_regenie_comparison",
        build_arguments=build_comparison_arguments,
        run=regenie_comparison.run_tool,
    ),
    BenchmarkToolName.PROFILE_COMPARISON.value: tooling_registry.ToolSpec(
        name=BenchmarkToolName.PROFILE_COMPARISON.value,
        config_name="benchmark_profile_comparison",
        build_arguments=build_profile_comparison_arguments,
        run=profile_comparison.run_tool,
    ),
    BenchmarkToolName.LINEAR_STARTUP.value: tooling_registry.ToolSpec(
        name=BenchmarkToolName.LINEAR_STARTUP.value,
        config_name="benchmark_linear_startup",
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
