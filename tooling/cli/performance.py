"""Grouped Hydra CLI for performance tooling."""

from __future__ import annotations

import enum
import typing

import hydra

from tooling.cli import performance_compare, performance_smoke
from tooling.common import hydra_compat as tooling_hydra_compat
from tooling.performance import jax_runtime

if typing.TYPE_CHECKING:
    import omegaconf


class PerformanceToolName(enum.StrEnum):
    """Available grouped performance tools."""

    SMOKE = "smoke"
    COMPARE = "compare"
    JAX_RUNTIME = "jax_runtime"


@hydra.main(version_base=None, config_path="../configs", config_name="performance")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Dispatch a performance tool from Hydra configuration."""
    tool_name = PerformanceToolName(str(config.tool.name))
    if tool_name == PerformanceToolName.SMOKE:
        performance_smoke.run_tool(performance_smoke.build_arguments_from_config(config))
        return
    if tool_name == PerformanceToolName.COMPARE:
        performance_compare.run_tool(performance_compare.build_arguments_from_config(config))
        return
    if tool_name == PerformanceToolName.JAX_RUNTIME:
        jax_runtime.run_tool(jax_runtime.build_arguments_from_config(config))
        return
    typing.assert_never(tool_name)


def main() -> None:
    """Run the grouped performance CLI from default Hydra configuration."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()


if __name__ == "__main__":
    main()
