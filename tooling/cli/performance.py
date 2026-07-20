"""Grouped Hydra CLI for performance tooling."""

from __future__ import annotations

import enum
import typing

import hydra

from tooling.cli import performance_compare, performance_smoke
from tooling.common import hydra_compat as tooling_hydra_compat
from tooling.common import registry as tooling_registry
from tooling.performance import jax_runtime

if typing.TYPE_CHECKING:
    import omegaconf


class PerformanceToolName(enum.StrEnum):
    """Available grouped performance tools."""

    SMOKE = "smoke"
    COMPARE = "compare"
    JAX_RUNTIME = "jax_runtime"


TOOLS: dict[str, tooling_registry.ToolSpec[typing.Any]] = {
    PerformanceToolName.SMOKE.value: tooling_registry.ToolSpec(
        build_arguments=performance_smoke.build_arguments_from_config,
        run=performance_smoke.run_tool,
    ),
    PerformanceToolName.COMPARE.value: tooling_registry.ToolSpec(
        build_arguments=performance_compare.build_arguments_from_config,
        run=performance_compare.run_tool,
    ),
    PerformanceToolName.JAX_RUNTIME.value: tooling_registry.ToolSpec(
        build_arguments=jax_runtime.build_arguments_from_config,
        run=jax_runtime.run_tool,
    ),
}


@hydra.main(version_base=None, config_path="../configs", config_name="performance")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Dispatch a performance tool from Hydra configuration."""
    tooling_registry.dispatch_tool(config, TOOLS)


def main() -> None:
    """Run the grouped performance CLI from default Hydra configuration."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()


if __name__ == "__main__":
    main()
