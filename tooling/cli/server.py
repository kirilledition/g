"""Grouped Hydra CLI for server setup tooling."""

from __future__ import annotations

import enum
import typing

import hydra

from tooling.common import hydra_compat as tooling_hydra_compat
from tooling.common import registry as tooling_registry
from tooling.server import bootstrap_tools, nsight_tools

if typing.TYPE_CHECKING:
    import omegaconf


class ServerToolName(enum.StrEnum):
    """Available grouped server tools."""

    BOOTSTRAP_TOOLS = "bootstrap_tools"
    NSIGHT_TOOLS = "nsight_tools"


TOOLS: dict[str, tooling_registry.ToolSpec[typing.Any]] = {
    ServerToolName.BOOTSTRAP_TOOLS.value: tooling_registry.ToolSpec(
        name=ServerToolName.BOOTSTRAP_TOOLS.value,
        config_name="server_bootstrap_tools",
        build_arguments=bootstrap_tools.build_arguments_from_config,
        run=bootstrap_tools.run_tool,
    ),
    ServerToolName.NSIGHT_TOOLS.value: tooling_registry.ToolSpec(
        name=ServerToolName.NSIGHT_TOOLS.value,
        config_name="server_nsight_tools",
        build_arguments=nsight_tools.build_arguments_from_config,
        run=nsight_tools.run_tool,
    ),
}


@hydra.main(version_base=None, config_path="../configs", config_name="server")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Dispatch a server setup tool from Hydra configuration."""
    tooling_registry.dispatch_tool(config, TOOLS)


def main() -> None:
    """Run the grouped server CLI from default Hydra configuration."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()


if __name__ == "__main__":
    main()
