"""Grouped Hydra CLI for server setup tooling."""

from __future__ import annotations

import enum
import typing

import hydra

from tooling.common import hydra_compat as tooling_hydra_compat
from tooling.server import bootstrap_tools, nsight_tools

if typing.TYPE_CHECKING:
    import omegaconf


class ServerToolName(enum.StrEnum):
    """Available grouped server tools."""

    BOOTSTRAP_TOOLS = "bootstrap_tools"
    NSIGHT_TOOLS = "nsight_tools"


@hydra.main(version_base=None, config_path="../configs", config_name="server")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Dispatch a server setup tool from Hydra configuration."""
    tool_name = ServerToolName(str(config.tool.name))
    if tool_name == ServerToolName.BOOTSTRAP_TOOLS:
        bootstrap_tools.run_tool(bootstrap_tools.build_arguments_from_config(config))
        return
    if tool_name == ServerToolName.NSIGHT_TOOLS:
        nsight_tools.run_tool(nsight_tools.build_arguments_from_config(config))
        return
    typing.assert_never(tool_name)


def main() -> None:
    """Run the grouped server CLI from default Hydra configuration."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()


if __name__ == "__main__":
    main()
