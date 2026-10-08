"""Grouped Hydra CLI for data preparation tooling."""

from __future__ import annotations

import enum
import typing

import hydra

from tooling.common import hydra_compat as tooling_hydra_compat
from tooling.common import registry as tooling_registry

if typing.TYPE_CHECKING:
    import omegaconf


class DataToolName(enum.StrEnum):
    """Available grouped data tools."""

    FETCH = "fetch"
    SIMULATE = "simulate"
    REGENIE_BASELINE = "regenie_baseline"
    ALL_OF_US = "all_of_us"


def load_tool(tool_name: DataToolName) -> tooling_registry.ToolSpec[typing.Any]:
    """Import only the requested tool so minimal installations need no dev stack."""
    match tool_name:
        case DataToolName.ALL_OF_US:
            from tooling.data import all_of_us as data_all_of_us

            return tooling_registry.ToolSpec(
                build_arguments=data_all_of_us.build_arguments_from_config,
                run=data_all_of_us.run_tool,
            )
        case DataToolName.FETCH:
            from tooling.data import fetch as data_fetch

            return tooling_registry.ToolSpec(
                build_arguments=data_fetch.build_arguments_from_config,
                run=data_fetch.run_tool,
            )
        case DataToolName.SIMULATE:
            from tooling.data import simulate as data_simulate

            return tooling_registry.ToolSpec(
                build_arguments=data_simulate.build_arguments_from_config,
                run=data_simulate.run_tool,
            )
        case DataToolName.REGENIE_BASELINE:
            from tooling.data import regenie_baseline

            return tooling_registry.ToolSpec(
                build_arguments=regenie_baseline.build_arguments_from_config,
                run=regenie_baseline.run_tool,
            )
    typing.assert_never(tool_name)


@hydra.main(version_base=None, config_path="../configs", config_name="data")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Dispatch a data preparation tool from Hydra configuration."""
    if "tool" not in config or "name" not in config.tool:
        raise KeyError("Grouped tooling configs must contain tool.name.")
    try:
        tool_name = DataToolName(str(config.tool.name))
    except ValueError as error:
        accepted_names = ", ".join(sorted(name.value for name in DataToolName))
        raise KeyError(f"Unknown tool.name `{config.tool.name}`. Accepted values: {accepted_names}.") from error
    tooling_registry.dispatch_tool(config, {tool_name.value: load_tool(tool_name)})


def main() -> None:
    """Run the grouped data CLI from default Hydra configuration."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()


if __name__ == "__main__":
    main()
