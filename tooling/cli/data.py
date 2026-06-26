"""Grouped Hydra CLI for data preparation tooling."""

from __future__ import annotations

import enum
import typing

import hydra

from tooling.common import hydra_compat as tooling_hydra_compat
from tooling.common import registry as tooling_registry
from tooling.data import fetch as data_fetch
from tooling.data import simulate as data_simulate

if typing.TYPE_CHECKING:
    import omegaconf


class DataToolName(enum.StrEnum):
    """Available grouped data tools."""

    FETCH = "fetch"
    SIMULATE = "simulate"


TOOLS: dict[str, tooling_registry.ToolSpec[typing.Any]] = {
    DataToolName.FETCH.value: tooling_registry.ToolSpec(
        name=DataToolName.FETCH.value,
        config_name="data_fetch",
        build_arguments=data_fetch.build_arguments_from_config,
        run=data_fetch.run_tool,
    ),
    DataToolName.SIMULATE.value: tooling_registry.ToolSpec(
        name=DataToolName.SIMULATE.value,
        config_name="data_simulate",
        build_arguments=data_simulate.build_arguments_from_config,
        run=data_simulate.run_tool,
    ),
}


@hydra.main(version_base=None, config_path="../configs", config_name="data")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Dispatch a data preparation tool from Hydra configuration."""
    tooling_registry.dispatch_tool(config, TOOLS)


def main() -> None:
    """Run the grouped data CLI from default Hydra configuration."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()


if __name__ == "__main__":
    main()
