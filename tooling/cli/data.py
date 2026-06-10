"""Grouped Hydra CLI for data preparation tooling."""

from __future__ import annotations

import enum
import typing

import hydra

from tooling.common import hydra_compat as tooling_hydra_compat
from tooling.data import fetch as data_fetch
from tooling.data import simulate as data_simulate

if typing.TYPE_CHECKING:
    import omegaconf


class DataToolName(enum.StrEnum):
    """Available grouped data tools."""

    FETCH = "fetch"
    SIMULATE = "simulate"


@hydra.main(version_base=None, config_path="../configs", config_name="data")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Dispatch a data preparation tool from Hydra configuration."""
    tool_name = DataToolName(str(config.tool.name))
    if tool_name == DataToolName.FETCH:
        data_fetch.run_tool(data_fetch.build_arguments_from_config(config))
        return
    if tool_name == DataToolName.SIMULATE:
        data_simulate.run_tool(data_simulate.build_arguments_from_config(config))
        return
    typing.assert_never(tool_name)


def main() -> None:
    """Run the grouped data CLI from default Hydra configuration."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()


if __name__ == "__main__":
    main()
