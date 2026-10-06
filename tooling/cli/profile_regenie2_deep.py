#!/usr/bin/env python3
"""Hydra entrypoint for deep profiling of REGENIE and the GWAS engine."""

from __future__ import annotations

import typing

import hydra

from tooling.common import hydra_compat as tooling_hydra_compat
from tooling.profile_deep import config as profile_deep_config
from tooling.profile_deep import runner as profile_deep_runner

if typing.TYPE_CHECKING:
    import omegaconf


@hydra.main(version_base=None, config_path="../configs", config_name="profile_regenie2_deep")
def hydra_main(config: omegaconf.DictConfig) -> None:
    """Run the deep profiling campaign through Hydra."""
    profile_deep_runner.run_tool(profile_deep_config.build_arguments_from_config(config), hydra_config=config)


def main() -> None:
    """Run the landau deep profiling campaign."""
    tooling_hydra_compat.apply_argparse_help_patch()
    hydra_main()


if __name__ == "__main__":
    main()
