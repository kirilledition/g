"""Hydra composition support for development tooling."""

from __future__ import annotations

import typing

import hydra

if typing.TYPE_CHECKING:
    import omegaconf

CONFIG_MODULE = "tooling.configs"


def compose_config(
    *,
    config_name: str,
    overrides: typing.Sequence[str] | None = None,
    include_hydra_config: bool = False,
) -> omegaconf.DictConfig:
    """Compose a tooling config with Hydra.

    Args:
        config_name: Config name in the tooling config package.
        overrides: Optional Hydra overrides.
        include_hydra_config: Whether to include Hydra's own config node.

    Returns:
        Composed Hydra configuration.

    """
    override_list = list(overrides) if overrides is not None else []
    with hydra.initialize_config_module(config_module=CONFIG_MODULE, version_base=None):
        return hydra.compose(
            config_name=config_name,
            overrides=override_list,
            return_hydra_config=include_hydra_config,
        )
