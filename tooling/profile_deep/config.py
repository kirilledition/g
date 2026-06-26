"""Deep-profile Hydra configuration adaptation."""

from __future__ import annotations

import typing

import tooling.cli.profile_regenie2_deep as profile_regenie2_deep

if typing.TYPE_CHECKING:
    import omegaconf

    from tooling.profile_deep import models as profile_deep_models


def build_arguments_from_config(config: omegaconf.DictConfig) -> profile_deep_models.ProfileArguments:
    """Build deep-profile arguments from a composed Hydra config."""
    return profile_regenie2_deep.build_arguments_from_config(config)


def build_arguments_from_overrides(
    overrides: typing.Sequence[str] | None = None,
) -> profile_deep_models.ProfileArguments:
    """Build deep-profile arguments from Hydra overrides."""
    return profile_regenie2_deep.build_arguments_from_overrides(overrides)


def apply_smoke_overrides(arguments: profile_deep_models.ProfileArguments) -> profile_deep_models.ProfileArguments:
    """Apply smoke-mode overrides."""
    return profile_regenie2_deep.apply_smoke_overrides(arguments)
