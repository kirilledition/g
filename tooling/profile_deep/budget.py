"""Deep-profile campaign budget helpers."""

from __future__ import annotations

import typing

import tooling.cli.profile_regenie2_deep as profile_regenie2_deep

if typing.TYPE_CHECKING:
    from pathlib import Path

    from tooling.profile_deep import models as profile_deep_models


def build_campaign_budget(
    *,
    arguments: profile_deep_models.ProfileArguments,
    output_directory: Path,
) -> profile_deep_models.CampaignBudget:
    """Build the deep-profile campaign budget."""
    return profile_regenie2_deep.build_campaign_budget(arguments=arguments, output_directory=output_directory)


def enforce_campaign_budget(
    arguments: profile_deep_models.ProfileArguments, budget: profile_deep_models.CampaignBudget
) -> None:
    """Enforce campaign budget limits."""
    profile_regenie2_deep.enforce_campaign_budget(arguments, budget)
