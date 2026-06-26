"""Deep-profile orchestration entrypoint."""

from __future__ import annotations

import typing

import tooling.cli.profile_regenie2_deep as profile_regenie2_deep

if typing.TYPE_CHECKING:
    from tooling.profile_deep import models as profile_deep_models


def run_tool(arguments: profile_deep_models.ProfileArguments) -> None:
    """Run the deep-profile workflow."""
    profile_regenie2_deep.run_tool(arguments)
