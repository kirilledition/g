"""Deep-profile orchestration entrypoint."""

from __future__ import annotations

import tooling.cli.profile_regenie2_deep as profile_regenie2_deep


def run_tool(arguments: profile_regenie2_deep.ProfileArguments) -> None:
    """Run the deep-profile workflow."""
    profile_regenie2_deep.run_tool(arguments)
