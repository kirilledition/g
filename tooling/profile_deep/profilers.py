"""Deep-profile external profiler helpers."""

from __future__ import annotations

import typing

import tooling.cli.profile_regenie2_deep as profile_regenie2_deep

if typing.TYPE_CHECKING:
    from tooling.profile_deep import models as profile_deep_models


def build_profiler_tool_status(
    arguments: profile_deep_models.ProfileArguments,
) -> dict[str, profile_deep_models.ProfilerToolStatus]:
    """Detect external profiler tool availability."""
    return profile_regenie2_deep.build_profiler_tool_status(arguments)
