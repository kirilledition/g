"""Deep-profile external profiler helpers."""

from __future__ import annotations

import tooling.cli.profile_regenie2_deep as profile_regenie2_deep


def build_profiler_tool_status(
    arguments: profile_regenie2_deep.ProfileArguments,
) -> dict[str, profile_regenie2_deep.ProfilerToolStatus]:
    """Detect external profiler tool availability."""
    return profile_regenie2_deep.build_profiler_tool_status(arguments)
