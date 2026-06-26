"""Deep-profile artifact manifest helpers."""

from __future__ import annotations

import typing

import tooling.cli.profile_regenie2_deep as profile_regenie2_deep

if typing.TYPE_CHECKING:
    from pathlib import Path

    from tooling.profile_deep import models as profile_deep_models


def collect_artifact_manifest(
    *,
    output_directory: Path,
    profiler_tool_status: dict[str, profile_deep_models.ProfilerToolStatus],
    summary_payload: dict[str, typing.Any] | None = None,
    profile_plan: profile_deep_models.ProfilePlan | None = None,
) -> dict[str, typing.Any]:
    """Build a structured artifact manifest for one profile campaign."""
    return profile_regenie2_deep.collect_artifact_manifest(
        output_directory=output_directory,
        profiler_tool_status=profiler_tool_status,
        summary_payload=summary_payload,
        profile_plan=profile_plan,
    )


def write_artifact_manifest(
    *,
    output_directory: Path,
    profiler_tool_status: dict[str, profile_deep_models.ProfilerToolStatus],
    summary_payload: dict[str, typing.Any] | None = None,
    profile_plan: profile_deep_models.ProfilePlan | None = None,
) -> None:
    """Write the profile artifact manifest."""
    profile_regenie2_deep.write_artifact_manifest(
        output_directory=output_directory,
        profiler_tool_status=profiler_tool_status,
        summary_payload=summary_payload,
        profile_plan=profile_plan,
    )
