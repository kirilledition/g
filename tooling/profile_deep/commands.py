"""Deep-profile command construction helpers."""

from __future__ import annotations

import typing

import tooling.cli.profile_regenie2_deep as profile_regenie2_deep

if typing.TYPE_CHECKING:
    from pathlib import Path

    from tooling.common import g_regenie as tooling_g_regenie


def build_g_step2_regenie_run_spec(
    *,
    baseline_paths: typing.Any,
    candidate: profile_regenie2_deep.Step2Candidate,
    output_prefix: Path,
    variant_limit: int | None,
    jax_cache_directory: Path | None,
    stage_timing_path: Path | None,
) -> tooling_g_regenie.RegenieRunSpec:
    """Build the shared REGENIE run spec for a deep-profile child command."""
    return profile_regenie2_deep.build_g_step2_regenie_run_spec(
        baseline_paths=baseline_paths,
        candidate=candidate,
        output_prefix=output_prefix,
        variant_limit=variant_limit,
        jax_cache_directory=jax_cache_directory,
        stage_timing_path=stage_timing_path,
    )


def build_g_step2_child_command(
    *,
    baseline_paths: typing.Any,
    candidate: profile_regenie2_deep.Step2Candidate,
    output_prefix: Path,
    variant_limit: int | None,
    cache_directory: Path | None = None,
    stage_timing_path: Path | None = None,
    trace_directory: Path | None = None,
    memory_profile_path: Path | None = None,
    diagnostic_options: dict[str, object] | None = None,
) -> list[str]:
    """Build one isolated Python child command for a g REGENIE step 2 run."""
    return profile_regenie2_deep.build_g_step2_child_command(
        baseline_paths=baseline_paths,
        candidate=candidate,
        output_prefix=output_prefix,
        variant_limit=variant_limit,
        cache_directory=cache_directory,
        stage_timing_path=stage_timing_path,
        trace_directory=trace_directory,
        memory_profile_path=memory_profile_path,
        diagnostic_options=diagnostic_options,
    )
