"""Deep-profile report helpers."""

from __future__ import annotations

import typing

import tooling.cli.profile_regenie2_deep as profile_regenie2_deep

if typing.TYPE_CHECKING:
    from tooling.profile_deep import models as profile_deep_models


def build_summary_markdown(
    *,
    aggregate_results: list[profile_deep_models.AggregateResult],
    comparisons: dict[str, dict[str, float]],
    stage_totals: dict[str, float],
    comparison_notes: profile_deep_models.RuntimeComparisonNotes | None = None,
    regenie_baseline_scope: profile_deep_models.RegenieBaselineScope | None = None,
    logging_perturbation_results: list[dict[str, object]] | None = None,
    binary_correction_diagnostics: dict[str, object] | None = None,
) -> str:
    """Build the deep-profile Markdown summary."""
    return profile_regenie2_deep.build_summary_markdown(
        aggregate_results=aggregate_results,
        comparisons=comparisons,
        stage_totals=stage_totals,
        comparison_notes=comparison_notes,
        regenie_baseline_scope=regenie_baseline_scope,
        logging_perturbation_results=logging_perturbation_results,
        binary_correction_diagnostics=binary_correction_diagnostics,
    )
