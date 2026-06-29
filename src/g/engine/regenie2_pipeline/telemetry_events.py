"""Telemetry and logging event helpers for REGENIE step 2 pipelines."""

from __future__ import annotations

import logging
import typing

from g import types

if typing.TYPE_CHECKING:
    from g.engine.regenie2_pipeline import context as pipeline_context

logger = logging.getLogger(__name__)


def log_sample_alignment_completed(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    sample_count: int | None,
    covariate_count: int | None,
    phenotype_name: str | None,
    phenotype_count: int | None,
    phenotype_group_count: int | None,
) -> None:
    """Emit sample-alignment telemetry with mode-specific fields."""
    if context.telemetry_session is None:
        return
    context.telemetry_session.log_sample_alignment_completed(
        association_mode=context.association_mode,
        phenotype=phenotype_name,
        phenotype_count=phenotype_count,
        sample_count=sample_count,
        covariate_count=covariate_count,
        phenotype_group_count=phenotype_group_count,
    )


def log_multi_phenotype_sample_summary(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    sample_mode: types.MultiPhenotypeSampleMode,
    sample_counts: tuple[int, ...],
    sample_set_fingerprints: tuple[str | None, ...],
    phenotype_group_count: int,
) -> None:
    """Emit a user-visible summary of multi-phenotype sample semantics."""
    sample_counts_differ = len(set(sample_counts)) > 1
    if sample_mode == types.MultiPhenotypeSampleMode.COMPLETE_CASE:
        logger.info(
            "Analyzed %s phenotypes in complete-case sample mode; one shared sample set was used.",
            len(sample_counts),
        )
    else:
        sample_count_summary = (
            "sample counts differ across phenotypes"
            if sample_counts_differ
            else "sample counts do not differ across phenotypes"
        )
        logger.info(
            "Analyzed %s phenotypes in per-phenotype sample mode; %s.",
            len(sample_counts),
            sample_count_summary,
        )
    if context.telemetry_session is None:
        return
    context.telemetry_session.log_multi_phenotype_sample_summary(
        association_mode=context.association_mode,
        sample_mode=sample_mode,
        sample_counts=sample_counts,
        sample_set_fingerprints=sample_set_fingerprints,
        phenotype_group_count=phenotype_group_count,
    )


def log_prediction_source_loaded(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    phenotype_name: str | None,
    phenotype_count: int | None,
) -> None:
    """Emit prediction-source telemetry with mode-specific fields."""
    if context.telemetry_session is None:
        return
    context.telemetry_session.log_prediction_source_loaded(
        association_mode=context.association_mode,
        phenotype=phenotype_name,
        phenotype_count=phenotype_count,
    )
