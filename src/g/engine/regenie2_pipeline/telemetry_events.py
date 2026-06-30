"""Telemetry and logging event helpers for REGENIE step 2 pipelines."""

from __future__ import annotations

import typing

from g import _core, types

if typing.TYPE_CHECKING:
    from g.engine.regenie2_pipeline import context as pipeline_context


def emit_pipeline_telemetry_diagnostic_event(
    level: str,
    event: str,
    message: str,
    fields: typing.Mapping[str, object],
) -> None:
    """Emit one structured pipeline telemetry diagnostic through native tracing."""
    _core.emit_diagnostic_event_fields(level, event, message, fields)


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
        message = (
            f"Analyzed {len(sample_counts)} phenotypes in complete-case sample mode; one shared sample set was used."
        )
    else:
        sample_count_summary = (
            "sample counts differ across phenotypes"
            if sample_counts_differ
            else "sample counts do not differ across phenotypes"
        )
        message = f"Analyzed {len(sample_counts)} phenotypes in per-phenotype sample mode; {sample_count_summary}."
    emit_pipeline_telemetry_diagnostic_event(
        "info",
        "pipeline_multi_phenotype_sample_summary",
        message,
        {
            "phenotype_count": len(sample_counts),
            "phenotype_group_count": phenotype_group_count,
            "sample_counts_differ": sample_counts_differ,
            "sample_mode": sample_mode.value,
        },
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
