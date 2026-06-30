"""Telemetry and logging event helpers for REGENIE step 2 pipelines."""

from __future__ import annotations

import typing

from g import _core, types
from g.engine import run_events

if typing.TYPE_CHECKING:
    from g.engine.regenie2_pipeline import context as pipeline_context


def emit_pipeline_telemetry_diagnostic_event_payload(payload: typing.Mapping[str, object]) -> None:
    """Emit one pipeline telemetry diagnostic payload through native tracing."""
    _core.emit_diagnostic_event_fields(
        str(payload["level"]),
        str(payload["event_name"]),
        str(payload["message"]),
        typing.cast("typing.Mapping[str, object]", payload["fields"]),
    )


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
    emit_pipeline_telemetry_diagnostic_event_payload(
        run_events.build_pipeline_multi_phenotype_sample_summary_diagnostic_payload(
            phenotype_count=len(sample_counts),
            phenotype_group_count=phenotype_group_count,
            sample_counts_differ=sample_counts_differ,
            sample_mode=sample_mode,
        )
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
