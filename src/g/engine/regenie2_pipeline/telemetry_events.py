"""Telemetry and logging event helpers for REGENIE step 2 pipelines."""

from __future__ import annotations

import typing

from g.engine import run_events, telemetry

if typing.TYPE_CHECKING:
    from g import _core, types
    from g.engine.regenie2_pipeline import context as pipeline_context

type TelemetrySession = telemetry.TelemetrySession


def native_pipeline_diagnostic_policy() -> _core.NativePipelineDiagnosticPolicy:
    """Build the native pipeline diagnostic policy handle."""
    return run_events.native_pipeline_diagnostic_policy()


def native_run_event_telemetry_policy() -> _core.NativeRunEventTelemetryPolicy:
    """Build the native run-event telemetry policy handle."""
    return run_events.native_run_event_telemetry_policy()


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
    native_run_event_telemetry_policy().record_sample_alignment_completed_telemetry_event(
        context.telemetry_session,
        context.association_mode.value,
        phenotype_name,
        phenotype_count,
        sample_count,
        covariate_count,
        phenotype_group_count,
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
    native_pipeline_diagnostic_policy().record_pipeline_multi_phenotype_sample_summary_diagnostic_event(
        phenotype_count=len(sample_counts),
        phenotype_group_count=phenotype_group_count,
        sample_counts_differ=sample_counts_differ,
        sample_mode=sample_mode.value,
    )
    native_run_event_telemetry_policy().record_multi_phenotype_sample_summary_telemetry_event(
        context.telemetry_session,
        context.association_mode.value,
        sample_mode.value,
        sample_counts,
        sample_set_fingerprints,
        phenotype_group_count,
    )


def log_prediction_source_loaded(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    phenotype_name: str | None,
    phenotype_count: int | None,
) -> None:
    """Emit prediction-source telemetry with mode-specific fields."""
    native_run_event_telemetry_policy().record_prediction_source_loaded_telemetry_event(
        context.telemetry_session,
        context.association_mode.value,
        phenotype_name,
        phenotype_count,
    )
