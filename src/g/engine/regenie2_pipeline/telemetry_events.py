"""Telemetry and logging event helpers for REGENIE step 2 pipelines."""

from __future__ import annotations

import typing

from g import _core

if typing.TYPE_CHECKING:
    from g import types
    from g.engine.regenie2_pipeline import context as pipeline_context

type TelemetrySession = object


def native_pipeline_diagnostic_policy() -> _core.NativePipelineDiagnosticPolicy:
    """Build the native pipeline diagnostic policy handle."""
    return _core.NativePipelineDiagnosticPolicy()


def native_run_event_telemetry_policy() -> _core.NativeRunEventTelemetryPolicy:
    """Build the native run-event telemetry policy handle."""
    return _core.NativeRunEventTelemetryPolicy()


def record_pipeline_single_trait_started(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    phenotype_name: str,
    pipeline_label: str,
) -> None:
    """Record that a single-trait pipeline has started."""
    native_pipeline_diagnostic_policy().record_pipeline_single_trait_started_diagnostic_event(
        association_mode=context.association_mode.value,
        phenotype_name=phenotype_name,
        pipeline_label=pipeline_label,
    )


def record_pipeline_single_trait_input_load_started(
    *,
    phenotype_name: str,
    pipeline_label: str,
) -> None:
    """Record that single-trait native input loading has started."""
    native_pipeline_diagnostic_policy().record_pipeline_single_trait_input_load_started_diagnostic_event(
        phenotype_name=phenotype_name,
        pipeline_label=pipeline_label,
    )


def record_pipeline_single_trait_input_aligned(
    *,
    covariate_count: int,
    phenotype_name: str,
    pipeline_label: str,
    sample_count: int,
) -> None:
    """Record that single-trait native input alignment has completed."""
    native_pipeline_diagnostic_policy().record_pipeline_single_trait_input_aligned_diagnostic_event(
        covariate_count=covariate_count,
        phenotype_name=phenotype_name,
        pipeline_label=pipeline_label,
        sample_count=sample_count,
    )


def record_pipeline_single_trait_prediction_source_load_started(
    *,
    phenotype_name: str,
    pipeline_label: str,
) -> None:
    """Record that single-trait prediction source loading has started."""
    native_pipeline_diagnostic_policy().record_pipeline_single_trait_prediction_source_load_started_diagnostic_event(
        phenotype_name=phenotype_name,
        pipeline_label=pipeline_label,
    )


def record_pipeline_single_trait_preflight_started(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    phenotype_name: str,
    pipeline_label: str,
) -> None:
    """Record that single-trait preflight has started."""
    native_pipeline_diagnostic_policy().record_pipeline_single_trait_preflight_started_diagnostic_event(
        phenotype_name=phenotype_name,
        pipeline_label=pipeline_label,
        trusted_no_missing_diploid=context.effective_trusted_no_missing_diploid,
        variant_limit=context.variant_limit,
    )


def record_pipeline_single_trait_preflight_completed(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    phenotype_name: str,
    pipeline_label: str,
    sample_count: int,
    covariate_count: int,
    chromosome_count: int,
) -> None:
    """Record that single-trait preflight has completed."""
    native_pipeline_diagnostic_policy().record_pipeline_single_trait_preflight_completed_diagnostic_event(
        chromosome_count=chromosome_count,
        covariate_count=covariate_count,
        phenotype_name=phenotype_name,
        pipeline_label=pipeline_label,
        sample_count=sample_count,
    )
    native_run_event_telemetry_policy().record_single_trait_preflight_completed_telemetry_event(
        context.telemetry_session,
        context.association_mode.value,
        phenotype_name,
        sample_count,
        covariate_count,
        chromosome_count,
    )


def record_pipeline_multi_trait_started(
    *,
    association_mode: types.AssociationMode,
    phenotype_count: int,
    sample_mode: types.MultiPhenotypeSampleMode,
) -> None:
    """Record that a multi-trait pipeline has started."""
    native_pipeline_diagnostic_policy().record_pipeline_multi_trait_started_diagnostic_event(
        association_mode=association_mode.value,
        phenotype_count=phenotype_count,
        sample_mode=sample_mode.value,
    )


def record_pipeline_multi_trait_input_load_started(phenotype_count: int) -> None:
    """Record that multi-trait input loading has started."""
    native_pipeline_diagnostic_policy().record_pipeline_multi_trait_input_load_started_diagnostic_event(
        phenotype_count=phenotype_count,
    )


def record_pipeline_multi_trait_input_aligned(
    *,
    covariate_count: int,
    phenotype_count: int,
    sample_count: int,
) -> None:
    """Record that multi-trait input alignment has completed."""
    native_pipeline_diagnostic_policy().record_pipeline_multi_trait_input_aligned_diagnostic_event(
        covariate_count=covariate_count,
        phenotype_count=phenotype_count,
        sample_count=sample_count,
    )


def record_pipeline_multi_trait_prediction_source_load_started(phenotype_count: int) -> None:
    """Record that multi-trait prediction source loading has started."""
    native_pipeline_diagnostic_policy().record_pipeline_multi_trait_prediction_source_load_started_diagnostic_event(
        phenotype_count=phenotype_count,
    )


def record_pipeline_grouped_per_phenotype_started(
    *,
    association_mode: types.AssociationMode,
    phenotype_count: int,
    sample_mode: types.MultiPhenotypeSampleMode,
) -> None:
    """Record that grouped per-phenotype execution has started."""
    native_pipeline_diagnostic_policy().record_pipeline_grouped_per_phenotype_started_diagnostic_event(
        association_mode=association_mode.value,
        phenotype_count=phenotype_count,
        sample_mode=sample_mode.value,
    )


def record_pipeline_grouped_per_phenotype_groups_prepared(
    *,
    phenotype_count: int,
    phenotype_group_count: int,
) -> None:
    """Record that grouped per-phenotype inputs were grouped."""
    native_pipeline_diagnostic_policy().record_pipeline_grouped_per_phenotype_groups_prepared_diagnostic_event(
        phenotype_count=phenotype_count,
        phenotype_group_count=phenotype_group_count,
    )


def record_pipeline_grouped_union_delivery_selected(
    *,
    grouped_sample_count: int,
    phenotype_group_count: int,
    union_sample_count: int,
) -> None:
    """Record that grouped union delivery was selected."""
    native_pipeline_diagnostic_policy().record_pipeline_grouped_union_delivery_selected_diagnostic_event(
        grouped_sample_count=grouped_sample_count,
        phenotype_group_count=phenotype_group_count,
        union_sample_count=union_sample_count,
    )


def record_pipeline_multi_group_preflight_started(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    phenotype_count: int,
    sample_count: int,
) -> None:
    """Record that multi-group preflight has started."""
    native_pipeline_diagnostic_policy().record_pipeline_multi_group_preflight_started_diagnostic_event(
        phenotype_count=phenotype_count,
        sample_count=sample_count,
        trusted_no_missing_diploid=context.effective_trusted_no_missing_diploid,
        variant_limit=context.variant_limit,
    )


def record_pipeline_multi_group_preflight_completed(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    phenotype_count: int,
    sample_count: int,
) -> None:
    """Record that multi-group preflight has completed."""
    native_pipeline_diagnostic_policy().record_pipeline_multi_group_preflight_completed_diagnostic_event(
        phenotype_count=phenotype_count,
        sample_count=sample_count,
        trusted_no_missing_diploid=context.effective_trusted_no_missing_diploid,
        variant_limit=context.variant_limit,
    )
    native_run_event_telemetry_policy().record_multi_phenotype_preflight_completed_telemetry_event(
        context.telemetry_session,
        context.association_mode.value,
        phenotype_count,
        sample_count,
    )


def record_gpu_genotype_format_resolved(
    telemetry_session: TelemetrySession | None,
    *,
    requested_gpu_genotype_format: types.GpuGenotypeFormat,
    resolved_gpu_genotype_format: types.GpuGenotypeFormat,
    resolution_reason: str,
    fallback_error: str | None,
) -> None:
    """Record a resolved GPU genotype format decision."""
    native_pipeline_diagnostic_policy().record_pipeline_gpu_genotype_format_resolved_diagnostic_event(
        requested_gpu_genotype_format=requested_gpu_genotype_format.value,
        resolved_gpu_genotype_format=resolved_gpu_genotype_format.value,
        resolution_reason=resolution_reason,
        fallback_error=fallback_error,
    )
    native_run_event_telemetry_policy().record_gpu_genotype_format_resolved_telemetry_event(
        telemetry_session,
        requested_gpu_genotype_format.value,
        resolved_gpu_genotype_format.value,
        resolution_reason,
        fallback_error,
    )


def record_association_backend_selected(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    phenotype_name: str | None,
    phenotype_count: int | None,
) -> None:
    """Record selected association backend telemetry."""
    native_run_event_telemetry_policy().record_association_backend_selected_telemetry_event(
        context.telemetry_session,
        context.association_mode.value,
        context.backend_plan.backend_kind.value,
        context.backend_plan.jax_device.value,
        context.backend_plan.genotype_format.value,
        phenotype_name,
        phenotype_count,
    )


def record_bgen_engine_open_started(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    pipeline_label: str,
    phenotype_name: str | None,
    phenotype_count: int | None,
) -> None:
    """Record that BGEN engine opening has started."""
    native_pipeline_diagnostic_policy().record_pipeline_bgen_engine_open_started_diagnostic_event(
        phenotype_count=phenotype_count,
        phenotype_name=phenotype_name,
        pipeline_label=pipeline_label,
        trusted_no_missing_diploid=context.effective_trusted_no_missing_diploid,
        variant_limit=context.variant_limit,
    )


def record_bgen_engine_opened(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    engine: _core.Regenie2RunEngine,
    pipeline_label: str,
    phenotype_name: str | None,
    phenotype_count: int | None,
) -> None:
    """Record that BGEN engine opening completed."""
    native_pipeline_diagnostic_policy().record_pipeline_bgen_engine_opened_diagnostic_event(
        phenotype_count=phenotype_count,
        phenotype_name=phenotype_name,
        pipeline_label=pipeline_label,
        sample_count=int(engine.sample_count),
        variant_count=int(engine.variant_count),
    )
    native_run_event_telemetry_policy().record_bgen_engine_opened_telemetry_event(
        context.telemetry_session,
        context.association_mode.value,
        context.backend_plan.backend_kind.value,
        int(engine.sample_count),
        int(engine.variant_count),
        phenotype_name,
        phenotype_count,
    )


def record_prevalidated_bgen_engine_used(
    *,
    phenotype_count: int | None,
    phenotype_name: str | None,
    pipeline_label: str,
) -> None:
    """Record that a prevalidated BGEN engine is being reused."""
    native_pipeline_diagnostic_policy().record_pipeline_prevalidated_bgen_engine_used_diagnostic_event(
        phenotype_count=phenotype_count,
        phenotype_name=phenotype_name,
        pipeline_label=pipeline_label,
    )


def record_pipeline_output_resume_committed_chunks(
    *,
    committed_chunk_count: int,
    output_index: int,
) -> None:
    """Record committed chunk count discovered during resume."""
    native_pipeline_diagnostic_policy().record_pipeline_output_resume_committed_chunks_diagnostic_event(
        committed_chunk_count=committed_chunk_count,
        output_index=output_index,
    )


def record_pipeline_output_writer_sessions_create_started(
    *,
    association_mode: types.AssociationMode,
    output_count: int,
) -> None:
    """Record that output writer session creation has started."""
    native_pipeline_diagnostic_policy().record_pipeline_output_writer_sessions_create_started_diagnostic_event(
        association_mode=association_mode.value,
        output_count=output_count,
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
