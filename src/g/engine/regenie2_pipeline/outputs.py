"""Output lifecycle helpers for REGENIE step 2 pipelines."""

from __future__ import annotations

import dataclasses
import json
import time
import typing

from g import _core, execution_plan, types
from g.engine import timing as engine_timing
from g.engine.native_dispatch import groups as native_dispatch_groups

if typing.TYPE_CHECKING:
    from g.engine.regenie2_pipeline import context as pipeline_context


class OutputPreparationBgenEngineProtocol(typing.Protocol):
    """Minimal engine shape needed for output preparation."""

    variant_count: int


type OutputPreparationGroupInput = tuple[
    tuple[str, ...],
    tuple[str, ...],
    int,
    str,
    str | None,
    tuple[int, ...] | None,
    tuple[str, ...] | None,
    str | None,
    str | None,
    str | None,
    str | None,
]


def open_pipeline_bgen_engine(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    pipeline_label: str,
    phenotype_name: str | None,
    phenotype_count: int | None,
) -> _core.NativeRunEngineSession:
    """Open the native BGEN engine and emit shared telemetry."""
    engine_start_time = time.perf_counter()
    _core.record_pipeline_bgen_engine_open_started_diagnostic_event(
        phenotype_count=phenotype_count,
        phenotype_name=phenotype_name,
        pipeline_label=pipeline_label,
        trusted_no_missing_diploid=context.effective_trusted_no_missing_diploid,
        variant_limit=context.variant_limit,
    )
    _core.record_association_backend_selected_telemetry(
        context.telemetry_session,
        context.association_mode.value,
        context.backend_plan.backend_kind.value,
        context.backend_plan.jax_device.value,
        context.backend_plan.genotype_format.value,
        phenotype_name,
        phenotype_count,
    )
    engine = context.engine_session
    engine.open_bgen_engine(
        str(context.genotype_source_config.source_path),
        chunk_size=context.chunk_size,
        variant_limit=context.variant_limit,
        trusted_no_missing_diploid=context.effective_trusted_no_missing_diploid,
        trusted_bgen_validation_mode=(
            context.trusted_bgen_validation_mode.value if context.effective_trusted_no_missing_diploid else None
        ),
    )
    engine_timing.record_stage_duration(
        context.stage_timing_recorder,
        "bgen_engine_open_index_setup",
        engine_start_time,
    )
    _core.record_pipeline_bgen_engine_opened_diagnostic_event(
        phenotype_count=phenotype_count,
        phenotype_name=phenotype_name,
        pipeline_label=pipeline_label,
        sample_count=int(engine.sample_count),
        variant_count=int(engine.variant_count),
    )
    _core.record_bgen_engine_opened_telemetry(
        context.telemetry_session,
        context.association_mode.value,
        context.backend_plan.backend_kind.value,
        int(engine.sample_count),
        int(engine.variant_count),
        phenotype_name,
        phenotype_count,
    )
    return engine


def build_binary_kernel_config_json(
    *,
    context: pipeline_context.Regenie2PipelineContext,
) -> str | None:
    """Serialize Python-owned binary kernel config for native manifest preparation."""
    if not context.is_binary_trait or context.binary_kernel_config is None:
        return None
    return json.dumps(
        dataclasses.asdict(context.binary_kernel_config),
        sort_keys=True,
        separators=(",", ":"),
    )


def build_output_preparation_group(
    *,
    phenotype_names: tuple[str, ...],
    covariate_names: tuple[str, ...],
    sample_count: int,
    output_sample_mode: types.MultiPhenotypeSampleMode,
    phenotype_compute_group: execution_plan.PhenotypeComputeGroup | None,
) -> OutputPreparationGroupInput:
    """Build compact runtime data for native output preparation."""
    return (
        phenotype_names,
        covariate_names,
        sample_count,
        output_sample_mode.value,
        None if phenotype_compute_group is None else phenotype_compute_group.group_mode.value,
        None if phenotype_compute_group is None else phenotype_compute_group.phenotype_indices,
        None if phenotype_compute_group is None else phenotype_compute_group.phenotype_names,
        None if phenotype_compute_group is None else phenotype_compute_group.sample_mode.value,
        None if phenotype_compute_group is None else phenotype_compute_group.sample_set_fingerprint,
        None if phenotype_compute_group is None else phenotype_compute_group.covariate_design_fingerprint,
        None if phenotype_compute_group is None else phenotype_compute_group.prediction_alignment_fingerprint,
    )


def prepare_output_bundles(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    engine: OutputPreparationBgenEngineProtocol,
    output_groups: tuple[OutputPreparationGroupInput, ...],
) -> tuple[_core.NativePreparedOutputBundle, ...]:
    """Prepare output runs and writer sessions through the native lifecycle."""
    return context.engine_session.prepare_output_bundles_from_runtime_plan(
        output_groups,
        int(engine.variant_count),
        context.effective_trusted_no_missing_diploid,
        native_dispatch_groups.resolve_sample_key_mode(context.alignment_config).value,
        build_binary_kernel_config_json(context=context),
        context.requested_gpu_genotype_format.value,
        context.gpu_genotype_format.value,
        context.score_dtype.value,
        context.firth_dtype.value,
        None if context.stage_timing_recorder is None else context.stage_timing_recorder.native_recorder,
    )


def existing_manifest_from_prepared_run(
    prepared_run: _core.NativeRunLifecyclePhenotypeRun,
) -> dict[str, typing.Any] | None:
    """Return an existing persisted manifest mapping for GPU-format planning."""
    existing_manifest = prepared_run.existing_manifest_payload()
    if existing_manifest is None:
        return None
    if not isinstance(existing_manifest, dict):
        message = "Native prepared run existing manifest payload must be a mapping."
        raise TypeError(message)
    return existing_manifest
