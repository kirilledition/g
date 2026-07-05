"""Output lifecycle helpers for REGENIE step 2 pipelines."""

from __future__ import annotations

import time
import typing

from g import _core, execution_plan
from g.engine import timing as engine_timing
from g.engine.native_dispatch import engine as native_dispatch_engine
from g.engine.native_dispatch import groups as native_dispatch_groups
from g.io import output

if typing.TYPE_CHECKING:
    from g.engine.regenie2_pipeline import context as pipeline_context

type RunManifestHeaderInput = output.RunManifestHeaderInput
type ManifestFileFingerprintCache = output.ManifestFileFingerprintCache
type MultiPhenotypeSampleMode = output.MultiPhenotypeSampleMode

SINGLE_PHENOTYPE_SAMPLE_MODE = output.MultiPhenotypeSampleMode.SINGLE_PHENOTYPE
PER_PHENOTYPE_SAMPLE_MODE = output.MultiPhenotypeSampleMode.PER_PHENOTYPE
COMPLETE_CASE_SAMPLE_MODE = output.MultiPhenotypeSampleMode.COMPLETE_CASE
JAX_ENABLE_X64 = True


def open_pipeline_bgen_engine(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    pipeline_label: str,
    phenotype_name: str | None,
    phenotype_count: int | None,
) -> _core.Regenie2RunEngine:
    """Open the native BGEN engine and emit shared telemetry."""
    engine_start_time = time.perf_counter()
    _core.record_pipeline_bgen_engine_open_started_diagnostic_event(
        phenotype_count=phenotype_count,
        phenotype_name=phenotype_name,
        pipeline_label=pipeline_label,
        trusted_no_missing_diploid=context.effective_trusted_no_missing_diploid,
        variant_limit=context.variant_limit,
    )
    _core.record_association_backend_selected_telemetry_event(
        context.telemetry_session,
        context.association_mode.value,
        context.backend_plan.backend_kind.value,
        context.backend_plan.jax_device.value,
        context.backend_plan.genotype_format.value,
        phenotype_name,
        phenotype_count,
    )
    engine = native_dispatch_engine.build_bgen_run_engine(
        genotype_source_config=context.genotype_source_config,
        chunk_size=context.chunk_size,
        variant_limit=context.variant_limit,
        trusted_no_missing_diploid=context.effective_trusted_no_missing_diploid,
        trusted_bgen_validation_mode=context.trusted_bgen_validation_mode,
        trusted_bgen_validator=None,
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
    _core.record_bgen_engine_opened_telemetry_event(
        context.telemetry_session,
        context.association_mode.value,
        context.backend_plan.backend_kind.value,
        int(engine.sample_count),
        int(engine.variant_count),
        phenotype_name,
        phenotype_count,
    )
    return engine


def use_prepared_pipeline_bgen_engine(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    engine: _core.Regenie2RunEngine,
    pipeline_label: str,
    phenotype_name: str | None,
    phenotype_count: int | None,
) -> _core.Regenie2RunEngine:
    """Reuse a prevalidated BGEN engine and emit shared telemetry."""
    _core.record_pipeline_prevalidated_bgen_engine_used_diagnostic_event(
        phenotype_count=phenotype_count,
        phenotype_name=phenotype_name,
        pipeline_label=pipeline_label,
    )
    _core.record_association_backend_selected_telemetry_event(
        context.telemetry_session,
        context.association_mode.value,
        context.backend_plan.backend_kind.value,
        context.backend_plan.jax_device.value,
        context.backend_plan.genotype_format.value,
        phenotype_name,
        phenotype_count,
    )
    _core.record_pipeline_bgen_engine_opened_diagnostic_event(
        phenotype_count=phenotype_count,
        phenotype_name=phenotype_name,
        pipeline_label=pipeline_label,
        sample_count=int(engine.sample_count),
        variant_count=int(engine.variant_count),
    )
    _core.record_bgen_engine_opened_telemetry_event(
        context.telemetry_session,
        context.association_mode.value,
        context.backend_plan.backend_kind.value,
        int(engine.sample_count),
        int(engine.variant_count),
        phenotype_name,
        phenotype_count,
    )
    return engine


def build_pipeline_manifest_header(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    phenotype_name: str,
    covariate_names: tuple[str, ...],
    sample_count: int,
    variant_count: int,
    multi_phenotype_sample_mode: MultiPhenotypeSampleMode,
    phenotype_compute_group: execution_plan.PhenotypeComputeGroup | None,
) -> RunManifestHeaderInput:
    """Build the current manifest header for one output run."""
    phenotype_compute_group_id = (
        None
        if phenotype_compute_group is None
        else execution_plan.build_phenotype_compute_group_id(phenotype_compute_group)
    )
    return output.build_current_run_manifest_header(
        association_mode=context.association_mode,
        association_backend_kind=context.backend_plan.backend_kind,
        bgen_path=context.genotype_source_config.source_path,
        sample_path=context.genotype_source_config.sample_path,
        phenotype_path=context.phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=context.covariate_path,
        covariate_names=covariate_names,
        prediction_list_path=context.prediction_list_path,
        prediction_input_phenotype_names=(
            (phenotype_name,) if phenotype_compute_group is None else phenotype_compute_group.phenotype_names
        ),
        fingerprint_cache=context.input_fingerprint_cache,
        sample_count=sample_count,
        variant_count=variant_count,
        chunk_size=context.chunk_size,
        variant_limit=context.variant_limit,
        binary_correction_plan=context.correction_plan,
        trusted_no_missing_diploid=context.effective_trusted_no_missing_diploid,
        sample_key_mode=native_dispatch_groups.resolve_sample_key_mode(context.alignment_config),
        binary_kernel_config=context.binary_kernel_config if context.is_binary_trait else None,
        bgen_decode_tile_variant_count=context.bgen_decode_tile_variant_count,
        trusted_bgen_validation_mode=context.trusted_bgen_validation_mode,
        jax_device=context.jax_device,
        jax_enable_x64=JAX_ENABLE_X64,
        jax_matmul_precision=context.jax_matmul_precision,
        requested_gpu_genotype_format=context.requested_gpu_genotype_format,
        gpu_genotype_format=context.gpu_genotype_format,
        score_dtype=context.score_dtype,
        firth_dtype=context.firth_dtype,
        multi_phenotype_sample_mode=multi_phenotype_sample_mode,
        phenotype_compute_group_id=phenotype_compute_group_id,
        sample_set_fingerprint=None
        if phenotype_compute_group is None
        else phenotype_compute_group.sample_set_fingerprint,
        covariate_design_fingerprint=(
            None if phenotype_compute_group is None else phenotype_compute_group.covariate_design_fingerprint
        ),
        prediction_alignment_fingerprint=(
            None if phenotype_compute_group is None else phenotype_compute_group.prediction_alignment_fingerprint
        ),
        output_format=context.writer_settings.output_format,
        finalize_parquet=context.writer_settings.finalize_parquet,
        writer_thread_count=context.writer_settings.writer_thread_count,
        writer_queue_depth=context.writer_settings.writer_queue_depth,
        chunks_per_arrow_file=context.writer_settings.chunks_per_arrow_file,
        arrow_compression=context.writer_settings.arrow_compression,
        parquet_compression=context.writer_settings.parquet_compression,
        output_statistic_dtype=context.writer_settings.output_statistic_dtype,
    )


def initialize_pipeline_output_runs(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    phenotype_names: tuple[str, ...],
    current_headers_by_trait: tuple[RunManifestHeaderInput, ...],
) -> _core.NativeRunLifecycleOutputInitialization:
    """Validate/write output manifests and return committed chunk sets."""
    native_initialization = context.lifecycle_session.initialize_output_runs(
        phenotype_names,
        current_headers_by_trait,
    )
    if context.lifecycle_session.output_resume:
        for output_index, committed_chunk_identifier_set in enumerate(
            native_initialization.committed_chunk_identifier_sets()
        ):
            committed_chunk_count = len(committed_chunk_identifier_set)
            _core.record_pipeline_output_resume_committed_chunks_diagnostic_event(
                committed_chunk_count=committed_chunk_count,
                output_index=output_index,
            )
    return native_initialization


def validate_pipeline_resume_compatibility(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    phenotype_names: tuple[str, ...],
    current_headers_by_trait: tuple[RunManifestHeaderInput, ...],
) -> None:
    """Validate all resume manifests before any output run is mutated."""
    if not context.lifecycle_session.output_resume:
        return
    context.lifecycle_session.validate_output_resume_compatibility(phenotype_names, current_headers_by_trait)


def create_pipeline_writer_sessions(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    prepared_runs_by_trait: tuple[_core.NativeRunLifecyclePhenotypeRun, ...],
) -> tuple[typing.Any, ...]:
    """Create output writer sessions and record preparation timing."""
    writer_start_time = time.perf_counter()
    _core.record_pipeline_output_writer_sessions_create_started_diagnostic_event(
        association_mode=context.association_mode.value,
        output_count=len(prepared_runs_by_trait),
    )
    writer_sessions = tuple(
        _core.OutputWriterSession(
            prepared_run.run_directory,
            prepared_run.chunks_directory,
            context.association_mode.value,
            writer_thread_count=context.writer_settings.writer_thread_count,
            writer_queue_depth=context.writer_settings.writer_queue_depth,
            output_format=context.writer_settings.output_format.value,
            output_statistic_dtype=context.writer_settings.output_statistic_dtype.value,
            finalize_parquet=context.writer_settings.finalize_parquet,
            chunks_per_arrow_file=context.writer_settings.chunks_per_arrow_file,
            arrow_compression=context.writer_settings.arrow_compression.value,
            parquet_compression=context.writer_settings.parquet_compression.value,
            collect_stage_timings=engine_timing.should_collect_exact_stage_timings(context.stage_timing_recorder),
        )
        for prepared_run in prepared_runs_by_trait
    )
    engine_timing.record_stage_duration(context.stage_timing_recorder, "output_writer_preparation", writer_start_time)
    return writer_sessions


def committed_chunk_identifiers(
    initialization: _core.NativeRunLifecycleOutputInitialization,
    output_index: int,
) -> set[int]:
    """Return committed chunk identifiers for one initialized output."""
    return {int(chunk_identifier) for chunk_identifier in initialization.committed_chunk_identifiers(output_index)}


def committed_chunk_identifier_sets(
    initialization: _core.NativeRunLifecycleOutputInitialization,
) -> tuple[set[int], ...]:
    """Return committed chunk identifiers for each initialized output."""
    return tuple(
        {int(chunk_identifier) for chunk_identifier in chunk_identifier_set}
        for chunk_identifier_set in initialization.committed_chunk_identifier_sets()
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
