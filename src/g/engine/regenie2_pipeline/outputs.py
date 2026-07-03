"""Output lifecycle helpers for REGENIE step 2 pipelines."""

from __future__ import annotations

import time
import typing
from dataclasses import dataclass

from g import _core, execution_plan, types
from g.engine.regenie2_pipeline import bgen_engine, inputs, runtime_policy, telemetry_events
from g.runner import outputs as output
from g.runner import timing

if typing.TYPE_CHECKING:
    from g.engine.regenie2_pipeline import context as pipeline_context

type OutputRunPaths = output.OutputRunPaths
type OutputWriterSettings = output.OutputWriterSettings
type RunManifestHeaderInput = output.RunManifestHeaderInput
type ManifestFileFingerprintCache = output.ManifestFileFingerprintCache
type MultiPhenotypeSampleMode = output.MultiPhenotypeSampleMode

SINGLE_PHENOTYPE_SAMPLE_MODE = output.MultiPhenotypeSampleMode.SINGLE_PHENOTYPE
PER_PHENOTYPE_SAMPLE_MODE = output.MultiPhenotypeSampleMode.PER_PHENOTYPE
COMPLETE_CASE_SAMPLE_MODE = output.MultiPhenotypeSampleMode.COMPLETE_CASE


@dataclass(frozen=True)
class InitializedPipelineOutputRuns:
    """Native initialization result for pipeline output runs.

    Attributes:
        native_initialization: Native output initialization result handle.

    """

    native_initialization: _core.NativePipelineOutputInitialization

    @property
    def committed_chunk_identifier_sets(self) -> tuple[set[int], ...]:
        """Return committed chunk identifiers for each initialized output."""
        return tuple(
            {int(chunk_identifier) for chunk_identifier in chunk_identifier_set}
            for chunk_identifier_set in self.native_initialization.committed_chunk_identifier_sets()
        )

    def committed_chunk_identifiers(self, output_index: int) -> set[int]:
        """Return committed chunk identifiers for one initialized output.

        Args:
            output_index: Output index in the native initialization result.

        Returns:
            Committed chunk identifiers for the requested output.

        """
        return {
            int(chunk_identifier)
            for chunk_identifier in self.native_initialization.committed_chunk_identifiers(output_index)
        }


def log_association_backend_selected(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    phenotype_name: str | None,
    phenotype_count: int | None,
) -> None:
    """Emit telemetry for the concrete association backend selection."""
    telemetry_events.native_run_event_telemetry_policy().record_association_backend_selected_telemetry_event(
        context.telemetry_session,
        context.association_mode.value,
        context.backend_plan.backend_kind.value,
        context.backend_plan.jax_device.value,
        context.backend_plan.genotype_format.value,
        phenotype_name,
        phenotype_count,
    )


def log_bgen_engine_opened(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    engine: _core.Regenie2RunEngine,
    phenotype_name: str | None,
    phenotype_count: int | None,
) -> None:
    """Emit telemetry for an opened BGEN engine."""
    telemetry_events.native_run_event_telemetry_policy().record_bgen_engine_opened_telemetry_event(
        context.telemetry_session,
        context.association_mode.value,
        context.backend_plan.backend_kind.value,
        int(engine.sample_count),
        int(engine.variant_count),
        phenotype_name,
        phenotype_count,
    )


def open_pipeline_bgen_engine(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    pipeline_label: str,
    phenotype_name: str | None,
    phenotype_count: int | None,
) -> _core.Regenie2RunEngine:
    """Open the native BGEN engine and emit shared telemetry."""
    engine_start_time = time.perf_counter()
    native_pipeline_diagnostic_policy = telemetry_events.native_pipeline_diagnostic_policy()
    native_pipeline_diagnostic_policy.record_pipeline_bgen_engine_open_started_diagnostic_event(
        phenotype_count=phenotype_count,
        phenotype_name=phenotype_name,
        pipeline_label=pipeline_label,
        trusted_no_missing_diploid=context.effective_trusted_no_missing_diploid,
        variant_limit=context.variant_limit,
    )
    log_association_backend_selected(context=context, phenotype_name=phenotype_name, phenotype_count=phenotype_count)
    engine = bgen_engine.build_bgen_run_engine(
        genotype_source_config=context.genotype_source_config,
        chunk_size=context.chunk_size,
        variant_limit=context.variant_limit,
        trusted_no_missing_diploid=context.effective_trusted_no_missing_diploid,
        trusted_bgen_validation_mode=context.trusted_bgen_validation_mode,
        trusted_bgen_validator=None,
    )
    timing.record_stage_duration(context.stage_timing_recorder, "bgen_engine_open_index_setup", engine_start_time)
    native_pipeline_diagnostic_policy.record_pipeline_bgen_engine_opened_diagnostic_event(
        phenotype_count=phenotype_count,
        phenotype_name=phenotype_name,
        pipeline_label=pipeline_label,
        sample_count=int(engine.sample_count),
        variant_count=int(engine.variant_count),
    )
    log_bgen_engine_opened(
        context=context,
        engine=engine,
        phenotype_name=phenotype_name,
        phenotype_count=phenotype_count,
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
    native_pipeline_diagnostic_policy = telemetry_events.native_pipeline_diagnostic_policy()
    native_pipeline_diagnostic_policy.record_pipeline_prevalidated_bgen_engine_used_diagnostic_event(
        phenotype_count=phenotype_count,
        phenotype_name=phenotype_name,
        pipeline_label=pipeline_label,
    )
    log_association_backend_selected(context=context, phenotype_name=phenotype_name, phenotype_count=phenotype_count)
    native_pipeline_diagnostic_policy.record_pipeline_bgen_engine_opened_diagnostic_event(
        phenotype_count=phenotype_count,
        phenotype_name=phenotype_name,
        pipeline_label=pipeline_label,
        sample_count=int(engine.sample_count),
        variant_count=int(engine.variant_count),
    )
    log_bgen_engine_opened(
        context=context,
        engine=engine,
        phenotype_name=phenotype_name,
        phenotype_count=phenotype_count,
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
        sample_key_mode=inputs.resolve_sample_key_mode(context.alignment_config),
        binary_kernel_config=context.binary_kernel_config if context.is_binary_trait else None,
        bgen_decode_tile_variant_count=context.bgen_decode_tile_variant_count,
        trusted_bgen_validation_mode=context.trusted_bgen_validation_mode,
        jax_device=context.jax_device,
        jax_enable_x64=runtime_policy.JAX_ENABLE_X64,
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
    output_run_paths_by_trait: tuple[OutputRunPaths, ...],
    existing_manifests_by_trait: tuple[dict[str, typing.Any] | None, ...],
    current_headers_by_trait: tuple[RunManifestHeaderInput, ...],
    resume: bool,
    resume_mode: types.ResumeMode,
    runtime_compatibility_token: _core.NativeRuntimeCompatibilityToken,
) -> InitializedPipelineOutputRuns:
    """Validate/write output manifests and return committed chunk sets."""
    native_preparation_batch = build_pipeline_output_preparation_batch(
        output_run_paths_by_trait=output_run_paths_by_trait,
        existing_manifests_by_trait=existing_manifests_by_trait,
        current_headers_by_trait=current_headers_by_trait,
        resume=resume,
        resume_mode=resume_mode,
    )
    native_initialization = native_preparation_batch.initialize(runtime_compatibility_token)
    if resume:
        native_pipeline_diagnostic_policy = telemetry_events.native_pipeline_diagnostic_policy()
        for output_index, committed_chunk_identifier_set in enumerate(
            native_initialization.committed_chunk_identifier_sets()
        ):
            committed_chunk_count = len(committed_chunk_identifier_set)
            native_pipeline_diagnostic_policy.record_pipeline_output_resume_committed_chunks_diagnostic_event(
                committed_chunk_count=committed_chunk_count,
                output_index=output_index,
            )
    return InitializedPipelineOutputRuns(native_initialization=native_initialization)


def validate_pipeline_resume_compatibility(
    *,
    output_run_paths_by_trait: tuple[OutputRunPaths, ...],
    existing_manifests_by_trait: tuple[dict[str, typing.Any] | None, ...],
    current_headers_by_trait: tuple[RunManifestHeaderInput, ...],
    resume_mode: types.ResumeMode,
) -> None:
    """Validate all resume manifests before any output run is mutated."""
    native_preparation_batch = build_pipeline_output_preparation_batch(
        output_run_paths_by_trait=output_run_paths_by_trait,
        existing_manifests_by_trait=existing_manifests_by_trait,
        current_headers_by_trait=current_headers_by_trait,
        resume=True,
        resume_mode=resume_mode,
    )
    native_preparation_batch.validate_resume_compatibility()


def build_pipeline_output_preparation_batch(
    *,
    output_run_paths_by_trait: tuple[OutputRunPaths, ...],
    existing_manifests_by_trait: tuple[dict[str, typing.Any] | None, ...],
    current_headers_by_trait: tuple[RunManifestHeaderInput, ...],
    resume: bool,
    resume_mode: types.ResumeMode,
) -> _core.NativePipelineOutputPreparationBatch:
    """Build the native output-preparation batch handle.

    Args:
        output_run_paths_by_trait: Output run paths in trait order.
        existing_manifests_by_trait: Existing manifest mappings in trait order.
        current_headers_by_trait: Current manifest headers in trait order.
        resume: Whether this batch will resume existing output runs.
        resume_mode: Resume validation policy.

    Returns:
        Native output preparation batch handle.

    """
    return output.build_native_pipeline_output_preparation_batch(
        output_run_paths_by_trait=output_run_paths_by_trait,
        existing_manifests_by_trait=existing_manifests_by_trait,
        current_headers_by_trait=current_headers_by_trait,
        resume=resume,
        resume_mode=resume_mode,
    )


def notify_output_runs_initialized(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    phenotype_names: tuple[str, ...],
) -> None:
    """Notify the runner that manifest compatibility has passed."""
    if context.output_initialized_callback is None:
        return
    context.output_initialized_callback(phenotype_names)


def create_pipeline_writer_sessions(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    output_run_paths_by_trait: tuple[OutputRunPaths, ...],
) -> tuple[typing.Any, ...]:
    """Create output writer sessions and record preparation timing."""
    writer_start_time = time.perf_counter()
    telemetry_events.native_pipeline_diagnostic_policy().record_pipeline_output_writer_sessions_create_started_diagnostic_event(
        association_mode=context.association_mode.value,
        output_count=len(output_run_paths_by_trait),
    )
    writer_sessions = tuple(
        output.create_output_writer_session(
            output_run_paths,
            context.association_mode,
            writer_thread_count=context.writer_settings.writer_thread_count,
            writer_queue_depth=context.writer_settings.writer_queue_depth,
            finalize_parquet=context.writer_settings.finalize_parquet,
            output_format=context.writer_settings.output_format,
            chunks_per_arrow_file=context.writer_settings.chunks_per_arrow_file,
            arrow_compression=context.writer_settings.arrow_compression,
            parquet_compression=context.writer_settings.parquet_compression,
            output_statistic_dtype=context.writer_settings.output_statistic_dtype,
            collect_stage_timings=timing.should_collect_exact_stage_timings(context.stage_timing_recorder),
        )
        for output_run_paths in output_run_paths_by_trait
    )
    timing.record_stage_duration(context.stage_timing_recorder, "output_writer_preparation", writer_start_time)
    return writer_sessions


def build_manifest_file_fingerprint_cache() -> ManifestFileFingerprintCache:
    """Build a run-scoped manifest fingerprint cache."""
    return output.ManifestFileFingerprintCache()
