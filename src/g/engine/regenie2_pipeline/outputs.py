"""Output lifecycle helpers for REGENIE step 2 pipelines."""

from __future__ import annotations

import json
import logging
import time
import typing
from dataclasses import dataclass

from g import _core, execution_plan, types
from g.engine import timing
from g.engine.native_dispatch import engine as native_dispatch_engine
from g.engine.native_dispatch import groups as native_dispatch_groups
from g.io import output

if typing.TYPE_CHECKING:
    from g.engine.regenie2_pipeline import context as pipeline_context

logger = logging.getLogger(__name__)


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
    if context.telemetry_session is None:
        return
    context.telemetry_session.log_association_backend_selected(
        association_mode=context.association_mode,
        association_backend_kind=context.backend_plan.backend_kind,
        device=context.backend_plan.jax_device,
        genotype_format=context.backend_plan.genotype_format,
        phenotype=phenotype_name,
        phenotype_count=phenotype_count,
    )


def log_bgen_engine_opened(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    engine: _core.Regenie2RunEngine,
    phenotype_name: str | None,
    phenotype_count: int | None,
) -> None:
    """Emit telemetry for an opened BGEN engine."""
    if context.telemetry_session is None:
        return
    context.telemetry_session.log_bgen_engine_opened(
        association_mode=context.association_mode,
        association_backend_kind=context.backend_plan.backend_kind,
        sample_count=int(engine.sample_count),
        variant_count=int(engine.variant_count),
        phenotype=phenotype_name,
        phenotype_count=phenotype_count,
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
    logger.debug("Opening native BGEN engine for %s pipeline.", pipeline_label)
    log_association_backend_selected(context=context, phenotype_name=phenotype_name, phenotype_count=phenotype_count)
    engine = native_dispatch_engine.build_bgen_run_engine(
        genotype_source_config=context.genotype_source_config,
        chunk_size=context.chunk_size,
        variant_limit=context.variant_limit,
        trusted_no_missing_diploid=context.effective_trusted_no_missing_diploid,
        trusted_bgen_validation_mode=context.trusted_bgen_validation_mode,
        trusted_bgen_validator=None,
    )
    timing.record_stage_duration(context.stage_timing_recorder, "bgen_engine_open_index_setup", engine_start_time)
    logger.debug(
        "Native BGEN engine opened for %s pipeline: sample_count=%s variant_count=%s.",
        pipeline_label,
        engine.sample_count,
        engine.variant_count,
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
    logger.debug("Using prevalidated native BGEN engine for %s pipeline.", pipeline_label)
    log_association_backend_selected(context=context, phenotype_name=phenotype_name, phenotype_count=phenotype_count)
    logger.debug(
        "Native BGEN engine opened for %s pipeline: sample_count=%s variant_count=%s.",
        pipeline_label,
        engine.sample_count,
        engine.variant_count,
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
    multi_phenotype_sample_mode: output.MultiPhenotypeSampleMode,
    phenotype_compute_group: execution_plan.PhenotypeComputeGroup | None,
) -> output.CurrentRunManifestHeader:
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
    output_run_paths_by_trait: tuple[output.OutputRunPaths, ...],
    existing_manifests_by_trait: tuple[dict[str, typing.Any] | None, ...],
    current_headers_by_trait: tuple[output.RunManifestHeaderInput, ...],
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
        for committed_chunk_identifier_set in native_initialization.committed_chunk_identifier_sets():
            logger.info("Resuming run with %d previously committed chunks.", len(committed_chunk_identifier_set))
    return InitializedPipelineOutputRuns(native_initialization=native_initialization)


def validate_pipeline_resume_compatibility(
    *,
    output_run_paths_by_trait: tuple[output.OutputRunPaths, ...],
    existing_manifests_by_trait: tuple[dict[str, typing.Any] | None, ...],
    current_headers_by_trait: tuple[output.RunManifestHeaderInput, ...],
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
    output_run_paths_by_trait: tuple[output.OutputRunPaths, ...],
    existing_manifests_by_trait: tuple[dict[str, typing.Any] | None, ...],
    current_headers_by_trait: tuple[output.RunManifestHeaderInput, ...],
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
    return _core.NativePipelineOutputPreparationBatch(
        tuple(str(output_run_paths.run_directory) for output_run_paths in output_run_paths_by_trait),
        tuple(str(output_run_paths.chunks_directory) for output_run_paths in output_run_paths_by_trait),
        tuple(
            None if existing_manifest is None else json.dumps(existing_manifest, sort_keys=True)
            for existing_manifest in existing_manifests_by_trait
        ),
        tuple(
            json.dumps(output.run_manifest_header_input_to_mapping(current_header), sort_keys=True)
            for current_header in current_headers_by_trait
        ),
        resume,
        resume_mode.value,
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
    output_run_paths_by_trait: tuple[output.OutputRunPaths, ...],
) -> tuple[typing.Any, ...]:
    """Create output writer sessions and record preparation timing."""
    writer_start_time = time.perf_counter()
    logger.debug("Creating output writer(s) for %s pipeline.", context.association_mode.value)
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
