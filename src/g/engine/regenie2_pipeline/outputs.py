"""Output lifecycle helpers for REGENIE step 2 pipelines."""

from __future__ import annotations

import time
import typing

from g import _core, execution_plan, io, types
from g.engine import timing as engine_timing
from g.engine.native_dispatch import groups as native_dispatch_groups

if typing.TYPE_CHECKING:
    from g.engine.regenie2_pipeline import context as pipeline_context

type RunManifestHeaderInput = io.RunManifestHeaderInput
type ManifestFileFingerprintCache = io.ManifestFileFingerprintCache


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
    _core.record_association_backend_selected_telemetry(
        context.telemetry_session,
        context.association_mode.value,
        context.backend_plan.backend_kind.value,
        context.backend_plan.jax_device.value,
        context.backend_plan.genotype_format.value,
        phenotype_name,
        phenotype_count,
    )
    engine = _core.Regenie2RunEngine(
        str(context.genotype_source_config.source_path),
        chunk_size=context.chunk_size,
        variant_limit=context.variant_limit,
        trusted_no_missing_diploid=context.effective_trusted_no_missing_diploid,
    )
    if context.effective_trusted_no_missing_diploid:
        engine.validate_trusted_no_missing_diploid_with_default_cache(
            str(context.genotype_source_config.source_path),
            context.trusted_bgen_validation_mode.value,
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


def build_pipeline_manifest_header(
    *,
    context: pipeline_context.Regenie2PipelineContext,
    phenotype_name: str,
    covariate_names: tuple[str, ...],
    sample_count: int,
    variant_count: int,
    multi_phenotype_sample_mode: types.MultiPhenotypeSampleMode,
    phenotype_compute_group: execution_plan.PhenotypeComputeGroup | None,
) -> RunManifestHeaderInput:
    """Build the current manifest header for one output run."""
    return io.build_current_run_manifest_header(
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
        jax_enable_x64=True,
        jax_matmul_precision=context.jax_matmul_precision,
        requested_gpu_genotype_format=context.requested_gpu_genotype_format,
        gpu_genotype_format=context.gpu_genotype_format,
        score_dtype=context.score_dtype,
        firth_dtype=context.firth_dtype,
        multi_phenotype_sample_mode=multi_phenotype_sample_mode,
        phenotype_compute_group_mode=None if phenotype_compute_group is None else phenotype_compute_group.group_mode,
        phenotype_compute_group_indices=None
        if phenotype_compute_group is None
        else phenotype_compute_group.phenotype_indices,
        phenotype_compute_group_names=None
        if phenotype_compute_group is None
        else phenotype_compute_group.phenotype_names,
        phenotype_compute_group_sample_mode=None
        if phenotype_compute_group is None
        else phenotype_compute_group.sample_mode,
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
