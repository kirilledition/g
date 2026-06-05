"""Native-driven REGENIE step 2 pipeline wrappers."""

from __future__ import annotations

import hashlib
import logging
import time
import typing
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from g import _core, types
from g.compute.regenie2_binary import api as regenie2_binary
from g.compute.regenie2_binary import config as regenie2_binary_config
from g.compute.regenie2_linear import api as regenie2_linear
from g.engine import callbacks, native_dispatch, preflight, shutdown, telemetry, timing
from g.io import output, source

REGENIE_COMPUTE_PATCH_TARGETS = (regenie2_binary, regenie2_linear)
logger = logging.getLogger(__name__)


def run_regenie2_linear_bgen_pipeline(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    phenotype_path: Path,
    phenotype_name: str,
    prediction_list_path: Path,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    chunk_size: int,
    variant_limit: int | None,
    output_run_paths: output.OutputRunPaths,
    staging_depth: int = 1,
    existing_manifest: dict[str, typing.Any] | None = None,
    resume: bool = False,
    resume_mode: types.ResumeMode = types.ResumeMode.FAST,
    finalize_parquet: bool = False,
    writer_thread_count: int = output.PACKAGED_WRITER_THREAD_COUNT,
    writer_queue_depth: int = output.PACKAGED_WRITER_QUEUE_DEPTH,
    chunks_per_arrow_file: int = output.PACKAGED_CHUNKS_PER_ARROW_FILE,
    arrow_compression: types.ArrowCompression = types.ArrowCompression.ZSTD,
    trusted_no_missing_diploid: bool = False,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode = types.TrustedBgenValidationMode.CACHE_ON_MISS,
    bgen_decode_tile_variant_count: int = output.PACKAGED_BGEN_DECODE_TILE_VARIANT_COUNT,
    bgen_simd: types.BgenSimdMode = output.PACKAGED_BGEN_SIMD,
    jax_device: types.Device = types.Device.CPU,
    jax_matmul_precision: types.JaxMatmulPrecision | None = None,
    jax_enable_x64: bool = output.PACKAGED_JAX_ENABLE_X64,
    score_dtype: types.FloatingPointDtype = output.PACKAGED_SCORE_DTYPE,
    firth_dtype: types.FloatingPointDtype = output.PACKAGED_FIRTH_DTYPE,
    output_format: types.OutputFormat = types.OutputFormat.PARQUET,
    stage_timing_recorder: timing.StageTimingRecorder | None = None,
    telemetry_session: telemetry.TelemetrySession | None = None,
    alignment_config: native_dispatch.SampleAlignmentConfigProtocol | None = None,
) -> Path | None:
    """Run the native BGEN pipeline for quantitative REGENIE step 2."""
    logger.info("Starting linear REGENIE step 2 BGEN pipeline.")
    stage_timing_recorder = stage_timing_recorder or timing.build_stage_timing_recorder()
    use_variant_major = True
    engine_start_time = time.perf_counter()
    logger.debug("Opening native BGEN engine for linear pipeline.")
    engine = native_dispatch.build_bgen_run_engine(
        genotype_source_config=genotype_source_config,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
        trusted_bgen_validation_mode=trusted_bgen_validation_mode,
    )
    timing.record_stage_duration(stage_timing_recorder, "bgen_engine_open_index_setup", engine_start_time)
    logger.debug(
        "Native BGEN engine opened for linear pipeline: sample_count=%s variant_count=%s.",
        engine.sample_count,
        engine.variant_count,
    )
    if telemetry_session is not None:
        telemetry_session.log_event(
            "bgen_engine_opened",
            association_mode=types.AssociationMode.REGENIE2_LINEAR.value,
            phenotype=phenotype_name,
            sample_count=int(engine.sample_count),
            variant_count=int(engine.variant_count),
        )
    alignment_start_time = time.perf_counter()
    logger.debug("Loading aligned native sample, phenotype, and covariate inputs for linear pipeline.")
    run_input = native_dispatch.load_native_bgen_run_input(
        genotype_source_config=genotype_source_config,
        engine=engine,
        phenotype_path=phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=False,
        alignment_config=alignment_config,
    )
    timing.record_stage_duration(stage_timing_recorder, "sample_phenotype_covariate_alignment", alignment_start_time)
    logger.debug(
        "Aligned linear pipeline inputs: sample_count=%s covariate_count=%s.",
        int(run_input.sample_indices.shape[0]),
        len(run_input.native_aligned_sample_data.covariate_names),
    )
    if telemetry_session is not None:
        telemetry_session.log_event(
            "sample_alignment_completed",
            association_mode=types.AssociationMode.REGENIE2_LINEAR.value,
            phenotype=phenotype_name,
            sample_count=int(run_input.sample_indices.shape[0]),
            covariate_count=len(run_input.native_aligned_sample_data.covariate_names),
        )
    current_header = output.build_current_run_manifest_header(
        association_mode=types.AssociationMode.REGENIE2_LINEAR,
        bgen_path=genotype_source_config.source_path,
        sample_path=source.resolve_bgen_sample_path(
            genotype_source_config.source_path,
            genotype_source_config.sample_path,
        ),
        phenotype_path=phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=covariate_path,
        covariate_names=tuple(run_input.native_aligned_sample_data.covariate_names),
        prediction_list_path=prediction_list_path,
        sample_count=int(run_input.sample_indices.shape[0]),
        variant_count=int(engine.variant_count),
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        binary_correction_plan=types.BinaryCorrectionPlan(),
        trusted_no_missing_diploid=trusted_no_missing_diploid,
        sample_key_mode=native_dispatch.resolve_sample_key_mode(alignment_config),
        bgen_decode_tile_variant_count=bgen_decode_tile_variant_count,
        bgen_simd=bgen_simd,
        trusted_bgen_validation_mode=trusted_bgen_validation_mode,
        jax_device=jax_device,
        jax_matmul_precision=jax_matmul_precision,
        jax_enable_x64=jax_enable_x64,
        score_dtype=score_dtype,
        firth_dtype=firth_dtype,
        output_format=output_format,
        finalize_parquet=finalize_parquet,
        writer_thread_count=writer_thread_count,
        writer_queue_depth=writer_queue_depth,
        chunks_per_arrow_file=chunks_per_arrow_file,
        arrow_compression=arrow_compression,
    )
    initialized_output_run = output.initialize_output_run(
        output_run_paths=output_run_paths,
        existing_manifest=existing_manifest,
        current_header=current_header,
        resume=resume,
        resume_mode=resume_mode,
    )
    writer_start_time = time.perf_counter()
    logger.debug("Creating output writer for linear pipeline.")
    writer_session = output.create_output_writer_session(
        output_run_paths,
        types.AssociationMode.REGENIE2_LINEAR,
        writer_thread_count=writer_thread_count,
        writer_queue_depth=writer_queue_depth,
        finalize_parquet=finalize_parquet,
        chunks_per_arrow_file=chunks_per_arrow_file,
        arrow_compression=arrow_compression,
        collect_stage_timings=stage_timing_recorder is not None,
    )
    timing.record_stage_duration(stage_timing_recorder, "output_writer_preparation", writer_start_time)
    prediction_start_time = time.perf_counter()
    logger.debug("Loading REGENIE prediction source for linear pipeline.")
    prediction_source = native_dispatch.build_regenie_prediction_source(
        prediction_list_path=prediction_list_path,
        phenotype_name=phenotype_name,
        run_input=run_input,
        alignment_config=alignment_config,
    )
    timing.record_stage_duration(stage_timing_recorder, "prediction_source_load", prediction_start_time)
    if telemetry_session is not None:
        telemetry_session.log_event(
            "prediction_source_loaded",
            association_mode=types.AssociationMode.REGENIE2_LINEAR.value,
            phenotype=phenotype_name,
        )
    preflight_start_time = time.perf_counter()
    logger.debug("Running preflight validation for linear pipeline.")
    preflight_report = preflight.run_regenie2_preflight(
        run_input=run_input,
        prediction_source=prediction_source,
        engine=engine,
        variant_limit=variant_limit,
        is_binary_trait=False,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
    )
    timing.record_stage_duration(stage_timing_recorder, "preflight_validation", preflight_start_time)
    logger.debug(
        "Preflight validation passed for linear pipeline: sample_count=%s covariate_count=%s chromosome_count=%s.",
        preflight_report.sample_count,
        preflight_report.covariate_count,
        preflight_report.chromosome_count,
    )
    if telemetry_session is not None:
        telemetry_session.log_event(
            "preflight_completed",
            association_mode=types.AssociationMode.REGENIE2_LINEAR.value,
            phenotype=phenotype_name,
            sample_count=preflight_report.sample_count,
            covariate_count=preflight_report.covariate_count,
            chromosome_count=preflight_report.chromosome_count,
        )
    callback = callbacks.LinearRegenie2PipelineCallback(
        run_input=run_input,
        prediction_source=prediction_source,
        writer_session=writer_session,
        staging_depth=staging_depth,
        score_dtype=score_dtype,
        stage_timing_recorder=stage_timing_recorder,
        telemetry_session=telemetry_session,
    )
    return native_dispatch.run_bgen_engine_with_callback(
        engine=engine,
        run_input=run_input,
        committed_chunk_identifiers=set(initialized_output_run.committed_chunk_identifiers),
        writer_session=writer_session,
        callback=callback,
        stage_timing_recorder=stage_timing_recorder,
        variant_major_dosage=use_variant_major,
    )


def run_regenie2_binary_bgen_pipeline(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    phenotype_path: Path,
    phenotype_name: str,
    prediction_list_path: Path,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    chunk_size: int,
    variant_limit: int | None,
    output_run_paths: output.OutputRunPaths,
    staging_depth: int = 1,
    existing_manifest: dict[str, typing.Any] | None = None,
    resume: bool = False,
    resume_mode: types.ResumeMode = types.ResumeMode.FAST,
    finalize_parquet: bool = False,
    writer_thread_count: int = output.PACKAGED_WRITER_THREAD_COUNT,
    writer_queue_depth: int = output.PACKAGED_WRITER_QUEUE_DEPTH,
    chunks_per_arrow_file: int = output.PACKAGED_CHUNKS_PER_ARROW_FILE,
    arrow_compression: types.ArrowCompression = types.ArrowCompression.ZSTD,
    trusted_no_missing_diploid: bool = False,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode = types.TrustedBgenValidationMode.CACHE_ON_MISS,
    bgen_decode_tile_variant_count: int = output.PACKAGED_BGEN_DECODE_TILE_VARIANT_COUNT,
    bgen_simd: types.BgenSimdMode = output.PACKAGED_BGEN_SIMD,
    jax_device: types.Device = types.Device.CPU,
    jax_matmul_precision: types.JaxMatmulPrecision | None = None,
    jax_enable_x64: bool = output.PACKAGED_JAX_ENABLE_X64,
    score_dtype: types.FloatingPointDtype = output.PACKAGED_SCORE_DTYPE,
    firth_dtype: types.FloatingPointDtype = output.PACKAGED_FIRTH_DTYPE,
    output_format: types.OutputFormat = types.OutputFormat.PARQUET,
    correction_plan: types.BinaryCorrectionPlan = types.BinaryCorrectionPlan(),
    kernel_config: regenie2_binary_config.BinaryKernelConfig | None = None,
    gpu_genotype_format: types.GpuGenotypeFormat = types.GpuGenotypeFormat.DOSAGE,
    null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy = (
        types.NullLogisticNonconvergencePolicy.FAIL
    ),
    stage_timing_recorder: timing.StageTimingRecorder | None = None,
    telemetry_session: telemetry.TelemetrySession | None = None,
    alignment_config: native_dispatch.SampleAlignmentConfigProtocol | None = None,
) -> Path | None:
    """Run the native BGEN pipeline for binary REGENIE step 2."""
    logger.info("Starting binary REGENIE step 2 BGEN pipeline.")
    stage_timing_recorder = stage_timing_recorder or timing.build_stage_timing_recorder()
    resolved_kernel_config = kernel_config or regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG
    use_variant_major = True
    use_packed8 = gpu_genotype_format == types.GpuGenotypeFormat.PACKED8
    effective_trusted_no_missing_diploid = trusted_no_missing_diploid or use_packed8
    engine_start_time = time.perf_counter()
    logger.debug("Opening native BGEN engine for binary pipeline.")
    engine = native_dispatch.build_bgen_run_engine(
        genotype_source_config=genotype_source_config,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        trusted_no_missing_diploid=effective_trusted_no_missing_diploid,
        trusted_bgen_validation_mode=trusted_bgen_validation_mode,
    )
    timing.record_stage_duration(stage_timing_recorder, "bgen_engine_open_index_setup", engine_start_time)
    logger.debug(
        "Native BGEN engine opened for binary pipeline: sample_count=%s variant_count=%s.",
        engine.sample_count,
        engine.variant_count,
    )
    if telemetry_session is not None:
        telemetry_session.log_event(
            "bgen_engine_opened",
            association_mode=types.AssociationMode.REGENIE2_BINARY.value,
            phenotype=phenotype_name,
            sample_count=int(engine.sample_count),
            variant_count=int(engine.variant_count),
        )
    alignment_start_time = time.perf_counter()
    logger.debug("Loading aligned native sample, phenotype, and covariate inputs for binary pipeline.")
    run_input = native_dispatch.load_native_bgen_run_input(
        genotype_source_config=genotype_source_config,
        engine=engine,
        phenotype_path=phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=True,
        alignment_config=alignment_config,
    )
    timing.record_stage_duration(stage_timing_recorder, "sample_phenotype_covariate_alignment", alignment_start_time)
    logger.debug(
        "Aligned binary pipeline inputs: sample_count=%s covariate_count=%s.",
        int(run_input.sample_indices.shape[0]),
        len(run_input.native_aligned_sample_data.covariate_names),
    )
    if telemetry_session is not None:
        telemetry_session.log_event(
            "sample_alignment_completed",
            association_mode=types.AssociationMode.REGENIE2_BINARY.value,
            phenotype=phenotype_name,
            sample_count=int(run_input.sample_indices.shape[0]),
            covariate_count=len(run_input.native_aligned_sample_data.covariate_names),
        )
    current_header = output.build_current_run_manifest_header(
        association_mode=types.AssociationMode.REGENIE2_BINARY,
        bgen_path=genotype_source_config.source_path,
        sample_path=source.resolve_bgen_sample_path(
            genotype_source_config.source_path,
            genotype_source_config.sample_path,
        ),
        phenotype_path=phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=covariate_path,
        covariate_names=tuple(run_input.native_aligned_sample_data.covariate_names),
        prediction_list_path=prediction_list_path,
        sample_count=int(run_input.sample_indices.shape[0]),
        variant_count=int(engine.variant_count),
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        binary_correction_plan=correction_plan,
        trusted_no_missing_diploid=effective_trusted_no_missing_diploid,
        sample_key_mode=native_dispatch.resolve_sample_key_mode(alignment_config),
        binary_kernel_config=resolved_kernel_config,
        bgen_decode_tile_variant_count=bgen_decode_tile_variant_count,
        bgen_simd=bgen_simd,
        trusted_bgen_validation_mode=trusted_bgen_validation_mode,
        jax_device=jax_device,
        jax_matmul_precision=jax_matmul_precision,
        jax_enable_x64=jax_enable_x64,
        gpu_genotype_format=gpu_genotype_format,
        score_dtype=score_dtype,
        firth_dtype=firth_dtype,
        output_format=output_format,
        finalize_parquet=finalize_parquet,
        writer_thread_count=writer_thread_count,
        writer_queue_depth=writer_queue_depth,
        chunks_per_arrow_file=chunks_per_arrow_file,
        arrow_compression=arrow_compression,
    )
    initialized_output_run = output.initialize_output_run(
        output_run_paths=output_run_paths,
        existing_manifest=existing_manifest,
        current_header=current_header,
        resume=resume,
        resume_mode=resume_mode,
    )
    writer_start_time = time.perf_counter()
    logger.debug("Creating output writer for binary pipeline.")
    writer_session = output.create_output_writer_session(
        output_run_paths,
        types.AssociationMode.REGENIE2_BINARY,
        writer_thread_count=writer_thread_count,
        writer_queue_depth=writer_queue_depth,
        finalize_parquet=finalize_parquet,
        chunks_per_arrow_file=chunks_per_arrow_file,
        arrow_compression=arrow_compression,
        collect_stage_timings=stage_timing_recorder is not None,
    )
    timing.record_stage_duration(stage_timing_recorder, "output_writer_preparation", writer_start_time)
    prediction_start_time = time.perf_counter()
    logger.debug("Loading REGENIE prediction source for binary pipeline.")
    prediction_source = native_dispatch.build_regenie_prediction_source(
        prediction_list_path=prediction_list_path,
        phenotype_name=phenotype_name,
        run_input=run_input,
        alignment_config=alignment_config,
    )
    timing.record_stage_duration(stage_timing_recorder, "prediction_source_load", prediction_start_time)
    if telemetry_session is not None:
        telemetry_session.log_event(
            "prediction_source_loaded",
            association_mode=types.AssociationMode.REGENIE2_BINARY.value,
            phenotype=phenotype_name,
        )
    preflight_start_time = time.perf_counter()
    logger.debug("Running preflight validation for binary pipeline.")
    preflight_report = preflight.run_regenie2_preflight(
        run_input=run_input,
        prediction_source=prediction_source,
        engine=engine,
        variant_limit=variant_limit,
        is_binary_trait=True,
        trusted_no_missing_diploid=effective_trusted_no_missing_diploid,
    )
    timing.record_stage_duration(stage_timing_recorder, "preflight_validation", preflight_start_time)
    logger.debug(
        "Preflight validation passed for binary pipeline: sample_count=%s covariate_count=%s chromosome_count=%s.",
        preflight_report.sample_count,
        preflight_report.covariate_count,
        preflight_report.chromosome_count,
    )
    if telemetry_session is not None:
        telemetry_session.log_event(
            "preflight_completed",
            association_mode=types.AssociationMode.REGENIE2_BINARY.value,
            phenotype=phenotype_name,
            sample_count=preflight_report.sample_count,
            covariate_count=preflight_report.covariate_count,
            chromosome_count=preflight_report.chromosome_count,
        )
    callback = callbacks.BinaryRegenie2PipelineCallback(
        run_input=run_input,
        prediction_source=prediction_source,
        writer_session=writer_session,
        correction_plan=correction_plan,
        kernel_config=resolved_kernel_config,
        null_logistic_nonconvergence_policy=null_logistic_nonconvergence_policy,
        staging_depth=staging_depth,
        score_dtype=score_dtype,
        stage_timing_recorder=stage_timing_recorder,
        telemetry_session=telemetry_session,
    )
    return native_dispatch.run_bgen_engine_with_callback(
        engine=engine,
        run_input=run_input,
        committed_chunk_identifiers=set(initialized_output_run.committed_chunk_identifiers),
        writer_session=writer_session,
        callback=callback,
        stage_timing_recorder=stage_timing_recorder,
        variant_major_dosage=use_variant_major,
        variant_major_packed8_probability_pairs=use_packed8,
    )


@dataclass(frozen=True)
class MultiTraitPreflightRunInput:
    """Single-trait view over a multi-trait run input for preflight validation."""

    phenotype_vector: typing.Any
    covariate_matrix: typing.Any


@dataclass(frozen=True)
class SingleTraitPredictionView:
    """Single-trait view over a multi-trait prediction source."""

    prediction_source: typing.Any
    trait_index: int

    def get_chromosome_predictions(self, chromosome: str) -> typing.Any:
        """Return one trait's aligned LOCO predictions for preflight validation."""
        return self.prediction_source.get_chromosome_predictions(chromosome)[self.trait_index]


@dataclass(frozen=True)
class GroupedAlignedSampleData:
    """Python-owned multi-trait alignment assembled from per-phenotype alignments.

    Attributes:
        phenotype_names: Phenotype names in trait-major matrix order.
        sample_indices: BGEN sample indices shared by all grouped phenotypes.
        family_identifiers: Family identifiers for the shared sample order.
        individual_identifiers: Individual identifiers for the shared sample order.
        phenotype_matrix: Trait-major phenotype matrix.
        covariate_names: Covariate names shared by all grouped phenotypes.
        covariate_matrix: Shared design matrix.
        is_binary_trait: Whether phenotypes are binary traits.

    """

    phenotype_names: tuple[str, ...]
    sample_indices: npt.NDArray[np.int64]
    family_identifiers: tuple[str, ...]
    individual_identifiers: tuple[str, ...]
    phenotype_matrix: npt.NDArray[np.float32]
    covariate_names: tuple[str, ...]
    covariate_matrix: npt.NDArray[np.float32]
    is_binary_trait: bool


@dataclass(frozen=True)
class GroupedSingleTraitPredictionSource:
    """Trait-major prediction source assembled from single-trait sources."""

    prediction_sources: tuple[typing.Any, ...]

    def get_chromosome_predictions(self, chromosome: str) -> npt.NDArray[np.float32]:
        """Return trait-major aligned LOCO predictions for one chromosome."""
        return np.stack(
            tuple(
                np.asarray(prediction_source.get_chromosome_predictions(chromosome), dtype=np.float32)
                for prediction_source in self.prediction_sources
            ),
            axis=0,
        )


def run_regenie2_multi_phenotype_linear_bgen_pipeline(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    phenotype_path: Path,
    phenotype_names: tuple[str, ...],
    prediction_list_path: Path,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    chunk_size: int,
    variant_limit: int | None,
    output_run_paths_by_phenotype: tuple[output.OutputRunPaths, ...],
    staging_depth: int = 1,
    existing_manifests_by_phenotype: tuple[dict[str, typing.Any] | None, ...] | None = None,
    resume: bool = False,
    resume_mode: types.ResumeMode = types.ResumeMode.FAST,
    finalize_parquet: bool = False,
    writer_thread_count: int = output.PACKAGED_WRITER_THREAD_COUNT,
    writer_queue_depth: int = output.PACKAGED_WRITER_QUEUE_DEPTH,
    chunks_per_arrow_file: int = output.PACKAGED_CHUNKS_PER_ARROW_FILE,
    arrow_compression: types.ArrowCompression = types.ArrowCompression.ZSTD,
    trusted_no_missing_diploid: bool = False,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode = types.TrustedBgenValidationMode.CACHE_ON_MISS,
    bgen_decode_tile_variant_count: int = output.PACKAGED_BGEN_DECODE_TILE_VARIANT_COUNT,
    bgen_simd: types.BgenSimdMode = output.PACKAGED_BGEN_SIMD,
    jax_device: types.Device = types.Device.CPU,
    jax_matmul_precision: types.JaxMatmulPrecision | None = None,
    jax_enable_x64: bool = output.PACKAGED_JAX_ENABLE_X64,
    score_dtype: types.FloatingPointDtype = output.PACKAGED_SCORE_DTYPE,
    firth_dtype: types.FloatingPointDtype = output.PACKAGED_FIRTH_DTYPE,
    output_format: types.OutputFormat = types.OutputFormat.PARQUET,
    stage_timing_recorder: timing.StageTimingRecorder | None = None,
    telemetry_session: telemetry.TelemetrySession | None = None,
    alignment_config: native_dispatch.SampleAlignmentConfigProtocol | None = None,
    sample_mode: types.MultiPhenotypeSampleMode | None = None,
) -> tuple[Path | None, ...]:
    """Run the complete-case native BGEN pipeline once for multiple quantitative phenotypes."""
    return run_regenie2_multi_phenotype_bgen_pipeline(
        genotype_source_config=genotype_source_config,
        phenotype_path=phenotype_path,
        phenotype_names=phenotype_names,
        prediction_list_path=prediction_list_path,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        output_run_paths_by_phenotype=output_run_paths_by_phenotype,
        staging_depth=staging_depth,
        existing_manifests_by_phenotype=existing_manifests_by_phenotype,
        resume=resume,
        resume_mode=resume_mode,
        finalize_parquet=finalize_parquet,
        writer_thread_count=writer_thread_count,
        writer_queue_depth=writer_queue_depth,
        chunks_per_arrow_file=chunks_per_arrow_file,
        arrow_compression=arrow_compression,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
        trusted_bgen_validation_mode=trusted_bgen_validation_mode,
        bgen_decode_tile_variant_count=bgen_decode_tile_variant_count,
        bgen_simd=bgen_simd,
        jax_device=jax_device,
        jax_matmul_precision=jax_matmul_precision,
        jax_enable_x64=jax_enable_x64,
        score_dtype=score_dtype,
        firth_dtype=firth_dtype,
        output_format=output_format,
        correction_plan=types.BinaryCorrectionPlan(),
        kernel_config=None,
        null_logistic_nonconvergence_policy=types.NullLogisticNonconvergencePolicy.FAIL,
        stage_timing_recorder=stage_timing_recorder,
        telemetry_session=telemetry_session,
        alignment_config=alignment_config,
        sample_mode=sample_mode,
        association_mode=types.AssociationMode.REGENIE2_LINEAR,
    )


def run_regenie2_multi_phenotype_binary_bgen_pipeline(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    phenotype_path: Path,
    phenotype_names: tuple[str, ...],
    prediction_list_path: Path,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    chunk_size: int,
    variant_limit: int | None,
    output_run_paths_by_phenotype: tuple[output.OutputRunPaths, ...],
    staging_depth: int = 1,
    existing_manifests_by_phenotype: tuple[dict[str, typing.Any] | None, ...] | None = None,
    resume: bool = False,
    resume_mode: types.ResumeMode = types.ResumeMode.FAST,
    finalize_parquet: bool = False,
    writer_thread_count: int = output.PACKAGED_WRITER_THREAD_COUNT,
    writer_queue_depth: int = output.PACKAGED_WRITER_QUEUE_DEPTH,
    chunks_per_arrow_file: int = output.PACKAGED_CHUNKS_PER_ARROW_FILE,
    arrow_compression: types.ArrowCompression = types.ArrowCompression.ZSTD,
    trusted_no_missing_diploid: bool = False,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode = types.TrustedBgenValidationMode.CACHE_ON_MISS,
    bgen_decode_tile_variant_count: int = output.PACKAGED_BGEN_DECODE_TILE_VARIANT_COUNT,
    bgen_simd: types.BgenSimdMode = output.PACKAGED_BGEN_SIMD,
    jax_device: types.Device = types.Device.CPU,
    jax_matmul_precision: types.JaxMatmulPrecision | None = None,
    jax_enable_x64: bool = output.PACKAGED_JAX_ENABLE_X64,
    score_dtype: types.FloatingPointDtype = output.PACKAGED_SCORE_DTYPE,
    firth_dtype: types.FloatingPointDtype = output.PACKAGED_FIRTH_DTYPE,
    output_format: types.OutputFormat = types.OutputFormat.PARQUET,
    correction_plan: types.BinaryCorrectionPlan = types.BinaryCorrectionPlan(),
    kernel_config: regenie2_binary_config.BinaryKernelConfig | None = None,
    null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy = (
        types.NullLogisticNonconvergencePolicy.FAIL
    ),
    stage_timing_recorder: timing.StageTimingRecorder | None = None,
    telemetry_session: telemetry.TelemetrySession | None = None,
    alignment_config: native_dispatch.SampleAlignmentConfigProtocol | None = None,
    sample_mode: types.MultiPhenotypeSampleMode | None = None,
) -> tuple[Path | None, ...]:
    """Run the complete-case native BGEN pipeline once for multiple binary phenotypes."""
    resolved_kernel_config = kernel_config or regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG
    return run_regenie2_multi_phenotype_bgen_pipeline(
        genotype_source_config=genotype_source_config,
        phenotype_path=phenotype_path,
        phenotype_names=phenotype_names,
        prediction_list_path=prediction_list_path,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        output_run_paths_by_phenotype=output_run_paths_by_phenotype,
        staging_depth=staging_depth,
        existing_manifests_by_phenotype=existing_manifests_by_phenotype,
        resume=resume,
        resume_mode=resume_mode,
        finalize_parquet=finalize_parquet,
        writer_thread_count=writer_thread_count,
        writer_queue_depth=writer_queue_depth,
        chunks_per_arrow_file=chunks_per_arrow_file,
        arrow_compression=arrow_compression,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
        trusted_bgen_validation_mode=trusted_bgen_validation_mode,
        bgen_decode_tile_variant_count=bgen_decode_tile_variant_count,
        bgen_simd=bgen_simd,
        jax_device=jax_device,
        jax_matmul_precision=jax_matmul_precision,
        jax_enable_x64=jax_enable_x64,
        score_dtype=score_dtype,
        firth_dtype=firth_dtype,
        output_format=output_format,
        correction_plan=correction_plan,
        kernel_config=resolved_kernel_config,
        null_logistic_nonconvergence_policy=null_logistic_nonconvergence_policy,
        stage_timing_recorder=stage_timing_recorder,
        telemetry_session=telemetry_session,
        alignment_config=alignment_config,
        sample_mode=sample_mode,
        association_mode=types.AssociationMode.REGENIE2_BINARY,
    )


def run_regenie2_multi_phenotype_bgen_pipeline(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    phenotype_path: Path,
    phenotype_names: tuple[str, ...],
    prediction_list_path: Path,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    chunk_size: int,
    variant_limit: int | None,
    output_run_paths_by_phenotype: tuple[output.OutputRunPaths, ...],
    staging_depth: int,
    existing_manifests_by_phenotype: tuple[dict[str, typing.Any] | None, ...] | None,
    resume: bool,
    resume_mode: types.ResumeMode,
    finalize_parquet: bool,
    writer_thread_count: int,
    writer_queue_depth: int,
    chunks_per_arrow_file: int,
    arrow_compression: types.ArrowCompression,
    trusted_no_missing_diploid: bool,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode,
    bgen_decode_tile_variant_count: int,
    bgen_simd: types.BgenSimdMode,
    jax_device: types.Device,
    jax_matmul_precision: types.JaxMatmulPrecision | None,
    jax_enable_x64: bool,
    score_dtype: types.FloatingPointDtype,
    firth_dtype: types.FloatingPointDtype,
    output_format: types.OutputFormat,
    correction_plan: types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig | None,
    null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    telemetry_session: telemetry.TelemetrySession | None,
    alignment_config: native_dispatch.SampleAlignmentConfigProtocol | None,
    sample_mode: types.MultiPhenotypeSampleMode | None,
    association_mode: types.AssociationMode,
) -> tuple[Path | None, ...]:
    """Shared implementation for multi-phenotype BGEN pipelines."""
    if sample_mode == types.MultiPhenotypeSampleMode.PER_PHENOTYPE:
        return run_regenie2_grouped_per_phenotype_bgen_pipeline(
            genotype_source_config=genotype_source_config,
            phenotype_path=phenotype_path,
            phenotype_names=phenotype_names,
            prediction_list_path=prediction_list_path,
            covariate_path=covariate_path,
            covariate_names=covariate_names,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
            output_run_paths_by_phenotype=output_run_paths_by_phenotype,
            staging_depth=staging_depth,
            existing_manifests_by_phenotype=existing_manifests_by_phenotype,
            resume=resume,
            resume_mode=resume_mode,
            finalize_parquet=finalize_parquet,
            writer_thread_count=writer_thread_count,
            writer_queue_depth=writer_queue_depth,
            chunks_per_arrow_file=chunks_per_arrow_file,
            arrow_compression=arrow_compression,
            trusted_no_missing_diploid=trusted_no_missing_diploid,
            trusted_bgen_validation_mode=trusted_bgen_validation_mode,
            bgen_decode_tile_variant_count=bgen_decode_tile_variant_count,
            bgen_simd=bgen_simd,
            jax_device=jax_device,
            jax_matmul_precision=jax_matmul_precision,
            jax_enable_x64=jax_enable_x64,
            score_dtype=score_dtype,
            firth_dtype=firth_dtype,
            output_format=output_format,
            correction_plan=correction_plan,
            kernel_config=kernel_config,
            null_logistic_nonconvergence_policy=null_logistic_nonconvergence_policy,
            stage_timing_recorder=stage_timing_recorder,
            telemetry_session=telemetry_session,
            alignment_config=alignment_config,
            association_mode=association_mode,
        )
    if sample_mode != types.MultiPhenotypeSampleMode.COMPLETE_CASE:
        message = "Multi-phenotype sample mode must be per-phenotype or complete-case."
        raise ValueError(message)
    logger.info("Starting multi-phenotype REGENIE step 2 BGEN pipeline.")
    stage_timing_recorder = stage_timing_recorder or timing.build_stage_timing_recorder()
    resolved_kernel_config = kernel_config or regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG
    existing_manifests = existing_manifests_by_phenotype or tuple(None for _ in phenotype_names)
    engine_start_time = time.perf_counter()
    logger.debug("Opening native BGEN engine for multi-phenotype pipeline.")
    engine = native_dispatch.build_bgen_run_engine(
        genotype_source_config=genotype_source_config,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
        trusted_bgen_validation_mode=trusted_bgen_validation_mode,
    )
    timing.record_stage_duration(stage_timing_recorder, "bgen_engine_open_index_setup", engine_start_time)
    logger.debug(
        "Native BGEN engine opened for multi-phenotype pipeline: sample_count=%s variant_count=%s.",
        engine.sample_count,
        engine.variant_count,
    )
    if telemetry_session is not None:
        telemetry_session.log_event(
            "bgen_engine_opened",
            association_mode=association_mode.value,
            phenotype_count=len(phenotype_names),
            sample_count=int(engine.sample_count),
            variant_count=int(engine.variant_count),
        )
    alignment_start_time = time.perf_counter()
    logger.debug("Loading aligned native sample, phenotype, and covariate inputs for multi-phenotype pipeline.")
    run_input = native_dispatch.load_native_bgen_multi_run_input(
        genotype_source_config=genotype_source_config,
        engine=engine,
        phenotype_path=phenotype_path,
        phenotype_names=phenotype_names,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=association_mode == types.AssociationMode.REGENIE2_BINARY,
        alignment_config=alignment_config,
    )
    timing.record_stage_duration(stage_timing_recorder, "sample_phenotype_covariate_alignment", alignment_start_time)
    logger.debug(
        "Aligned multi-phenotype pipeline inputs: sample_count=%s phenotype_count=%s covariate_count=%s.",
        int(run_input.sample_indices.shape[0]),
        len(run_input.phenotype_names),
        len(run_input.native_multi_aligned_sample_data.covariate_names),
    )
    if telemetry_session is not None:
        telemetry_session.log_event(
            "sample_alignment_completed",
            association_mode=association_mode.value,
            phenotype_count=len(run_input.phenotype_names),
            sample_count=int(run_input.sample_indices.shape[0]),
            covariate_count=len(run_input.native_multi_aligned_sample_data.covariate_names),
        )
    prediction_start_time = time.perf_counter()
    logger.debug("Loading REGENIE prediction source for multi-phenotype pipeline.")
    prediction_source = native_dispatch.build_multi_regenie_prediction_source(
        prediction_list_path=prediction_list_path,
        run_input=run_input,
        alignment_config=alignment_config,
    )
    timing.record_stage_duration(stage_timing_recorder, "prediction_source_load", prediction_start_time)
    return run_prepared_multi_phenotype_bgen_group(
        engine=engine,
        run_input=run_input,
        prediction_source=prediction_source,
        genotype_source_config=genotype_source_config,
        phenotype_path=phenotype_path,
        phenotype_names=phenotype_names,
        prediction_list_path=prediction_list_path,
        covariate_path=covariate_path,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        output_run_paths_by_phenotype=output_run_paths_by_phenotype,
        staging_depth=staging_depth,
        existing_manifests=existing_manifests,
        resume=resume,
        resume_mode=resume_mode,
        finalize_parquet=finalize_parquet,
        writer_thread_count=writer_thread_count,
        writer_queue_depth=writer_queue_depth,
        chunks_per_arrow_file=chunks_per_arrow_file,
        arrow_compression=arrow_compression,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
        trusted_bgen_validation_mode=trusted_bgen_validation_mode,
        bgen_decode_tile_variant_count=bgen_decode_tile_variant_count,
        bgen_simd=bgen_simd,
        jax_device=jax_device,
        jax_matmul_precision=jax_matmul_precision,
        jax_enable_x64=jax_enable_x64,
        score_dtype=score_dtype,
        firth_dtype=firth_dtype,
        output_format=output_format,
        correction_plan=correction_plan,
        resolved_kernel_config=resolved_kernel_config,
        null_logistic_nonconvergence_policy=null_logistic_nonconvergence_policy,
        stage_timing_recorder=stage_timing_recorder,
        telemetry_session=telemetry_session,
        alignment_config=alignment_config,
        association_mode=association_mode,
        output_sample_mode=output.MultiPhenotypeSampleMode.COMPLETE_CASE_INTERSECTION,
    )


def run_regenie2_grouped_per_phenotype_bgen_pipeline(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    phenotype_path: Path,
    phenotype_names: tuple[str, ...],
    prediction_list_path: Path,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    chunk_size: int,
    variant_limit: int | None,
    output_run_paths_by_phenotype: tuple[output.OutputRunPaths, ...],
    staging_depth: int,
    existing_manifests_by_phenotype: tuple[dict[str, typing.Any] | None, ...] | None,
    resume: bool,
    resume_mode: types.ResumeMode,
    finalize_parquet: bool,
    writer_thread_count: int,
    writer_queue_depth: int,
    chunks_per_arrow_file: int,
    arrow_compression: types.ArrowCompression,
    trusted_no_missing_diploid: bool,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode,
    bgen_decode_tile_variant_count: int,
    bgen_simd: types.BgenSimdMode,
    jax_device: types.Device,
    jax_matmul_precision: types.JaxMatmulPrecision | None,
    jax_enable_x64: bool,
    score_dtype: types.FloatingPointDtype,
    firth_dtype: types.FloatingPointDtype,
    output_format: types.OutputFormat,
    correction_plan: types.BinaryCorrectionPlan,
    kernel_config: regenie2_binary_config.BinaryKernelConfig | None,
    null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    telemetry_session: telemetry.TelemetrySession | None,
    alignment_config: native_dispatch.SampleAlignmentConfigProtocol | None,
    association_mode: types.AssociationMode,
) -> tuple[Path | None, ...]:
    """Group independently aligned phenotypes and run one BGEN pass per compatible group."""
    logger.info("Starting grouped per-phenotype REGENIE step 2 BGEN pipeline.")
    stage_timing_recorder = stage_timing_recorder or timing.build_stage_timing_recorder()
    resolved_kernel_config = kernel_config or regenie2_binary_config.DEFAULT_BINARY_KERNEL_CONFIG
    existing_manifests = existing_manifests_by_phenotype or tuple(None for _ in phenotype_names)
    engine_start_time = time.perf_counter()
    logger.debug("Opening native BGEN engine for grouped per-phenotype pipeline.")
    engine = native_dispatch.build_bgen_run_engine(
        genotype_source_config=genotype_source_config,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
        trusted_bgen_validation_mode=trusted_bgen_validation_mode,
    )
    timing.record_stage_duration(stage_timing_recorder, "bgen_engine_open_index_setup", engine_start_time)
    if telemetry_session is not None:
        telemetry_session.log_event(
            "bgen_engine_opened",
            association_mode=association_mode.value,
            phenotype_count=len(phenotype_names),
            sample_count=int(engine.sample_count),
            variant_count=int(engine.variant_count),
        )
    alignment_start_time = time.perf_counter()
    indexed_run_inputs = tuple(
        (
            phenotype_index,
            native_dispatch.load_native_bgen_run_input(
                genotype_source_config=genotype_source_config,
                engine=engine,
                phenotype_path=phenotype_path,
                phenotype_name=phenotype_name,
                covariate_path=covariate_path,
                covariate_names=covariate_names,
                is_binary_trait=association_mode == types.AssociationMode.REGENIE2_BINARY,
                alignment_config=alignment_config,
            ),
        )
        for phenotype_index, phenotype_name in enumerate(phenotype_names)
    )
    timing.record_stage_duration(stage_timing_recorder, "sample_phenotype_covariate_alignment", alignment_start_time)
    grouped_run_inputs = group_per_phenotype_run_inputs(indexed_run_inputs)
    logger.info(
        "Prepared %s compatible per-phenotype group(s) for %s phenotype(s).",
        len(grouped_run_inputs),
        len(phenotype_names),
    )
    if telemetry_session is not None:
        telemetry_session.log_event(
            "sample_alignment_completed",
            association_mode=association_mode.value,
            phenotype_count=len(phenotype_names),
            phenotype_group_count=len(grouped_run_inputs),
        )

    final_parquet_paths_by_index: list[Path | None] = [None] * len(phenotype_names)
    for grouped_run_input in grouped_run_inputs:
        group_indices = tuple(phenotype_index for phenotype_index, _run_input in grouped_run_input)
        group_single_run_inputs = tuple(run_input for _phenotype_index, run_input in grouped_run_input)
        prediction_start_time = time.perf_counter()
        group_prediction_source = build_grouped_single_trait_prediction_source(
            prediction_list_path=prediction_list_path,
            phenotype_names=tuple(phenotype_names[phenotype_index] for phenotype_index in group_indices),
            run_inputs=group_single_run_inputs,
            alignment_config=alignment_config,
        )
        timing.record_stage_duration(stage_timing_recorder, "prediction_source_load", prediction_start_time)
        group_multi_run_input = build_grouped_native_bgen_multi_run_input(
            phenotype_names=tuple(phenotype_names[phenotype_index] for phenotype_index in group_indices),
            run_inputs=group_single_run_inputs,
        )
        group_final_parquet_paths = run_prepared_multi_phenotype_bgen_group(
            engine=engine,
            run_input=group_multi_run_input,
            prediction_source=group_prediction_source,
            genotype_source_config=genotype_source_config,
            phenotype_path=phenotype_path,
            phenotype_names=group_multi_run_input.phenotype_names,
            prediction_list_path=prediction_list_path,
            covariate_path=covariate_path,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
            output_run_paths_by_phenotype=tuple(
                output_run_paths_by_phenotype[phenotype_index] for phenotype_index in group_indices
            ),
            staging_depth=staging_depth,
            existing_manifests=tuple(existing_manifests[phenotype_index] for phenotype_index in group_indices),
            resume=resume,
            resume_mode=resume_mode,
            finalize_parquet=finalize_parquet,
            writer_thread_count=writer_thread_count,
            writer_queue_depth=writer_queue_depth,
            chunks_per_arrow_file=chunks_per_arrow_file,
            arrow_compression=arrow_compression,
            trusted_no_missing_diploid=trusted_no_missing_diploid,
            trusted_bgen_validation_mode=trusted_bgen_validation_mode,
            bgen_decode_tile_variant_count=bgen_decode_tile_variant_count,
            bgen_simd=bgen_simd,
            jax_device=jax_device,
            jax_matmul_precision=jax_matmul_precision,
            jax_enable_x64=jax_enable_x64,
            score_dtype=score_dtype,
            firth_dtype=firth_dtype,
            output_format=output_format,
            correction_plan=correction_plan,
            resolved_kernel_config=resolved_kernel_config,
            null_logistic_nonconvergence_policy=null_logistic_nonconvergence_policy,
            stage_timing_recorder=stage_timing_recorder,
            telemetry_session=telemetry_session,
            alignment_config=alignment_config,
            association_mode=association_mode,
            output_sample_mode=output.MultiPhenotypeSampleMode.SINGLE_PHENOTYPE,
        )
        for phenotype_index, final_parquet_path in zip(group_indices, group_final_parquet_paths, strict=True):
            final_parquet_paths_by_index[phenotype_index] = final_parquet_path
    return tuple(final_parquet_paths_by_index)


def group_per_phenotype_run_inputs(
    indexed_run_inputs: tuple[tuple[int, native_dispatch.NativeBgenRunInput], ...],
) -> tuple[tuple[tuple[int, native_dispatch.NativeBgenRunInput], ...], ...]:
    """Group independently aligned phenotypes by identical sample and covariate alignment."""
    groups_by_fingerprint: dict[str, list[tuple[int, native_dispatch.NativeBgenRunInput]]] = {}
    group_order: list[str] = []
    for indexed_run_input in indexed_run_inputs:
        _phenotype_index, run_input = indexed_run_input
        fingerprint = compute_sample_covariate_alignment_fingerprint(run_input)
        if fingerprint not in groups_by_fingerprint:
            groups_by_fingerprint[fingerprint] = []
            group_order.append(fingerprint)
        groups_by_fingerprint[fingerprint].append(indexed_run_input)
    return tuple(tuple(groups_by_fingerprint[fingerprint]) for fingerprint in group_order)


def compute_sample_covariate_alignment_fingerprint(run_input: native_dispatch.NativeBgenRunInput) -> str:
    """Fingerprint sample indices and covariate values for per-phenotype grouping."""
    sample_indices = np.ascontiguousarray(run_input.sample_indices, dtype=np.int64)
    covariate_matrix = np.ascontiguousarray(np.asarray(run_input.covariate_matrix, dtype=np.float32))
    digest = hashlib.sha256()
    digest.update(str(sample_indices.shape).encode())
    digest.update(str(sample_indices.dtype).encode())
    digest.update(sample_indices.tobytes())
    digest.update(str(covariate_matrix.shape).encode())
    digest.update(str(covariate_matrix.dtype).encode())
    digest.update(covariate_matrix.tobytes())
    return digest.hexdigest()


def build_grouped_native_bgen_multi_run_input(
    *,
    phenotype_names: tuple[str, ...],
    run_inputs: tuple[native_dispatch.NativeBgenRunInput, ...],
) -> native_dispatch.NativeBgenMultiRunInput:
    """Build a multi-trait run input from independently aligned compatible traits."""
    first_run_input = run_inputs[0]
    grouped_aligned_sample_data = GroupedAlignedSampleData(
        phenotype_names=phenotype_names,
        sample_indices=first_run_input.sample_indices,
        family_identifiers=tuple(first_run_input.native_aligned_sample_data.family_identifiers),
        individual_identifiers=tuple(first_run_input.native_aligned_sample_data.individual_identifiers),
        phenotype_matrix=np.stack(
            tuple(np.asarray(run_input.phenotype_vector, dtype=np.float32) for run_input in run_inputs),
            axis=0,
        ),
        covariate_names=tuple(first_run_input.native_aligned_sample_data.covariate_names),
        covariate_matrix=np.asarray(first_run_input.covariate_matrix, dtype=np.float32),
        is_binary_trait=first_run_input.is_binary_trait,
    )
    return native_dispatch.NativeBgenMultiRunInput(
        native_multi_aligned_sample_data=typing.cast("typing.Any", grouped_aligned_sample_data),
        phenotype_names=phenotype_names,
        sample_indices=np.ascontiguousarray(grouped_aligned_sample_data.sample_indices, dtype=np.int64),
        family_identifiers=grouped_aligned_sample_data.family_identifiers,
        individual_identifiers=grouped_aligned_sample_data.individual_identifiers,
        phenotype_matrix=jnp.asarray(grouped_aligned_sample_data.phenotype_matrix, dtype=jnp.float32),
        covariate_matrix=jnp.asarray(grouped_aligned_sample_data.covariate_matrix, dtype=jnp.float32),
        is_binary_trait=grouped_aligned_sample_data.is_binary_trait,
    )


def build_grouped_single_trait_prediction_source(
    *,
    prediction_list_path: Path,
    phenotype_names: tuple[str, ...],
    run_inputs: tuple[native_dispatch.NativeBgenRunInput, ...],
    alignment_config: native_dispatch.SampleAlignmentConfigProtocol | None,
) -> GroupedSingleTraitPredictionSource:
    """Build one trait-major prediction source from independently aligned traits."""
    prediction_sources = tuple(
        native_dispatch.build_regenie_prediction_source(
            prediction_list_path=prediction_list_path,
            phenotype_name=phenotype_name,
            run_input=run_input,
            alignment_config=alignment_config,
        )
        for phenotype_name, run_input in zip(phenotype_names, run_inputs, strict=True)
    )
    return GroupedSingleTraitPredictionSource(prediction_sources=prediction_sources)


def run_prepared_multi_phenotype_bgen_group(
    *,
    engine: _core.Regenie2RunEngine,
    run_input: native_dispatch.NativeBgenMultiRunInput,
    prediction_source: typing.Any,
    genotype_source_config: source.GenotypeSourceConfig,
    phenotype_path: Path,
    phenotype_names: tuple[str, ...],
    prediction_list_path: Path,
    covariate_path: Path | None,
    chunk_size: int,
    variant_limit: int | None,
    output_run_paths_by_phenotype: tuple[output.OutputRunPaths, ...],
    staging_depth: int,
    existing_manifests: tuple[dict[str, typing.Any] | None, ...],
    resume: bool,
    resume_mode: types.ResumeMode,
    finalize_parquet: bool,
    writer_thread_count: int,
    writer_queue_depth: int,
    chunks_per_arrow_file: int,
    arrow_compression: types.ArrowCompression,
    trusted_no_missing_diploid: bool,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode,
    bgen_decode_tile_variant_count: int,
    bgen_simd: types.BgenSimdMode,
    jax_device: types.Device,
    jax_matmul_precision: types.JaxMatmulPrecision | None,
    jax_enable_x64: bool,
    score_dtype: types.FloatingPointDtype,
    firth_dtype: types.FloatingPointDtype,
    output_format: types.OutputFormat,
    correction_plan: types.BinaryCorrectionPlan,
    resolved_kernel_config: regenie2_binary_config.BinaryKernelConfig,
    null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    telemetry_session: telemetry.TelemetrySession | None,
    alignment_config: native_dispatch.SampleAlignmentConfigProtocol | None,
    association_mode: types.AssociationMode,
    output_sample_mode: output.MultiPhenotypeSampleMode,
) -> tuple[Path | None, ...]:
    """Run one prepared compatible phenotype group through one BGEN pass."""
    current_headers = tuple(
        output.build_current_run_manifest_header(
            association_mode=association_mode,
            bgen_path=genotype_source_config.source_path,
            sample_path=source.resolve_bgen_sample_path(
                genotype_source_config.source_path,
                genotype_source_config.sample_path,
            ),
            phenotype_path=phenotype_path,
            phenotype_name=phenotype_name,
            covariate_path=covariate_path,
            covariate_names=tuple(run_input.native_multi_aligned_sample_data.covariate_names),
            prediction_list_path=prediction_list_path,
            sample_count=int(run_input.sample_indices.shape[0]),
            variant_count=int(engine.variant_count),
            chunk_size=chunk_size,
            variant_limit=variant_limit,
            binary_correction_plan=correction_plan,
            trusted_no_missing_diploid=trusted_no_missing_diploid,
            sample_key_mode=native_dispatch.resolve_sample_key_mode(alignment_config),
            binary_kernel_config=(
                resolved_kernel_config if association_mode == types.AssociationMode.REGENIE2_BINARY else None
            ),
            bgen_decode_tile_variant_count=bgen_decode_tile_variant_count,
            bgen_simd=bgen_simd,
            trusted_bgen_validation_mode=trusted_bgen_validation_mode,
            jax_device=jax_device,
            jax_matmul_precision=jax_matmul_precision,
            jax_enable_x64=jax_enable_x64,
            score_dtype=score_dtype,
            firth_dtype=firth_dtype,
            multi_phenotype_sample_mode=output_sample_mode,
            output_format=output_format,
            finalize_parquet=finalize_parquet,
            writer_thread_count=writer_thread_count,
            writer_queue_depth=writer_queue_depth,
            chunks_per_arrow_file=chunks_per_arrow_file,
            arrow_compression=arrow_compression,
        )
        for phenotype_name in phenotype_names
    )
    initialized_output_runs = tuple(
        output.initialize_output_run(
            output_run_paths=output_run_paths,
            existing_manifest=existing_manifest,
            current_header=current_header,
            resume=resume,
            resume_mode=resume_mode,
        )
        for output_run_paths, existing_manifest, current_header in zip(
            output_run_paths_by_phenotype,
            existing_manifests,
            current_headers,
            strict=True,
        )
    )
    committed_chunk_identifier_sets = tuple(
        set(initialized_output_run.committed_chunk_identifiers) for initialized_output_run in initialized_output_runs
    )
    writer_start_time = time.perf_counter()
    logger.debug("Creating output writers for multi-phenotype pipeline.")
    writer_sessions = tuple(
        output.create_output_writer_session(
            output_run_paths,
            association_mode,
            writer_thread_count=writer_thread_count,
            writer_queue_depth=writer_queue_depth,
            finalize_parquet=finalize_parquet,
            chunks_per_arrow_file=chunks_per_arrow_file,
            arrow_compression=arrow_compression,
            collect_stage_timings=stage_timing_recorder is not None,
        )
        for output_run_paths in output_run_paths_by_phenotype
    )
    timing.record_stage_duration(stage_timing_recorder, "output_writer_preparation", writer_start_time)
    if telemetry_session is not None:
        telemetry_session.log_event(
            "prediction_source_loaded",
            association_mode=association_mode.value,
            phenotype_count=len(run_input.phenotype_names),
        )
    preflight_start_time = time.perf_counter()
    logger.debug("Running preflight validation for multi-phenotype pipeline.")
    run_multi_preflight(
        run_input=run_input,
        prediction_source=prediction_source,
        engine=engine,
        variant_limit=variant_limit,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
    )
    timing.record_stage_duration(stage_timing_recorder, "preflight_validation", preflight_start_time)
    logger.debug("Preflight validation passed for multi-phenotype pipeline.")
    if telemetry_session is not None:
        telemetry_session.log_event(
            "preflight_completed",
            association_mode=association_mode.value,
            phenotype_count=len(run_input.phenotype_names),
            sample_count=int(run_input.sample_indices.shape[0]),
        )
    if association_mode == types.AssociationMode.REGENIE2_BINARY:
        callback = callbacks.MultiBinaryRegenie2PipelineCallback(
            run_input=run_input,
            prediction_source=prediction_source,
            writer_sessions=writer_sessions,
            committed_chunk_identifier_sets=committed_chunk_identifier_sets,
            correction_plan=correction_plan,
            kernel_config=resolved_kernel_config,
            null_logistic_nonconvergence_policy=null_logistic_nonconvergence_policy,
            staging_depth=staging_depth,
            score_dtype=score_dtype,
            stage_timing_recorder=stage_timing_recorder,
            telemetry_session=telemetry_session,
        )
    else:
        callback = callbacks.MultiLinearRegenie2PipelineCallback(
            run_input=run_input,
            prediction_source=prediction_source,
            writer_sessions=writer_sessions,
            committed_chunk_identifier_sets=committed_chunk_identifier_sets,
            staging_depth=staging_depth,
            score_dtype=score_dtype,
            stage_timing_recorder=stage_timing_recorder,
            telemetry_session=telemetry_session,
        )
    committed_by_every_phenotype = set.intersection(*committed_chunk_identifier_sets)
    return run_bgen_engine_with_multi_callback(
        engine=engine,
        run_input=run_input,
        committed_chunk_identifiers=committed_by_every_phenotype,
        writer_sessions=writer_sessions,
        callback=callback,
        stage_timing_recorder=stage_timing_recorder,
        variant_major_dosage=True,
    )


def run_multi_preflight(
    *,
    run_input: native_dispatch.NativeBgenMultiRunInput,
    prediction_source: typing.Any,
    engine: _core.Regenie2RunEngine,
    variant_limit: int | None,
    trusted_no_missing_diploid: bool,
) -> None:
    """Run existing single-trait preflight checks for every trait in a multi run."""
    for trait_index in range(len(run_input.phenotype_names)):
        preflight.run_regenie2_preflight(
            run_input=MultiTraitPreflightRunInput(
                phenotype_vector=run_input.phenotype_matrix[trait_index],
                covariate_matrix=run_input.covariate_matrix,
            ),
            prediction_source=SingleTraitPredictionView(prediction_source, trait_index),
            engine=engine,
            variant_limit=variant_limit,
            is_binary_trait=run_input.is_binary_trait,
            trusted_no_missing_diploid=trusted_no_missing_diploid,
        )


def run_bgen_engine_with_multi_callback(
    *,
    engine: _core.Regenie2RunEngine,
    run_input: native_dispatch.NativeBgenMultiRunInput,
    committed_chunk_identifiers: set[int] | None,
    writer_sessions: tuple[typing.Any, ...],
    callback: object,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    variant_major_dosage: bool = False,
) -> tuple[Path | None, ...]:
    """Run native BGEN chunk delivery once and close all per-phenotype writers."""
    callback_finished = False
    try:
        if stage_timing_recorder is not None:
            engine.reset_profile()
        engine_delivery_start_time = time.perf_counter()
        committed_chunk_identifier_list = sorted(committed_chunk_identifiers or set())
        logger.debug(
            "Starting multi-phenotype native BGEN delivery: committed_chunk_count=%s variant_major_dosage=%s.",
            len(committed_chunk_identifier_list),
            variant_major_dosage,
        )
        if variant_major_dosage:
            processed_chunk_count = engine.run_bgen_variant_major_dosage_buffered_chunks(
                run_input.sample_indices,
                callback,
                committed_chunk_identifiers=committed_chunk_identifier_list,
            )
        else:
            processed_chunk_count = engine.run_bgen_dosage_buffered_chunks(
                run_input.sample_indices,
                callback,
                committed_chunk_identifiers=committed_chunk_identifier_list,
            )
        timing.record_stage_duration(stage_timing_recorder, "native_engine_delivery", engine_delivery_start_time)
        logger.debug("Multi-phenotype native BGEN delivery finished: processed_chunk_count=%s.", processed_chunk_count)
        if stage_timing_recorder is not None:
            stage_timing_recorder.set_native_bgen_profile(engine.profile_snapshot())
        native_dispatch.finish_callback_drain(callback=callback, stage_timing_recorder=stage_timing_recorder)
        callback_finished = True
        writer_finish_start_time = time.perf_counter()
        logger.debug("Finishing multi-phenotype output writers and optional Parquet finalization.")
        final_parquet_paths = tuple(
            None if (final_path := writer_session.finish()) is None else Path(final_path)
            for writer_session in writer_sessions
        )
        timing.record_stage_duration(
            stage_timing_recorder, "writer_finish_and_parquet_finalization", writer_finish_start_time
        )
    except shutdown.GracefulShutdownRequested as shutdown_request:
        logger.info("Multi-phenotype native BGEN delivery interrupted by %s.", shutdown_request.signal_name)
        try:
            if not callback_finished:
                native_dispatch.finish_callback_drain(callback=callback, stage_timing_recorder=stage_timing_recorder)
            writer_finish_start_time = time.perf_counter()
            for writer_session in writer_sessions:
                writer_session.finish_interrupted(shutdown_request.signal_name)
            timing.record_stage_duration(stage_timing_recorder, "writer_finish_interrupted", writer_finish_start_time)
        except BaseException:
            native_dispatch.abort_callback(callback)
            for writer_session in writer_sessions:
                native_dispatch.abort_writer_session(writer_session)
            timing.write_stage_timing_snapshot(stage_timing_recorder, None)
            raise
        timing.write_stage_timing_snapshot(stage_timing_recorder, None)
        raise
    except BaseException:
        logger.exception("Multi-phenotype native BGEN delivery failed.")
        native_dispatch.abort_callback(callback)
        for writer_session in writer_sessions:
            native_dispatch.abort_writer_session(writer_session)
        timing.write_stage_timing_snapshot(stage_timing_recorder, None)
        raise
    timing.write_stage_timing_snapshot(stage_timing_recorder, None)
    logger.info("Multi-phenotype native BGEN pipeline finished.")
    return final_parquet_paths
