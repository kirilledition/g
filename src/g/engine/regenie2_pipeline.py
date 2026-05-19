"""Native-driven REGENIE step 2 pipeline wrappers."""

from __future__ import annotations

import time
import typing
from dataclasses import dataclass
from pathlib import Path

import g.compute.regenie2_binary as regenie2_binary
import g.compute.regenie2_binary_types as regenie2_binary_types
import g.compute.regenie2_linear as regenie2_linear
import g.engine.callbacks as callbacks
import g.engine.native_dispatch as native_dispatch
import g.engine.preflight as preflight
import g.engine.timing as timing
import g.engine.trusted_validation as trusted_validation
import g.engine.warm_cache as warm_cache
from g import _core, types
from g.io import output, source

REGENIE_COMPUTE_PATCH_TARGETS = (regenie2_binary, regenie2_linear)
StageTimingSnapshot = timing.StageTimingSnapshot
StageTimingRecorder = timing.StageTimingRecorder
build_stage_timing_recorder = timing.build_stage_timing_recorder
record_stage_duration = timing.record_stage_duration
write_stage_timing_snapshot = timing.write_stage_timing_snapshot
TRUSTED_BGEN_VALIDATION_SCHEMA_VERSION = trusted_validation.TRUSTED_BGEN_VALIDATION_SCHEMA_VERSION
assume_trusted_no_missing_diploid_validated = trusted_validation.assume_trusted_no_missing_diploid_validated
trusted_bgen_validation_cache_directory = trusted_validation.trusted_bgen_validation_cache_directory
build_trusted_bgen_validation_fingerprint = trusted_validation.build_trusted_bgen_validation_fingerprint
trusted_bgen_validation_cache_path = trusted_validation.trusted_bgen_validation_cache_path
validate_trusted_bgen_with_cache = trusted_validation.validate_trusted_bgen_with_cache

PreprocessedDosageChunkWorkItem = callbacks.PreprocessedDosageChunkWorkItem
PreprocessedVariantMajorDosageChunkWorkItem = callbacks.PreprocessedVariantMajorDosageChunkWorkItem
RegeniePredictionSourceProtocol = callbacks.RegeniePredictionSourceProtocol
NativeBgenCallbackRunner = callbacks.NativeBgenCallbackRunner
LinearRegenie2PipelineCallback = callbacks.LinearRegenie2PipelineCallback
BinaryRegenie2PipelineCallback = callbacks.BinaryRegenie2PipelineCallback
MultiLinearRegenie2PipelineCallback = callbacks.MultiLinearRegenie2PipelineCallback
MultiBinaryRegenie2PipelineCallback = callbacks.MultiBinaryRegenie2PipelineCallback
block_until_ready = callbacks.block_until_ready
record_binary_chunk_diagnostics = callbacks.record_binary_chunk_diagnostics
put_genotype_matrix_on_device = callbacks.put_genotype_matrix_on_device
write_regenie2_native_chunk_with_optional_timing = callbacks.write_regenie2_native_chunk_with_optional_timing
get_metadata_chromosome = callbacks.get_metadata_chromosome
WarmCacheShape = warm_cache.WarmCacheShape
WarmCacheReport = warm_cache.WarmCacheReport
build_warm_cache_shapes = warm_cache.build_warm_cache_shapes
build_synthetic_genotype_matrix = warm_cache.build_synthetic_genotype_matrix
warm_regenie2_linear_bgen_cache = warm_cache.warm_regenie2_linear_bgen_cache
warm_regenie2_binary_bgen_cache = warm_cache.warm_regenie2_binary_bgen_cache
first_engine_chromosome = warm_cache.first_engine_chromosome
NativeBgenRunInput = native_dispatch.NativeBgenRunInput
NativeBgenMultiRunInput = native_dispatch.NativeBgenMultiRunInput
build_native_bgen_run_input = native_dispatch.build_native_bgen_run_input
load_native_aligned_sample_data = native_dispatch.load_native_aligned_sample_data
build_regenie_prediction_source = native_dispatch.build_regenie_prediction_source
build_multi_regenie_prediction_source = native_dispatch.build_multi_regenie_prediction_source
SampleAlignmentConfigProtocol = native_dispatch.SampleAlignmentConfigProtocol
resolve_sample_key_mode = native_dispatch.resolve_sample_key_mode
resolve_allow_duplicate_iid_alignment = native_dispatch.resolve_allow_duplicate_iid_alignment


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
    writer_thread_count: int = output.DEFAULT_WRITER_THREAD_COUNT,
    writer_queue_depth: int = output.DEFAULT_WRITER_QUEUE_DEPTH,
    chunks_per_arrow_file: int = output.DEFAULT_CHUNKS_PER_ARROW_FILE,
    arrow_compression: types.ArrowCompression = types.ArrowCompression.ZSTD,
    trusted_no_missing_diploid: bool = False,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode = types.TrustedBgenValidationMode.CACHE_ON_MISS,
    stage_timing_recorder: StageTimingRecorder | None = None,
    alignment_config: SampleAlignmentConfigProtocol | None = None,
) -> Path | None:
    """Run the native BGEN pipeline for quantitative REGENIE step 2."""
    stage_timing_recorder = stage_timing_recorder or build_stage_timing_recorder()
    use_variant_major = trusted_no_missing_diploid
    engine_start_time = time.perf_counter()
    engine = build_bgen_run_engine(
        genotype_source_config=genotype_source_config,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
        trusted_bgen_validation_mode=trusted_bgen_validation_mode,
    )
    record_stage_duration(stage_timing_recorder, "bgen_engine_open_index_setup", engine_start_time)
    alignment_start_time = time.perf_counter()
    run_input = load_native_bgen_run_input(
        genotype_source_config=genotype_source_config,
        engine=engine,
        phenotype_path=phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=False,
        alignment_config=alignment_config,
    )
    record_stage_duration(stage_timing_recorder, "sample_phenotype_covariate_alignment", alignment_start_time)
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
        sample_key_mode=resolve_sample_key_mode(alignment_config),
        allow_duplicate_iid_alignment=resolve_allow_duplicate_iid_alignment(alignment_config),
    )
    initialized_output_run = output.initialize_output_run(
        output_run_paths=output_run_paths,
        existing_manifest=existing_manifest,
        current_header=current_header,
        resume=resume,
        resume_mode=resume_mode,
    )
    writer_start_time = time.perf_counter()
    writer_session = output.create_output_writer_session(
        output_run_paths,
        types.AssociationMode.REGENIE2_LINEAR,
        writer_thread_count=writer_thread_count,
        writer_queue_depth=writer_queue_depth,
        finalize_parquet=finalize_parquet,
        chunks_per_arrow_file=chunks_per_arrow_file,
        arrow_compression=arrow_compression,
    )
    record_stage_duration(stage_timing_recorder, "output_writer_preparation", writer_start_time)
    prediction_start_time = time.perf_counter()
    prediction_source = build_regenie_prediction_source(
        prediction_list_path=prediction_list_path,
        phenotype_name=phenotype_name,
        run_input=run_input,
        alignment_config=alignment_config,
    )
    record_stage_duration(stage_timing_recorder, "prediction_source_load", prediction_start_time)
    preflight_start_time = time.perf_counter()
    preflight.run_regenie2_preflight(
        run_input=run_input,
        prediction_source=prediction_source,
        engine=engine,
        is_binary_trait=False,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
    )
    record_stage_duration(stage_timing_recorder, "preflight_validation", preflight_start_time)
    callback = LinearRegenie2PipelineCallback(
        run_input=run_input,
        prediction_source=prediction_source,
        writer_session=writer_session,
        staging_depth=staging_depth,
        stage_timing_recorder=stage_timing_recorder,
    )
    return run_bgen_engine_with_callback(
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
    writer_thread_count: int = output.DEFAULT_WRITER_THREAD_COUNT,
    writer_queue_depth: int = output.DEFAULT_WRITER_QUEUE_DEPTH,
    chunks_per_arrow_file: int = output.DEFAULT_CHUNKS_PER_ARROW_FILE,
    arrow_compression: types.ArrowCompression = types.ArrowCompression.ZSTD,
    trusted_no_missing_diploid: bool = False,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode = types.TrustedBgenValidationMode.CACHE_ON_MISS,
    correction_plan: types.BinaryCorrectionPlan = types.BinaryCorrectionPlan(),
    kernel_config: regenie2_binary_types.BinaryKernelConfig | None = None,
    stage_timing_recorder: StageTimingRecorder | None = None,
    alignment_config: SampleAlignmentConfigProtocol | None = None,
) -> Path | None:
    """Run the native BGEN pipeline for binary REGENIE step 2."""
    stage_timing_recorder = stage_timing_recorder or build_stage_timing_recorder()
    resolved_kernel_config = kernel_config or regenie2_binary.DEFAULT_BINARY_KERNEL_CONFIG
    use_variant_major = trusted_no_missing_diploid
    engine_start_time = time.perf_counter()
    engine = build_bgen_run_engine(
        genotype_source_config=genotype_source_config,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
        trusted_bgen_validation_mode=trusted_bgen_validation_mode,
    )
    record_stage_duration(stage_timing_recorder, "bgen_engine_open_index_setup", engine_start_time)
    alignment_start_time = time.perf_counter()
    run_input = load_native_bgen_run_input(
        genotype_source_config=genotype_source_config,
        engine=engine,
        phenotype_path=phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=True,
        alignment_config=alignment_config,
    )
    record_stage_duration(stage_timing_recorder, "sample_phenotype_covariate_alignment", alignment_start_time)
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
        trusted_no_missing_diploid=trusted_no_missing_diploid,
        sample_key_mode=resolve_sample_key_mode(alignment_config),
        allow_duplicate_iid_alignment=resolve_allow_duplicate_iid_alignment(alignment_config),
    )
    initialized_output_run = output.initialize_output_run(
        output_run_paths=output_run_paths,
        existing_manifest=existing_manifest,
        current_header=current_header,
        resume=resume,
        resume_mode=resume_mode,
    )
    writer_start_time = time.perf_counter()
    writer_session = output.create_output_writer_session(
        output_run_paths,
        types.AssociationMode.REGENIE2_BINARY,
        writer_thread_count=writer_thread_count,
        writer_queue_depth=writer_queue_depth,
        finalize_parquet=finalize_parquet,
        chunks_per_arrow_file=chunks_per_arrow_file,
        arrow_compression=arrow_compression,
    )
    record_stage_duration(stage_timing_recorder, "output_writer_preparation", writer_start_time)
    prediction_start_time = time.perf_counter()
    prediction_source = build_regenie_prediction_source(
        prediction_list_path=prediction_list_path,
        phenotype_name=phenotype_name,
        run_input=run_input,
        alignment_config=alignment_config,
    )
    record_stage_duration(stage_timing_recorder, "prediction_source_load", prediction_start_time)
    preflight_start_time = time.perf_counter()
    preflight.run_regenie2_preflight(
        run_input=run_input,
        prediction_source=prediction_source,
        engine=engine,
        is_binary_trait=True,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
    )
    record_stage_duration(stage_timing_recorder, "preflight_validation", preflight_start_time)
    callback = BinaryRegenie2PipelineCallback(
        run_input=run_input,
        prediction_source=prediction_source,
        writer_session=writer_session,
        correction_plan=correction_plan,
        kernel_config=resolved_kernel_config,
        staging_depth=staging_depth,
        stage_timing_recorder=stage_timing_recorder,
    )
    return run_bgen_engine_with_callback(
        engine=engine,
        run_input=run_input,
        committed_chunk_identifiers=set(initialized_output_run.committed_chunk_identifiers),
        writer_session=writer_session,
        callback=callback,
        stage_timing_recorder=stage_timing_recorder,
        variant_major_dosage=use_variant_major,
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
    writer_thread_count: int = output.DEFAULT_WRITER_THREAD_COUNT,
    writer_queue_depth: int = output.DEFAULT_WRITER_QUEUE_DEPTH,
    chunks_per_arrow_file: int = output.DEFAULT_CHUNKS_PER_ARROW_FILE,
    arrow_compression: types.ArrowCompression = types.ArrowCompression.ZSTD,
    trusted_no_missing_diploid: bool = False,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode = types.TrustedBgenValidationMode.CACHE_ON_MISS,
    stage_timing_recorder: StageTimingRecorder | None = None,
    alignment_config: SampleAlignmentConfigProtocol | None = None,
) -> tuple[Path | None, ...]:
    """Run the native BGEN pipeline once for multiple quantitative phenotypes."""
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
        correction_plan=types.BinaryCorrectionPlan(),
        stage_timing_recorder=stage_timing_recorder,
        alignment_config=alignment_config,
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
    writer_thread_count: int = output.DEFAULT_WRITER_THREAD_COUNT,
    writer_queue_depth: int = output.DEFAULT_WRITER_QUEUE_DEPTH,
    chunks_per_arrow_file: int = output.DEFAULT_CHUNKS_PER_ARROW_FILE,
    arrow_compression: types.ArrowCompression = types.ArrowCompression.ZSTD,
    trusted_no_missing_diploid: bool = False,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode = types.TrustedBgenValidationMode.CACHE_ON_MISS,
    correction_plan: types.BinaryCorrectionPlan = types.BinaryCorrectionPlan(),
    stage_timing_recorder: StageTimingRecorder | None = None,
    alignment_config: SampleAlignmentConfigProtocol | None = None,
) -> tuple[Path | None, ...]:
    """Run the native BGEN pipeline once for multiple binary phenotypes."""
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
        correction_plan=correction_plan,
        stage_timing_recorder=stage_timing_recorder,
        alignment_config=alignment_config,
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
    correction_plan: types.BinaryCorrectionPlan,
    stage_timing_recorder: StageTimingRecorder | None,
    alignment_config: SampleAlignmentConfigProtocol | None,
    association_mode: types.AssociationMode,
) -> tuple[Path | None, ...]:
    """Shared implementation for native multi-phenotype BGEN pipelines."""
    stage_timing_recorder = stage_timing_recorder or build_stage_timing_recorder()
    use_variant_major = trusted_no_missing_diploid
    existing_manifests = existing_manifests_by_phenotype or tuple(None for _ in phenotype_names)
    engine_start_time = time.perf_counter()
    engine = build_bgen_run_engine(
        genotype_source_config=genotype_source_config,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
        trusted_bgen_validation_mode=trusted_bgen_validation_mode,
    )
    record_stage_duration(stage_timing_recorder, "bgen_engine_open_index_setup", engine_start_time)
    alignment_start_time = time.perf_counter()
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
    record_stage_duration(stage_timing_recorder, "sample_phenotype_covariate_alignment", alignment_start_time)
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
            sample_key_mode=resolve_sample_key_mode(alignment_config),
            allow_duplicate_iid_alignment=resolve_allow_duplicate_iid_alignment(alignment_config),
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
    writer_sessions = tuple(
        output.create_output_writer_session(
            output_run_paths,
            association_mode,
            writer_thread_count=writer_thread_count,
            writer_queue_depth=writer_queue_depth,
            finalize_parquet=finalize_parquet,
            chunks_per_arrow_file=chunks_per_arrow_file,
            arrow_compression=arrow_compression,
        )
        for output_run_paths in output_run_paths_by_phenotype
    )
    record_stage_duration(stage_timing_recorder, "output_writer_preparation", writer_start_time)
    prediction_start_time = time.perf_counter()
    prediction_source = build_multi_regenie_prediction_source(
        prediction_list_path=prediction_list_path,
        run_input=run_input,
        alignment_config=alignment_config,
    )
    record_stage_duration(stage_timing_recorder, "prediction_source_load", prediction_start_time)
    preflight_start_time = time.perf_counter()
    run_multi_preflight(
        run_input=run_input,
        prediction_source=prediction_source,
        engine=engine,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
    )
    record_stage_duration(stage_timing_recorder, "preflight_validation", preflight_start_time)
    if association_mode == types.AssociationMode.REGENIE2_BINARY:
        callback = MultiBinaryRegenie2PipelineCallback(
            run_input=run_input,
            prediction_source=prediction_source,
            writer_sessions=writer_sessions,
            committed_chunk_identifier_sets=committed_chunk_identifier_sets,
            correction_plan=correction_plan,
            staging_depth=staging_depth,
            stage_timing_recorder=stage_timing_recorder,
        )
    else:
        callback = MultiLinearRegenie2PipelineCallback(
            run_input=run_input,
            prediction_source=prediction_source,
            writer_sessions=writer_sessions,
            committed_chunk_identifier_sets=committed_chunk_identifier_sets,
            staging_depth=staging_depth,
            stage_timing_recorder=stage_timing_recorder,
        )
    committed_by_every_phenotype = set.intersection(*committed_chunk_identifier_sets)
    return run_bgen_engine_with_multi_callback(
        engine=engine,
        run_input=run_input,
        committed_chunk_identifiers=committed_by_every_phenotype,
        writer_sessions=writer_sessions,
        callback=callback,
        stage_timing_recorder=stage_timing_recorder,
        variant_major_dosage=use_variant_major,
    )


def run_multi_preflight(
    *,
    run_input: NativeBgenMultiRunInput,
    prediction_source: typing.Any,
    engine: _core.Regenie2RunEngine,
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
            is_binary_trait=run_input.is_binary_trait,
            trusted_no_missing_diploid=trusted_no_missing_diploid,
        )


def load_native_bgen_run_input(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    engine: _core.Regenie2RunEngine,
    phenotype_path: Path,
    phenotype_name: str,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    is_binary_trait: bool,
    alignment_config: SampleAlignmentConfigProtocol | None = None,
) -> NativeBgenRunInput:
    """Load native-aligned samples and JAX compute inputs for a native BGEN run."""
    return native_dispatch.load_native_bgen_run_input(
        genotype_source_config=genotype_source_config,
        engine=engine,
        phenotype_path=phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=is_binary_trait,
        alignment_config=alignment_config,
        build_native_bgen_run_input_callable=build_native_bgen_run_input,
        load_aligned_sample_data_callable=load_native_aligned_sample_data,
    )


def build_bgen_run_engine(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    chunk_size: int,
    variant_limit: int | None,
    trusted_no_missing_diploid: bool = False,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode = types.TrustedBgenValidationMode.CACHE_ON_MISS,
) -> _core.Regenie2RunEngine:
    """Open the native BGEN run engine once for alignment and chunk delivery."""
    return native_dispatch.build_bgen_run_engine(
        genotype_source_config=genotype_source_config,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
        trusted_bgen_validation_mode=trusted_bgen_validation_mode,
        trusted_bgen_validator=validate_trusted_bgen_with_cache,
    )


def run_bgen_engine_with_callback(
    *,
    engine: _core.Regenie2RunEngine,
    run_input: NativeBgenRunInput,
    committed_chunk_identifiers: set[int] | None,
    writer_session: typing.Any,
    callback: object,
    stage_timing_recorder: StageTimingRecorder | None,
    variant_major_dosage: bool = False,
) -> Path | None:
    """Run native BGEN chunk delivery and close the output writer."""
    return native_dispatch.run_bgen_engine_with_callback(
        engine=engine,
        run_input=run_input,
        committed_chunk_identifiers=committed_chunk_identifiers,
        writer_session=writer_session,
        callback=callback,
        stage_timing_recorder=stage_timing_recorder,
        variant_major_dosage=variant_major_dosage,
        stage_timing_snapshot_writer=write_stage_timing_snapshot,
    )


def run_bgen_engine_with_multi_callback(
    *,
    engine: _core.Regenie2RunEngine,
    run_input: NativeBgenMultiRunInput,
    committed_chunk_identifiers: set[int] | None,
    writer_sessions: tuple[typing.Any, ...],
    callback: object,
    stage_timing_recorder: StageTimingRecorder | None,
    variant_major_dosage: bool = False,
) -> tuple[Path | None, ...]:
    """Run native BGEN chunk delivery once and close all per-phenotype writers."""
    try:
        if stage_timing_recorder is not None:
            engine.reset_profile()
        engine_delivery_start_time = time.perf_counter()
        committed_chunk_identifier_list = sorted(committed_chunk_identifiers or set())
        if variant_major_dosage:
            engine.run_bgen_variant_major_dosage_buffered_chunks(
                run_input.sample_indices,
                callback,
                committed_chunk_identifiers=committed_chunk_identifier_list,
            )
        else:
            engine.run_bgen_dosage_buffered_chunks(
                run_input.sample_indices,
                callback,
                committed_chunk_identifiers=committed_chunk_identifier_list,
            )
        record_stage_duration(stage_timing_recorder, "native_engine_delivery", engine_delivery_start_time)
        if stage_timing_recorder is not None:
            stage_timing_recorder.set_native_bgen_profile(engine.profile_snapshot())
        callback_finish_start_time = time.perf_counter()
        typing.cast("typing.Any", callback).finish()
        record_stage_duration(stage_timing_recorder, "callback_drain", callback_finish_start_time)
        writer_finish_start_time = time.perf_counter()
        final_parquet_paths = tuple(
            None if (final_path := writer_session.finish()) is None else Path(final_path)
            for writer_session in writer_sessions
        )
        record_stage_duration(stage_timing_recorder, "writer_finish_and_parquet_finalization", writer_finish_start_time)
    except Exception:
        abort_callback = getattr(callback, "abort", None)
        if callable(abort_callback):
            abort_callback()
        for writer_session in writer_sessions:
            writer_session.abort()
        write_stage_timing_snapshot(stage_timing_recorder, None)
        raise
    write_stage_timing_snapshot(stage_timing_recorder, None)
    return final_parquet_paths
