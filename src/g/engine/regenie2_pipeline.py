"""Native-driven REGENIE step 2 pipeline wrappers."""

from __future__ import annotations

import time
import typing

import g.compute.regenie2_binary as regenie2_binary
import g.compute.regenie2_linear as regenie2_linear
import g.engine.callbacks as callbacks
import g.engine.native_dispatch as native_dispatch
import g.engine.preflight as preflight
import g.engine.timing as timing
import g.engine.trusted_validation as trusted_validation
import g.engine.warm_cache as warm_cache
from g import _core, types
from g.io import output, source

if typing.TYPE_CHECKING:
    from pathlib import Path

REGENIE_COMPUTE_PATCH_TARGETS = (regenie2_binary, regenie2_linear)
StageTimingSnapshot = timing.StageTimingSnapshot
StageTimingRecorder = timing.StageTimingRecorder
build_stage_timing_recorder_from_environment = timing.build_stage_timing_recorder_from_environment
record_stage_duration = timing.record_stage_duration
write_stage_timing_snapshot_from_environment = timing.write_stage_timing_snapshot_from_environment
ASSUME_TRUSTED_NO_MISSING_DIPLOID_VALIDATED_ENVIRONMENT_VARIABLE = (
    trusted_validation.ASSUME_TRUSTED_NO_MISSING_DIPLOID_VALIDATED_ENVIRONMENT_VARIABLE
)
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
build_native_bgen_run_input = native_dispatch.build_native_bgen_run_input
load_native_aligned_sample_data = native_dispatch.load_native_aligned_sample_data
build_regenie_prediction_source = native_dispatch.build_regenie_prediction_source
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
    committed_chunk_identifiers: set[int] | None = None,
    finalize_parquet: bool = False,
    writer_thread_count: int = output.DEFAULT_WRITER_THREAD_COUNT,
    writer_queue_depth: int = output.DEFAULT_WRITER_QUEUE_DEPTH,
    trusted_no_missing_diploid: bool = False,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode = types.TrustedBgenValidationMode.CACHE_ON_MISS,
    stage_timing_recorder: StageTimingRecorder | None = None,
    alignment_config: SampleAlignmentConfigProtocol | None = None,
) -> Path | None:
    """Run the native BGEN pipeline for quantitative REGENIE step 2."""
    stage_timing_recorder = stage_timing_recorder or build_stage_timing_recorder_from_environment()
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
    output.write_run_manifest_header(
        output_run_paths=output_run_paths,
        association_mode=types.AssociationMode.REGENIE2_LINEAR,
        bgen_path=genotype_source_config.source_path,
        sample_count=int(engine.sample_count),
        variant_count=int(engine.variant_count),
        chunk_size=chunk_size,
        binary_correction_plan=types.BinaryCorrectionPlan(),
        trusted_no_missing_diploid=trusted_no_missing_diploid,
    )
    writer_start_time = time.perf_counter()
    writer_session = output.create_output_writer_session(
        output_run_paths,
        types.AssociationMode.REGENIE2_LINEAR,
        writer_thread_count=writer_thread_count,
        writer_queue_depth=writer_queue_depth,
        finalize_parquet=finalize_parquet,
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
        committed_chunk_identifiers=committed_chunk_identifiers,
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
    committed_chunk_identifiers: set[int] | None = None,
    finalize_parquet: bool = False,
    writer_thread_count: int = output.DEFAULT_WRITER_THREAD_COUNT,
    writer_queue_depth: int = output.DEFAULT_WRITER_QUEUE_DEPTH,
    trusted_no_missing_diploid: bool = False,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode = types.TrustedBgenValidationMode.CACHE_ON_MISS,
    correction_plan: types.BinaryCorrectionPlan = types.BinaryCorrectionPlan(),
    stage_timing_recorder: StageTimingRecorder | None = None,
    alignment_config: SampleAlignmentConfigProtocol | None = None,
) -> Path | None:
    """Run the native BGEN pipeline for binary REGENIE step 2."""
    stage_timing_recorder = stage_timing_recorder or build_stage_timing_recorder_from_environment()
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
    output.write_run_manifest_header(
        output_run_paths=output_run_paths,
        association_mode=types.AssociationMode.REGENIE2_BINARY,
        bgen_path=genotype_source_config.source_path,
        sample_count=int(engine.sample_count),
        variant_count=int(engine.variant_count),
        chunk_size=chunk_size,
        binary_correction_plan=correction_plan,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
    )
    writer_start_time = time.perf_counter()
    writer_session = output.create_output_writer_session(
        output_run_paths,
        types.AssociationMode.REGENIE2_BINARY,
        writer_thread_count=writer_thread_count,
        writer_queue_depth=writer_queue_depth,
        finalize_parquet=finalize_parquet,
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
        staging_depth=staging_depth,
        stage_timing_recorder=stage_timing_recorder,
    )
    return run_bgen_engine_with_callback(
        engine=engine,
        run_input=run_input,
        committed_chunk_identifiers=committed_chunk_identifiers,
        writer_session=writer_session,
        callback=callback,
        stage_timing_recorder=stage_timing_recorder,
        variant_major_dosage=use_variant_major,
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
        stage_timing_snapshot_writer=write_stage_timing_snapshot_from_environment,
    )
