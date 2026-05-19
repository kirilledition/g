"""Native-driven REGENIE step 2 pipeline wrappers."""

from __future__ import annotations

import time
import typing
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from g import _core, types
from g.compute import regenie2_binary, regenie2_linear
import g.engine.callbacks as callbacks
import g.engine.preflight as preflight
import g.engine.timing as timing
import g.engine.trusted_validation as trusted_validation
from g.io import output, source

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


@dataclass(frozen=True)
class NativeBgenRunInput:
    """Sample-aligned inputs retained in native form for BGEN REGENIE step 2.

    Attributes:
        native_aligned_sample_data: Rust-owned aligned sample identifiers and matrices.
        sample_indices: BGEN sample indices for native chunk delivery.
        phenotype_vector: JAX phenotype vector.
        covariate_matrix: JAX design matrix.
        is_binary_trait: Whether the run is for a binary trait.

    """

    native_aligned_sample_data: _core.NativeAlignedSampleData
    sample_indices: npt.NDArray[np.int64]
    phenotype_vector: jax.Array
    covariate_matrix: jax.Array
    is_binary_trait: bool


def build_native_bgen_run_input(
    native_aligned_sample_data: _core.NativeAlignedSampleData,
) -> NativeBgenRunInput:
    """Build Python/JAX views over Rust-owned aligned sample data."""
    return NativeBgenRunInput(
        native_aligned_sample_data=native_aligned_sample_data,
        sample_indices=np.ascontiguousarray(native_aligned_sample_data.sample_indices, dtype=np.int64),
        phenotype_vector=jnp.asarray(native_aligned_sample_data.phenotype_vector, dtype=jnp.float32),
        covariate_matrix=jnp.asarray(native_aligned_sample_data.covariate_matrix, dtype=jnp.float32),
        is_binary_trait=native_aligned_sample_data.is_binary_trait,
    )


def load_native_aligned_sample_data_from_individual_identifier_table(
    *,
    sample_table: typing.Any,
    phenotype_path: Path,
    phenotype_name: str,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    is_binary_trait: bool,
) -> _core.NativeAlignedSampleData:
    """Load Rust-owned aligned sample data from explicit sample identifiers."""
    return _core.align_sample_data(
        np.ascontiguousarray(sample_table.get_column("sample_index").to_numpy(), dtype=np.int64),
        typing.cast("list[str]", sample_table.get_column("family_identifier").to_list()),
        typing.cast("list[str]", sample_table.get_column("individual_identifier").to_list()),
        str(phenotype_path),
        phenotype_name,
        str(covariate_path) if covariate_path is not None else None,
        list(covariate_names) if covariate_names is not None else None,
        is_binary_trait,
    )


def load_native_aligned_sample_data_from_sample_file(
    *,
    sample_path: Path,
    expected_sample_count: int,
    phenotype_path: Path,
    phenotype_name: str,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    is_binary_trait: bool,
) -> _core.NativeAlignedSampleData:
    """Load Rust-owned aligned sample data through Oxford sample-file parsing."""
    return _core.align_sample_data_from_sample_file(
        str(sample_path),
        expected_sample_count,
        str(phenotype_path),
        phenotype_name,
        str(covariate_path) if covariate_path is not None else None,
        list(covariate_names) if covariate_names is not None else None,
        is_binary_trait,
    )


@dataclass(frozen=True)
class WarmCacheShape:
    """One genotype matrix shape warmed for the JAX compilation cache."""

    sample_count: int
    variant_count: int


@dataclass(frozen=True)
class WarmCacheReport:
    """Summary of warmed REGENIE step 2 JAX cache entries."""

    warmed_shapes: tuple[WarmCacheShape, ...]


def build_warm_cache_shapes(
    *,
    engine: _core.Regenie2RunEngine,
    chunk_size: int,
    variant_limit: int | None,
    sample_count: int,
) -> tuple[WarmCacheShape, ...]:
    """Build the full and tail chunk shapes that should be warmed."""
    chunk_specs = _core.plan_genotype_chunks(
        engine.variant_count,
        chunk_size,
        engine.chromosome_boundary_indices(),
        variant_limit=variant_limit,
        committed_chunk_identifiers=None,
    )
    variant_counts = []
    for chunk_spec in chunk_specs:
        variant_count = int(chunk_spec.variant_stop_index - chunk_spec.variant_start_index)
        if variant_count > 0 and variant_count not in variant_counts:
            variant_counts.append(variant_count)
    variant_counts.sort(reverse=True)
    return tuple(
        WarmCacheShape(sample_count=sample_count, variant_count=variant_count) for variant_count in variant_counts[:2]
    )


def build_synthetic_genotype_matrix(
    *,
    phenotype_vector: jax.Array,
    variant_count: int,
    is_binary_trait: bool,
) -> jax.Array:
    """Build deterministic genotype inputs for cache warming."""
    if is_binary_trait:
        genotype_vector = jnp.asarray(phenotype_vector, dtype=jnp.float32) * 2.0
    else:
        sample_index = jnp.arange(phenotype_vector.shape[0], dtype=jnp.float32)
        genotype_vector = jnp.mod(sample_index, 3.0)
        genotype_vector = genotype_vector - jnp.mean(genotype_vector)
    return jnp.tile(genotype_vector[:, None], (1, variant_count))


def warm_regenie2_linear_bgen_cache(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    phenotype_path: Path,
    phenotype_name: str,
    prediction_list_path: Path,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    chunk_size: int,
    variant_limit: int | None,
    trusted_no_missing_diploid: bool = False,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode = types.TrustedBgenValidationMode.CACHE_ON_MISS,
) -> WarmCacheReport:
    """Warm full and tail JAX compilation-cache shapes for quantitative REGENIE step 2."""
    engine = build_bgen_run_engine(
        genotype_source_config=genotype_source_config,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
        trusted_bgen_validation_mode=trusted_bgen_validation_mode,
    )
    run_input = load_native_bgen_run_input(
        genotype_source_config=genotype_source_config,
        engine=engine,
        phenotype_path=phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=False,
    )
    prediction_source = build_regenie_prediction_source(
        prediction_list_path=prediction_list_path,
        phenotype_name=phenotype_name,
        run_input=run_input,
    )
    chromosome = first_engine_chromosome(engine)
    regenie_state = regenie2_linear.prepare_regenie2_linear_state(
        covariate_matrix=run_input.covariate_matrix,
        phenotype_vector=run_input.phenotype_vector,
    )
    chromosome_state = regenie2_linear.prepare_regenie2_linear_chromosome_state(
        regenie_state,
        jax.device_put(prediction_source.get_chromosome_predictions(chromosome)),
    )
    shapes = build_warm_cache_shapes(
        engine=engine,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        sample_count=int(run_input.sample_indices.shape[0]),
    )
    for shape in shapes:
        genotype_matrix = build_synthetic_genotype_matrix(
            phenotype_vector=run_input.phenotype_vector,
            variant_count=shape.variant_count,
            is_binary_trait=False,
        )
        result = regenie2_linear.compute_regenie2_linear_chunk_from_chromosome_state(
            chromosome_state=chromosome_state,
            genotype_matrix=genotype_matrix,
        )
        block_until_ready(result.log10_p_value)
    return WarmCacheReport(warmed_shapes=shapes)


def warm_regenie2_binary_bgen_cache(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    phenotype_path: Path,
    phenotype_name: str,
    prediction_list_path: Path,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    chunk_size: int,
    variant_limit: int | None,
    correction_plan: types.BinaryCorrectionPlan,
    trusted_no_missing_diploid: bool = False,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode = types.TrustedBgenValidationMode.CACHE_ON_MISS,
) -> WarmCacheReport:
    """Warm full and tail JAX compilation-cache shapes for binary REGENIE step 2."""
    engine = build_bgen_run_engine(
        genotype_source_config=genotype_source_config,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
        trusted_bgen_validation_mode=trusted_bgen_validation_mode,
    )
    run_input = load_native_bgen_run_input(
        genotype_source_config=genotype_source_config,
        engine=engine,
        phenotype_path=phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=True,
    )
    prediction_source = build_regenie_prediction_source(
        prediction_list_path=prediction_list_path,
        phenotype_name=phenotype_name,
        run_input=run_input,
    )
    chromosome = first_engine_chromosome(engine)
    regenie_state = regenie2_binary.prepare_regenie2_binary_state(
        covariate_matrix=run_input.covariate_matrix,
        phenotype_vector=run_input.phenotype_vector,
    )
    chromosome_state = regenie2_binary.prepare_regenie2_binary_chromosome_state(
        regenie_state,
        jax.device_put(prediction_source.get_chromosome_predictions(chromosome)),
    )
    shapes = build_warm_cache_shapes(
        engine=engine,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        sample_count=int(run_input.sample_indices.shape[0]),
    )
    for shape in shapes:
        genotype_matrix = build_synthetic_genotype_matrix(
            phenotype_vector=run_input.phenotype_vector,
            variant_count=shape.variant_count,
            is_binary_trait=True,
        )
        result = regenie2_binary.compute_regenie2_binary_chunk_from_chromosome_state(
            chromosome_state=chromosome_state,
            genotype_matrix=genotype_matrix,
            correction_plan=correction_plan,
        )
        block_until_ready(result.log10_p_value)
    return WarmCacheReport(warmed_shapes=shapes)


def first_engine_chromosome(engine: _core.Regenie2RunEngine) -> str:
    """Return the first chromosome label from the native BGEN engine."""
    chromosome_values, _, _, _, _ = engine.variant_metadata_slice(0, 1)
    if not chromosome_values:
        message = "Cannot warm REGENIE step 2 cache for an empty BGEN dataset."
        raise ValueError(message)
    return chromosome_values[0]


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
) -> Path | None:
    """Run the native BGEN pipeline for quantitative REGENIE step 2."""
    stage_timing_recorder = stage_timing_recorder or build_stage_timing_recorder_from_environment()
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
) -> NativeBgenRunInput:
    """Load native-aligned samples and JAX compute inputs for a native BGEN run."""
    source.validate_genotype_source_config(genotype_source_config)
    resolved_sample_path = source.resolve_bgen_sample_path(
        genotype_source_config.source_path,
        genotype_source_config.sample_path,
    )
    if resolved_sample_path is not None:
        native_aligned_sample_data = load_native_aligned_sample_data_from_sample_file(
            sample_path=resolved_sample_path,
            expected_sample_count=engine.sample_count,
            phenotype_path=phenotype_path,
            phenotype_name=phenotype_name,
            covariate_path=covariate_path,
            covariate_names=covariate_names,
            is_binary_trait=is_binary_trait,
        )
        return build_native_bgen_run_input(native_aligned_sample_data)
    if engine.contains_embedded_samples:
        sample_table = source.build_sample_identifier_table(np.asarray(engine.sample_identifiers(), dtype=np.str_))
        native_aligned_sample_data = load_native_aligned_sample_data_from_individual_identifier_table(
            sample_table=sample_table,
            phenotype_path=phenotype_path,
            phenotype_name=phenotype_name,
            covariate_path=covariate_path,
            covariate_names=covariate_names,
            is_binary_trait=is_binary_trait,
        )
        return build_native_bgen_run_input(native_aligned_sample_data)
    message = "BGEN file does not contain samples and no .sample file was found."
    raise ValueError(message)


def build_regenie_prediction_source(
    *,
    prediction_list_path: Path,
    phenotype_name: str,
    run_input: NativeBgenRunInput,
) -> _core.RegeniePredictionSource:
    """Load Rust-owned REGENIE step 1 predictions aligned to the run samples."""
    return _core.RegeniePredictionSource.from_native_aligned_sample_data(
        str(prediction_list_path),
        phenotype_name,
        run_input.native_aligned_sample_data,
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
    engine = _core.Regenie2RunEngine(
        str(genotype_source_config.source_path),
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
    )
    if trusted_no_missing_diploid:
        validate_trusted_bgen_with_cache(
            engine=engine,
            bgen_path=genotype_source_config.source_path,
            validation_mode=trusted_bgen_validation_mode,
        )
    return engine


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
    try:
        if stage_timing_recorder is not None:
            engine.reset_profile()
        engine_delivery_start_time = time.perf_counter()
        sample_indices = run_input.sample_indices
        committed_chunk_identifier_list = sorted(committed_chunk_identifiers or set())
        if variant_major_dosage:
            engine.run_bgen_variant_major_dosage_buffered_chunks(
                sample_indices,
                callback,
                committed_chunk_identifiers=committed_chunk_identifier_list,
            )
        else:
            engine.run_bgen_dosage_buffered_chunks(
                sample_indices,
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
        final_parquet_path = writer_session.finish()
        record_stage_duration(stage_timing_recorder, "writer_finish_and_parquet_finalization", writer_finish_start_time)
    except Exception:
        abort_callback = getattr(callback, "abort", None)
        if callable(abort_callback):
            abort_callback()
        writer_session.abort()
        write_stage_timing_snapshot_from_environment(stage_timing_recorder)
        raise
    write_stage_timing_snapshot_from_environment(stage_timing_recorder)
    if final_parquet_path is None:
        return None
    return Path(final_parquet_path)
