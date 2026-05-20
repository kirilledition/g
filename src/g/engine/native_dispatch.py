"""Native BGEN engine dispatch helpers for REGENIE step 2 pipelines."""

from __future__ import annotations

import contextlib
import time
import typing
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from g import _core, types
from g.engine import shutdown, timing, trusted_validation
from g.io import source


class SampleAlignmentConfigProtocol(typing.Protocol):
    """Sample identity alignment settings accepted by native dispatch."""

    sample_key_mode: types.SampleKeyMode


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


@dataclass(frozen=True)
class NativeBgenMultiRunInput:
    """Sample-aligned inputs for a shared multi-phenotype native BGEN run.

    Attributes:
        native_multi_aligned_sample_data: Rust-owned complete-case aligned multi-phenotype data.
        phenotype_names: Phenotype names in trait-major matrix order.
        sample_indices: BGEN sample indices for native chunk delivery.
        family_identifiers: Family identifiers for the shared sample set.
        individual_identifiers: Individual identifiers for the shared sample set.
        phenotype_matrix: JAX trait-major phenotype matrix.
        covariate_matrix: JAX shared design matrix.
        is_binary_trait: Whether the run is for binary traits.

    """

    native_multi_aligned_sample_data: _core.NativeMultiAlignedSampleData
    phenotype_names: tuple[str, ...]
    sample_indices: npt.NDArray[np.int64]
    family_identifiers: tuple[str, ...]
    individual_identifiers: tuple[str, ...]
    phenotype_matrix: jax.Array
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


def build_native_bgen_multi_run_input(
    native_multi_aligned_sample_data: _core.NativeMultiAlignedSampleData,
) -> NativeBgenMultiRunInput:
    """Build Python/JAX views over Rust-owned complete-case multi-phenotype data."""
    return NativeBgenMultiRunInput(
        native_multi_aligned_sample_data=native_multi_aligned_sample_data,
        phenotype_names=tuple(native_multi_aligned_sample_data.phenotype_names),
        sample_indices=np.ascontiguousarray(native_multi_aligned_sample_data.sample_indices, dtype=np.int64),
        family_identifiers=tuple(native_multi_aligned_sample_data.family_identifiers),
        individual_identifiers=tuple(native_multi_aligned_sample_data.individual_identifiers),
        phenotype_matrix=jnp.asarray(native_multi_aligned_sample_data.phenotype_matrix, dtype=jnp.float32),
        covariate_matrix=jnp.asarray(native_multi_aligned_sample_data.covariate_matrix, dtype=jnp.float32),
        is_binary_trait=native_multi_aligned_sample_data.is_binary_trait,
    )


def resolve_sample_key_mode(alignment_config: SampleAlignmentConfigProtocol | None) -> types.SampleKeyMode:
    """Resolve the sample key mode for native calls."""
    if alignment_config is None:
        return types.SampleKeyMode.IID
    return alignment_config.sample_key_mode


def load_native_aligned_sample_data(
    *,
    engine: _core.Regenie2RunEngine,
    sample_path: Path | None,
    phenotype_path: Path,
    phenotype_name: str,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    is_binary_trait: bool,
    alignment_config: SampleAlignmentConfigProtocol | None = None,
) -> _core.NativeAlignedSampleData:
    """Load Rust-owned aligned sample data from a sample file or embedded BGEN samples."""
    return engine.align_sample_data(
        str(sample_path) if sample_path is not None else None,
        str(phenotype_path),
        phenotype_name,
        str(covariate_path) if covariate_path is not None else None,
        list(covariate_names) if covariate_names is not None else None,
        is_binary_trait,
        sample_key_mode=resolve_sample_key_mode(alignment_config).value,
    )


def load_native_multi_aligned_sample_data(
    *,
    engine: _core.Regenie2RunEngine,
    sample_path: Path | None,
    phenotype_path: Path,
    phenotype_names: tuple[str, ...],
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    is_binary_trait: bool,
    alignment_config: SampleAlignmentConfigProtocol | None = None,
) -> _core.NativeMultiAlignedSampleData:
    """Load Rust-owned complete-case multi-phenotype sample data."""
    return engine.align_multi_sample_data(
        str(sample_path) if sample_path is not None else None,
        str(phenotype_path),
        list(phenotype_names),
        str(covariate_path) if covariate_path is not None else None,
        list(covariate_names) if covariate_names is not None else None,
        is_binary_trait,
        sample_key_mode=resolve_sample_key_mode(alignment_config).value,
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
    build_native_bgen_run_input_callable: typing.Callable[[_core.NativeAlignedSampleData], NativeBgenRunInput]
    | None = None,
    load_aligned_sample_data_callable: typing.Callable[..., _core.NativeAlignedSampleData] | None = None,
) -> NativeBgenRunInput:
    """Load native-aligned samples and JAX compute inputs for a native BGEN run."""
    source.validate_genotype_source_config(genotype_source_config)
    resolved_sample_path = source.resolve_bgen_sample_path(
        genotype_source_config.source_path,
        genotype_source_config.sample_path,
    )
    resolved_build_native_bgen_run_input = build_native_bgen_run_input_callable or build_native_bgen_run_input
    resolved_load_aligned_sample_data = load_aligned_sample_data_callable or load_native_aligned_sample_data
    native_aligned_sample_data = resolved_load_aligned_sample_data(
        engine=engine,
        sample_path=resolved_sample_path,
        phenotype_path=phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=is_binary_trait,
        alignment_config=alignment_config,
    )
    return resolved_build_native_bgen_run_input(native_aligned_sample_data)


def load_native_bgen_multi_run_input(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    engine: _core.Regenie2RunEngine,
    phenotype_path: Path,
    phenotype_names: tuple[str, ...],
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    is_binary_trait: bool,
    alignment_config: SampleAlignmentConfigProtocol | None = None,
) -> NativeBgenMultiRunInput:
    """Load native complete-case multi-phenotype samples and JAX compute inputs."""
    source.validate_genotype_source_config(genotype_source_config)
    resolved_sample_path = source.resolve_bgen_sample_path(
        genotype_source_config.source_path,
        genotype_source_config.sample_path,
    )
    native_multi_aligned_sample_data = load_native_multi_aligned_sample_data(
        engine=engine,
        sample_path=resolved_sample_path,
        phenotype_path=phenotype_path,
        phenotype_names=phenotype_names,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=is_binary_trait,
        alignment_config=alignment_config,
    )
    return build_native_bgen_multi_run_input(native_multi_aligned_sample_data)


def build_regenie_prediction_source(
    *,
    prediction_list_path: Path,
    phenotype_name: str,
    run_input: NativeBgenRunInput,
    alignment_config: SampleAlignmentConfigProtocol | None = None,
) -> _core.RegeniePredictionSource:
    """Load Rust-owned REGENIE step 1 predictions aligned to the run samples."""
    return _core.RegeniePredictionSource.from_native_aligned_sample_data(
        str(prediction_list_path),
        phenotype_name,
        run_input.native_aligned_sample_data,
        sample_key_mode=resolve_sample_key_mode(alignment_config).value,
    )


def build_multi_regenie_prediction_source(
    *,
    prediction_list_path: Path,
    run_input: NativeBgenMultiRunInput,
    alignment_config: SampleAlignmentConfigProtocol | None = None,
) -> _core.MultiRegeniePredictionSource:
    """Load native multi-trait REGENIE step 1 predictions aligned to shared samples."""
    return _core.MultiRegeniePredictionSource.from_native_multi_aligned_sample_data(
        str(prediction_list_path),
        run_input.native_multi_aligned_sample_data,
        sample_key_mode=resolve_sample_key_mode(alignment_config).value,
    )


def build_bgen_run_engine(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    chunk_size: int,
    variant_limit: int | None,
    trusted_no_missing_diploid: bool = False,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode = types.TrustedBgenValidationMode.CACHE_ON_MISS,
    trusted_bgen_validator: typing.Callable[..., None] | None = None,
) -> _core.Regenie2RunEngine:
    """Open the native BGEN run engine once for alignment and chunk delivery."""
    engine = _core.Regenie2RunEngine(
        str(genotype_source_config.source_path),
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
    )
    if trusted_no_missing_diploid:
        resolved_trusted_bgen_validator = trusted_bgen_validator or trusted_validation.validate_trusted_bgen_with_cache
        resolved_trusted_bgen_validator(
            engine=engine,
            bgen_path=genotype_source_config.source_path,
            validation_mode=trusted_bgen_validation_mode,
        )
    return engine


def finish_callback_drain(
    *,
    callback: object,
    stage_timing_recorder: timing.StageTimingRecorder | None,
) -> None:
    """Wait for queued callback work to drain."""
    callback_finish_start_time = time.perf_counter()
    typing.cast("typing.Any", callback).finish()
    timing.record_stage_duration(stage_timing_recorder, "callback_drain", callback_finish_start_time)


def finish_writer_session(
    *,
    writer_session: typing.Any,
    stage_timing_recorder: timing.StageTimingRecorder | None,
) -> str | None:
    """Finish the writer session and optionally finalize Parquet output."""
    writer_finish_start_time = time.perf_counter()
    final_parquet_path = writer_session.finish()
    timing.record_stage_duration(
        stage_timing_recorder, "writer_finish_and_parquet_finalization", writer_finish_start_time
    )
    return typing.cast("str | None", final_parquet_path)


def finish_writer_session_interrupted(
    *,
    writer_session: typing.Any,
    shutdown_request: shutdown.GracefulShutdownRequested,
    stage_timing_recorder: timing.StageTimingRecorder | None,
) -> None:
    """Flush writer output for an interrupted run without final Parquet."""
    writer_finish_start_time = time.perf_counter()
    writer_session.finish_interrupted(shutdown_request.signal_name)
    timing.record_stage_duration(stage_timing_recorder, "writer_finish_interrupted", writer_finish_start_time)


def abort_callback(callback: object) -> None:
    """Request callback worker shutdown when supported."""
    abort_callback_method = getattr(callback, "abort", None)
    if callable(abort_callback_method):
        with contextlib.suppress(Exception):
            abort_callback_method()


def abort_writer_session(writer_session: typing.Any) -> None:
    """Abort one writer session."""
    with contextlib.suppress(Exception):
        writer_session.abort()


def run_bgen_engine_with_callback(
    *,
    engine: _core.Regenie2RunEngine,
    run_input: NativeBgenRunInput,
    committed_chunk_identifiers: set[int] | None,
    writer_session: typing.Any,
    callback: object,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    variant_major_dosage: bool = False,
    stage_timing_snapshot_writer: typing.Callable[
        [timing.StageTimingRecorder | None, Path | None], None
    ] = timing.write_stage_timing_snapshot,
) -> Path | None:
    """Run native BGEN chunk delivery and close the output writer."""
    callback_finished = False
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
        timing.record_stage_duration(stage_timing_recorder, "native_engine_delivery", engine_delivery_start_time)
        if stage_timing_recorder is not None:
            stage_timing_recorder.set_native_bgen_profile(engine.profile_snapshot())
        finish_callback_drain(callback=callback, stage_timing_recorder=stage_timing_recorder)
        callback_finished = True
        final_parquet_path = finish_writer_session(
            writer_session=writer_session,
            stage_timing_recorder=stage_timing_recorder,
        )
    except shutdown.GracefulShutdownRequested as shutdown_request:
        try:
            if not callback_finished:
                finish_callback_drain(callback=callback, stage_timing_recorder=stage_timing_recorder)
            finish_writer_session_interrupted(
                writer_session=writer_session,
                shutdown_request=shutdown_request,
                stage_timing_recorder=stage_timing_recorder,
            )
        except BaseException:
            abort_callback(callback)
            abort_writer_session(writer_session)
            stage_timing_snapshot_writer(stage_timing_recorder, None)
            raise
        stage_timing_snapshot_writer(stage_timing_recorder, None)
        raise
    except BaseException:
        abort_callback(callback)
        abort_writer_session(writer_session)
        stage_timing_snapshot_writer(stage_timing_recorder, None)
        raise
    stage_timing_snapshot_writer(stage_timing_recorder, None)
    if final_parquet_path is None:
        return None
    return Path(final_parquet_path)
