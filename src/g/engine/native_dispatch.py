"""Native BGEN engine dispatch helpers for REGENIE step 2 pipelines."""

from __future__ import annotations

import contextlib
import logging
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

logger = logging.getLogger(__name__)


class SampleAlignmentConfigProtocol(typing.Protocol):
    """Sample identity alignment settings accepted by native dispatch."""

    sample_key_mode: types.SampleKeyMode


class MultiRegeniePredictionSourceProtocol(typing.Protocol):
    """Prediction source interface used by grouped native run inputs."""

    def get_chromosome_predictions(self, chromosome: str) -> npt.NDArray[np.float32]:
        """Return trait-major LOCO predictions for one chromosome."""
        ...


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
    """Sample-aligned inputs for an opt-in complete-case multi-phenotype native BGEN run.

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


@dataclass(frozen=True)
class NativeBgenGroupedRunInput:
    """One native-planned group of compatible per-phenotype run inputs.

    Attributes:
        phenotype_indices: Original phenotype indices included in this group.
        run_input: Multi-trait run input for the compatible phenotype group.
        prediction_source: Native multi-trait prediction source aligned to the group.

    """

    phenotype_indices: tuple[int, ...]
    run_input: NativeBgenMultiRunInput
    prediction_source: MultiRegeniePredictionSourceProtocol


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


def build_native_bgen_grouped_run_inputs(
    native_grouped_aligned_sample_data: _core.NativeGroupedAlignedSampleData,
    prediction_sources: list[_core.MultiRegeniePredictionSource],
) -> tuple[NativeBgenGroupedRunInput, ...]:
    """Build Python/JAX views over native grouped per-phenotype alignment data."""
    if len(native_grouped_aligned_sample_data.groups) != len(prediction_sources):
        message = (
            "Grouped prediction source count does not match grouped aligned sample data count: "
            f"{len(prediction_sources)} prediction source(s), "
            f"{len(native_grouped_aligned_sample_data.groups)} aligned group(s)."
        )
        raise ValueError(message)
    grouped_run_inputs: list[NativeBgenGroupedRunInput] = []
    for native_group, prediction_source in zip(
        native_grouped_aligned_sample_data.groups,
        prediction_sources,
        strict=True,
    ):
        grouped_run_inputs.append(
            NativeBgenGroupedRunInput(
                phenotype_indices=tuple(int(phenotype_index) for phenotype_index in native_group.phenotype_indices),
                run_input=build_native_bgen_multi_run_input(native_group.aligned_sample_data),
                prediction_source=prediction_source,
            )
        )
    return tuple(grouped_run_inputs)


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


def load_native_bgen_grouped_run_inputs(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    engine: _core.Regenie2RunEngine,
    phenotype_path: Path,
    phenotype_names: tuple[str, ...],
    prediction_list_path: Path,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    is_binary_trait: bool,
    alignment_config: SampleAlignmentConfigProtocol | None = None,
) -> tuple[NativeBgenGroupedRunInput, ...]:
    """Load native grouped per-phenotype samples and JAX compute inputs."""
    source.validate_genotype_source_config(genotype_source_config)
    resolved_sample_path = source.resolve_bgen_sample_path(
        genotype_source_config.source_path,
        genotype_source_config.sample_path,
    )
    native_grouped_aligned_sample_data = engine.align_grouped_sample_data(
        str(resolved_sample_path) if resolved_sample_path is not None else None,
        str(phenotype_path),
        list(phenotype_names),
        str(covariate_path) if covariate_path is not None else None,
        list(covariate_names) if covariate_names is not None else None,
        is_binary_trait,
        sample_key_mode=resolve_sample_key_mode(alignment_config).value,
    )
    prediction_sources = _core.MultiRegeniePredictionSource.from_native_grouped_aligned_sample_data(
        str(prediction_list_path),
        native_grouped_aligned_sample_data,
        sample_key_mode=resolve_sample_key_mode(alignment_config).value,
    )
    return build_native_bgen_grouped_run_inputs(native_grouped_aligned_sample_data, prediction_sources)


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
    logger.debug("Constructing native BGEN run engine.")
    engine = _core.Regenie2RunEngine(
        str(genotype_source_config.source_path),
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
    )
    if trusted_no_missing_diploid:
        logger.debug("Validating trusted no-missing diploid BGEN mode.")
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
    logger.debug("Draining native callback worker queues.")
    typing.cast("typing.Any", callback).finish()
    timing.record_stage_duration(stage_timing_recorder, "callback_drain", callback_finish_start_time)


def finish_writer_session(
    *,
    writer_session: typing.Any,
    stage_timing_recorder: timing.StageTimingRecorder | None,
) -> str | None:
    """Finish the writer session and optionally finalize Parquet output."""
    writer_finish_start_time = time.perf_counter()
    logger.debug("Finishing output writer and optional Parquet finalization.")
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
    logger.info("Flushing interrupted output writer after %s.", shutdown_request.signal_name)
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
    variant_major_packed8_probability_pairs: bool = False,
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
        logger.debug(
            "Starting native BGEN delivery: committed_chunk_count=%s variant_major_packed8_probability_pairs=%s.",
            len(committed_chunk_identifier_list),
            variant_major_packed8_probability_pairs,
        )
        if variant_major_packed8_probability_pairs:
            processed_chunk_count = engine.run_bgen_variant_major_packed8_probability_pair_buffered_chunks(
                sample_indices,
                callback,
                committed_chunk_identifiers=committed_chunk_identifier_list,
            )
        else:
            processed_chunk_count = engine.run_bgen_variant_major_dosage_buffered_chunks(
                sample_indices,
                callback,
                committed_chunk_identifiers=committed_chunk_identifier_list,
            )
        timing.record_stage_duration(stage_timing_recorder, "native_engine_delivery", engine_delivery_start_time)
        logger.debug("Native BGEN delivery finished: processed_chunk_count=%s.", processed_chunk_count)
        if stage_timing_recorder is not None:
            stage_timing_recorder.set_native_bgen_profile(engine.profile_snapshot())
        finish_callback_drain(callback=callback, stage_timing_recorder=stage_timing_recorder)
        callback_finished = True
        final_parquet_path = finish_writer_session(
            writer_session=writer_session,
            stage_timing_recorder=stage_timing_recorder,
        )
    except shutdown.GracefulShutdownRequested as shutdown_request:
        logger.info("Native BGEN delivery interrupted by %s.", shutdown_request.signal_name)
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
        logger.exception("Native BGEN delivery failed.")
        abort_callback(callback)
        abort_writer_session(writer_session)
        stage_timing_snapshot_writer(stage_timing_recorder, None)
        raise
    stage_timing_snapshot_writer(stage_timing_recorder, None)
    logger.info("Native BGEN pipeline finished.")
    if final_parquet_path is None:
        return None
    return Path(final_parquet_path)
