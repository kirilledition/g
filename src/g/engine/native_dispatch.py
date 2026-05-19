"""Native BGEN engine dispatch helpers for REGENIE step 2 pipelines."""

from __future__ import annotations

import time
import typing
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

import g._core as core
import g.engine.timing as timing
import g.engine.trusted_validation as trusted_validation
import g.io.source as source
import g.types as g_types

StageTimingRecorder = timing.StageTimingRecorder
record_stage_duration = timing.record_stage_duration


class SampleAlignmentConfigProtocol(typing.Protocol):
    """Sample identity alignment settings accepted by native dispatch."""

    sample_key_mode: g_types.SampleKeyMode
    allow_duplicate_iid_alignment: bool


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

    native_aligned_sample_data: core.NativeAlignedSampleData
    sample_indices: npt.NDArray[np.int64]
    phenotype_vector: jax.Array
    covariate_matrix: jax.Array
    is_binary_trait: bool


def build_native_bgen_run_input(
    native_aligned_sample_data: core.NativeAlignedSampleData,
) -> NativeBgenRunInput:
    """Build Python/JAX views over Rust-owned aligned sample data."""
    return NativeBgenRunInput(
        native_aligned_sample_data=native_aligned_sample_data,
        sample_indices=np.ascontiguousarray(native_aligned_sample_data.sample_indices, dtype=np.int64),
        phenotype_vector=jnp.asarray(native_aligned_sample_data.phenotype_vector, dtype=jnp.float32),
        covariate_matrix=jnp.asarray(native_aligned_sample_data.covariate_matrix, dtype=jnp.float32),
        is_binary_trait=native_aligned_sample_data.is_binary_trait,
    )


def resolve_sample_key_mode(alignment_config: SampleAlignmentConfigProtocol | None) -> g_types.SampleKeyMode:
    """Resolve the sample key mode for native calls."""
    if alignment_config is None:
        return g_types.SampleKeyMode.IID
    return alignment_config.sample_key_mode


def resolve_allow_duplicate_iid_alignment(alignment_config: SampleAlignmentConfigProtocol | None) -> bool:
    """Resolve whether duplicate-IID compatibility alignment is enabled."""
    if alignment_config is None:
        return False
    return alignment_config.allow_duplicate_iid_alignment


def load_native_aligned_sample_data(
    *,
    engine: core.Regenie2RunEngine,
    sample_path: Path | None,
    phenotype_path: Path,
    phenotype_name: str,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    is_binary_trait: bool,
    alignment_config: SampleAlignmentConfigProtocol | None = None,
) -> core.NativeAlignedSampleData:
    """Load Rust-owned aligned sample data from a sample file or embedded BGEN samples."""
    return engine.align_sample_data(
        str(sample_path) if sample_path is not None else None,
        str(phenotype_path),
        phenotype_name,
        str(covariate_path) if covariate_path is not None else None,
        list(covariate_names) if covariate_names is not None else None,
        is_binary_trait,
        sample_key_mode=resolve_sample_key_mode(alignment_config).value,
        allow_duplicate_iid_alignment=resolve_allow_duplicate_iid_alignment(alignment_config),
    )


def load_native_bgen_run_input(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    engine: core.Regenie2RunEngine,
    phenotype_path: Path,
    phenotype_name: str,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    is_binary_trait: bool,
    alignment_config: SampleAlignmentConfigProtocol | None = None,
    build_native_bgen_run_input_callable: typing.Callable[
        [core.NativeAlignedSampleData], NativeBgenRunInput
    ] = build_native_bgen_run_input,
    load_aligned_sample_data_callable: typing.Callable[
        ..., core.NativeAlignedSampleData
    ] = load_native_aligned_sample_data,
) -> NativeBgenRunInput:
    """Load native-aligned samples and JAX compute inputs for a native BGEN run."""
    source.validate_genotype_source_config(genotype_source_config)
    resolved_sample_path = source.resolve_bgen_sample_path(
        genotype_source_config.source_path,
        genotype_source_config.sample_path,
    )
    native_aligned_sample_data = load_aligned_sample_data_callable(
        engine=engine,
        sample_path=resolved_sample_path,
        phenotype_path=phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=is_binary_trait,
        alignment_config=alignment_config,
    )
    return build_native_bgen_run_input_callable(native_aligned_sample_data)


def build_regenie_prediction_source(
    *,
    prediction_list_path: Path,
    phenotype_name: str,
    run_input: NativeBgenRunInput,
    alignment_config: SampleAlignmentConfigProtocol | None = None,
) -> core.RegeniePredictionSource:
    """Load Rust-owned REGENIE step 1 predictions aligned to the run samples."""
    return core.RegeniePredictionSource.from_native_aligned_sample_data(
        str(prediction_list_path),
        phenotype_name,
        run_input.native_aligned_sample_data,
        sample_key_mode=resolve_sample_key_mode(alignment_config).value,
        allow_duplicate_iid_alignment=resolve_allow_duplicate_iid_alignment(alignment_config),
    )


def build_bgen_run_engine(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    chunk_size: int,
    variant_limit: int | None,
    trusted_no_missing_diploid: bool = False,
    trusted_bgen_validation_mode: g_types.TrustedBgenValidationMode = g_types.TrustedBgenValidationMode.CACHE_ON_MISS,
    trusted_bgen_validator: typing.Callable[..., None] = trusted_validation.validate_trusted_bgen_with_cache,
) -> core.Regenie2RunEngine:
    """Open the native BGEN run engine once for alignment and chunk delivery."""
    engine = core.Regenie2RunEngine(
        str(genotype_source_config.source_path),
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
    )
    if trusted_no_missing_diploid:
        trusted_bgen_validator(
            engine=engine,
            bgen_path=genotype_source_config.source_path,
            validation_mode=trusted_bgen_validation_mode,
        )
    return engine


def run_bgen_engine_with_callback(
    *,
    engine: core.Regenie2RunEngine,
    run_input: NativeBgenRunInput,
    committed_chunk_identifiers: set[int] | None,
    writer_session: typing.Any,
    callback: object,
    stage_timing_recorder: StageTimingRecorder | None,
    variant_major_dosage: bool = False,
    stage_timing_snapshot_writer: typing.Callable[
        [StageTimingRecorder | None], None
    ] = timing.write_stage_timing_snapshot_from_environment,
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
        stage_timing_snapshot_writer(stage_timing_recorder)
        raise
    stage_timing_snapshot_writer(stage_timing_recorder)
    if final_parquet_path is None:
        return None
    return Path(final_parquet_path)
