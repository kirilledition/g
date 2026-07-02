"""Genotype transfer and chunk metadata helpers for callback workers."""

from __future__ import annotations

import time
import typing

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

import g.engine.callbacks.diagnostics as diagnostics
import g.engine.callbacks.shared as shared
from g import _core, types
from g.engine import timing

HostGenotypeBuffer = shared.HostGenotypeBuffer
HostOrDeviceFloatArray = shared.HostOrDeviceFloatArray
LinearChunkStatsArrays = shared.LinearChunkStatsArrays
BinaryChunkStatsArrays = shared.BinaryChunkStatsArrays
PublicStatisticArray = npt.NDArray[np.float32] | npt.NDArray[np.float64]
block_until_ready = diagnostics.block_until_ready
get_metadata_chromosome = shared.get_metadata_chromosome


class TransferMetadataArrayProtocol(typing.Protocol):
    """Array contract required for transfer metadata summaries."""

    shape: typing.Any
    dtype: typing.Any


class ChunkStatsComputeArraysProtocol(typing.Protocol):
    """Native chunk-stat contract for compute-needed arrays."""

    def compute_arrays(
        self,
        *,
        include_imputed_dosage_square_sum: bool,
        include_sparse_firth_candidate: bool,
    ) -> typing.Mapping[str, object]:
        """Return chunk-stat arrays needed by JAX compute paths."""


def put_compute_array_on_device(array: HostOrDeviceFloatArray) -> jax.Array:
    """Place an aligned host/JAX input array on the active JAX device."""
    return typing.cast("jax.Array", jax.device_put(array))


def record_transfer_metadata_for_array(
    *,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    transfer_name: str,
    array_role: str,
    array: TransferMetadataArrayProtocol,
) -> None:
    """Record conservative transfer size metadata when diagnostics are active."""
    if stage_timing_recorder is None:
        return
    try:
        numpy_dtype = np.dtype(array.dtype)
    except TypeError:
        return
    shape_dimensions = tuple(int(dimension) for dimension in array.shape)
    stage_timing_recorder.add_transfer_metadata_for_shape(
        transfer_name=transfer_name,
        array_role=array_role,
        dtype_name=numpy_dtype.name,
        shape_dimensions=shape_dimensions,
        item_size=int(numpy_dtype.itemsize),
    )


def put_genotype_matrix_on_device(
    genotype_matrix: jax.Array | HostGenotypeBuffer,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    chunk_metadata: typing.Any | None,
    *,
    array_role: str,
) -> jax.Array:
    """Transfer a genotype chunk to the active JAX device with optional timing."""
    if stage_timing_recorder is None:
        return typing.cast("jax.Array", jax.device_put(genotype_matrix))
    start_time = time.perf_counter()
    genotype_device_array = jax.device_put(genotype_matrix)
    if timing.should_collect_exact_stage_timings(stage_timing_recorder):
        block_until_ready(genotype_device_array)
    record_stage_duration_with_optional_chunk(
        stage_timing_recorder=stage_timing_recorder,
        stage_name="host_to_device_transfer",
        start_time=start_time,
        chunk_metadata=chunk_metadata,
    )
    record_transfer_metadata_for_array(
        stage_timing_recorder=stage_timing_recorder,
        transfer_name="host_to_device_transfer",
        array_role=array_role,
        array=genotype_matrix,
    )
    return genotype_device_array


def put_chunk_array_on_device(
    array: jax.Array | npt.NDArray[typing.Any],
    stage_timing_recorder: timing.StageTimingRecorder | None,
    chunk_metadata: typing.Any,
    *,
    array_role: str,
) -> jax.Array:
    """Transfer one chunk-scoped array to the active JAX device with timing."""
    if stage_timing_recorder is None:
        return typing.cast("jax.Array", jax.device_put(array))
    start_time = time.perf_counter()
    device_array = jax.device_put(array)
    if timing.should_collect_exact_stage_timings(stage_timing_recorder):
        block_until_ready(device_array)
    record_stage_duration_with_optional_chunk(
        stage_timing_recorder=stage_timing_recorder,
        stage_name="host_to_device_transfer",
        start_time=start_time,
        chunk_metadata=chunk_metadata,
    )
    record_transfer_metadata_for_array(
        stage_timing_recorder=stage_timing_recorder,
        transfer_name="host_to_device_transfer",
        array_role=array_role,
        array=array,
    )
    return device_array


def block_compute_result_for_timing(
    *,
    result_ready_value: jax.Array,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    start_time: float,
    chunk_metadata: typing.Any | None,
) -> None:
    """Synchronize chunk compute only when detailed stage timings are enabled."""
    if stage_timing_recorder is None:
        return
    if timing.should_collect_exact_stage_timings(stage_timing_recorder):
        block_until_ready(result_ready_value)
    record_stage_duration_with_optional_chunk(
        stage_timing_recorder=stage_timing_recorder,
        stage_name="jax_compute",
        start_time=start_time,
        chunk_metadata=chunk_metadata,
    )


def build_chunk_timing_identity(metadata: typing.Any) -> timing.ChunkTimingIdentity:
    """Build per-chunk timing identity fields from native metadata."""
    native_identity = build_native_callback_chunk_identity(metadata)
    return timing.ChunkTimingIdentity(
        chunk_identifier=native_identity.chunk_identifier,
        chromosome=native_identity.chromosome,
        variant_start_index=native_identity.variant_start_index,
        variant_stop_index=native_identity.variant_stop_index,
        variant_count=native_identity.variant_count,
    )


def build_native_callback_chunk_identity(metadata: typing.Any) -> _core.NativeCallbackChunkIdentity:
    """Build the native callback chunk identity from metadata attributes."""
    return _core.build_callback_chunk_identity(
        chromosome=get_metadata_chromosome(metadata),
        variant_start_index=int(metadata.variant_start_index),
        variant_stop_index=int(metadata.variant_stop_index),
    )


def record_stage_duration_with_optional_chunk(
    *,
    stage_timing_recorder: timing.StageTimingRecorder | None,
    stage_name: str,
    start_time: float,
    chunk_metadata: typing.Any | None,
) -> None:
    """Record a stage duration globally and optionally against one chunk."""
    if stage_timing_recorder is None:
        return
    if chunk_metadata is None:
        timing.record_stage_duration(stage_timing_recorder, stage_name, start_time)
        return
    timing.record_chunk_stage_duration(
        stage_timing_recorder,
        chunk_identity=build_chunk_timing_identity(chunk_metadata),
        stage_name=stage_name,
        start_time=start_time,
    )


def resolve_public_statistic_jax_dtype(output_statistic_dtype: types.FloatingPointDtype) -> typing.Any:
    """Resolve the configured public statistic dtype for JAX materialization."""
    if output_statistic_dtype == types.FloatingPointDtype.FLOAT32:
        return jnp.float32
    if output_statistic_dtype == types.FloatingPointDtype.FLOAT64:
        return jnp.float64
    message = f"Unsupported public statistic output dtype: {output_statistic_dtype!s}"
    raise ValueError(message)


def resolve_public_statistic_numpy_dtype(
    output_statistic_dtype: types.FloatingPointDtype,
) -> type[np.float32] | type[np.float64]:
    """Resolve the configured public statistic dtype for NumPy host arrays."""
    if output_statistic_dtype == types.FloatingPointDtype.FLOAT32:
        return np.float32
    if output_statistic_dtype == types.FloatingPointDtype.FLOAT64:
        return np.float64
    message = f"Unsupported public statistic output dtype: {output_statistic_dtype!s}"
    raise ValueError(message)


def narrow_public_statistic_array_on_device(
    array: jax.Array,
    output_statistic_dtype: types.FloatingPointDtype,
) -> jax.Array:
    """Cast public result statistics to the configured native writer dtype before host transfer."""
    return jnp.asarray(array, dtype=resolve_public_statistic_jax_dtype(output_statistic_dtype))


def select_active_trait_rows_on_device(
    array: jax.Array,
    *,
    active_trait_indices: tuple[int, ...],
    total_trait_count: int,
) -> jax.Array:
    """Return active trait rows without materializing inactive traits on host."""
    if len(active_trait_indices) == total_trait_count and active_trait_indices == tuple(range(total_trait_count)):
        return array
    active_trait_index_array = jnp.asarray(active_trait_indices, dtype=jnp.int32)
    return jnp.take(array, active_trait_index_array, axis=0)


def cast_statistic_array_for_native_writer(
    array: object,
    output_statistic_dtype: types.FloatingPointDtype,
) -> PublicStatisticArray:
    """Cast computed statistics to the configured public native writer schema dtype."""
    return typing.cast(
        "PublicStatisticArray",
        np.asarray(array, dtype=resolve_public_statistic_numpy_dtype(output_statistic_dtype)),
    )


def cast_statistic_array_for_native_writer_float32(array: object) -> npt.NDArray[np.float32]:
    """Cast computed statistics to the float32 native writer schema dtype."""
    return typing.cast("npt.NDArray[np.float32]", np.asarray(array, dtype=np.float32))


def cast_statistic_array_for_native_writer_float64(array: object) -> npt.NDArray[np.float64]:
    """Cast computed statistics to the float64 native writer schema dtype."""
    return typing.cast("npt.NDArray[np.float64]", np.asarray(array, dtype=np.float64))


def get_chunk_stats_compute_arrays(
    chunk_stats: ChunkStatsComputeArraysProtocol,
    *,
    include_imputed_dosage_square_sum: bool,
    include_sparse_firth_candidate: bool,
) -> typing.Mapping[str, object]:
    """Return compute-needed native stat arrays through the bundled binding."""
    return chunk_stats.compute_arrays(
        include_imputed_dosage_square_sum=include_imputed_dosage_square_sum,
        include_sparse_firth_candidate=include_sparse_firth_candidate,
    )


def get_linear_chunk_stats_arrays(chunk_stats: ChunkStatsComputeArraysProtocol) -> LinearChunkStatsArrays:
    """Return the native stat arrays needed by linear variant-major compute."""
    compute_arrays = get_chunk_stats_compute_arrays(
        chunk_stats,
        include_imputed_dosage_square_sum=True,
        include_sparse_firth_candidate=False,
    )
    return LinearChunkStatsArrays(
        dosage_sum=typing.cast("npt.NDArray[np.float32]", compute_arrays["dosage_sum"]),
        observation_count=typing.cast("npt.NDArray[np.int32]", compute_arrays["observation_count"]),
        imputed_dosage_square_sum=typing.cast(
            "npt.NDArray[np.float32]",
            compute_arrays["imputed_dosage_square_sum"],
        ),
    )


def get_binary_chunk_stats_arrays(
    chunk_stats: _core.ChunkStats,
    *,
    include_sparse_firth_candidate: bool,
) -> BinaryChunkStatsArrays:
    """Return the native stat arrays needed by binary variant-major compute."""
    compute_arrays = get_chunk_stats_compute_arrays(
        chunk_stats,
        include_imputed_dosage_square_sum=False,
        include_sparse_firth_candidate=include_sparse_firth_candidate,
    )
    sparse_candidate_mask: npt.NDArray[np.bool_] | None = None
    if include_sparse_firth_candidate:
        sparse_candidate_mask = typing.cast(
            "npt.NDArray[np.bool_]",
            compute_arrays["is_rare_sparse_firth_candidate"],
        )
    return BinaryChunkStatsArrays(
        dosage_sum=typing.cast("npt.NDArray[np.float32]", compute_arrays["dosage_sum"]),
        observation_count=typing.cast("npt.NDArray[np.int32]", compute_arrays["observation_count"]),
        sparse_candidate_mask=sparse_candidate_mask,
    )


def build_projected_variant_major_dosage_chunk_stats(
    genotype_matrix_by_variant: npt.NDArray[np.float32],
) -> _core.ChunkStats:
    """Build native chunk stats for a projected variant-major dosage buffer."""
    return _core.summarize_variant_major_dosage_chunk_stats(
        np.ascontiguousarray(genotype_matrix_by_variant, dtype=np.float32)
    )


__all__ = [
    "block_compute_result_for_timing",
    "build_chunk_timing_identity",
    "build_native_callback_chunk_identity",
    "build_projected_variant_major_dosage_chunk_stats",
    "cast_statistic_array_for_native_writer",
    "cast_statistic_array_for_native_writer_float32",
    "cast_statistic_array_for_native_writer_float64",
    "get_binary_chunk_stats_arrays",
    "get_chunk_stats_compute_arrays",
    "get_linear_chunk_stats_arrays",
    "narrow_public_statistic_array_on_device",
    "put_chunk_array_on_device",
    "put_compute_array_on_device",
    "put_genotype_matrix_on_device",
    "record_stage_duration_with_optional_chunk",
    "record_transfer_metadata_for_array",
    "select_active_trait_rows_on_device",
]
